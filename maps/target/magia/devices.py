"""Concrete tile-local Devices owned by the MAGIA target."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import ceil
from typing import Any

from maps.graph import TensorDType
from maps.hardware import (
    DMADevice,
    DMAJob,
    DeviceKind,
    FixedDeviceAssignment,
    ScalarDevice,
    SystolicDevice,
    VectorDevice,
    WorkKind,
    WorkSignature,
    same_dtype_signatures,
)

L1_CORE_TRANSFER_LATENCY = 4
REDMULE_ARRAY_WIDTH = 24
REDMULE_ARRAY_HEIGHT = 8

SPATZ_LANES = 4
SPATZ_LANE_WIDTH_BYTES = 4
SPATZ_VLEN_BITS = 512
SPATZ_FREQUENCY_HZ = 200_000_000
SPATZ_VLSU_PORTS = 4
SPATZ_TCDM_BANKS = 32
SPATZ_TCDM_BANK_WIDTH_BYTES = 4
SPATZ_VLSU_SWITCH_SAME_CYCLES = 2
SPATZ_VLSU_SWITCH_STORE_TO_LOAD_CYCLES = 3
SPATZ_VLSU_SWITCH_LOAD_TO_STORE_CYCLES = 7

_FLOAT_DTYPES = (TensorDType.FLOAT16, TensorDType.FLOAT32)
_UNARY_CORE_WORK = (
    WorkKind.ALL_REDUCE_MAX,
    WorkKind.ALL_REDUCE_SUM,
    WorkKind.ABS,
    WorkKind.EXP,
    WorkKind.GROUP_REDUCE,
    WorkKind.IM2COL,
    WorkKind.LOG,
    WorkKind.NEG,
    WorkKind.OUTPUT_REFORMAT,
    WorkKind.RELU,
    WorkKind.REDUCE_MAX,
    WorkKind.REDUCE_SUM,
    WorkKind.RESHAPE,
    WorkKind.SIGMOID,
    WorkKind.SLICE,
    WorkKind.SQRT,
    WorkKind.TRANSPOSE,
)
_BINARY_CORE_WORK = (
    WorkKind.ADD,
    WorkKind.DIV,
    WorkKind.MUL,
    WorkKind.POW,
    WorkKind.SUB,
)


_CORE_CAPABILITIES = (
    same_dtype_signatures(_UNARY_CORE_WORK, (1,), _FLOAT_DTYPES)
    | same_dtype_signatures(_BINARY_CORE_WORK, (2,), _FLOAT_DTYPES)
    | same_dtype_signatures((WorkKind.MUL,), (1,), _FLOAT_DTYPES)
    | same_dtype_signatures((WorkKind.DEPTHWISE_CONV,), (2, 3), _FLOAT_DTYPES)
    | same_dtype_signatures((WorkKind.GROUP_NORMALIZE,), (5,), _FLOAT_DTYPES)
    | frozenset(
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT32,) * input_count,
            (TensorDType.FLOAT32,),
        )
        for input_count in (2, 3)
    )
)

IDMA_READ_DEVICE = DMADevice(
    name="idma_read",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.READJOB,
    burst_bytes=4,
)
IDMA_WRITE_DEVICE = DMADevice(
    name="idma_write",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.WRITEJOB,
    burst_bytes=8,
)
CORE_DEVICE = ScalarDevice(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.ALL_REDUCE_MAX: 1,
        WorkKind.ALL_REDUCE_SUM: 1,
        WorkKind.GEMM: 1,
        WorkKind.GROUP_NORMALIZE: 1,
        WorkKind.GROUP_REDUCE: 1,
        WorkKind.ABS: 1,
        WorkKind.ADD: 1 / (1 + 3 * L1_CORE_TRANSFER_LATENCY),
        WorkKind.DIV: 1,
        WorkKind.CONV2D: 1,
        WorkKind.DEPTHWISE_CONV: 1,
        WorkKind.EXP: 1,
        WorkKind.LOG: 1 / 176,
        WorkKind.MUL: 1,
        WorkKind.NEG: 1,
        WorkKind.POW: 1,
        WorkKind.RELU: 1,
        WorkKind.REDUCE_MAX: 1,
        WorkKind.REDUCE_SUM: 1,
        WorkKind.RESHAPE: 1,
        WorkKind.SIGMOID: 1,
        WorkKind.SLICE: 1,
        WorkKind.SQRT: 1,
        WorkKind.SUB: 1,
        WorkKind.TRANSPOSE: 1,
        WorkKind.IM2COL: 1,
        WorkKind.OUTPUT_REFORMAT: 1,
    },
    capabilities=_CORE_CAPABILITIES,
)
REDMULE_DEVICE = SystolicDevice(
    name="redmule",
    kind=DeviceKind.SYSTOLIC,
    throughput={WorkKind.GEMM: REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT},
    capabilities=frozenset(
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT16,) * input_count,
            (TensorDType.FLOAT16,),
        )
        for input_count in (2, 3)
    ),
    array_width=REDMULE_ARRAY_WIDTH,
    array_height=REDMULE_ARRAY_HEIGHT,
)


@dataclass(frozen=True)
class _KernelProfile:
    compute_passes: int
    lmul: int = 8
    reduction: bool = False


class _MemoryAction(Enum):
    LOAD = auto()
    STORE = auto()


_KERNEL_PROFILES = {
    WorkKind.ADD: _KernelProfile(1),
    WorkKind.SUB: _KernelProfile(1),
    WorkKind.DIV: _KernelProfile(1),
    WorkKind.RELU: _KernelProfile(1),
    WorkKind.EXP: _KernelProfile(5),
    WorkKind.SIGMOID: _KernelProfile(6),
    WorkKind.REDUCE_SUM: _KernelProfile(1, reduction=True),
    WorkKind.REDUCE_MAX: _KernelProfile(1, reduction=True),
}
_CAST_PROFILE = _KernelProfile(1)
_BINARY_SPATZ_WORK = frozenset({WorkKind.ADD, WorkKind.SUB, WorkKind.DIV})


def _spatz_capabilities() -> frozenset[WorkSignature]:
    capabilities = {
        WorkSignature(
            work_kind,
            (dtype, dtype) if work_kind in _BINARY_SPATZ_WORK else (dtype,),
            (dtype,),
        )
        for work_kind in _KERNEL_PROFILES
        for dtype in _FLOAT_DTYPES
    }
    capabilities.update(
        {
            WorkSignature(WorkKind.CAST, (source,), (target,))
            for source, target in (
                (TensorDType.FLOAT16, TensorDType.FLOAT32),
                (TensorDType.FLOAT32, TensorDType.FLOAT16),
            )
        }
    )
    return frozenset(capabilities)


@dataclass(frozen=True)
class SpatzDevice(VectorDevice):
    """Conservative serial cost model for a tile-local Spatz kernel."""

    lanes: int = SPATZ_LANES
    lane_width_bytes: int = SPATZ_LANE_WIDTH_BYTES
    vlen_bits: int = SPATZ_VLEN_BITS
    frequency_hz: int = SPATZ_FREQUENCY_HZ
    vlsu_ports: int = SPATZ_VLSU_PORTS
    tcdm_banks: int = SPATZ_TCDM_BANKS
    tcdm_bank_width_bytes: int = SPATZ_TCDM_BANK_WIDTH_BYTES
    vlsu_switch_same: int = SPATZ_VLSU_SWITCH_SAME_CYCLES
    vlsu_switch_store_to_load: int = SPATZ_VLSU_SWITCH_STORE_TO_LOAD_CYCLES
    vlsu_switch_load_to_store: int = SPATZ_VLSU_SWITCH_LOAD_TO_STORE_CYCLES

    def cycles(self, work: Any) -> int:
        work_kind = work.work_kind
        profile = _CAST_PROFILE if work_kind is WorkKind.CAST else _KERNEL_PROFILES.get(work_kind)
        if profile is None:
            raise ValueError(f"device {self.name} does not support {work_kind.name} work")
        elem_bytes = work.output_slices[0].tensor.elem_bytes
        elements_per_cycle = (self.lanes * self.lane_width_bytes) // elem_bytes
        compute_cycles = profile.compute_passes * ceil(
            work.operation_count() / elements_per_cycle
        )
        if profile.reduction:
            compute_cycles *= 3
        actions = self._memory_actions(work, profile)
        memory_cycles = sum(
            ceil(num_bytes / (self.vlsu_ports * self.lane_width_bytes))
            for _, num_bytes in actions
        )
        switch_cycles = sum(
            self._switch_cycles(previous, current)
            for (previous, _), (current, _) in zip(actions, actions[1:])
        )
        return self.startup_cycles + compute_cycles + memory_cycles + switch_cycles

    def _memory_actions(
        self,
        work: Any,
        profile: _KernelProfile,
    ) -> tuple[tuple[_MemoryAction, int], ...]:
        max_instruction_bytes = (self.vlen_bits // 8) * profile.lmul
        streams = tuple(
            (_MemoryAction.LOAD, ref.num_bytes) for ref in work.input_slices
        )
        streams += tuple(
            (_MemoryAction.STORE, ref.num_bytes) for ref in work.output_slices
        )
        strip_mined_streams = []
        for direction, num_bytes in streams:
            remaining = num_bytes
            instructions = []
            while remaining:
                instruction_bytes = min(remaining, max_instruction_bytes)
                instructions.append((direction, instruction_bytes))
                remaining -= instruction_bytes
            strip_mined_streams.append(tuple(instructions))
        actions = []
        instruction_count = max((len(stream) for stream in strip_mined_streams), default=0)
        for instruction_index in range(instruction_count):
            for stream in strip_mined_streams:
                if instruction_index < len(stream):
                    actions.append(stream[instruction_index])
        return tuple(actions)

    def _switch_cycles(
        self,
        previous: _MemoryAction,
        current: _MemoryAction,
    ) -> int:
        if previous == current:
            return self.vlsu_switch_same
        if previous is _MemoryAction.STORE:
            return self.vlsu_switch_store_to_load
        return self.vlsu_switch_load_to_store


SPATZ_DEVICE = SpatzDevice(
    name="spatz",
    kind=DeviceKind.VECTOR,
    throughput={work_kind: 1 for work_kind in (*_KERNEL_PROFILES, WorkKind.CAST)},
    capabilities=_spatz_capabilities(),
)
TILE_DEVICES = (
    IDMA_READ_DEVICE,
    IDMA_WRITE_DEVICE,
    CORE_DEVICE,
    SPATZ_DEVICE,
    REDMULE_DEVICE,
)
DEVICE_ASSIGNMENT = FixedDeviceAssignment(
    {
        signature: device.name
        for device in (REDMULE_DEVICE, CORE_DEVICE, SPATZ_DEVICE)
        for signature in device.capabilities
    }
)

__all__ = [
    "CORE_DEVICE",
    "DEVICE_ASSIGNMENT",
    "IDMA_READ_DEVICE",
    "IDMA_WRITE_DEVICE",
    "REDMULE_DEVICE",
    "SPATZ_DEVICE",
    "SpatzDevice",
    "TILE_DEVICES",
]
