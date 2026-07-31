"""Spatz vector device model derived from the Magia GVSOC configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from MAPS.arch import DeviceKind, VectorDevice, WorkKind, WorkSignature
from MAPS.core.dtype import TensorDType

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


@dataclass(frozen=True)
class _KernelProfile:
    compute_passes: int
    lmul: int = 8
    reduction: bool = False


_KERNEL_PROFILES = {
    WorkKind.ADD: _KernelProfile(compute_passes=1),
    WorkKind.SUB: _KernelProfile(compute_passes=1),
    WorkKind.DIV: _KernelProfile(compute_passes=1),
    WorkKind.RELU: _KernelProfile(compute_passes=1),
    WorkKind.EXP: _KernelProfile(compute_passes=5),
    WorkKind.SIGMOID: _KernelProfile(compute_passes=6),
    WorkKind.REDUCE_SUM: _KernelProfile(compute_passes=1, reduction=True),
    WorkKind.REDUCE_MAX: _KernelProfile(compute_passes=1, reduction=True),
}

_BINARY_WORK_KINDS = frozenset({WorkKind.ADD, WorkKind.SUB, WorkKind.DIV})


def _spatz_capabilities() -> frozenset[WorkSignature]:
    capabilities = {
        WorkSignature(
            work_kind=work_kind,
            input_dtypes=(dtype, dtype) if work_kind in _BINARY_WORK_KINDS else (dtype,),
            output_dtypes=(dtype,),
        )
        for work_kind in _KERNEL_PROFILES
        for dtype in (TensorDType.FLOAT16, TensorDType.FLOAT32)
    }
    capabilities.update(
        {
            WorkSignature(
                work_kind=WorkKind.CAST,
                input_dtypes=(TensorDType.FLOAT16,),
                output_dtypes=(TensorDType.FLOAT32,),
            ),
            WorkSignature(
                work_kind=WorkKind.CAST,
                input_dtypes=(TensorDType.FLOAT32,),
                output_dtypes=(TensorDType.FLOAT16,),
            ),
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

    def cycles(self, work: object) -> int:
        work_kind = work.work_kind
        profile = _KERNEL_PROFILES.get(work_kind)
        if profile is None:
            raise ValueError(f"device {self.name} does not support {work_kind.name} work")

        elem_bytes = work.output_slices[0].tensor.elem_bytes
        elements_per_cycle = (self.lanes * self.lane_width_bytes) // elem_bytes
        if elements_per_cycle == 0:
            raise ValueError(
                f"device {self.name} does not support {elem_bytes}-byte elements"
            )
        compute_chunks = ceil(work.operation_count() / elements_per_cycle)
        compute_cycles = profile.compute_passes * compute_chunks
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
        work: object,
        profile: _KernelProfile,
    ) -> tuple[tuple[str, int], ...]:
        max_instruction_bytes = (self.vlen_bits // 8) * profile.lmul
        streams = tuple(("load", ref.num_bytes) for ref in work.input_slices)
        streams += tuple(("store", ref.num_bytes) for ref in work.output_slices)

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

    def _switch_cycles(self, previous: str, current: str) -> int:
        if previous == current:
            return self.vlsu_switch_same
        if previous == "store":
            return self.vlsu_switch_store_to_load
        return self.vlsu_switch_load_to_store


SPATZ_DEVICE = SpatzDevice(
    name="spatz",
    kind=DeviceKind.VECTOR,
    throughput={work_kind: 1 for work_kind in _KERNEL_PROFILES},
    capabilities=_spatz_capabilities(),
)
