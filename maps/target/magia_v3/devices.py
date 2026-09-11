"""Concrete tile-local Devices owned by the MAGIA-v3 Target."""

from dataclasses import dataclass, replace
from math import ceil
from types import MappingProxyType
from typing import Any, Mapping

from maps.graph import TensorDType
from maps.hardware import (
    CollectiveCost,
    FixedDeviceAssignment,
    Tile,
    WorkKind,
    WorkSignature,
)
from maps.target.magia.devices import (
    CORE_DEVICE as MAGIA_V2_CORE_DEVICE,
    IDMA_READ_DEVICE,
    IDMA_WRITE_DEVICE as MAGIA_V2_IDMA_WRITE_DEVICE,
    REDMULE_DEVICE as MAGIA_V2_REDMULE_DEVICE,
    SPATZ_DEVICE as MAGIA_V2_SPATZ_DEVICE,
    SpatzDevice,
)

_FP16_IM2COL = WorkSignature(
    WorkKind.IM2COL,
    (TensorDType.FLOAT16,),
    (TensorDType.FLOAT16,),
)
_FP16_BITS = 16
_SPATZ_KERNEL_LMUL = 8

IDMA_WRITE_DEVICE = replace(
    MAGIA_V2_IDMA_WRITE_DEVICE,
    throughput={**MAGIA_V2_IDMA_WRITE_DEVICE.throughput, WorkKind.IM2COL: 4.0},
    startup_cycles=1_800,
    capabilities=frozenset({_FP16_IM2COL}),
)


@dataclass(frozen=True)
class MagiaV3SpatzDevice(SpatzDevice):
    """GVSoC-calibrated timing for the MAGIA-v3 tile-local SDK tasks."""

    kernel_startup_cycles: Mapping[WorkKind, int] = MappingProxyType({})
    gemm_loop_cycles: float = 0.0
    mul_vector_block_cycles: tuple[float, float] = (0.0, 0.0)
    broadcast_vector_block_cycles: Mapping[WorkKind, float] = MappingProxyType({})
    broadcast_scalar_element_cycles: Mapping[WorkKind, float] = MappingProxyType({})
    reduction_output_cycles: float = 0.0

    @staticmethod
    def _broadcast_geometry(work: Any) -> tuple[int, int, bool]:
        output = work.output_slices[0].tensor_slice
        output_elements = output.num_elements
        broadcast_inputs = tuple(
            ref
            for ref in work.input_slices
            if ref.tensor_slice.num_elements < output_elements
        )
        if not broadcast_inputs:
            return 1, output_elements, False

        broadcast = broadcast_inputs[0].tensor_slice
        if broadcast.dims[-1].length == 1 and all(
            broadcast_dim.length == output_dim.length
            for broadcast_dim, output_dim in zip(
                broadcast.dims[:-1], output.dims[:-1]
            )
        ):
            return broadcast.num_elements, output.dims[-1].length, True

        row_len = 1
        for broadcast_dim, output_dim in zip(
            reversed(broadcast.dims), reversed(output.dims)
        ):
            if broadcast_dim.length != output_dim.length:
                break
            row_len *= output_dim.length
        return output_elements // row_len, row_len, False

    def cycles(self, work: Any) -> int:
        work_kind = work.work_kind
        if work_kind not in self.kernel_startup_cycles:
            return super().cycles(work)
        operation_count = work.operation_count()
        if operation_count == 0:
            return 0
        throughput = self.throughput[work_kind]
        if work_kind is WorkKind.GEMM:
            batch_size, m_size, n_size, k_size = work.dimensions()
            vector_elements = (
                self.vlen_bits // _FP16_BITS
            ) * _SPATZ_KERNEL_LMUL
            vector_blocks = ceil(n_size / vector_elements)
            loop_work = batch_size * m_size * k_size * vector_blocks
            return self.kernel_startup_cycles[work_kind] + ceil(
                self.gemm_loop_cycles * loop_work + operation_count / throughput
            )
        elif work_kind in {WorkKind.MUL, WorkKind.SUB, WorkKind.DIV}:
            rows, row_len, scalar_broadcast = self._broadcast_geometry(work)
            if row_len % 2 != 0:
                return self.kernel_startup_cycles[work_kind] + ceil(
                    self.broadcast_scalar_element_cycles[work_kind]
                    * operation_count
                )

            vector_elements = (
                self.vlen_bits // _FP16_BITS
            ) * _SPATZ_KERNEL_LMUL
            vector_blocks = rows * ceil(row_len / vector_elements)
            if work_kind is WorkKind.MUL:
                block_cycles = self.mul_vector_block_cycles[scalar_broadcast]
            else:
                block_cycles = self.broadcast_vector_block_cycles[work_kind]
            return self.kernel_startup_cycles[work_kind] + ceil(
                block_cycles * vector_blocks
            )
        elif work_kind is WorkKind.REDUCE_SUM:
            output_elements = work.output_slices[0].tensor_slice.num_elements
            return self.kernel_startup_cycles[work_kind] + ceil(
                operation_count / throughput
                + self.reduction_output_cycles * output_elements
            )
        return self.kernel_startup_cycles[work_kind] + ceil(
            operation_count / throughput
        )

    def collective_cycles(
        self,
        work_kind: WorkKind,
        element_count: int,
        participants: tuple[Tile, ...],
    ) -> int:
        runtime_overhead = 650
        if len(participants) <= 1:
            return runtime_overhead
        return runtime_overhead + super().collective_cycles(
            work_kind, element_count, participants
        )


# Fitted to warm tile-local task timings from the checked-in 4x4 MobileViT GVSoC
# calibration, subtracting event_imiss stalls during analysis. Communication-bearing
# IM2COL and collective costs remain separate.
_KERNEL_STARTUP_CYCLES = MappingProxyType(
    {
        WorkKind.GEMM: 3_500,
        WorkKind.ADD: 2_400,
        WorkKind.MUL: 2_200,
        WorkKind.SUB: 2_818,
        WorkKind.DIV: 3_156,
        WorkKind.RELU: 2_200,
        WorkKind.SOFTMAX_EXP: 2_500,
        WorkKind.GROUP_REDUCE: 2_100,
        WorkKind.GROUP_CENTERED_REDUCE: 2_100,
        WorkKind.GROUP_NORMALIZE: 2_500,
        WorkKind.REDUCE_SUM: 2_060,
        WorkKind.REDUCE_MAX: 2_200,
    }
)

SPATZ_DEVICE = MagiaV3SpatzDevice(
    name=MAGIA_V2_SPATZ_DEVICE.name,
    kind=MAGIA_V2_SPATZ_DEVICE.kind,
    throughput={
        **MAGIA_V2_SPATZ_DEVICE.throughput,
        WorkKind.GEMM: 7.45,
        WorkKind.ADD: 2.25,
        WorkKind.MUL: 3.0,
        WorkKind.SUB: 0.196,
        WorkKind.DIV: 0.1135,
        WorkKind.RELU: 5.10,
        WorkKind.SOFTMAX_EXP: 0.107,
        WorkKind.GROUP_REDUCE: 4.0,
        WorkKind.GROUP_CENTERED_REDUCE: 1.484,
        WorkKind.GROUP_NORMALIZE: 0.583,
        WorkKind.REDUCE_SUM: 1 / 7.4,
        WorkKind.REDUCE_MAX: 0.0922,
    },
    startup_cycles=0,
    vlen_bits=256,
    capabilities=frozenset(),
    collective_costs={
        WorkKind.ALL_REDUCE_SUM: CollectiveCost(
            participant_rounds=2,
            hop_cycles=1,
        ),
        WorkKind.ALL_REDUCE_MAX: CollectiveCost(
            participant_rounds=2,
            hop_cycles=1,
        ),
    },
    kernel_startup_cycles=_KERNEL_STARTUP_CYCLES,
    # The task's GEMM loop has both fixed scalar/load work for each MxK
    # iteration and per-lane FMA work. It repeats both for every LMUL=8
    # output-vector block.
    gemm_loop_cycles=5.8,
    # The broadcast tasks execute one vector loop per row and vector block. Short
    # rows therefore cost almost as much as a full LMUL=8 vector. Tuple entries
    # distinguish row-vector and per-row-scalar broadcast modes.
    mul_vector_block_cycles=(37.6, 20.3),
    broadcast_vector_block_cycles=MappingProxyType(
        {
            WorkKind.SUB: 77.33,
            WorkKind.DIV: 52.0,
        }
    ),
    # Odd row lengths use the SDK's scalar fallback to avoid a Spatz VLSU
    # alignment bug. MUL has a much cheaper scalar instruction path than the
    # software-expanded FP16 SUB/DIV path.
    broadcast_scalar_element_cycles=MappingProxyType(
        {
            WorkKind.MUL: 17.38,
            WorkKind.SUB: 575.3,
            WorkKind.DIV: 581.6,
        }
    ),
    # Each ReduceSum output owns a separately initialized accumulator.
    reduction_output_cycles=33.5,
)
SPATZ_DEVICE = replace(
    SPATZ_DEVICE,
    capabilities=frozenset(
        signature
        for signature in MAGIA_V2_SPATZ_DEVICE.capabilities
        if all(
            dtype is TensorDType.FLOAT16
            for dtype in signature.input_dtypes + signature.output_dtypes
        )
        and signature.work_kind
        in {
            WorkKind.ADD,
            WorkKind.MUL,
            WorkKind.SUB,
            WorkKind.DIV,
            WorkKind.RELU,
            WorkKind.SOFTMAX_EXP,
            WorkKind.GROUP_REDUCE,
            WorkKind.GROUP_CENTERED_REDUCE,
            WorkKind.GROUP_NORMALIZE,
            WorkKind.REDUCE_MAX,
            WorkKind.REDUCE_SUM,
            WorkKind.ALL_REDUCE_SUM,
            WorkKind.ALL_REDUCE_MAX,
            WorkKind.GEMM,
        }
    )
    | frozenset(
        WorkSignature(
            work_kind,
            (TensorDType.FLOAT16,) * input_count,
            (TensorDType.FLOAT16,),
        )
        for work_kind, input_count in (
            (WorkKind.MUL, 2),
            (WorkKind.SUB, 2),
            (WorkKind.DIV, 2),
            (WorkKind.SOFTMAX_EXP, 1),
            (WorkKind.GROUP_REDUCE, 1),
            (WorkKind.GROUP_CENTERED_REDUCE, 2),
            (WorkKind.GROUP_NORMALIZE, 5),
            (WorkKind.REDUCE_SUM, 1),
            (WorkKind.ALL_REDUCE_SUM, 1),
            (WorkKind.ALL_REDUCE_MAX, 1),
            (WorkKind.GEMM, 2),
            (WorkKind.GEMM, 3),
        )
    ),
)
REDMULE_DEVICE = replace(
    MAGIA_V2_REDMULE_DEVICE,
    capabilities=MAGIA_V2_REDMULE_DEVICE.capabilities - SPATZ_DEVICE.capabilities,
)
CORE_DEVICE = replace(
    MAGIA_V2_CORE_DEVICE,
    capabilities=(
        MAGIA_V2_CORE_DEVICE.capabilities
        - SPATZ_DEVICE.capabilities
        - IDMA_WRITE_DEVICE.capabilities
    ),
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
        for device in (IDMA_WRITE_DEVICE, REDMULE_DEVICE, CORE_DEVICE, SPATZ_DEVICE)
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
    "TILE_DEVICES",
]
