"""Tile-local compute device capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from math import ceil
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, cast

if TYPE_CHECKING:
    from maps.graph import Node, TensorDType
    from maps.hardware.tile import Tile


class DeviceKind(Enum):
    SCALAR = auto()
    VECTOR = auto()
    SYSTOLIC = auto()
    DMA = auto()
    MATRIX = auto()

class WorkKind(Enum):
    GEMM = auto()
    CONV2D = auto()
    ELEMENTWISE = auto()
    GROUP_NORMALIZE = auto()
    GROUP_REDUCE = auto()
    GROUP_CENTERED_REDUCE = auto()
    ALL_REDUCE_SUM = auto()
    ALL_REDUCE_MAX = auto()
    ABS = auto()
    ADD = auto()
    DIV = auto()
    DEPTHWISE_CONV = auto()
    LOG = auto()
    MUL = auto()
    NEG = auto()
    POW = auto()
    RELU = auto()
    REDUCE_SUM = auto()
    REDUCE_MAX = auto()
    RESHAPE = auto()
    EXP = auto()
    SOFTMAX_EXP = auto()
    SIGMOID = auto()
    SLICE = auto()
    SPLIT = auto()
    SQRT = auto()
    SUB = auto()
    TRANSPOSE = auto()
    IM2COL = auto()
    WEIGHT_PACK = auto()
    OUTPUT_REFORMAT = auto()
    CAST = auto()
    DMA = auto()

class DMAJob(Enum):
    READJOB = auto()
    WRITEJOB = auto()


@dataclass(frozen=True)
class WorkSignature:
    """Complete graph-visible type contract for one operation."""

    work_kind: WorkKind
    input_dtypes: tuple[TensorDType, ...]
    output_dtypes: tuple[TensorDType, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dtypes", tuple(self.input_dtypes))
        object.__setattr__(self, "output_dtypes", tuple(self.output_dtypes))

    @classmethod
    def from_node(cls, node: "Node") -> "WorkSignature":
        """Derive the ordered typed-work identity of a primitive Node."""

        work_kind = getattr(node.payload, "work_kind", None)
        if not isinstance(work_kind, WorkKind):
            raise ValueError(f"node {node.name} does not declare a WorkKind")

        untyped_tensors = tuple(
            tensor.name
            for tensor in node.inputs + node.outputs
            if tensor.dtype is None
        )
        if untyped_tensors:
            names = ", ".join(untyped_tensors)
            raise ValueError(f"node {node.name} has untyped tensors: {names}")

        return cls(
            work_kind=work_kind,
            input_dtypes=tuple(
                cast("TensorDType", tensor.dtype) for tensor in node.inputs
            ),
            output_dtypes=tuple(
                cast("TensorDType", tensor.dtype) for tensor in node.outputs
            ),
        )


@dataclass(frozen=True)
class CollectiveCost:
    """Target-selected cost assumptions for one SDK collective implementation."""

    participant_rounds: int
    hop_cycles: int

    def __post_init__(self) -> None:
        if self.participant_rounds <= 0:
            raise ValueError("participant_rounds must be > 0")
        if self.hop_cycles < 0:
            raise ValueError("hop_cycles must be >= 0")

    def cycles(
        self,
        work_kind: WorkKind,
        element_count: int,
        participants: tuple[Tile, ...],
        elements_per_cycle: float,
        startup_cycles: int,
    ) -> int:
        """Price the concrete participant group using target-owned assumptions."""

        if len(participants) <= 1:
            return 0
        del work_kind
        transfer_cycles = startup_cycles + ceil(
            element_count
            * self.participant_rounds
            * (len(participants) - 1)
            / elements_per_cycle
        )
        diameter = max(
            abs(left.x - right.x) + abs(left.y - right.y)
            for left in participants
            for right in participants
        )
        return transfer_cycles + self.hop_cycles * diameter


def same_dtype_signatures(
    work_kinds: tuple[WorkKind, ...],
    input_counts: tuple[int, ...],
    dtypes: tuple[TensorDType, ...],
) -> frozenset[WorkSignature]:
    """Build exact signatures whose operands share one TensorDType."""

    return frozenset(
        WorkSignature(work_kind, (dtype,) * input_count, (dtype,))
        for work_kind in work_kinds
        for input_count in input_counts
        for dtype in dtypes
    )


@dataclass(frozen=True)
class FixedDeviceAssignment:
    """Stable Device names selected for exact Work Signatures."""

    assignments: Mapping[WorkSignature, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assignments = dict(self.assignments)
        if any(not device_name for device_name in assignments.values()):
            raise ValueError("assigned device name must not be empty")
        object.__setattr__(self, "assignments", MappingProxyType(assignments))


@dataclass(frozen=True)
class Device:
    """Base class for tile-local device models."""

    name: str
    kind: DeviceKind
    throughput: dict[WorkKind, float]
    startup_cycles: int = 0
    capabilities: frozenset[WorkSignature] = frozenset()
    collective_costs: Mapping[WorkKind, CollectiveCost] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self) is Device:
            raise TypeError("Device must be instantiated through a concrete device type")
        if not self.name:
            raise ValueError("device name must not be empty")
        if self.startup_cycles < 0:
            raise ValueError("device startup_cycles must be >= 0")
        if not self.throughput:
            raise ValueError("device throughput must not be empty")

        # check for invalid throughput value
        if any(value <= 0 for value in self.throughput.values()):
            raise ValueError("device throughput values must be > 0")
        object.__setattr__(self, "throughput", dict(self.throughput))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(
            self,
            "collective_costs",
            MappingProxyType(dict(self.collective_costs)),
        )

    def supports(self, signature: WorkSignature) -> bool:
        """Return whether this Device declares the exact typed capability."""

        return signature in self.capabilities

    def cycles(self, work: Any) -> int:
        raise NotImplementedError

    def collective_cycles(
        self,
        work_kind: WorkKind,
        element_count: int,
        participants: tuple[Tile, ...],
    ) -> int:
        """Estimate one synchronous collective supplied by this Device."""

        try:
            implementation_cost = self.collective_costs[work_kind]
        except KeyError as exc:
            raise ValueError(
                f"device {self.name} has no collective cost for {work_kind.name}"
            ) from exc
        return implementation_cost.cycles(
            work_kind,
            element_count,
            participants,
            self.throughput[work_kind],
            self.startup_cycles,
        )

    def temporary_l1_bytes(self, signature: WorkSignature) -> int:
        """Return reusable per-tile scratch required by this implementation."""

        del signature
        return 0

    def _throughput_cycles(self, work_kind: WorkKind, amount: int) -> int:
        if amount < 0:
            raise ValueError("device work amount must be >= 0")
        if work_kind not in self.throughput:
            raise ValueError(f"device {self.name} does not support {work_kind.name} work")
        compute_cycles = ceil(amount / self.throughput[work_kind])
        if compute_cycles < 0:
            raise ValueError("device cycle estimator must return >= 0")
        return self.startup_cycles + compute_cycles


@dataclass(frozen=True)
class ScalarDevice(Device):
    """Scalar/core device model using throughput-based timing."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not DeviceKind.SCALAR:
            raise ValueError("ScalarDevice must use DeviceKind.SCALAR")

    def cycles(self, work: Any) -> int:
        work_kind = work.work_kind
        amount = work.operation_count()
        return self._throughput_cycles(work_kind, amount)

@dataclass(frozen=True)
class DMADevice(Device):
    """DMA device model using throughput-based timing."""

    job: DMAJob = DMAJob.READJOB
    burst_bytes: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not DeviceKind.DMA:
            raise ValueError("DMADevice must use DeviceKind.DMA")
        if not isinstance(self.job, DMAJob):
            raise ValueError("Bad DMADevice job description, must be a DMAJob type")
        if self.burst_bytes is not None and self.burst_bytes <= 0:
            raise ValueError("DMA burst_bytes must be > 0 when specified")

    def cycles(self, work: Any) -> int:
        raise ValueError("DMA device cannot perform compute operations")




@dataclass(frozen=True)
class SystolicDevice(Device):
    """Systolic-array device model with GEMM-specific timing."""

    array_width: int = 1
    array_height: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not DeviceKind.SYSTOLIC:
            raise ValueError("SystolicDevice must use DeviceKind.SYSTOLIC")
        if self.array_width <= 0 or self.array_height <= 0:
            raise ValueError("systolic array dimensions must be > 0")

    def cycles(self, work: Any) -> int:
        batch_volume, m_size, n_size, k_size = work.dimensions()
        m_blocks = ceil(m_size / self.array_height)
        n_blocks = ceil(n_size / self.array_width)
        fill_and_drain_cycles = self.array_height + self.array_width - 2
        compute_cycles = batch_volume * m_blocks * n_blocks * (k_size + fill_and_drain_cycles)
        return self.startup_cycles + compute_cycles


@dataclass(frozen=True)
class MatrixDevice(Device):
    """Matrix-unit device model with GEMM-specific timing."""

    srcA_width: int = 1
    srcA_height: int = 1
    srcB_width: int = 1
    srcB_height: int = 1

    math_fidelity: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not DeviceKind.MATRIX:
            raise ValueError("MatrixDevice must use DeviceKind.MATRIX")
        if self.srcA_width <= 0 or self.srcA_height <= 0:
            raise ValueError("MatrixDevice srcA dimensions must be > 0")
        if self.srcB_width <= 0 or self.srcB_height <= 0:
            raise ValueError("MatrixDevice srcB dimensions must be > 0")
        if self.srcA_width != self.srcB_height:
            raise ValueError("The reduction dimension of srcs must agree")
        if self.math_fidelity <= 0:
            raise ValueError("MatrixDevice math_fidelity must be > 0")

    def cycles(self, work: Any) -> int:
        batch_volume, m_size, n_size, k_size = work.dimensions()
        m_blocks = ceil(m_size / self.srcA_height)
        n_blocks = ceil(n_size / self.srcB_width)
        k_blocks = ceil(k_size / self.srcA_width)

        return self.startup_cycles + batch_volume * m_blocks * n_blocks * k_blocks * self.math_fidelity


@dataclass(frozen=True)
class VectorDevice(Device):
    """Vector device model using throughput-based timing."""

    vector_length: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not DeviceKind.VECTOR:
            raise ValueError("VectorDevice must use DeviceKind.VECTOR")
        if self.vector_length <= 0:
            raise ValueError("vector_length must be > 0")

    def cycles(self, work: Any) -> int:
        work_kind = work.work_kind
        amount = work.operation_count()

        vector_ops = ceil(amount / self.vector_length)
        return self._throughput_cycles(work_kind, vector_ops)
