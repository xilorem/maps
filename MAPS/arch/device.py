"""Tile-local compute device capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import ceil


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
    SIGMOID = auto()
    SLICE = auto()
    SQRT = auto()
    SUB = auto()
    TRANSPOSE = auto()
    IM2COL = auto()
    WEIGHT_PACK = auto()
    OUTPUT_REFORMAT = auto()
    DMA = auto()

    @property
    def fallback_kind(self) -> "WorkKind":
        if self in {
            WorkKind.ABS,
            WorkKind.ADD,
            WorkKind.DIV,
            WorkKind.DEPTHWISE_CONV,
            WorkKind.EXP,
            WorkKind.GROUP_NORMALIZE,
            WorkKind.GROUP_REDUCE,
            WorkKind.LOG,
            WorkKind.MUL,
            WorkKind.NEG,
            WorkKind.POW,
            WorkKind.RELU,
            WorkKind.RESHAPE,
            WorkKind.SIGMOID,
            WorkKind.SLICE,
            WorkKind.SQRT,
            WorkKind.SUB,
            WorkKind.TRANSPOSE,
            WorkKind.IM2COL,
            WorkKind.WEIGHT_PACK,
            WorkKind.OUTPUT_REFORMAT,
        }:
            return WorkKind.ELEMENTWISE
        return self


class DMAJob(Enum):
    READJOB = auto()
    WRITEJOB = auto()


@dataclass(frozen=True)
class Device:
    """Base class for tile-local device models."""

    name: str
    kind: DeviceKind
    throughput: dict[WorkKind, int]
    startup_cycles: int = 0

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

    def supports(self, work_kind: WorkKind) -> bool:
        return work_kind in self.throughput or work_kind.fallback_kind in self.throughput

    def cycles(self, work: object) -> int:
        raise NotImplementedError

    def _throughput_cycles(self, work_kind: WorkKind, amount: int) -> int:
        if amount < 0:
            raise ValueError("device work amount must be >= 0")
        if not self.supports(work_kind):
            raise ValueError(f"device {self.name} does not support {work_kind.name} work")
        throughput_kind = work_kind if work_kind in self.throughput else work_kind.fallback_kind
        compute_cycles = ceil(amount / self.throughput[throughput_kind])
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

    def cycles(self, work: object) -> int:
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

    def cycles(self, work: object) -> int:
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

    def cycles(self, work: object) -> int:
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

    def cycles(self, work: object) -> int:
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

    def cycles(self, work: object) -> int:
        work_kind = work.work_kind
        amount = work.operation_count()

        vector_ops = ceil(amount / self.vector_length)
        return self._throughput_cycles(work_kind, vector_ops)
