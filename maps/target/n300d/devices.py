"""Concrete Tensix Devices owned by the N300D target."""

from maps.graph import TensorDType
from maps.hardware import (
    DMADevice,
    DMAJob,
    DeviceKind,
    FixedDeviceAssignment,
    MatrixDevice,
    ScalarDevice,
    VectorDevice,
    WorkKind,
    WorkSignature,
)

_LOCAL_WORK = (
    WorkKind.ABS,
    WorkKind.ADD,
    WorkKind.DEPTHWISE_CONV,
    WorkKind.DIV,
    WorkKind.EXP,
    WorkKind.GROUP_NORMALIZE,
    WorkKind.GROUP_REDUCE,
    WorkKind.LOG,
    WorkKind.MUL,
    WorkKind.NEG,
    WorkKind.POW,
    WorkKind.REDUCE_MAX,
    WorkKind.REDUCE_SUM,
    WorkKind.RELU,
    WorkKind.RESHAPE,
    WorkKind.SIGMOID,
    WorkKind.SLICE,
    WorkKind.SQRT,
    WorkKind.SUB,
    WorkKind.TRANSPOSE,
)
_UNARY_WORK = tuple(
    work_kind
    for work_kind in _LOCAL_WORK
    if work_kind not in {
        WorkKind.ADD,
        WorkKind.DEPTHWISE_CONV,
        WorkKind.DIV,
        WorkKind.GROUP_NORMALIZE,
        WorkKind.MUL,
        WorkKind.POW,
        WorkKind.SUB,
    }
)
_BINARY_WORK = (
    WorkKind.ADD,
    WorkKind.DIV,
    WorkKind.MUL,
    WorkKind.POW,
    WorkKind.SUB,
)
_FLOAT_DTYPES = (TensorDType.FLOAT16, TensorDType.FLOAT32)


def _same_dtype_signatures(
    work_kinds: tuple[WorkKind, ...],
    input_counts: tuple[int, ...],
) -> frozenset[WorkSignature]:
    return frozenset(
        WorkSignature(work_kind, (dtype,) * input_count, (dtype,))
        for work_kind in work_kinds
        for input_count in input_counts
        for dtype in _FLOAT_DTYPES
    )


_LOCAL_CAPABILITIES = (
    _same_dtype_signatures(_UNARY_WORK, (1,))
    | _same_dtype_signatures(_BINARY_WORK, (2,))
    | _same_dtype_signatures((WorkKind.MUL,), (1,))
    | _same_dtype_signatures((WorkKind.DEPTHWISE_CONV,), (2, 3))
    | _same_dtype_signatures((WorkKind.GROUP_NORMALIZE,), (5,))
)
_GEMM_CAPABILITIES = _same_dtype_signatures((WorkKind.GEMM,), (2, 3))

READ_CORE = DMADevice(
    name="tensix_read_core",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.READJOB,
)
WRITE_CORE = DMADevice(
    name="tensix_write_core",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.WRITEJOB,
)
SCALAR_DEVICE = ScalarDevice(
    name="tensix_scalar",
    kind=DeviceKind.SCALAR,
    throughput={work_kind: 1 for work_kind in _LOCAL_WORK},
    capabilities=_LOCAL_CAPABILITIES,
)
VECTOR_DEVICE = VectorDevice(
    name="tensix_vector",
    kind=DeviceKind.VECTOR,
    throughput={work_kind: 1 for work_kind in _LOCAL_WORK},
    capabilities=_LOCAL_CAPABILITIES,
    vector_length=32,
)
MATRIX_DEVICE = MatrixDevice(
    name="tensix_matrix",
    kind=DeviceKind.MATRIX,
    throughput={WorkKind.GEMM: 1},
    capabilities=_GEMM_CAPABILITIES,
    srcA_width=16,
    srcA_height=8,
    srcB_width=16,
    srcB_height=16,
    math_fidelity=1,
)
TILE_DEVICES = (
    READ_CORE,
    WRITE_CORE,
    SCALAR_DEVICE,
    VECTOR_DEVICE,
    MATRIX_DEVICE,
)
DEVICE_ASSIGNMENT = FixedDeviceAssignment(
    {
        signature: VECTOR_DEVICE.name
        for signature in VECTOR_DEVICE.capabilities
    }
    | {
        signature: MATRIX_DEVICE.name
        for signature in MATRIX_DEVICE.capabilities
    }
)

__all__ = [
    "DEVICE_ASSIGNMENT",
    "MATRIX_DEVICE",
    "READ_CORE",
    "SCALAR_DEVICE",
    "TILE_DEVICES",
    "VECTOR_DEVICE",
    "WRITE_CORE",
]
