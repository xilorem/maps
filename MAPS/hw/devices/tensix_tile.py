"""Tensix tile-local placeholder device models."""

from __future__ import annotations

from MAPS.arch import (
    DMADevice,
    DMAJob,
    DeviceKind,
    FixedDeviceAssignment,
    MatrixDevice,
    ScalarDevice,
    VectorDevice,
    WorkKind,
)
from MAPS.core.dtype import TensorDType
from MAPS.hw.devices.capabilities import same_dtype_signatures


_TENSIX_LOCAL_WORK = (
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
_UNARY_TENSIX_WORK = tuple(
    work_kind
    for work_kind in _TENSIX_LOCAL_WORK
    if work_kind
    not in {
        WorkKind.ADD,
        WorkKind.DEPTHWISE_CONV,
        WorkKind.DIV,
        WorkKind.GROUP_NORMALIZE,
        WorkKind.MUL,
        WorkKind.POW,
        WorkKind.SUB,
    }
)
_BINARY_TENSIX_WORK = (
    WorkKind.ADD,
    WorkKind.DIV,
    WorkKind.MUL,
    WorkKind.POW,
    WorkKind.SUB,
)


_TENSIX_LOCAL_CAPABILITIES = (
    same_dtype_signatures(
        _UNARY_TENSIX_WORK,
        (1,),
        (TensorDType.FLOAT16, TensorDType.FLOAT32),
    )
    | same_dtype_signatures(
        _BINARY_TENSIX_WORK,
        (2,),
        (TensorDType.FLOAT16, TensorDType.FLOAT32),
    )
    | same_dtype_signatures(
        (WorkKind.MUL,),
        (1,),
        (TensorDType.FLOAT16, TensorDType.FLOAT32),
    )
    | same_dtype_signatures(
        (WorkKind.DEPTHWISE_CONV,),
        (2, 3),
        (TensorDType.FLOAT16, TensorDType.FLOAT32),
    )
    | same_dtype_signatures(
        (WorkKind.GROUP_NORMALIZE,),
        (5,),
        (TensorDType.FLOAT16, TensorDType.FLOAT32),
    )
)
_TENSIX_GEMM_CAPABILITIES = same_dtype_signatures(
    (WorkKind.GEMM,),
    (2, 3),
    (TensorDType.FLOAT16, TensorDType.FLOAT32),
)

TENSIX_READ_CORE = DMADevice(
    name="tensix_read_core",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.READJOB,
)

TENSIX_WRITE_CORE = DMADevice(
    name="tensix_write_core",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.WRITEJOB,
)

TENSIX_SCALAR_DEVICE = ScalarDevice(
    name="tensix_scalar",
    kind=DeviceKind.SCALAR,
    throughput={work_kind: 1 for work_kind in _TENSIX_LOCAL_WORK},
    capabilities=_TENSIX_LOCAL_CAPABILITIES,
)

TENSIX_VECTOR_DEVICE = VectorDevice(
    name="tensix_vector",
    kind=DeviceKind.VECTOR,
    throughput={work_kind: 1 for work_kind in _TENSIX_LOCAL_WORK},
    capabilities=_TENSIX_LOCAL_CAPABILITIES,
    vector_length=32,
)

TENSIX_MATRIX_DEVICE = MatrixDevice(
    name="tensix_matrix",
    kind=DeviceKind.MATRIX,
    throughput={
        WorkKind.GEMM: 1,
    },
    capabilities=_TENSIX_GEMM_CAPABILITIES,
    srcA_width=16,
    srcA_height=8,
    srcB_width=16,
    srcB_height=16,
    math_fidelity=1,
)

TENSIX_DEVICE_ASSIGNMENT = FixedDeviceAssignment(
    {
        signature: TENSIX_VECTOR_DEVICE.name
        for signature in TENSIX_VECTOR_DEVICE.capabilities
    }
    | {
        signature: TENSIX_MATRIX_DEVICE.name
        for signature in TENSIX_MATRIX_DEVICE.capabilities
    }
)
