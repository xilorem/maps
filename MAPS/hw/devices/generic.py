"""Generic tile-local devices."""

from __future__ import annotations

from MAPS.arch import (
    DMADevice,
    DMAJob,
    DeviceKind,
    FixedDeviceAssignment,
    ScalarDevice,
    WorkKind,
)
from MAPS.core.dtype import TensorDType
from MAPS.hw.devices.capabilities import same_dtype_signatures


_FLOAT_DTYPES = (TensorDType.FLOAT16, TensorDType.FLOAT32)
_UNARY_WORK = (
    WorkKind.ABS,
    WorkKind.EXP,
    WorkKind.GROUP_REDUCE,
    WorkKind.LOG,
    WorkKind.NEG,
    WorkKind.REDUCE_MAX,
    WorkKind.REDUCE_SUM,
    WorkKind.RELU,
    WorkKind.RESHAPE,
    WorkKind.SIGMOID,
    WorkKind.SLICE,
    WorkKind.SQRT,
    WorkKind.TRANSPOSE,
)
_BINARY_WORK = (
    WorkKind.ADD,
    WorkKind.DIV,
    WorkKind.MUL,
    WorkKind.POW,
    WorkKind.SUB,
)


_GENERIC_SCALAR_CAPABILITIES = (
    same_dtype_signatures(_UNARY_WORK, (1,), _FLOAT_DTYPES)
    | same_dtype_signatures(_BINARY_WORK, (2,), _FLOAT_DTYPES)
    | same_dtype_signatures((WorkKind.MUL,), (1,), _FLOAT_DTYPES)
    | same_dtype_signatures(
        (WorkKind.CONV2D, WorkKind.DEPTHWISE_CONV),
        (2, 3),
        _FLOAT_DTYPES,
    )
    | same_dtype_signatures((WorkKind.GEMM,), (2, 3), _FLOAT_DTYPES)
    | same_dtype_signatures((WorkKind.GROUP_NORMALIZE,), (5,), _FLOAT_DTYPES)
)

IDMA_READ_DEVICE = DMADevice(
    name="idma_read",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.READJOB,
)

IDMA_WRITE_DEVICE = DMADevice(
    name="idma_write",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.WRITEJOB,
)

GENERIC_SCALAR_DEVICE = ScalarDevice(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.GEMM: 1,
        WorkKind.GROUP_NORMALIZE: 1,
        WorkKind.GROUP_REDUCE: 1,
        WorkKind.ABS: 1,
        WorkKind.ADD: 1,
        WorkKind.DIV: 1,
        WorkKind.CONV2D: 1,
        WorkKind.DEPTHWISE_CONV: 1,
        WorkKind.LOG: 1,
        WorkKind.MUL: 1,
        WorkKind.NEG: 1,
        WorkKind.POW: 1,
        WorkKind.RELU: 1,
        WorkKind.REDUCE_SUM: 1,
        WorkKind.REDUCE_MAX: 1,
        WorkKind.RESHAPE: 1,
        WorkKind.EXP: 1,
        WorkKind.SIGMOID: 1,
        WorkKind.SLICE: 1,
        WorkKind.SQRT: 1,
        WorkKind.SUB: 1,
        WorkKind.TRANSPOSE: 1,
    },
    capabilities=_GENERIC_SCALAR_CAPABILITIES,
)

GENERIC_DEVICE_ASSIGNMENT = FixedDeviceAssignment(
    {
        signature: GENERIC_SCALAR_DEVICE.name
        for signature in GENERIC_SCALAR_DEVICE.capabilities
    }
)
