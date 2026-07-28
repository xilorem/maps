"""Generic tile-local devices."""

from __future__ import annotations

from MAPS.arch import DMADevice, DMAJob, DeviceKind, ScalarDevice, WorkKind

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

SCALAR_DEVICE = ScalarDevice(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.ELEMENTWISE: 1,
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
)

GENERIC_SCALAR_DEVICE = ScalarDevice(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.GEMM: 1,
        WorkKind.ELEMENTWISE: 1,
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
)
