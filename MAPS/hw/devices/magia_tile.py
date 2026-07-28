"""MAGIA tile-local device definitions."""

from __future__ import annotations

from MAPS.arch import DMADevice, DMAJob, DeviceKind, ScalarDevice, WorkKind
from MAPS.hw.devices.redmule import REDMULE_DEVICE

L1_CORE_TRANSFER_LATENCY = 4


MAGIA_IDMA_READ_DEVICE = DMADevice(
    name="idma_read",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.READJOB,
    burst_bytes=4,
)

MAGIA_IDMA_WRITE_DEVICE = DMADevice(
    name="idma_write",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.WRITEJOB,
    burst_bytes=8,
)

# Keep every scalar operation explicit so measured MAGIA rates can be updated
# independently without changing operation lowering or planner code.
MAGIA_CORE_DEVICE = ScalarDevice(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.ELEMENTWISE: 1,
        WorkKind.GROUP_NORMALIZE: 1,
        WorkKind.GROUP_REDUCE: 1,
        WorkKind.ABS: 1,
        WorkKind.ADD: 1/(1 + 2 * L1_CORE_TRANSFER_LATENCY + 1 * L1_CORE_TRANSFER_LATENCY), # op time + inputs read time + output write time
        WorkKind.DIV: 1,
        WorkKind.CONV2D: 1,
        WorkKind.DEPTHWISE_CONV: 1,
        WorkKind.EXP: 1,
        WorkKind.LOG: 1/(176), # approx from magia traces
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
    },
)

MAGIA_REDMULE_DEVICE = REDMULE_DEVICE

MAGIA_TILE_DEVICES = (
    MAGIA_IDMA_READ_DEVICE,
    MAGIA_IDMA_WRITE_DEVICE,
    MAGIA_CORE_DEVICE,
    MAGIA_REDMULE_DEVICE,
)
