"""RedMulE systolic device model."""

from __future__ import annotations

from MAPS.arch import DeviceKind, SystolicDevice, WorkKind

REDMULE_ARRAY_WIDTH = 24
REDMULE_ARRAY_HEIGHT = 8


REDMULE_DEVICE = SystolicDevice(
    name="redmule",
    kind=DeviceKind.SYSTOLIC,
    throughput={
        WorkKind.GEMM: REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT,
        WorkKind.CONV2D: REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT,
    },
    array_width=REDMULE_ARRAY_WIDTH,
    array_height=REDMULE_ARRAY_HEIGHT,
)
