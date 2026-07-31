"""Migration bridge to Hardware-owned Device contracts."""

from maps.hardware.device import (
    DMADevice,
    DMAJob,
    Device,
    DeviceKind,
    FixedDeviceAssignment,
    MatrixDevice,
    ScalarDevice,
    SystolicDevice,
    VectorDevice,
    WorkKind,
    WorkSignature,
)

__all__ = [
    "DMADevice",
    "DMAJob",
    "Device",
    "DeviceKind",
    "FixedDeviceAssignment",
    "MatrixDevice",
    "ScalarDevice",
    "SystolicDevice",
    "VectorDevice",
    "WorkKind",
    "WorkSignature",
]
