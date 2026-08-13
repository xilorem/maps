"""Concrete tile-local Devices owned by the MAGIA-v3 Target."""

from dataclasses import replace

from maps.graph import TensorDType
from maps.hardware import FixedDeviceAssignment
from maps.target.magia.devices import (
    CORE_DEVICE,
    IDMA_READ_DEVICE,
    IDMA_WRITE_DEVICE,
    REDMULE_DEVICE,
    SPATZ_DEVICE as MAGIA_V2_SPATZ_DEVICE,
)


SPATZ_DEVICE = replace(
    MAGIA_V2_SPATZ_DEVICE,
    vlen_bits=256,
    capabilities=frozenset(
        signature
        for signature in MAGIA_V2_SPATZ_DEVICE.capabilities
        if all(
            dtype is TensorDType.FLOAT16
            for dtype in signature.input_dtypes + signature.output_dtypes
        )
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
        for device in (REDMULE_DEVICE, CORE_DEVICE, SPATZ_DEVICE)
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
