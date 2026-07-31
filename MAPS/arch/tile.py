"""Tile-level hardware metadata."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch.device import DMADevice, DMAJob, Device, WorkSignature
from MAPS.arch.execution import FixedDeviceAssignment
from MAPS.arch.memory import L1Memory


@dataclass(frozen=True)
class Tile:
    """One physical tile in the mesh."""

    tile_id: int
    x: int
    y: int
    memory: L1Memory
    devices: tuple[Device, ...]
    device_assignment: FixedDeviceAssignment = FixedDeviceAssignment()

    def __post_init__(self) -> None:
        # check for valid tile identification and mesh position
        if self.tile_id < 0:
            raise ValueError("tile_id must be >= 0")
        if self.x < 0 or self.y < 0:
            raise ValueError("tile coordinates must be >= 0")
        if not self.devices:
            raise ValueError("tile devices must not be empty")

        devices_by_name = {device.name: device for device in self.devices}
        if len(devices_by_name) != len(self.devices):
            duplicate_name = next(
                device.name
                for index, device in enumerate(self.devices)
                if device.name in {prior.name for prior in self.devices[:index]}
            )
            raise ValueError(f"duplicate device name: {duplicate_name}")

        for signature, device_name in self.device_assignment.assignments.items():
            device = devices_by_name.get(device_name)
            if device is None:
                raise ValueError(
                    f"fixed assignment references unknown device {device_name}"
                )
            if not device.supports(signature):
                raise ValueError(
                    f"device {device_name} does not declare capability for {signature}"
                )

    def assigned_device(self, signature: WorkSignature) -> Device:
        """Return the Device named by this Tile's fixed assignment."""

        device_name = self.device_assignment.assignments[signature]
        return next(device for device in self.devices if device.name == device_name)
    
    def dma_devices(self, job: DMAJob) -> tuple[DMADevice, ...]:
        """Return the set of DMADevices associated with a tile for a
        specific DMAjob"""

        return tuple(device for device in self.devices 
            if isinstance(device, DMADevice) and device.job == job)
