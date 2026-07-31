"""Direct Conv2D cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.ops.common.cost import OpCostModel
from MAPS.ops.defs.direct_conv import Conv2DTileWork


@dataclass(frozen=True)
class Conv2DCostModel(OpCostModel):
    """Compute-only Conv2D model backed by explicitly advertised devices.

    This provisional model accounts for MAC throughput only. Patch-address
    generation, packing, and boundary overhead are intentionally not modeled.
    """

    preferred_device_kinds: tuple[DeviceKind, ...] = (
        DeviceKind.MATRIX,
        DeviceKind.SYSTOLIC,
    )

    def cost(
        self,
        tile_work: Conv2DTileWork,
        tile: Tile,
        assigned_device: object | None = None,
    ) -> int:
        del assigned_device
        devices = tuple(
            device for device in tile.devices if device.supports(WorkKind.CONV2D)
        )
        preferred = tuple(
            device for device in devices if device.kind in self.preferred_device_kinds
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for CONV2D work")
        return min(
            device._throughput_cycles(WorkKind.CONV2D, tile_work.operation_count())
            for device in candidates
        )
