"""Direct Conv2D cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile
from MAPS.ops.common.cost import OpCostModel
from MAPS.ops.defs.direct_conv import Conv2DTileWork


@dataclass(frozen=True)
class Conv2DCostModel(OpCostModel):
    """Compute-only Conv2D model backed by explicitly advertised devices.

    This provisional model accounts for MAC throughput only. Patch-address
    generation, packing, and boundary overhead are intentionally not modeled.
    """

    def cost(
        self,
        tile_work: Conv2DTileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        if assigned_device not in tile.devices:
            raise ValueError(
                f"assigned device {assigned_device.name} is not present on tile "
                f"{tile.tile_id}"
            )
        return assigned_device.cycles(tile_work)
