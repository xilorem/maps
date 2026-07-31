"""Direct Conv2D cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile
from MAPS.ops.common.cost import OpCostModel, require_tile_device
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
        return require_tile_device(tile, assigned_device).cycles(tile_work)
