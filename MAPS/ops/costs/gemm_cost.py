"""GEMM-level aggregation built on top of concrete per-tile GEMM work."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile
from MAPS.ops.defs.gemm import GemmTileWork
from MAPS.ops.common.cost import OpCostModel


@dataclass(frozen=True)
class GemmCostModel(OpCostModel):
    """Compute-only GEMM cycle model backed by tile devices."""

    def cost(
        self,
        tile_work: GemmTileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        if assigned_device not in tile.devices:
            raise ValueError(
                f"assigned device {assigned_device.name} is not present on tile "
                f"{tile.tile_id}"
            )
        return assigned_device.cycles(tile_work)
