"""Cast cycle aggregation backed by the assigned tile Device."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile
from MAPS.ops.common.cost import OpCostModel, require_tile_device
from MAPS.ops.defs.cast import CastTileWork


@dataclass(frozen=True)
class CastCostModel(OpCostModel):
    """Compute-only Cast cycle model owned by the assigned Device."""

    def cost(
        self,
        tile_work: CastTileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        return require_tile_device(tile, assigned_device).cycles(tile_work)


__all__ = ["CastCostModel"]
