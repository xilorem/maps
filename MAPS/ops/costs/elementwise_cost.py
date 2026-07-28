"""Reusable elementwise cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Tile, WorkKind
from MAPS.ops.defs.elementwise import ElementwiseTileWork
from MAPS.ops.common.cost import OpCostModel


@dataclass(frozen=True)
class ElementwiseCostModel(OpCostModel):
    """Elementwise cycle model backed by tile devices."""

    work_kind: WorkKind = WorkKind.ELEMENTWISE

    def cost(self, tile_work: ElementwiseTileWork, tile: Tile) -> int:
        devices = tuple(device for device in tile.devices if device.supports(self.work_kind))
        if not devices:
            raise ValueError(f"tile {tile.tile_id} has no device for {self.work_kind.name} work")
        return min(device.cycles(tile_work) for device in devices)
