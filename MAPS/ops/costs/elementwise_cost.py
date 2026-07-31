"""Reusable elementwise cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile, WorkKind
from MAPS.ops.defs.elementwise import ElementwiseTileWork
from MAPS.ops.common.cost import OpCostModel, require_tile_device


@dataclass(frozen=True)
class ElementwiseCostModel(OpCostModel):
    """Elementwise cycle model backed by tile devices."""

    work_kind: WorkKind = WorkKind.ELEMENTWISE

    def cost(
        self,
        tile_work: ElementwiseTileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        return require_tile_device(tile, assigned_device).cycles(tile_work)
