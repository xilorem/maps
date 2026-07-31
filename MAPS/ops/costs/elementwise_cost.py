"""Reusable elementwise cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile, WorkKind
from MAPS.ops.defs.elementwise import ElementwiseTileWork
from MAPS.ops.common.cost import OpCostModel


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
        if assigned_device not in tile.devices:
            raise ValueError(
                f"assigned device {assigned_device.name} is not present on tile "
                f"{tile.tile_id}"
            )
        return assigned_device.cycles(tile_work)
