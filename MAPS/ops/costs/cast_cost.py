"""Cast cycle aggregation backed by the assigned tile Device."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile, WorkKind
from MAPS.ops.common.cost import OpCostModel
from MAPS.ops.defs.cast import CastTileWork


@dataclass(frozen=True)
class CastCostModel(OpCostModel):
    """Compute-only Cast cycle model owned by the assigned Device."""

    def cost(
        self,
        tile_work: CastTileWork,
        tile: Tile,
        assigned_device: Device | None = None,
    ) -> int:
        if assigned_device is None:
            raise ValueError("Cast costing requires a fixed assigned Device")
        if assigned_device not in tile.devices:
            raise ValueError(
                f"assigned device {assigned_device.name} is not present on tile "
                f"{tile.tile_id}"
            )
        if not assigned_device.supports(WorkKind.CAST):
            raise ValueError(
                f"assigned device {assigned_device.name} cannot cost Cast work"
            )
        return assigned_device.cycles(tile_work)


__all__ = ["CastCostModel"]
