"""Reduction cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile, WorkKind
from MAPS.ops.defs.reduction import ReductionTileWork
from MAPS.ops.common.cost import OpCostModel


@dataclass(frozen=True)
class ReductionCostModel(OpCostModel):
    """Tile-local reduction cycle model backed by tile devices."""

    work_kind: WorkKind

    def __post_init__(self) -> None:
        if self.work_kind not in (WorkKind.REDUCE_SUM, WorkKind.REDUCE_MAX):
            raise ValueError("ReductionCostModel work_kind must be REDUCE_SUM or REDUCE_MAX")

    def cost(
        self,
        tile_work: ReductionTileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        if assigned_device not in tile.devices:
            raise ValueError(
                f"assigned device {assigned_device.name} is not present on tile "
                f"{tile.tile_id}"
            )
        return assigned_device.cycles(tile_work)
