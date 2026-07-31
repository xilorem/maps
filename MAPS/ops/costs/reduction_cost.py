"""Reduction cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile, WorkKind
from MAPS.ops.defs.reduction import ReductionTileWork
from MAPS.ops.common.cost import OpCostModel, require_tile_device


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
        return require_tile_device(tile, assigned_device).cycles(tile_work)
