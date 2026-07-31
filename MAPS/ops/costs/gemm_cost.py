"""GEMM-level aggregation built on top of concrete per-tile GEMM work."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile, WorkKind
from MAPS.ops.defs.gemm import GemmTileWork
from MAPS.ops.common.cost import OpCostModel


@dataclass(frozen=True)
class GemmCostModel(OpCostModel):
    """Compute-only GEMM cycle model backed by tile devices."""

    def cost(
        self,
        tile_work: GemmTileWork,
        tile: Tile,
        assigned_device: Device | None = None,
    ) -> int:
        if assigned_device is None:
            raise ValueError("GEMM costing requires a fixed assigned Device")
        if assigned_device not in tile.devices:
            raise ValueError(
                f"assigned device {assigned_device.name} is not present on tile "
                f"{tile.tile_id}"
            )
        if not assigned_device.supports(WorkKind.GEMM):
            raise ValueError(
                f"assigned device {assigned_device.name} cannot cost GEMM work"
            )
        return assigned_device.cycles(tile_work)
