"""Typed cost-model contracts shared by every MAPS operation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from MAPS.arch import Device, Tile

if TYPE_CHECKING:
    from MAPS.core.graph import Node
    from MAPS.core.layout import TensorLayout
    from MAPS.ops.common.tile_work import TileWork


class OpCostModel(ABC):
    """Cost contract for one operation family.

    ``cost`` describes work local to one tile. ``placement_cost`` describes
    communication or another cost that can only be evaluated from the complete
    placement. Most operations only implement the former; collective operations
    typically return zero local cost and implement the latter.
    """

    @abstractmethod
    def cost(
        self,
        tile_work: "TileWork",
        tile: "Tile",
        assigned_device: "Device",
    ) -> int:
        """Return non-negative cycles spent on one tile's local work."""

    def placement_cost(
        self,
        *,
        node: "Node",
        output_layouts: tuple["TensorLayout", ...],
    ) -> int:
        """Return non-negative cycles that depend on the complete placement."""

        del node, output_layouts
        return 0


def require_tile_device(tile: Tile, assigned_device: Device) -> Device:
    """Return an assigned Device only when it belongs to the costing Tile."""

    if assigned_device not in tile.devices:
        raise ValueError(
            f"assigned device {assigned_device.name} is not present on tile "
            f"{tile.tile_id}"
        )
    return assigned_device
