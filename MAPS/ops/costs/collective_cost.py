"""Collective communication cost models."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Device, Tile
from MAPS.transitions.transport import TransportCostModel
from MAPS.core.graph import Node
from MAPS.core.layout import (
    TensorLayout,
    tensor_slice_num_bytes,
    tile_tensor_slice,
)
from MAPS.ops.common.cost import OpCostModel


@dataclass(frozen=True)
class AllReduceCostModel(OpCostModel):
    """Placement-sensitive allreduce latency model."""

    reduction: str
    collective_axis: str = "x"

    def __post_init__(self) -> None:
        if self.reduction not in {"sum", "max"}:
            raise ValueError("AllReduceCostModel reduction must be 'sum' or 'max'")
        if self.collective_axis not in {"x", "y"}:
            raise ValueError("AllReduceCostModel collective_axis must be 'x' or 'y'")

    def cost(
        self,
        tile_work: object,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        del assigned_device
        del tile_work, tile
        return 0

    def placement_cost(
        self,
        *,
        node: Node,
        output_layouts: tuple[TensorLayout, ...],
    ) -> int:
        if len(output_layouts) != 1:
            raise ValueError("AllReduceCostModel expects exactly one output layout")

        output_layout = output_layouts[0]
        output_tensor = node.outputs[0]
        model = TransportCostModel(mesh=output_layout.submesh.mesh)
        groups = _logical_collective_groups(output_layout, self.collective_axis)
        group_costs = []
        for group_tiles in groups:
            payload_tiles = tuple(
                tile
                for tile in group_tiles
                if tensor_slice_num_bytes(
                    output_tensor,
                    tile_tensor_slice(output_tensor, output_layout, tile),
                ) > 0
            )
            if len(payload_tiles) <= 1:
                group_costs.append(0)
                continue

            root_tile = payload_tiles[0]
            reduce_phase = max(
                (
                    model.l1_to_l1(
                        tile,
                        root_tile,
                        tensor_slice_num_bytes(
                            output_tensor,
                            tile_tensor_slice(output_tensor, output_layout, tile),
                        ),
                    )
                    for tile in payload_tiles[1:]
                ),
                default=0,
            )
            broadcast_phase = max(
                (
                    model.l1_to_l1(
                        root_tile,
                        tile,
                        tensor_slice_num_bytes(
                            output_tensor,
                            tile_tensor_slice(output_tensor, output_layout, tile),
                        ),
                    )
                    for tile in payload_tiles[1:]
                ),
                default=0,
            )
            group_costs.append(reduce_phase + broadcast_phase)
        return max(group_costs, default=0)


def _logical_collective_groups(
    layout: TensorLayout,
    collective_axis: str,
) -> tuple[tuple[object, ...], ...]:
    logical_width = layout.effective_logical_width
    logical_height = layout.effective_logical_height
    groups: dict[int, list[object]] = {}
    for tile_ordinal, tile in enumerate(layout.submesh.tiles):
        logical_x = tile_ordinal % logical_width
        logical_y = tile_ordinal // logical_width
        group_key = logical_y if collective_axis == "x" else logical_x
        groups.setdefault(group_key, []).append(tile)

    return tuple(
        tuple(group_tiles)
        for _, group_tiles in sorted(groups.items())
    )
