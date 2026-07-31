"""Generic planner-side node cost estimation entry points."""

from __future__ import annotations

from MAPS.arch import WorkKind, WorkSignature
from MAPS.core.graph import Node
from MAPS.core.layout import TensorLayout


def cost_estimator(
    node: Node,
    output_layouts: tuple[TensorLayout, ...],
) -> int:
    """Estimate one node's bottleneck compute cost for virtual planning."""

    cost_model = node.payload.cost_model
    output_layout = node.payload.single_output_layout(output_layouts)
    submesh = output_layout.submesh
    tile_work = tuple(
        (
            tile,
            node.payload.build_tile_work(output_layouts=output_layouts, tile=tile),
        )
        for tile in submesh.tiles
    )
    tile_cost = max(
        (
            cost_model.cost(
                work,
                tile,
                (
                    tile.assigned_device(WorkSignature.from_node(node))
                    if node.payload.work_kind is WorkKind.GEMM
                    else None
                ),
            )
            for tile, work in tile_work
        ),
        default=0,
    )
    return tile_cost + int(
        cost_model.placement_cost(node=node, output_layouts=output_layouts)
    )


def placement_cost_estimator(
    node: Node,
    output_layouts: tuple[TensorLayout, ...],
) -> int:
    """Estimate the placement-specific component of one node cost."""

    return int(
        node.payload.cost_model.placement_cost(
            node=node,
            output_layouts=output_layouts,
        )
    )
