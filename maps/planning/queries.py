"""Read-only queries over Planning Stage Plans and Graph Nodes."""

from __future__ import annotations

from maps.hardware import Tile
from maps.graph import Node
from MAPS.core.layout import TensorSlice
from maps.planning.stages import StagePlan


def node_output_layouts(plan: StagePlan, node: Node) -> tuple:
    """Return the output layouts chosen for ``node`` within ``plan``."""

    return plan.node_output_layouts[plan_node_index(plan, node)]


def plan_node_index(plan: StagePlan, node: Node) -> int:
    """Return the identity-based position of ``node`` within a stage plan."""

    for node_idx, candidate in enumerate(plan.nodes):
        if candidate is node:
            return node_idx
    raise ValueError(f"node {node.name} is not present in stage plan {plan.stage_id}")


def node_output_index(node: Node, tensor: object) -> int:
    """Return the output position at which ``node`` produces ``tensor``."""

    for output_idx, candidate in enumerate(node.outputs):
        if candidate == tensor:
            return output_idx
    raise ValueError(
        f"tensor {getattr(tensor, 'name', tensor)} is not an output of node {node.name}"
    )


def required_input_slices(
    tensor: object,
    destination_node: Node,
    destination_output_layouts: tuple,
) -> tuple[tuple[Tile, TensorSlice], ...]:
    """Return the slice of ``tensor`` required by every destination tile."""

    required_slices = []
    submesh = destination_output_layouts[0].submesh
    for tile in submesh.tiles:
        tile_work = destination_node.payload.build_tile_work(
            output_layouts=destination_output_layouts,
            tile=tile,
        )
        for reference in tile_work.input_slices:
            if reference.tensor is tensor:
                required_slices.append((tile, reference.tensor_slice))
                break
    return tuple(required_slices)
