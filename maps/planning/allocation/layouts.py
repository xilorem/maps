"""Topological Stage layout resolution and concrete locality checks."""

from __future__ import annotations

from typing import cast

from maps.graph import Node
from MAPS.core.layout import TensorLayout, TensorSlice, tile_tensor_slice
from maps.operations import find_layout_relation
from maps.operations import OpPayload, TileWork


def resolve_stage_layouts(
    stage_nodes: tuple[Node, ...],
    submesh,
    logical_shape: tuple[int, int],
) -> tuple[tuple[TensorLayout, ...], ...]:
    """Resolve one coherent set of output layouts in stage execution order."""

    producer_output_by_tensor: dict[object, tuple[Node, int]] = {}
    layouts_by_node: dict[int, tuple[TensorLayout, ...]] = {}
    resolved: list[tuple[TensorLayout, ...]] = []
    for node in stage_nodes:
        payload = cast(OpPayload, node.payload)
        standalone = list(
            payload.output_layouts(submesh, logical_shape=logical_shape)
        )
        derived_by_output: dict[int, TensorLayout] = {}
        for input_index, tensor in enumerate(node.inputs):
            producer_info = producer_output_by_tensor.get(tensor)
            if producer_info is None:
                continue
            producer, producer_output_index = producer_info
            relation = find_layout_relation(
                node.payload,
                input_index=input_index,
                output_index=0,
            )
            if relation is None:
                continue
            incoming_layout = layouts_by_node[id(producer)][producer_output_index]
            derived = relation.output_layout_from_input_layout(incoming_layout)
            previous = derived_by_output.get(relation.output_index)
            if previous is not None and previous != derived:
                raise ValueError(
                    f"node {node.name} has conflicting stage-local layout relations"
                )
            derived_by_output[relation.output_index] = derived
        for output_index, derived in derived_by_output.items():
            derived.validate_for(node.outputs[output_index])
            standalone[output_index] = derived
        node_layouts = tuple(standalone)
        layouts_by_node[id(node)] = node_layouts
        resolved.append(node_layouts)
        for output_index, tensor in enumerate(node.outputs):
            producer_output_by_tensor[tensor] = (node, output_index)
    return tuple(resolved)


def verify_stage_locality(
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple[TensorLayout, ...], ...],
    submesh,
    node_tile_work: tuple[tuple[TileWork, ...], ...],
) -> None:
    """Require every local consumer read to fit its same-tile producer slice."""

    producer_by_tensor: dict[object, tuple[Node, int, tuple]] = {}
    for node_index, (node, layouts) in enumerate(
        zip(stage_nodes, node_output_layouts)
    ):
        for tile_index, tile in enumerate(submesh.tiles):
            work = node_tile_work[node_index][tile_index]
            required_by_tensor = {
                reference.tensor: reference.tensor_slice
                for reference in work.input_slices
            }
            for tensor in node.inputs:
                producer_info = producer_by_tensor.get(tensor)
                if producer_info is None:
                    continue
                producer, output_index, producer_layouts = producer_info
                produced_slice = tile_tensor_slice(
                    tensor,
                    producer_layouts[output_index],
                    tile,
                )
                required_slice = required_by_tensor[tensor]
                if not _contains(produced_slice, required_slice):
                    raise ValueError(
                        f"stage-local edge {producer.name}->{node.name} is not "
                        f"tile-local on tile {tile.tile_id}"
                    )
        for output_index, tensor in enumerate(node.outputs):
            producer_by_tensor[tensor] = (node, output_index, layouts)


def _contains(container: TensorSlice, contained: TensorSlice) -> bool:
    if container.rank != contained.rank:
        return False
    return all(
        outer.start <= inner.start
        and inner.start + inner.length <= outer.start + outer.length
        for outer, inner in zip(container.dims, contained.dims)
    )
