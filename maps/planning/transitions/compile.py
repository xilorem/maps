"""Compile graph communication once and bind it to physical tile placements."""

from __future__ import annotations

from typing import cast

from maps.graph import Graph, Node, Tensor
from maps.hardware import Tile
from maps.planning.mapping import (
    TensorLayout,
    TensorRange,
    TensorSlice,
    TensorSubSlice,
    tile_tensor_slice,
)
from maps.operations.contracts import OpPayload
from maps.planning.stages import node_output_index, node_output_layouts
from maps.planning.stages import StagePlacement, StagePlan

from .contracts import (
    InputDestination,
    InputTransition,
    IntermediateTransition,
    OutputSource,
    OutputTransition,
    Transfer,
    Transition,
    VirtualInputDestination,
    VirtualInputTransition,
    VirtualIntermediateTransition,
    VirtualOutputSource,
    VirtualOutputTransition,
    VirtualTransfer,
    VirtualTransition,
)


def build_virtual_transitions(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
) -> tuple[VirtualTransition, ...]:
    """Compile every graph boundary and cross-stage dependency."""

    tensor_id_by_identity = {
        id(tensor): tensor_id
        for tensor_id, tensor in enumerate(graph.tensors)
    }
    producer_by_tensor_identity = {
        id(tensor): node
        for node in graph.nodes
        for tensor in node.outputs
    }
    stage_id_by_node_identity = {
        id(node): stage_id
        for stage_id, plan in stage_plans.items()
        for node in plan.nodes
    }
    runtime_input_identities = {id(tensor) for tensor in graph.inputs}
    initializer_identities = {
        id(tensor)
        for tensor in graph.initializers
    } | {
        id(tensor)
        for tensor in graph.tensors
        if tensor.is_initializer
    }

    inputs: list[VirtualInputTransition] = []
    intermediates: list[VirtualIntermediateTransition] = []
    for destination_stage_id in sorted(stage_plans):
        destination_plan = stage_plans[destination_stage_id]
        destination_node = destination_plan.nodes[0]
        destination_layouts = node_output_layouts(
            destination_plan,
            destination_node,
        )
        for destination_input_index, tensor in enumerate(destination_node.inputs):
            tensor_identity = id(tensor)
            if tensor_identity in initializer_identities:
                continue
            destinations = _required_input_slices(
                tensor=tensor,
                destination_node=destination_node,
                destination_output_layouts=destination_layouts,
                destination_input_index=destination_input_index,
            )
            source_node = producer_by_tensor_identity.get(tensor_identity)
            if source_node is None:
                if tensor_identity in runtime_input_identities:
                    inputs.append(
                        VirtualInputTransition(
                            tensor=tensor,
                            tensor_id=tensor_id_by_identity[tensor_identity],
                            destination_stage_id=destination_stage_id,
                            destination_input_index=destination_input_index,
                            destinations=tuple(
                                VirtualInputDestination(
                                    virtual_tile_id=tile.tile_id,
                                    tensor_slice=tensor_slice,
                                )
                                for tile, tensor_slice in sorted(
                                    destinations,
                                    key=lambda item: item[0].tile_id,
                                )
                            ),
                        )
                    )
                continue

            source_stage_id = stage_id_by_node_identity[id(source_node)]
            if source_stage_id == destination_stage_id:
                continue
            source_output_index = node_output_index(source_node, tensor)
            source_layout = node_output_layouts(
                stage_plans[source_stage_id],
                source_node,
            )[source_output_index]
            intermediates.append(
                VirtualIntermediateTransition(
                    tensor=tensor,
                    tensor_id=tensor_id_by_identity[tensor_identity],
                    source_stage_id=source_stage_id,
                    source_output_index=source_output_index,
                    destination_stage_id=destination_stage_id,
                    destination_input_index=destination_input_index,
                    transfers=_build_virtual_transfers(
                        tensor,
                        source_layout,
                        destinations,
                    ),
                )
            )

    outputs = tuple(
        _build_virtual_output_transition(
            tensor,
            tensor_id_by_identity[id(tensor)],
            producer_by_tensor_identity[id(tensor)],
            stage_id_by_node_identity,
            stage_plans,
        )
        for tensor in graph.outputs
    )
    return tuple(inputs) + tuple(intermediates) + outputs


def bind_transitions(
    virtual_transitions: tuple[VirtualTransition, ...],
    placements: dict[int, StagePlacement],
) -> tuple[Transition, ...]:
    """Bind only virtual tile endpoints, retaining all collection positions."""

    transitions: list[Transition] = []
    for transition in virtual_transitions:
        if isinstance(transition, VirtualInputTransition):
            placement = placements[transition.destination_stage_id]
            transitions.append(
                InputTransition(
                    tensor_id=transition.tensor_id,
                    destination_stage_id=transition.destination_stage_id,
                    destination_input_index=transition.destination_input_index,
                    destinations=tuple(
                        InputDestination(
                            tile_id=placement.physical_tile_id(
                                destination.virtual_tile_id
                            ),
                            tensor_slice=destination.tensor_slice,
                        )
                        for destination in transition.destinations
                    ),
                )
            )
        elif isinstance(transition, VirtualIntermediateTransition):
            source_placement = placements[transition.source_stage_id]
            destination_placement = placements[transition.destination_stage_id]
            transitions.append(
                IntermediateTransition(
                    tensor_id=transition.tensor_id,
                    source_stage_id=transition.source_stage_id,
                    source_output_index=transition.source_output_index,
                    destination_stage_id=transition.destination_stage_id,
                    destination_input_index=transition.destination_input_index,
                    transfers=tuple(
                        Transfer(
                            source_tile_id=source_placement.physical_tile_id(
                                transfer.source_virtual_tile_id
                            ),
                            destination_tile_id=(
                                destination_placement.physical_tile_id(
                                    transfer.destination_virtual_tile_id
                                )
                            ),
                            source_subslice=transfer.source_subslice,
                            destination_subslice=transfer.destination_subslice,
                        )
                        for transfer in transition.transfers
                    ),
                )
            )
        else:
            source_placement = placements[transition.source_stage_id]
            transitions.append(
                OutputTransition(
                    tensor_id=transition.tensor_id,
                    source_stage_id=transition.source_stage_id,
                    source_output_index=transition.source_output_index,
                    sources=tuple(
                        OutputSource(
                            tile_id=source_placement.physical_tile_id(
                                source.virtual_tile_id
                            ),
                            tensor_slice=source.tensor_slice,
                        )
                        for source in transition.sources
                    ),
                )
            )
    return tuple(transitions)


def _build_virtual_output_transition(
    tensor: Tensor,
    tensor_id: int,
    source_node: Node,
    stage_id_by_node_identity: dict[int, int],
    stage_plans: dict[int, StagePlan],
) -> VirtualOutputTransition:
    source_stage_id = stage_id_by_node_identity[id(source_node)]
    source_output_index = node_output_index(source_node, tensor)
    source_layout = node_output_layouts(
        stage_plans[source_stage_id],
        source_node,
    )[source_output_index]
    return VirtualOutputTransition(
        tensor=tensor,
        tensor_id=tensor_id,
        source_stage_id=source_stage_id,
        source_output_index=source_output_index,
        sources=tuple(
            VirtualOutputSource(
                virtual_tile_id=tile.tile_id,
                tensor_slice=tensor_slice,
            )
            for tile, tensor_slice in tile_owned_slices(tensor, source_layout)
        ),
    )


def _build_virtual_transfers(
    tensor: Tensor,
    source_layout: TensorLayout,
    destinations: tuple[tuple[Tile, TensorSlice], ...],
) -> tuple[VirtualTransfer, ...]:
    transfers: list[VirtualTransfer] = []
    for source_tile, source_slice in tile_owned_slices(tensor, source_layout):
        for destination_tile, destination_slice in destinations:
            overlap = _intersect_slice(source_slice, destination_slice)
            if overlap is None:
                continue
            transfers.append(
                VirtualTransfer(
                    source_virtual_tile_id=source_tile.tile_id,
                    destination_virtual_tile_id=destination_tile.tile_id,
                    source_subslice=_relative_subslice(source_slice, overlap),
                    destination_subslice=_relative_subslice(
                        destination_slice,
                        overlap,
                    ),
                )
            )
    return tuple(sorted(transfers, key=_virtual_transfer_sort_key))


def _required_input_slices(
    tensor: Tensor,
    destination_node: Node,
    destination_output_layouts: tuple[TensorLayout, ...],
    destination_input_index: int,
) -> tuple[tuple[Tile, TensorSlice], ...]:
    payload = cast(OpPayload, destination_node.payload)
    destinations = []
    for tile in destination_output_layouts[0].submesh.tiles:
        tile_work = payload.build_tile_work(
            output_layouts=destination_output_layouts,
            tile=tile,
        )
        if destination_input_index >= len(tile_work.input_slices):
            continue
        reference = tile_work.input_slices[destination_input_index]
        if reference.tensor is not tensor:
            raise ValueError(
                f"tile work input {destination_input_index} does not match "
                f"node {destination_node.name}"
            )
        destinations.append((tile, reference.tensor_slice))
    return tuple(destinations)


def _virtual_transfer_sort_key(transfer: VirtualTransfer) -> tuple:
    return (
        transfer.source_virtual_tile_id,
        transfer.destination_virtual_tile_id,
        tuple(
            (dimension.start, dimension.length)
            for dimension in transfer.source_subslice.dims
        ),
        tuple(
            (dimension.start, dimension.length)
            for dimension in transfer.destination_subslice.dims
        ),
    )


def _intersect_slice(
    left: TensorSlice,
    right: TensorSlice,
) -> TensorSlice | None:
    if left.rank != right.rank:
        raise ValueError("cannot intersect slices with different ranks")
    dimensions = []
    for left_dimension, right_dimension in zip(left.dims, right.dims):
        start = max(left_dimension.start, right_dimension.start)
        end = min(
            left_dimension.start + left_dimension.length,
            right_dimension.start + right_dimension.length,
        )
        if start >= end:
            return None
        dimensions.append(TensorRange(start=start, length=end - start))
    return TensorSlice(rank=left.rank, dims=tuple(dimensions))


def _relative_subslice(
    parent: TensorSlice,
    child: TensorSlice,
) -> TensorSubSlice:
    if parent.rank != child.rank:
        raise ValueError("cannot build subslice from slices with different ranks")
    return TensorSubSlice(
        parent=parent,
        dims=tuple(
            TensorRange(
                start=child_dimension.start - parent_dimension.start,
                length=child_dimension.length,
            )
            for parent_dimension, child_dimension in zip(
                parent.dims,
                child.dims,
            )
        ),
    )


def tile_owned_slices(tensor: Tensor, layout: TensorLayout) -> tuple[tuple[Tile, TensorSlice], ...]:
    """Return the concrete slice owned by each tile in one submesh."""

    owned: list[tuple[Tile, TensorSlice]] = []
    for tile in layout.submesh.tiles:
        owned.append(
            (
                tile,
                tile_tensor_slice(
                    tensor=tensor,
                    layout=layout,
                    tile=tile,
                ),
            )
        )
    return tuple(owned)
