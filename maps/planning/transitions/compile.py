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
    bounding_tensor_slice,
    tile_tensor_slice,
)
from maps.operations.contracts import OpPayload, TileWork, input_slices_for_tensor
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


ResidentDestinations = tuple[tuple[Tile, TensorSlice], ...]
StageDemands = tuple[tuple[Tensor, ResidentDestinations], ...]


class VirtualTransitionCompiler:
    """Reuse graph topology and resident demands across allocation trials.

    The compiler belongs to one graph and fixed stage formation. Only resident
    demands are retained; potentially large pairwise transfer lists are returned
    to the caller rather than cached here.
    """

    def __init__(self, graph: Graph, stage_plans: dict[int, StagePlan]) -> None:
        self.graph = graph
        self.tensor_ids = {id(tensor): i for i, tensor in enumerate(graph.tensors)}
        self.producers = {
            id(tensor): node for node in graph.nodes for tensor in node.outputs
        }
        self.stage_ids = {
            id(node): stage for stage, plan in stage_plans.items() for node in plan.nodes
        }
        self.runtime_inputs = {id(tensor) for tensor in graph.inputs}
        self.initializers = {id(tensor) for tensor in graph.initializers} | {
            id(tensor) for tensor in graph.tensors if tensor.is_initializer
        }
        self.edges = tuple(sorted({
            (self.stage_ids[id(self.producers[id(tensor)])], stage)
            for stage, plan in stage_plans.items()
            for node in plan.nodes for tensor in node.inputs
            if id(tensor) not in self.initializers and id(tensor) in self.producers
            and self.stage_ids[id(self.producers[id(tensor)])] != stage
        }))
        # Retain each plan alongside its identity key to prevent id reuse.
        self._demands: dict[int, tuple[StagePlan, StageDemands]] = {}

    def set_demands(self, plan: StagePlan, demands: StageDemands) -> None:
        """Accept resident demands already derived during candidate analysis."""
        self._demands.setdefault(id(plan), (plan, demands))

    def demands(self, plan: StagePlan) -> StageDemands:
        """Return one resident bounding slice per external tensor and tile."""
        key = id(plan)
        if key not in self._demands:
            self._demands[key] = (plan, stage_resident_demands(
                plan.nodes, plan.node_output_layouts, self.initializers,
            ))
        return self._demands[key][1]

    def inputs(self, plan: StagePlan) -> tuple[VirtualInputTransition, ...]:
        return tuple(
            VirtualInputTransition(
                tensor=tensor,
                tensor_id=self.tensor_ids[id(tensor)],
                destination_stage_id=plan.stage_id,
                destinations=tuple(
                    VirtualInputDestination(virtual_tile_id=tile.tile_id, tensor_slice=slice_)
                    for tile, slice_ in sorted(destinations, key=lambda item: item[0].tile_id)
                ),
            )
            for tensor, destinations in self.demands(plan)
            if id(tensor) not in self.producers and id(tensor) in self.runtime_inputs
        )

    def outputs(self, plan: StagePlan) -> tuple[VirtualOutputTransition, ...]:
        return tuple(
            _build_virtual_output_transition(
                tensor, self.tensor_ids[id(tensor)], self.producers[id(tensor)],
                self.stage_ids, {plan.stage_id: plan},
            )
            for tensor in self.graph.outputs
            if self.stage_ids[id(self.producers[id(tensor)])] == plan.stage_id
        )

    def _intermediate(
        self,
        tensor: Tensor,
        destinations: ResidentDestinations,
        source_plan: StagePlan,
        destination_plan: StagePlan,
    ) -> VirtualIntermediateTransition:
        producer = self.producers[id(tensor)]
        layout = node_output_layouts(source_plan, producer)[
            node_output_index(producer, tensor)
        ]
        return VirtualIntermediateTransition(
            tensor=tensor, tensor_id=self.tensor_ids[id(tensor)],
            source_stage_id=source_plan.stage_id,
            destination_stage_id=destination_plan.stage_id,
            transfers=_build_virtual_transfers(tensor, layout, destinations),
        )

    def intermediates(
        self, source: StagePlan, destination: StagePlan,
    ) -> tuple[VirtualIntermediateTransition, ...]:
        """Compile all tensors on one stage edge together."""
        return tuple(
            self._intermediate(tensor, destinations, source, destination)
            for tensor, destinations in self.demands(destination)
            if id(tensor) in self.producers
            and self.stage_ids[id(self.producers[id(tensor)])] == source.stage_id
        )

    def build(self, plans: dict[int, StagePlan]) -> tuple[VirtualTransition, ...]:
        inputs: list[VirtualInputTransition] = []
        intermediates: list[VirtualIntermediateTransition] = []
        for stage in sorted(plans):
            plan = plans[stage]
            inputs.extend(self.inputs(plan))
            for tensor, destinations in self.demands(plan):
                producer = self.producers.get(id(tensor))
                if producer is not None:
                    intermediates.append(self._intermediate(
                        tensor, destinations, plans[self.stage_ids[id(producer)]], plan,
                    ))
        outputs = tuple(
            _build_virtual_output_transition(
                tensor, self.tensor_ids[id(tensor)], self.producers[id(tensor)],
                self.stage_ids, plans,
            )
            for tensor in self.graph.outputs
        )
        return tuple(inputs) + tuple(intermediates) + outputs


def stage_resident_demands(
    nodes: tuple[Node, ...],
    layouts: tuple[tuple[TensorLayout, ...], ...],
    initializer_identities: set[int],
    node_tile_work: tuple[tuple[TileWork, ...], ...] | None = None,
) -> StageDemands:
    """Derive resident external inputs, optionally from existing tile work."""
    local_outputs = {id(tensor) for node in nodes for tensor in node.outputs}
    demanded: dict[int, tuple[Tensor, list[tuple[Tile, TensorSlice]]]] = {}
    for index, (node, output_layouts) in enumerate(zip(nodes, layouts)):
        for tensor in node.inputs:
            identity = id(tensor)
            if identity in initializer_identities or identity in local_outputs or tensor.is_initializer:
                continue
            _, slices = demanded.setdefault(identity, (tensor, []))
            slices.extend(_required_input_slices(
                tensor, node, output_layouts,
                None if node_tile_work is None else node_tile_work[index],
            ))
    return tuple((tensor, _resident_destinations(slices)) for tensor, slices in demanded.values())


def build_virtual_transitions(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
) -> tuple[VirtualTransition, ...]:
    """Compile every graph boundary and cross-stage dependency."""
    return VirtualTransitionCompiler(graph, stage_plans).build(stage_plans)


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
                    destination_stage_id=transition.destination_stage_id,
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


def _resident_destinations(
    destinations: list[tuple[Tile, TensorSlice]],
) -> tuple[tuple[Tile, TensorSlice], ...]:
    tile_by_id: dict[int, Tile] = {}
    slices_by_tile_id: dict[int, list[TensorSlice]] = {}
    for tile, tensor_slice in destinations:
        tile_by_id.setdefault(tile.tile_id, tile)
        slices_by_tile_id.setdefault(tile.tile_id, []).append(tensor_slice)
    return tuple(
        (
            tile_by_id[tile_id],
            bounding_tensor_slice(tuple(slices)),
        )
        for tile_id, slices in slices_by_tile_id.items()
    )


def _required_input_slices(
    tensor: Tensor,
    destination_node: Node,
    destination_output_layouts: tuple[TensorLayout, ...],
    tile_work: tuple[TileWork, ...] | None = None,
) -> tuple[tuple[Tile, TensorSlice], ...]:
    payload = cast(OpPayload, destination_node.payload)
    destinations = []
    for index, tile in enumerate(destination_output_layouts[0].submesh.tiles):
        work = (
            tile_work[index] if tile_work is not None
            else payload.build_tile_work(output_layouts=destination_output_layouts, tile=tile)
        )
        destinations.extend(
            (tile, tensor_slice)
            for tensor_slice in input_slices_for_tensor(work, tensor)
        )
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
