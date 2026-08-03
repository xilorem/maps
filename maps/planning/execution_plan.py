"""Physical Execution Plan models owned by Planning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import re
from typing import cast

from maps.graph import Graph, Node, Tensor
from maps.hardware import EndpointKind, Mesh, Tile
from maps.operations.contracts import OpPayload
from maps.planning.allocation.candidates import (
    permanent_l1_allocation_bytes,
    stage_l1_allocation_bytes,
)
from maps.planning.stage_latency import estimate_stage_latency
from maps.planning.mapping import (
    Submesh,
    TensorLayout,
    TensorRange,
    TensorSlice,
    bounding_tensor_slice,
    tensor_slice_num_bytes,
    tile_tensor_slice,
)
from maps.planning.placement.evaluation import evaluate_placement
from maps.planning.stages import (
    StagePlacement,
    StagePlan,
    node_output_layouts,
    required_input_slices,
    virtual_submesh,
)
from maps.planning.transitions import bind_transitions
from maps.planning.transitions.contracts import (
    InputDestination,
    InputTransition,
    IntermediateTransition,
    Transition,
    VirtualTransition,
)


@dataclass(frozen=True)
class CollectiveGroup:
    """Exact physical participants for one Intra-Stage Collective invocation."""

    tile_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.tile_ids:
            raise ValueError("Collective Groups must not be empty")
        if tuple(sorted(set(self.tile_ids))) != self.tile_ids:
            raise ValueError("Collective Group tile ids must be unique and sorted")


@dataclass(frozen=True)
class ExecutionContract:
    """Execution settings that affect planning and backend allocation."""

    num_token_slots: int = 2

    def __post_init__(self) -> None:
        if self.num_token_slots <= 0:
            raise ValueError("num_token_slots must be > 0")


@dataclass(frozen=True)
class InitializerInput:
    """Per-tile residency for an immutable input Tensor."""

    destinations: tuple[InputDestination, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TransitionSource:
    """Layer input supplied by an Input or Intermediate Transition."""

    transition_id: int

    def __post_init__(self) -> None:
        if self.transition_id < 0:
            raise ValueError("transition sources require transition_id >= 0")


@dataclass(frozen=True)
class LocalInput:
    """Layer input read from a previous Layer output in the same Stage."""

    layer_idx: int
    tensor_id: int

    def __post_init__(self) -> None:
        if self.layer_idx < 0 or self.tensor_id < 0:
            raise ValueError("layer_idx and tensor_id must be >= 0")


LayerInputSource = InitializerInput | TransitionSource | LocalInput


@dataclass(frozen=True)
class LayerInput:
    """One input of a Layer."""

    tensor_id: int
    source: LayerInputSource

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")

    @classmethod
    def initializer(
        cls,
        tensor_id: int,
        destinations: tuple[InputDestination, ...],
    ) -> "LayerInput":
        return cls(
            tensor_id=tensor_id,
            source=InitializerInput(destinations=destinations),
        )

    @classmethod
    def transition_source(
        cls,
        tensor_id: int,
        transition_id: int,
    ) -> "LayerInput":
        return cls(
            tensor_id=tensor_id,
            source=TransitionSource(transition_id=transition_id),
        )

    @classmethod
    def local(cls, tensor_id: int, layer_idx: int) -> "LayerInput":
        return cls(
            tensor_id=tensor_id,
            source=LocalInput(layer_idx=layer_idx, tensor_id=tensor_id),
        )


@dataclass(frozen=True)
class LayerOutput:
    """One output of a Layer."""

    tensor_id: int
    layout: TensorLayout

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")


@dataclass(frozen=True)
class Layer:
    """One scheduled Graph Node inside a Stage."""

    node: Node
    inputs: tuple[LayerInput, ...] = field(default_factory=tuple)
    outputs: tuple[LayerOutput, ...] = field(default_factory=tuple)
    device_name: str | None = None
    source_operation: str | None = None
    collective_groups: tuple[CollectiveGroup, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.source_operation is None:
            object.__setattr__(self, "source_operation", self.node.source_operation)
        elif self.source_operation != self.node.source_operation:
            raise ValueError("Layer source_operation must match its Graph Node")

    def validate_tensors(self, tensors: tuple[Tensor, ...]) -> None:
        """Validate bound Tensor ids and output layout compatibility."""

        for layer_input in self.inputs:
            if layer_input.tensor_id >= len(tensors):
                raise ValueError(f"input tensor_id out of range: {layer_input.tensor_id}")
        for layer_output in self.outputs:
            if layer_output.tensor_id >= len(tensors):
                raise ValueError(
                    f"output tensor_id out of range: {layer_output.tensor_id}"
                )
            layer_output.layout.validate_for(tensors[layer_output.tensor_id])


@dataclass(frozen=True)
class Stage:
    """One scheduled execution unit on a physical Submesh."""

    name: str
    submesh: Submesh
    layers: tuple[Layer, ...] = field(default_factory=tuple)
    virtual_to_physical: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if not self.layers:
            raise ValueError("stages must contain at least one layer")

    @property
    def physical_to_virtual(self) -> dict[int, int]:
        """Return physical tile ids keyed to their virtual tile ids."""

        return {
            physical_tile_id: virtual_tile_id
            for virtual_tile_id, physical_tile_id in self.virtual_to_physical.items()
        }

    def validate_tensors(self, tensors: tuple[Tensor, ...]) -> None:
        """Validate Layer Tensor ids and output layout compatibility."""

        for layer in self.layers:
            layer.validate_tensors(tensors)


@dataclass(frozen=True)
class ExecutionPlan:
    """One complete physical execution decision."""

    name: str
    mesh: Mesh
    tensors: tuple[Tensor, ...] = field(default_factory=tuple)
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    transitions: tuple[Transition, ...] = field(default_factory=tuple)
    execution: ExecutionContract = field(default_factory=ExecutionContract)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("execution plan name must not be empty")


@dataclass(frozen=True)
class ExecutionPlanConstructionContext:
    """Precomputed identity indexes required during Execution Plan construction.

    Graph Nodes and Tensors are immutable domain objects, but several
    construction decisions depend on object identity rather than value equality.
    Building these indexes once makes that rule explicit and prevents each
    construction component from reconstructing subtly different producer or
    Stage maps.
    """

    graph: Graph
    stage_formation: dict[int, tuple[Node, ...]]
    node_stage_ids: dict[int, int]
    node_stage_layer_ids: dict[int, int]
    node_graph_layer_ids: dict[int, int]
    tensor_id_by_tensor: dict[object, int]
    producer_by_tensor: dict[object, Node]


def build_construction_context(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
) -> ExecutionPlanConstructionContext:
    """Index Graph ownership and ordering for consistent construction."""

    stage_formation = _resolve_stage_formation(graph, stage_plans)
    return ExecutionPlanConstructionContext(
        graph=graph,
        stage_formation=stage_formation,
        node_stage_ids={
            id(node): stage_id
            for stage_id, stage_nodes in stage_formation.items()
            for node in stage_nodes
        },
        node_stage_layer_ids={
            id(node): layer_idx
            for stage_nodes in stage_formation.values()
            for layer_idx, node in enumerate(stage_nodes)
        },
        node_graph_layer_ids={
            id(node): layer_id
            for layer_id, node in enumerate(graph.nodes)
        },
        tensor_id_by_tensor={
            tensor: tensor_id
            for tensor_id, tensor in enumerate(graph.tensors)
        },
        producer_by_tensor={
            tensor: node
            for node in graph.nodes
            for tensor in node.outputs
        },
    )


def _resolve_stage_formation(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
) -> dict[int, tuple[Node, ...]]:
    """Validate and recover the selected nodes carried by stage plans."""

    if any(not plan.nodes for plan in stage_plans.values()):
        raise ValueError("every stage plan must contain its selected nodes")
    stage_formation = {
        stage_id: plan.nodes
        for stage_id, plan in stage_plans.items()
    }
    selected_node_ids = [
        id(node)
        for nodes in stage_formation.values()
        for node in nodes
    ]
    graph_node_ids = {id(node) for node in graph.nodes}
    if len(selected_node_ids) != len(set(selected_node_ids)):
        raise ValueError("stage plans contain a graph node more than once")
    if set(selected_node_ids) != graph_node_ids:
        raise ValueError("stage plans must cover every graph node exactly once")
    return stage_formation


def construct_execution_plan(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    virtual_transitions: tuple[VirtualTransition, ...],
    *,
    execution: ExecutionContract = ExecutionContract(),
) -> ExecutionPlan:
    """Combine retained decisions into one complete physical Execution Plan."""

    if set(placements) != set(stage_plans):
        raise ValueError("placements must contain exactly one entry per stage plan")

    context = build_construction_context(graph, stage_plans)
    transitions = bind_transitions(virtual_transitions, placements)
    transition_ids = _destination_transition_ids(transitions)
    initializer_ids = {
        id(tensor)
        for tensor in graph.initializers
    } | {
        id(tensor)
        for tensor in graph.tensors
        if tensor.is_initializer
    }
    stages = tuple(
        _build_stage(
            stage_id,
            stage_plans[stage_id],
            placements[stage_id],
            context,
            transition_ids,
            initializer_ids,
        )
        for stage_id in sorted(stage_plans)
    )
    return ExecutionPlan(
        name=graph.name,
        mesh=mesh,
        tensors=tuple(
            replace(tensor, is_initializer=id(tensor) in initializer_ids)
            for tensor in graph.tensors
        ),
        stages=stages,
        transitions=transitions,
        execution=execution,
    )


def _destination_transition_ids(
    transitions: tuple[Transition, ...],
) -> dict[tuple[int, int], int]:
    return {
        (transition.destination_stage_id, transition.tensor_id): transition_id
        for transition_id, transition in enumerate(transitions)
        if isinstance(transition, (InputTransition, IntermediateTransition))
    }


def _build_stage(
    stage_id: int,
    plan: StagePlan,
    placement: StagePlacement,
    context: ExecutionPlanConstructionContext,
    transition_ids: dict[tuple[int, int], int],
    initializer_ids: set[int],
) -> Stage:
    return Stage(
        name="+".join(node.name for node in plan.nodes),
        submesh=placement.physical_submesh,
        virtual_to_physical=placement.virtual_to_physical,
        layers=tuple(
            _build_layer(
                stage_id,
                layer_index,
                node,
                plan,
                placement,
                context,
                transition_ids,
                initializer_ids,
            )
            for layer_index, node in enumerate(plan.nodes)
        ),
    )


def _build_layer(
    stage_id: int,
    layer_index: int,
    node: Node,
    plan: StagePlan,
    placement: StagePlacement,
    context: ExecutionPlanConstructionContext,
    transition_ids: dict[tuple[int, int], int],
    initializer_ids: set[int],
) -> Layer:
    output_layouts = node_output_layouts(plan, node)
    return Layer(
        node=node,
        device_name=plan.device_names[layer_index],
        source_operation=node.source_operation,
        collective_groups=tuple(
            CollectiveGroup(
                tuple(
                    sorted(
                        placement.physical_tile_id(virtual_tile_id)
                        for virtual_tile_id in virtual_group.virtual_tile_ids
                    )
                )
            )
            for virtual_group in plan.virtual_collective_groups[layer_index]
        ),
        inputs=tuple(
            _build_layer_input(
                stage_id,
                layer_index,
                input_index,
                tensor,
                node,
                output_layouts,
                placement,
                context,
                transition_ids,
                initializer_ids,
            )
            for input_index, tensor in enumerate(node.inputs)
        ),
        outputs=tuple(
            LayerOutput(
                tensor_id=context.tensor_id_by_tensor[tensor],
                layout=layout,
            )
            for tensor, layout in zip(node.outputs, output_layouts)
        ),
    )


def _build_layer_input(
    stage_id: int,
    layer_index: int,
    input_index: int,
    tensor: object,
    node: Node,
    output_layouts: tuple,
    placement: StagePlacement,
    context: ExecutionPlanConstructionContext,
    transition_ids: dict[tuple[int, int], int],
    initializer_ids: set[int],
) -> LayerInput:
    tensor_id = context.tensor_id_by_tensor[tensor]
    if id(tensor) in initializer_ids:
        destinations = required_input_slices(
            tensor=tensor,
            destination_node=node,
            destination_output_layouts=output_layouts,
        )
        return LayerInput.initializer(
            tensor_id,
            tuple(
                InputDestination(
                    tile_id=placement.physical_tile_id(tile.tile_id),
                    tensor_slice=tensor_slice,
                )
                for tile, tensor_slice in destinations
            ),
        )

    producer = context.producer_by_tensor.get(tensor)
    if producer is not None and context.node_stage_ids[id(producer)] == stage_id:
        return LayerInput.local(
            tensor_id,
            context.node_stage_layer_ids[id(producer)],
        )

    return LayerInput.transition_source(
        tensor_id,
        transition_ids[(stage_id, tensor_id)],
    )


def print_submeshes(execution_plan: ExecutionPlan) -> None:
    """Print one Execution Plan's Stage placement on the attached NoC."""

    mesh = execution_plan.mesh
    submesh_labels_by_tile_id: dict[int, list[str]] = defaultdict(list)
    for stage in execution_plan.stages:
        label = str(stage.submesh.submesh_id)
        for tile in stage.submesh.tiles:
            submesh_labels_by_tile_id[tile.tile_id].append(label)

    labels_by_node_id: dict[int, list[str]] = defaultdict(list)
    for endpoint in mesh.noc.endpoints:
        if endpoint.kind is EndpointKind.L1 and endpoint.tile_id is not None:
            labels = submesh_labels_by_tile_id.get(endpoint.tile_id)
            if labels:
                labels_by_node_id[endpoint.node_id].append("/".join(labels))
        elif endpoint.kind is EndpointKind.L2:
            labels_by_node_id[endpoint.node_id].append(
                _compact_l2_label(endpoint.name or "L2")
            )
        else:
            labels_by_node_id[endpoint.node_id].append(endpoint.kind.name)

    max_x = max(node.x for node in mesh.noc.nodes)
    max_y = max(node.y for node in mesh.noc.nodes)
    cell_strings: dict[tuple[int, int], str] = {}
    max_cell_width = 2

    for node in mesh.noc.nodes:
        labels = labels_by_node_id.get(node.node_id)
        cell = "/".join(labels) if labels else "."
        cell_strings[(node.x, node.y)] = cell
        max_cell_width = max(max_cell_width, len(cell))

    for y in range(max_y + 1):
        row = " ".join(
            cell_strings[(x, y)].rjust(max_cell_width)
            for x in range(max_x + 1)
        )
        print(row)


def _compact_l2_label(label: str) -> str:
    match = re.fullmatch(r"l2_(\d+)", label)
    if match is None:
        return label
    return f"L{_base36(int(match.group(1)))}"


def _base36(value: int) -> str:
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value < 36:
        return digits[value]
    result = []
    while value:
        value, remainder = divmod(value, 36)
        result.append(digits[remainder])
    return "".join(reversed(result))


def print_execution_plan_stage_cost(
    execution_plan: ExecutionPlan,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    virtual_transitions: tuple[VirtualTransition, ...],
    stage_latency_weight: float = 1.0,
    communication_weight: float = 1.0,
) -> None:
    """Print the combined worst-stage latency and physical IO estimate.

    Stage Latency is evaluated from the final physical collective bindings.
    Physical IO remains a separate signal from the spatial placement.
    """

    stage_latencies = {
        plan.stage_id: estimate_stage_latency(
            stage_nodes=plan.nodes,
            node_output_layouts=plan.node_output_layouts,
            virtual_tiles=virtual_submesh(plan).tiles,
            device_names=plan.device_names,
            virtual_collective_groups=plan.virtual_collective_groups,
            physical_tiles_by_virtual_id={
                virtual_id: execution_plan.mesh.tile_by_id(physical_id)
                for virtual_id, physical_id in placements[
                    plan.stage_id
                ].virtual_to_physical.items()
            },
        )
        for plan in stage_plans.values()
    }
    worst_stage_latency = max(stage_latencies.values(), default=0)
    evaluation = evaluate_placement(
        execution_plan.mesh,
        stage_plans,
        placements,
        virtual_transitions,
        stage_latency_weight=stage_latency_weight,
        communication_weight=communication_weight,
    )
    worst_external_communication = max(
        (
            breakdown.l1_write + breakdown.l2_read + breakdown.l2_write
            for breakdown in evaluation.stage_breakdowns.values()
        ),
        default=0,
    )
    weighted_stage_bottleneck = max(
        (
            max(
                stage_latency_weight
                * stage_latencies[plan.stage_id],
                communication_weight
                * (
                    evaluation.stage_breakdowns[plan.stage_id].l1_write
                    + evaluation.stage_breakdowns[plan.stage_id].l2_read
                    + evaluation.stage_breakdowns[plan.stage_id].l2_write
                ),
            )
            for plan in stage_plans.values()
        ),
        default=0,
    )
    print(
        "[planner] execution_plan_stage_cost="
        f"{weighted_stage_bottleneck} "
        f"(worst_stage_latency={worst_stage_latency} "
        f"worst_external_communication={worst_external_communication})"
    )


def estimate_stage_l1_memory_for_tile(
    stage: Stage,
    execution_plan: ExecutionPlan,
    tile: Tile,
) -> int:
    """Estimate the backend's permanent allocation for one stage tile."""

    virtual_tile = virtual_tile_for_stage_tile(stage, execution_plan, tile)
    resident_slices: dict[int, list[TensorSlice]] = {}
    for layer in stage.layers:
        for binding_idx, binding in enumerate(layer.inputs):
            if isinstance(binding.source, TransitionSource):
                resident_slices.setdefault(binding.tensor_id, []).append(
                    infer_input_slice_for_tile(
                        layer,
                        binding_idx,
                        execution_plan,
                        virtual_tile,
                    )
                )
    resident_bounds = {
        tensor_id: bounding_tensor_slice(tuple(slices))
        for tensor_id, slices in resident_slices.items()
    }
    allocated_resident_tensors: set[int] = set()
    allocation_sizes = []
    for layer in stage.layers:
        for binding_idx, binding in enumerate(layer.inputs):
            if isinstance(binding.source, LocalInput):
                continue
            tensor = execution_plan.tensors[binding.tensor_id]
            if isinstance(binding.source, InitializerInput):
                destination = next(
                    (
                        destination
                        for destination in binding.source.destinations
                        if destination.tile_id == tile.tile_id
                    ),
                    None,
                )
                if destination is None:
                    continue
                tensor_slice = destination.tensor_slice
                slot_count = 1
            else:
                if binding.tensor_id in allocated_resident_tensors:
                    continue
                allocated_resident_tensors.add(binding.tensor_id)
                tensor_slice = resident_bounds[binding.tensor_id]
                slot_count = execution_plan.execution.num_token_slots
            allocation_sizes.append(
                tensor_slice_num_bytes(tensor, tensor_slice) * slot_count
            )
        for binding in layer.outputs:
            tensor = execution_plan.tensors[binding.tensor_id]
            tensor_slice = tile_tensor_slice(tensor, binding.layout, virtual_tile)
            allocation_sizes.append(
                tensor_slice_num_bytes(tensor, tensor_slice)
                * execution_plan.execution.num_token_slots
            )
    permanent_l1_bytes = permanent_l1_allocation_bytes(allocation_sizes)
    scratch_l1_bytes = max(
        (
            _execution_layer_scratch_l1_bytes(layer, tile, virtual_tile)
            for layer in stage.layers
        ),
        default=0,
    )
    return stage_l1_allocation_bytes(permanent_l1_bytes, scratch_l1_bytes)


def _execution_layer_scratch_l1_bytes(
    layer: Layer,
    physical_tile: Tile,
    virtual_tile: Tile,
) -> int:
    """Price scratch only after the Layer's execution contract is resolvable."""

    if (
        layer.device_name is None
        or not layer.outputs
        or not hasattr(layer.node.payload, "build_tile_work")
    ):
        return 0
    try:
        device = physical_tile.device_by_name(layer.device_name)
    except ValueError:
        return 0
    payload = cast(OpPayload, layer.node.payload)
    return device.temporary_l1_bytes(
        payload.build_tile_work(
            output_layouts=tuple(output.layout for output in layer.outputs),
            tile=virtual_tile,
        )
    )


def estimate_stage_l2_memory(
    stage: Stage,
    execution_plan: ExecutionPlan,
) -> int:
    """Estimate L2 storage needed for a stage's external input bindings."""

    runtime_bindings: dict[int, list[tuple[Layer, int]]] = {}
    for layer in stage.layers:
        for binding_idx, binding in enumerate(layer.inputs):
            is_runtime_input = (
                isinstance(binding.source, TransitionSource)
                and binding.source.transition_id < len(execution_plan.transitions)
                and isinstance(
                    execution_plan.transitions[binding.source.transition_id],
                    InputTransition,
                )
            )
            if not is_runtime_input:
                continue
            runtime_bindings.setdefault(
                cast(TransitionSource, binding.source).transition_id,
                [],
            ).append((layer, binding_idx))

    l2_memory = 0
    for bindings in runtime_bindings.values():
        tensor_id = bindings[0][0].inputs[bindings[0][1]].tensor_id
        tensor = execution_plan.tensors[tensor_id]
        max_binding_bytes = 0
        for tile in stage.submesh.tiles:
            slices = []
            for layer, binding_idx in bindings:
                virtual_tile = virtual_tile_for_stage_tile(
                    stage,
                    execution_plan,
                    tile,
                )
                slices.append(
                    infer_input_slice_for_tile(
                        layer,
                        binding_idx,
                        execution_plan,
                        virtual_tile,
                    )
                )
            max_binding_bytes = max(
                max_binding_bytes,
                tensor_slice_num_bytes(
                    tensor,
                    bounding_tensor_slice(tuple(slices)),
                ),
            )
        l2_memory += max_binding_bytes
    return l2_memory


def infer_input_slice_for_tile(
    layer: Layer,
    binding_idx: int,
    execution_plan: ExecutionPlan,
    tile: Tile,
) -> TensorSlice:
    """Infer an input slice from tile work, falling back to the full tensor."""

    tensor = execution_plan.tensors[layer.inputs[binding_idx].tensor_id]
    node = layer.node
    if node.payload is not None and layer.outputs:
        output_layouts = tuple(output.layout for output in layer.outputs)
        tile_work = cast(OpPayload, node.payload).build_tile_work(
            output_layouts=output_layouts,
            tile=tile,
        )
        for reference in tile_work.input_slices:
            if tensor == reference.tensor:
                return reference.tensor_slice
    return _default_tensor_slice(tensor)


def virtual_tile_for_stage_tile(
    stage: Stage,
    execution_plan: ExecutionPlan,
    tile: Tile,
) -> Tile:
    """Translate one physical stage tile to the virtual layout tile."""

    if not stage.physical_to_virtual:
        return tile
    return execution_plan.mesh.tile_by_id(stage.physical_to_virtual[tile.tile_id])


def _default_tensor_slice(tensor: Tensor) -> TensorSlice:
    """Return a slice covering an entire tensor."""

    return TensorSlice(
        rank=tensor.rank,
        dims=tuple(
            TensorRange(start=0, length=dimension)
            for dimension in tensor.dims
        ),
    )


__all__ = [
    "CollectiveGroup",
    "ExecutionContract",
    "ExecutionPlan",
    "InitializerInput",
    "Layer",
    "LayerInput",
    "LayerInputSource",
    "LayerOutput",
    "LocalInput",
    "Stage",
    "TransitionSource",
    "construct_execution_plan",
    "estimate_stage_l1_memory_for_tile",
    "estimate_stage_l2_memory",
    "print_execution_plan_stage_cost",
    "print_submeshes",
]
