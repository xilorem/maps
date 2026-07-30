"""Execution Plan lowering from canonical Virtual Transitions."""

from __future__ import annotations

from dataclasses import replace

from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node
from MAPS.pipeline.execution import ExecutionContract
from MAPS.pipeline.execution_plan import ExecutionPlan
from MAPS.pipeline.layer import Layer, LayerInput, LayerOutput
from MAPS.pipeline.stage import Stage
from MAPS.planner.contracts.queries import (
    node_output_layouts,
    required_input_slices,
)
from MAPS.planner.contracts.stages import StagePlacement, StagePlan
from MAPS.planner.lowering.context import (
    PipelineLoweringContext,
    build_lowering_context,
)
from MAPS.transitions import bind_transitions
from MAPS.transitions.contracts import (
    InputDestination,
    InputTransition,
    IntermediateTransition,
    Transition,
    VirtualTransition,
)


def lower_execution_plan(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    virtual_transitions: tuple[VirtualTransition, ...],
    *,
    execution: ExecutionContract = ExecutionContract(),
) -> ExecutionPlan:
    """Bind retained communication and lower one complete Execution Plan."""

    if set(placements) != set(stage_plans):
        raise ValueError("placements must contain exactly one entry per stage plan")

    context = build_lowering_context(graph, stage_plans)
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
        (transition.destination_stage_id, transition.destination_input_index): (
            transition_id
        )
        for transition_id, transition in enumerate(transitions)
        if isinstance(transition, (InputTransition, IntermediateTransition))
    }


def _build_stage(
    stage_id: int,
    plan: StagePlan,
    placement: StagePlacement,
    context: PipelineLoweringContext,
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
    context: PipelineLoweringContext,
    transition_ids: dict[tuple[int, int], int],
    initializer_ids: set[int],
) -> Layer:
    output_layouts = node_output_layouts(plan, node)
    return Layer(
        node=node,
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
    context: PipelineLoweringContext,
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

    if layer_index != 0:
        raise ValueError(
            "runtime and cross-stage inputs must target the first layer of a stage"
        )
    return LayerInput.transition_source(
        tensor_id,
        transition_ids[(stage_id, input_index)],
    )


__all__ = ["lower_execution_plan"]
