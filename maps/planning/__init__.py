"""Plan target-specialized Graphs for physical execution."""

from __future__ import annotations

from maps.graph import Graph
from maps.hardware import Mesh

from .execution_plan import (
    CollectiveGroup,
    ExecutionContract,
    ExecutionPlan,
    InitializerInput,
    Layer,
    LayerInput,
    LayerInputSource,
    LayerOutput,
    LocalInput,
    Stage,
    TransitionSource,
    construct_execution_plan,
    print_execution_plan_stage_cost,
)
from .stages import VirtualCollectiveGroup
from .options import (
    AllocationOptions,
    PlacementOptions,
    PlanningOptions,
    StageFormationOptions,
)
from .validation import (
    ConstraintReport,
    ConstraintViolation,
    PlanningConstraints,
    require_valid_execution_plan,
    validate_execution_plan,
)


def plan(
    graph: Graph,
    mesh: Mesh,
    options: PlanningOptions | None = None,
) -> ExecutionPlan:
    """Return a validated physical Execution Plan for a specialized Graph.

    Planning deliberately starts after model import, Graph Rewrites, and Target
    Specialization. It returns an in-memory plan and performs no Deployment or
    filesystem work.
    """

    from maps.planning.placement import place
    from maps.planning.transitions import build_virtual_transitions
    from maps.planning.allocation import allocate
    from maps.planning.stages import form_stages

    options = options or PlanningOptions()

    stage_formation = form_stages(
        graph,
        options.stage_formation,
    )
    stage_plans = allocate(
        graph,
        mesh,
        stage_formation=stage_formation,
        debug=options.allocation.print_progress,
        stage_latency_weight=options.allocation.stage_latency_weight,
        communication_weight=options.allocation.communication_weight,
        num_token_slots=options.execution.num_token_slots,
    )
    virtual_transitions = build_virtual_transitions(graph, stage_plans)
    placements = place(
        mesh,
        stage_plans,
        virtual_transitions,
        show_progress=options.placement.print_progress,
        print_placement=options.placement.print_placement,
        print_costs=options.placement.print_costs,
        stage_latency_weight=options.allocation.stage_latency_weight,
        communication_weight=options.allocation.communication_weight,
    )
    execution_plan = construct_execution_plan(
        graph,
        mesh,
        stage_plans,
        placements,
        virtual_transitions,
        execution=options.execution,
    )
    require_valid_execution_plan(
        execution_plan,
        PlanningConstraints(
            max_stage_operations=options.stage_formation.max_stage_operations,
        ),
        error_prefix="planner produced an invalid Execution Plan",
    )

    if options.print_execution_plan_cost:
        print_execution_plan_stage_cost(
            execution_plan,
            stage_plans,
            placements,
            virtual_transitions,
            stage_latency_weight=options.allocation.stage_latency_weight,
            communication_weight=options.allocation.communication_weight,
        )

    return execution_plan


__all__ = [
    "AllocationOptions",
    "ConstraintReport",
    "ConstraintViolation",
    "CollectiveGroup",
    "ExecutionContract",
    "ExecutionPlan",
    "InitializerInput",
    "Layer",
    "LayerInput",
    "LayerInputSource",
    "LayerOutput",
    "LocalInput",
    "PlacementOptions",
    "PlanningConstraints",
    "PlanningOptions",
    "Stage",
    "StageFormationOptions",
    "TransitionSource",
    "VirtualCollectiveGroup",
    "plan",
    "require_valid_execution_plan",
    "validate_execution_plan",
]
