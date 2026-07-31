"""Plan target-specialized Graphs for physical execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maps.graph import Graph
    from maps.hardware import Mesh
    from MAPS.pipeline import ExecutionPlan

    from .options import PlanningOptions


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

    from MAPS.planner.passes.execution_plan_lowering import lower_execution_plan
    from MAPS.planner.passes.execution_plan_validation import (
        require_valid_execution_plan,
    )
    from maps.planning.placement import place
    from MAPS.planner.reporting.execution_plan import print_execution_plan_stage_cost
    from MAPS.planner.validation.contracts import PlannerConstraints
    from maps.planning.transitions import build_virtual_transitions
    from maps.planning.allocation import allocate
    from maps.planning.stage_formation import form_stages

    from .options import PlanningOptions

    options = options or PlanningOptions()

    # TODO(maps-repository-architecture 11): migrate Execution Plan ownership.
    stage_formation = form_stages(
        graph,
        options.stage_formation,
    )
    stage_plans = allocate(
        graph,
        mesh,
        stage_formation=stage_formation,
        debug=options.allocation.print_progress,
        compute_weight=options.allocation.compute_weight,
        communication_weight=options.allocation.communication_weight,
        num_token_slots=options.execution.num_token_slots,
    )
    virtual_transitions = build_virtual_transitions(graph, stage_plans)
    placements = place(
        mesh,
        stage_plans,
        virtual_transitions,
        show_progress=options.placement.print_progress,
        print_mapping=options.placement.print_mapping,
        print_costs=options.placement.print_costs,
    )
    execution_plan = lower_execution_plan(
        graph,
        mesh,
        stage_plans,
        placements,
        virtual_transitions,
        execution=options.execution,
    )
    require_valid_execution_plan(
        execution_plan,
        PlannerConstraints(
            max_stage_nodes=options.stage_formation.max_stage_nodes,
        ),
        error_prefix="planner produced an invalid Execution Plan",
    )

    if options.print_execution_plan_cost:
        print_execution_plan_stage_cost(
            execution_plan,
            stage_plans,
            placements,
            virtual_transitions,
        )

    return execution_plan


def __getattr__(name: str):
    """Load public Planning contracts without coupling internal phase imports."""

    if name == "ExecutionPlan":
        from MAPS.pipeline import ExecutionPlan

        return ExecutionPlan
    if name in {
        "AllocationOptions",
        "PlacementOptions",
        "PlanningOptions",
        "StageFormationOptions",
    }:
        from . import options

        return getattr(options, name)
    raise AttributeError(name)


__all__ = [
    "AllocationOptions",
    "ExecutionPlan",
    "PlacementOptions",
    "PlanningOptions",
    "StageFormationOptions",
    "plan",
]
