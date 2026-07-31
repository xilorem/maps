"""Plan target-specialized Graphs for physical execution."""

from __future__ import annotations

from maps.graph import Graph
from maps.hardware import Mesh
from MAPS.pipeline import ExecutionPlan
from MAPS.planner.contracts.options import StageSelectionOptions
from MAPS.planner.device_assignment import assigned_device_name
from MAPS.planner.passes.execution_plan_lowering import lower_execution_plan
from MAPS.planner.passes.execution_plan_validation import require_valid_execution_plan
from MAPS.planner.passes.spatial_mapping import map_spatially
from MAPS.planner.passes.stage_selection import form_stages
from MAPS.planner.passes.workload_balancing import balance_workload
from MAPS.planner.reporting.execution_plan import print_execution_plan_stage_cost
from MAPS.planner.validation.contracts import PlannerConstraints
from MAPS.transitions import build_virtual_transitions

from .options import (
    AllocationOptions,
    PlacementOptions,
    PlanningOptions,
    StageFormationOptions,
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

    options = options or PlanningOptions()

    # TODO(maps-repository-architecture 09-11): contract these imports as Stage
    # formation, Allocation, Placement, Transitions, and Execution Plan ownership
    # migrate into this module. Keep their established composition order here.
    for node in graph.nodes:
        assigned_device_name(node, mesh.tiles)

    stage_formation = form_stages(
        graph,
        StageSelectionOptions(
            max_stage_nodes=options.stage_formation.max_stage_nodes,
        ),
    )
    stage_plans = balance_workload(
        graph,
        mesh,
        stage_selection=stage_formation,
        debug=options.allocation.print_progress,
        compute_weight=options.allocation.compute_weight,
        communication_weight=options.allocation.communication_weight,
        num_token_slots=options.execution.num_token_slots,
    )
    virtual_transitions = build_virtual_transitions(graph, stage_plans)
    placements = map_spatially(
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


__all__ = [
    "AllocationOptions",
    "ExecutionPlan",
    "PlacementOptions",
    "PlanningOptions",
    "StageFormationOptions",
    "plan",
]
