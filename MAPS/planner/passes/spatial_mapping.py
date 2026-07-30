"""Spatial-mapping pass facade."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.planner.contracts.stages import StagePlacement, StagePlan
from MAPS.planner.spatial.diagnostics import (
    print_placement_grid,
    print_spatial_mapping_details,
)
from MAPS.planner.spatial.evaluation import MappingEvaluator
from MAPS.planner.spatial.ownership import assign_stage_ownerships, stage_order
from MAPS.planner.spatial.regions import build_initial_stage_placements
from MAPS.planner.spatial.repair import improve_spatial_mapping
from MAPS.planner.spatial.traffic import build_virtual_traffic
from MAPS.transitions import build_virtual_transitions


def map_spatially(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    show_progress: bool = False,
    print_mapping: bool = True,
    print_costs: bool = False,
) -> dict[int, StagePlacement]:
    """Map virtual stage plans onto connected physical mesh regions.

    Contract:
        Stage plans must contain complete virtual layouts, their tile counts must
        fit on ``mesh``, and selected stages must be represented exactly once.
        Returned regions are disjoint and connected, with a bijective ownership
        map from every virtual stage tile to one physical tile.

    Behavior:
        The pass analyzes virtual traffic, constructs a feasible initial set of
        connected regions, assigns communication-aware virtual ownership,
        evaluates exact physical IO, and applies strictly improving local
        repairs until the objective stalls.

    Raises:
        ValueError: If requested stage tiles exceed the mesh or no connected
            feasible placement can be constructed.
    """

    node_stage_ids = {
        id(node): stage_id
        for stage_id, plan in stage_plans.items()
        for node in plan.nodes
    }
    tile_counts = {
        stage_id: plan.tile_count
        for stage_id, plan in stage_plans.items()
    }
    if sum(tile_counts.values()) > mesh.num_tiles:
        raise ValueError("requested stage tiles exceed available mesh tiles")

    virtual_transitions = build_virtual_transitions(graph, stage_plans)
    traffic = build_virtual_traffic(virtual_transitions, stage_plans)
    _debug(show_progress, "[spatial_mapping] phase=virtual_analysis")
    _debug(
        show_progress,
        "[spatial_mapping] "
        f"stage_order={stage_order(tile_counts, traffic)} "
        f"communication_degree={traffic.communication_degree} "
        f"bottleneck_risk={traffic.bottleneck_risk} "
        f"l2_pressure={traffic.l2_pressure}",
    )

    placements = build_initial_stage_placements(
        mesh,
        stage_plans,
        tile_counts,
        traffic,
        show_progress,
    )
    placements = assign_stage_ownerships(mesh, stage_plans, placements, traffic)
    evaluator = MappingEvaluator(
        graph,
        mesh,
        stage_plans,
        node_stage_ids,
    )
    evaluation = evaluator.evaluate(placements)
    _debug(
        show_progress,
        "[spatial_mapping] "
        f"phase=initial_mapping objective={evaluation.objective} "
        f"worst_tile={evaluation.worst_tile_id}",
    )
    placements = improve_spatial_mapping(
        graph,
        mesh,
        stage_plans,
        placements,
        traffic,
        node_stage_ids,
        evaluation,
        show_progress,
        evaluator=evaluator,
    )

    if print_costs:
        print_spatial_mapping_details(
            graph,
            mesh,
            stage_plans,
            placements,
            node_stage_ids,
            label="ownership_aware",
        )
    elif print_mapping:
        print_placement_grid(mesh, placements)
    return placements


def _debug(enabled: bool, message: str) -> None:
    """Print one high-level mapping trace line when enabled."""

    if enabled:
        print(message)


__all__ = ["map_spatially"]
