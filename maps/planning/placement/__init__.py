"""Place virtual Stage Plans onto connected physical mesh regions."""

from __future__ import annotations

from maps.hardware import Mesh
from maps.planning.stages import StagePlacement, StagePlan
from maps.planning.placement.diagnostics import (
    print_placement_grid,
    print_placement_details,
)
from maps.planning.placement.evaluation import PlacementEvaluator
from maps.planning.placement.ownership import assign_stage_ownerships, stage_order
from maps.planning.placement.regions import build_initial_stage_placements
from maps.planning.placement.repair import improve_placement
from maps.planning.placement.traffic import build_virtual_traffic
from maps.planning.transitions import VirtualTransition


def place(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    virtual_transitions: tuple[VirtualTransition, ...],
    show_progress: bool = False,
    print_placement: bool = True,
    print_costs: bool = False,
) -> dict[int, StagePlacement]:
    """Place virtual Stage Plans onto connected physical mesh regions.

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

    tile_counts = {
        stage_id: plan.tile_count
        for stage_id, plan in stage_plans.items()
    }
    if sum(tile_counts.values()) > mesh.num_tiles:
        raise ValueError("requested stage tiles exceed available mesh tiles")

    traffic = build_virtual_traffic(virtual_transitions, stage_plans)
    _debug(show_progress, "[placement] phase=virtual_analysis")
    _debug(
        show_progress,
        "[placement] "
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
    evaluator = PlacementEvaluator(
        mesh,
        stage_plans,
        virtual_transitions,
    )
    evaluation = evaluator.evaluate(placements)
    _debug(
        show_progress,
        "[placement] "
        f"phase=initial_placement objective={evaluation.objective} "
        f"worst_tile={evaluation.worst_tile_id}",
    )
    placements = improve_placement(
        mesh,
        stage_plans,
        placements,
        traffic,
        virtual_transitions,
        evaluation,
        show_progress,
        evaluator=evaluator,
    )

    if print_costs:
        print_placement_details(
            mesh,
            stage_plans,
            placements,
            virtual_transitions,
            label="ownership_aware",
        )
    elif print_placement:
        print_placement_grid(mesh, placements)
    return placements


def _debug(enabled: bool, message: str) -> None:
    """Print one high-level Placement trace line when enabled."""

    if enabled:
        print(message)


__all__ = ["place"]
