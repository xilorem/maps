"""Place virtual Stage Plans onto connected physical mesh regions."""

from __future__ import annotations

from math import ceil, sqrt

from maps.hardware import Mesh
from maps.planning.stages import StagePlacement, StagePlan
from maps.planning.placement.evaluation import (
    PlacementEvaluator,
    build_virtual_traffic,
    print_placement_grid,
    print_placement_details,
)
from maps.planning.placement.regions import (
    assign_stage_ownerships,
    build_initial_stage_placements,
    stage_order,
)
from maps.planning.placement.repair import improve_placement
from maps.planning.transitions import VirtualTransition


def place(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    virtual_transitions: tuple[VirtualTransition, ...],
    show_progress: bool = False,
    print_placement: bool = True,
    print_costs: bool = False,
    stage_latency_weight: float = 1.0,
    communication_weight: float = 1.0,
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
        compares compact and full-mesh seeds using analytical stage service
        estimates, and sweeps strictly improving local repairs across all stages.

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

    evaluator = PlacementEvaluator(
        mesh, stage_plans, virtual_transitions,
        stage_latency_weight, communication_weight,
    )
    domains: dict[str, frozenset[int] | None] = {"native": None}
    compact = compact_tile_domain(mesh, sum(tile_counts.values()))
    if len(compact) < mesh.num_tiles:
        domains["compact"] = compact
    best_placements = None
    best_evaluation = None
    for label, domain in domains.items():
        placements = build_initial_stage_placements(
            mesh, stage_plans, tile_counts, traffic, show_progress,
            allowed_tile_ids=domain,
        )
        placements = assign_stage_ownerships(mesh, stage_plans, placements, traffic)
        evaluation = evaluator.evaluate(placements)
        _debug(show_progress, f"[placement] seed={label} initial_objective={evaluation.objective}")
        placements = improve_placement(
            mesh, stage_plans, placements, traffic, virtual_transitions,
            evaluation, show_progress, evaluator=evaluator,
        )
        evaluation = evaluator.evaluate(placements)
        _debug(show_progress, f"[placement] seed={label} final_objective={evaluation.objective}")
        if best_evaluation is None or evaluation.objective < best_evaluation.objective:
            best_placements, best_evaluation = placements, evaluation
            best_label = label
    assert best_placements is not None and best_evaluation is not None
    placements = best_placements
    _debug(show_progress, f"[placement] selected_seed={best_label} objective={best_evaluation.objective}")

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


def compact_tile_domain(mesh: Mesh, used_tiles: int) -> frozenset[int]:
    """A left-edge rectangle with roughly twice the allocated area as slack.

    Its dimensions depend on allocation, not total mesh area, so a larger mesh
    can retain the same compact starting footprint. Thin meshes are supported.
    """
    side = max(1, ceil(sqrt(2 * used_tiles)))
    width, height = min(mesh.width, side), min(mesh.height, side)
    if width * height < used_tiles:
        width = min(mesh.width, max(width, ceil(used_tiles / height)))
        height = min(mesh.height, max(height, ceil(used_tiles / width)))
    return frozenset(mesh.tile_id(x, y) for y in range(height) for x in range(width))


def _debug(enabled: bool, message: str) -> None:
    """Print one high-level Placement trace line when enabled."""

    if enabled:
        print(message)


__all__ = ["place"]
