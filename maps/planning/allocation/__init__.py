"""Allocate virtual tiles, logical shapes, layouts, and Devices to Stages."""

from __future__ import annotations

from maps.graph import Graph
from maps.hardware import Mesh
from maps.planning.stages import StageFormation, StagePlan

from maps.planning.allocation.selection import (
    grow_stage_candidates,
    seed_stage_candidates,
)
from maps.planning.allocation.candidates import StageCandidateAnalyzer
from maps.planning.allocation.context import build_allocation_context
from maps.planning.allocation.diagnostics import print_stage_metric_breakdown


def allocate(
    graph: Graph,
    mesh: Mesh,
    stage_formation: StageFormation,
    debug: bool = False,
    compute_weight: float = 1.0,
    communication_weight: float = 1.0,
    num_token_slots: int = 2,
) -> dict[int, StagePlan]:
    """Choose virtual tile allocations and tensor layouts for all stages.

    Contract:
        ``stage_formation`` must cover every graph node exactly once.
        ``compute_weight`` and ``communication_weight`` weight their respective
        costs when comparing feasible allocations; they do not relax memory
        constraints.

    Behavior:
        The pass validates and classifies the graph, seeds each stage with its
        smallest L1-feasible tile count, greedily spends remaining mesh tiles to
        improve the ordered global bottleneck, then chooses the best logical
        layout for every final allocation.

    Returns:
        A stage-id mapping of virtual ``StagePlan`` objects.  Their layouts are
        final, but they contain no required physical placement decision.

    Raises:
        ValueError: If Stage formation is invalid or no complete L1-feasible
            allocation fits on the mesh.
    """

    context = build_allocation_context(graph, stage_formation)
    analyzer = StageCandidateAnalyzer(
        context.stage_formation,
        mesh,
        context.initializer_tensors,
        num_token_slots,
    )

    candidates = seed_stage_candidates(
        context,
        mesh,
        analyzer,
        debug,
    )

    candidates, evaluation = grow_stage_candidates(
        context,
        mesh,
        candidates,
        analyzer,
        compute_weight=compute_weight,
        communication_weight=communication_weight,
        debug=debug,
    )
    plans = {
        stage_id: candidate.plan
        for stage_id, candidate in candidates.items()
    }

    if debug:
        print(
            "[allocation] "
            f"final_tile_counts="
            f"{ {stage_id: plan.tile_count for stage_id, plan in plans.items()} }"
        )
        print(
            "[allocation] "
            f"final_logical_shapes="
            f"{ {stage_id: plan.logical_shape for stage_id, plan in plans.items()} }"
        )

    print_stage_metric_breakdown(
        enabled=debug,
        stage_formation=context.stage_formation,
        evaluation=evaluation,
    )
    return plans


__all__ = ["allocate"]
