"""Workload-balancing pass facade."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.planner.contracts.stages import StagePlan, StageSelection
from MAPS.planner.workload.allocation import (
    grow_stage_candidates,
    seed_stage_candidates,
)
from MAPS.planner.workload.candidates import StageCandidateAnalyzer
from MAPS.planner.workload.context import build_workload_context
from MAPS.planner.workload.diagnostics import print_stage_metric_breakdown


def balance_workload(
    graph: Graph,
    mesh: Mesh,
    stage_selection: StageSelection,
    debug: bool = False,
    compute_weight: float = 1.0,
    communication_weight: float = 1.0,
    num_token_slots: int = 2,
) -> dict[int, StagePlan]:
    """Choose virtual tile allocations and tensor layouts for all stages.

    Contract:
        ``stage_selection`` must cover every graph node exactly once.
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
        ValueError: If stage selection is invalid or no complete L1-feasible
            allocation fits on the mesh.
    """

    context = build_workload_context(graph, stage_selection)
    analyzer = StageCandidateAnalyzer(
        context.stage_selection,
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

    candidates = grow_stage_candidates(
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
            "[workload_balancing] "
            f"final_tile_counts="
            f"{ {stage_id: plan.tile_count for stage_id, plan in plans.items()} }"
        )
        print(
            "[workload_balancing] "
            f"final_logical_shapes="
            f"{ {stage_id: plan.logical_shape for stage_id, plan in plans.items()} }"
        )

    print_stage_metric_breakdown(
        enabled=debug,
        plans=plans,
        stage_selection=context.stage_selection,
        mesh=mesh,
        graph_inputs=context.graph_inputs,
        graph_outputs=context.graph_outputs,
        producer_stage_id_by_tensor=context.producer_stage_id_by_tensor,
        initializer_tensors=context.initializer_tensors,
    )
    return plans


__all__ = ["balance_workload"]
