"""L1-feasible seeding and greedy tile-allocation growth."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node
from MAPS.planner.contracts.stages import StageSelection
from MAPS.planner.workload.candidates import StageCandidate, StageCandidateAnalyzer
from MAPS.planner.workload.context import WorkloadContext
from MAPS.planner.workload.metrics import (
    SelectionEvaluation,
    evaluate_candidate_selection,
    selection_objective,
)


def seed_stage_candidates(
    context: WorkloadContext,
    mesh: Mesh,
    analyzer: StageCandidateAnalyzer,
    debug: bool,
) -> dict[int, StageCandidate]:
    """Select every stage's smallest L1-feasible candidate.

    Stages are seeded independently.  The combined result is legal only when
    the number of selected stages and the sum of their minimum allocations fit
    on the physical mesh.
    """

    stage_ids = tuple(context.stage_selection)
    if len(stage_ids) > mesh.num_tiles:
        raise ValueError(
            f"deterministic stage selection produced {len(stage_ids)} stages for "
            f"a {mesh.num_tiles}-tile target; every stage requires at least one "
            "tile. Lower stage_selection.max_stage_nodes to a value other than 1 "
            "only if it enables more compatible coalescing, or select another "
            "target. Workload balancing does not split or rewrite selected groups."
        )

    _debug(debug, "[workload_balancing] phase=initial_l1_seeding")
    candidates = {
        stage_id: initial_candidate_for_stage(
            mesh=mesh,
            stage_id=stage_id,
            stage_selection=context.stage_selection,
            analyzer=analyzer,
            debug=debug,
        )
        for stage_id in stage_ids
    }
    minimum_tile_count = _used_tile_count(candidates)
    if minimum_tile_count > mesh.num_tiles:
        raise ValueError("minimum L1-feasible tile counts exceed available tiles")
    return candidates


def grow_stage_candidates(
    context: WorkloadContext,
    mesh: Mesh,
    selected_candidates: dict[int, StageCandidate],
    analyzer: StageCandidateAnalyzer,
    compute_weight: float,
    communication_weight: float,
    debug: bool,
) -> tuple[dict[int, StageCandidate], SelectionEvaluation]:
    """Spend remaining tiles while improving the global bottleneck objective.

    On each iteration stages are ordered by their current bottleneck metric.
    Before a stage participates in one growth step, doubling its current tile
    count must improve the objective. A failed doubling probe permanently
    removes that stage from growth consideration.
    The first stage with a feasible allocation growth that lexicographically
    improves all ordered bottlenecks receives the smallest such growth.  Search
    stops when the mesh is full or no globally improving allocation exists.
    """

    selected_candidates = dict(selected_candidates)
    used_tiles = _used_tile_count(selected_candidates)
    active_stage_ids = set(context.stage_selection)

    _debug(debug, f"[workload_balancing] start used_tiles={used_tiles}/{mesh.num_tiles}")
    _debug(
        debug,
        "[workload_balancing] "
        f"initial_tile_counts={_candidate_tile_counts(selected_candidates)}",
    )
    _debug(debug, "[workload_balancing] phase=greedy_growth")

    current_evaluation = evaluate_candidate_selection(
        selected_candidates,
        mesh=mesh,
        compute_weight=compute_weight,
        communication_weight=communication_weight,
        graph=context.graph,
    )
    while used_tiles < mesh.num_tiles:
        current_metrics = current_evaluation.metrics

        stage_order = tuple(
            sorted(
                active_stage_ids,
                key=lambda stage_id: (-current_metrics[stage_id], stage_id),
            )
        )

        _debug(debug, f"[workload_balancing] used_tiles={used_tiles}/{mesh.num_tiles}")
        _debug(debug, f"[workload_balancing] current_selection_metrics={_format_metrics(current_metrics)}")
        _debug(debug, f"[workload_balancing] stage_order_by_workload={stage_order}")

        chosen_stage_id: int | None = None
        chosen_tile_count: int | None = None

        for stage_id in stage_order:
            _debug(
                debug,
                "[workload_balancing] "
                f"try_stage={stage_id} nodes={_stage_label(context.stage_selection[stage_id])} "
                f"current_tile_count={selected_candidates[stage_id].plan.tile_count} "
                f"current_logical_shape={selected_candidates[stage_id].plan.logical_shape} "
                f"current_metric={current_metrics[stage_id]}",
            )

            growth_arguments = dict(
                stage_id=stage_id,
                mesh=mesh,
                selected_candidates=selected_candidates,
                analyzer=analyzer,
                used_tiles=used_tiles,
                current_metric=current_metrics[stage_id],
                debug=debug,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
                graph=context.graph,
                current_selection_metrics=current_metrics,
            )
            current_tile_count = selected_candidates[stage_id].plan.tile_count
            doubled_current_count = current_tile_count * 2
            doubled_added_tiles = doubled_current_count - current_tile_count
            remaining_tiles = mesh.num_tiles - used_tiles
            if doubled_added_tiles <= remaining_tiles:
                doubling_growth = _growth_candidate_for_stage(
                    **growth_arguments,
                    candidate_counts=(doubled_current_count,),
                )
                if doubling_growth is None:
                    active_stage_ids.remove(stage_id)
                    _debug(
                        debug,
                        "[workload_balancing] "
                        f"stage={stage_id} doubled_current_tile_count="
                        f"{doubled_current_count} no_improvement prune_stage",
                    )
                    continue
                _debug(
                    debug,
                    "[workload_balancing] "
                    f"stage={stage_id} doubled_current_tile_count="
                    f"{doubled_current_count} improvement_available",
                )
            else:
                _debug(
                    debug,
                    "[workload_balancing] "
                    f"stage={stage_id} doubled_current_tile_count="
                    f"{doubled_current_count} outside_remaining_budget",
                )

            growth = _growth_candidate_for_stage(**growth_arguments)

            if growth is not None:
                chosen_stage_id = stage_id
                candidate, candidate_evaluation = growth
                chosen_tile_count = candidate.plan.tile_count
                selected_candidates[stage_id] = candidate
                current_evaluation = candidate_evaluation
                break

            _debug(debug, f"[workload_balancing] stage={stage_id} no_valid_growth")

        if chosen_stage_id is None or chosen_tile_count is None:
            _debug(debug, "[workload_balancing] no_global_improvement_available")
            break

        previous_count = used_tiles
        used_tiles = _used_tile_count(selected_candidates)

        _debug(
            debug,
            "[workload_balancing] "
            f"choose worst_stage={chosen_stage_id} new_tile_count={chosen_tile_count}",
        )

        assert used_tiles > previous_count
        _debug(
            debug,
            "[workload_balancing] "
            f"updated_tile_counts={_candidate_tile_counts(selected_candidates)}",
        )

    return selected_candidates, current_evaluation


def initial_candidate_for_stage(
    mesh: Mesh,
    stage_id: int,
    stage_selection: StageSelection,
    analyzer: StageCandidateAnalyzer,
    debug: bool = False,
) -> StageCandidate:
    """Return the smallest L1-feasible candidate for one stage."""

    stage_nodes = stage_selection[stage_id]
    for tile_count in range(1, mesh.num_tiles + 1):
        candidate = analyzer.candidate(stage_id, tile_count)
        if candidate is None:
            _debug(
                debug,
                "[workload_balancing] "
                f"seed stage={stage_id} tile_count={tile_count} skip=L1-infeasible",
            )
            continue
        _debug(
            debug,
            "[workload_balancing] "
            f"seed stage={stage_id} choose tile_count={tile_count} "
            f"logical_shape={candidate.plan.logical_shape}",
        )
        return candidate
    raise ValueError(
        f"stage {stage_id} nodes={tuple(node.name for node in stage_nodes)} "
        f"canonical_node_count={len(stage_nodes)} "
        f"contains_explicit_group={any('stage_group_id' in node.attributes for node in stage_nodes)} "
        f"has no L1-feasible layout on mesh {mesh.shape}; "
        f"attempted_tile_counts=1..{mesh.num_tiles} "
        "layout_families=all_rectangular_factorizations. The caller can lower "
        "stage_selection.max_stage_nodes or select another target."
    )


def _growth_candidate_for_stage(
    stage_id: int,
    mesh: Mesh,
    selected_candidates: dict[int, StageCandidate],
    analyzer: StageCandidateAnalyzer,
    used_tiles: int,
    current_metric: float,
    graph: Graph,
    debug: bool = False,
    compute_weight: float = 1.0,
    communication_weight: float = 1.0,
    current_selection_metrics: dict[int, float] | None = None,
    candidate_counts: tuple[int, ...] | None = None,
) -> tuple[StageCandidate, SelectionEvaluation] | None:
    """Return the first improving replacement candidate for one stage."""

    current_tile_count = selected_candidates[stage_id].plan.tile_count
    remaining_tiles = mesh.num_tiles - used_tiles
    if candidate_counts is None:
        candidate_counts = tuple(
            current_tile_count + added_tiles
            for added_tiles in range(1, remaining_tiles + 1)
        )
    else:
        candidate_counts = tuple(
            candidate_count
            for candidate_count in candidate_counts
            if current_tile_count < candidate_count
            and candidate_count - current_tile_count <= remaining_tiles
        )
    _debug(
        debug,
        "[workload_balancing] "
        f"stage={stage_id} candidate_tile_counts={candidate_counts}",
    )

    for candidate_count in candidate_counts:
        candidate = analyzer.candidate(stage_id, candidate_count)
        if candidate is None:
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_count={candidate_count} "
                "skip=L1-infeasible",
            )
            continue
        candidate_selection = dict(selected_candidates)
        candidate_selection[stage_id] = candidate
        candidate_evaluation = evaluate_candidate_selection(
            candidate_selection,
            mesh=mesh,
            compute_weight=compute_weight,
            communication_weight=communication_weight,
            graph=graph,
        )
        candidate_metrics = candidate_evaluation.metrics
        candidate_metric = candidate_metrics[stage_id]
        if current_selection_metrics is None:
            improved = candidate_metric < current_metric
        else:
            improved = (
                selection_objective(candidate_metrics)
                < selection_objective(current_selection_metrics)
            )
        if not improved:
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_count={candidate_count} "
                f"skip=no_metric_improvement candidate_metric={candidate_metric} "
                f"current_metric={current_metric}",
            )
            continue
        _debug(
            debug,
            "[workload_balancing] "
            f"stage={stage_id} candidate_tile_count={candidate_count} "
            f"accepted_improvement={current_metric - candidate_metric}",
        )
        return candidate, candidate_evaluation
    return None


def _candidate_tile_counts(
    candidates: dict[int, StageCandidate],
) -> dict[int, int]:
    """Derive selected tile counts from Stage Candidates."""

    return {
        stage_id: candidate.plan.tile_count
        for stage_id, candidate in candidates.items()
    }


def _used_tile_count(candidates: dict[int, StageCandidate]) -> int:
    """Return the number of tiles occupied by a candidate selection."""

    return sum(
        candidate.plan.tile_count
        for candidate in candidates.values()
    )


def _stage_label(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact label for one selected stage."""

    return "+".join(node.name for node in stage_nodes)


def _debug(enabled: bool, message: str) -> None:
    """Print one allocation trace line when diagnostics are enabled."""

    if enabled:
        print(message)


def _format_metrics(metrics: dict[int, float]) -> str:
    """Format per-stage metrics in deterministic stage order."""

    return "{" + ", ".join(
        f"{stage_id}: {metric}"
        for stage_id, metric in metrics.items()
    ) + "}"
