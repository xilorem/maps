"""L1-feasible seeding and greedy virtual tile-allocation growth."""

from __future__ import annotations

from dataclasses import dataclass

from maps.hardware import Mesh
from maps.graph import Graph, Node
from maps.planning.stages import (
    StageFormation,
    StagePlan,
    validate_stage_formation,
    virtual_submesh,
)
from maps.planning.allocation.candidates import StageCandidate, StageCandidateAnalyzer
from maps.planning.transitions import build_virtual_transitions


def seed_stage_candidates(
    context: AllocationContext,
    mesh: Mesh,
    analyzer: StageCandidateAnalyzer,
    debug: bool,
) -> dict[int, StageCandidate]:
    """Select every stage's smallest L1-feasible candidate.

    Stages are seeded independently.  The combined result is legal only when
    the number of selected stages and the sum of their minimum allocations fit
    on the physical mesh.
    """

    stage_ids = tuple(context.stage_formation)
    if len(stage_ids) > mesh.num_tiles:
        raise ValueError(
            f"deterministic Stage formation produced {len(stage_ids)} stages for "
            f"a {mesh.num_tiles}-tile target; every stage requires at least one "
            "tile. Raise stage_formation.max_stage_operations above 1 "
            "only if it enables more compatible coalescing, or select another "
            "target. Allocation does not split or rewrite formed Stages."
        )

    _debug(debug, "[allocation] phase=initial_l1_seeding")
    candidates = {
        stage_id: initial_candidate_for_stage(
            mesh=mesh,
            stage_id=stage_id,
            stage_formation=context.stage_formation,
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
    context: AllocationContext,
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
    active_stage_ids = set(context.stage_formation)

    _debug(debug, f"[allocation] start used_tiles={used_tiles}/{mesh.num_tiles}")
    _debug(
        debug,
        "[allocation] "
        f"initial_tile_counts={_candidate_tile_counts(selected_candidates)}",
    )
    _debug(debug, "[allocation] phase=greedy_growth")

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

        _debug(debug, f"[allocation] used_tiles={used_tiles}/{mesh.num_tiles}")
        _debug(debug, f"[allocation] current_selection_metrics={_format_metrics(current_metrics)}")
        _debug(debug, f"[allocation] stage_order_by_bottleneck={stage_order}")

        chosen_stage_id: int | None = None
        chosen_tile_count: int | None = None

        for stage_id in stage_order:
            _debug(
                debug,
                "[allocation] "
                f"try_stage={stage_id} nodes={_stage_label(context.stage_formation[stage_id])} "
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
                        "[allocation] "
                        f"stage={stage_id} doubled_current_tile_count="
                        f"{doubled_current_count} no_improvement prune_stage",
                    )
                    continue
                _debug(
                    debug,
                    "[allocation] "
                    f"stage={stage_id} doubled_current_tile_count="
                    f"{doubled_current_count} improvement_available",
                )
            else:
                _debug(
                    debug,
                    "[allocation] "
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

            _debug(debug, f"[allocation] stage={stage_id} no_valid_growth")

        if chosen_stage_id is None or chosen_tile_count is None:
            _debug(debug, "[allocation] no_global_improvement_available")
            break

        previous_count = used_tiles
        used_tiles = _used_tile_count(selected_candidates)

        _debug(
            debug,
            "[allocation] "
            f"choose worst_stage={chosen_stage_id} new_tile_count={chosen_tile_count}",
        )

        assert used_tiles > previous_count
        _debug(
            debug,
            "[allocation] "
            f"updated_tile_counts={_candidate_tile_counts(selected_candidates)}",
        )

    return selected_candidates, current_evaluation


def initial_candidate_for_stage(
    mesh: Mesh,
    stage_id: int,
    stage_formation: StageFormation,
    analyzer: StageCandidateAnalyzer,
    debug: bool = False,
) -> StageCandidate:
    """Return the smallest L1-feasible candidate for one stage."""

    stage_nodes = stage_formation[stage_id]
    for tile_count in range(1, mesh.num_tiles + 1):
        candidate = analyzer.candidate(stage_id, tile_count)
        if candidate is None:
            _debug(
                debug,
                "[allocation] "
                f"seed stage={stage_id} tile_count={tile_count} skip=L1-infeasible",
            )
            continue
        _debug(
            debug,
            "[allocation] "
            f"seed stage={stage_id} choose tile_count={tile_count} "
            f"logical_shape={candidate.plan.logical_shape}",
        )
        return candidate
    raise ValueError(
        f"stage {stage_id} nodes={tuple(node.name for node in stage_nodes)} "
        f"source_operations={tuple(dict.fromkeys(node.source_operation for node in stage_nodes))} "
        f"has no L1-feasible layout on mesh {mesh.shape}; "
        f"attempted_tile_counts=1..{mesh.num_tiles} "
        "layout_families=all_rectangular_factorizations. The caller can lower "
        "stage_formation.max_stage_operations or select another target."
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
        "[allocation] "
        f"stage={stage_id} candidate_tile_counts={candidate_counts}",
    )

    for candidate_count in candidate_counts:
        candidate = analyzer.candidate(stage_id, candidate_count)
        if candidate is None:
            _debug(
                debug,
                "[allocation] "
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
                "[allocation] "
                f"stage={stage_id} candidate_tile_count={candidate_count} "
                f"skip=no_metric_improvement candidate_metric={candidate_metric} "
                f"current_metric={current_metric}",
            )
            continue
        _debug(
            debug,
            "[allocation] "
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


@dataclass(frozen=True)
class AllocationContext:
    """Validated inputs shared by Stage Candidate Allocation."""

    graph: Graph
    stage_formation: StageFormation
    initializer_tensors: frozenset


def build_allocation_context(
    graph: Graph,
    stage_formation: StageFormation,
) -> AllocationContext:
    """Validate Stage coverage and retain intrinsic Allocation inputs."""

    resolved_selection = validate_stage_formation(graph, stage_formation)
    initializer_tensors = frozenset(graph.initializers)
    return AllocationContext(
        graph=graph,
        stage_formation=resolved_selection,
        initializer_tensors=initializer_tensors,
    )


@dataclass(frozen=True)
class StageMetricBreakdown:
    """Canonical intrinsic and communication costs for one selected Stage."""

    compute_cycles: int
    communication_cycles: int
    weighted_bottleneck: float


@dataclass(frozen=True)
class SelectionEvaluation:
    """Reusable global evaluation of one complete candidate selection."""

    stage_breakdowns: dict[int, StageMetricBreakdown]

    @property
    def metrics(self) -> dict[int, float]:
        """Return the weighted bottleneck used to order each Stage."""

        return {
            stage_id: breakdown.weighted_bottleneck
            for stage_id, breakdown in self.stage_breakdowns.items()
        }


def evaluate_candidate_selection(
    candidates: dict[int, StageCandidate],
    mesh: Mesh,
    compute_weight: float,
    communication_weight: float,
    graph: Graph,
) -> SelectionEvaluation:
    """Evaluate one complete Stage Candidate selection."""

    plans = {
        stage_id: candidate.plan
        for stage_id, candidate in candidates.items()
    }
    virtual_communication = _virtual_communication_cycles(graph, mesh, plans)
    return SelectionEvaluation(
        stage_breakdowns={
            stage_id: StageMetricBreakdown(
                compute_cycles=candidate.stage_compute,
                communication_cycles=max(
                    virtual_communication[stage_id].values(),
                    default=0,
                ),
                weighted_bottleneck=max(
                    compute_weight * candidate.stage_compute,
                    communication_weight
                    * max(virtual_communication[stage_id].values(), default=0),
                ),
            )
            for stage_id, candidate in candidates.items()
        }
    )


def _virtual_communication_cycles(
    graph: Graph,
    mesh: Mesh,
    plans: dict[int, StagePlan],
) -> dict[int, dict[int, int]]:
    """Estimate producer-side virtual-tile communication cycles."""

    # Virtual traffic is a pre-placement analysis shared by Allocation estimation
    # and Placement; it does not depend on physical mapping decisions.
    from maps.planning.placement.evaluation import build_virtual_traffic

    virtual_transitions = build_virtual_transitions(graph, plans)
    traffic = build_virtual_traffic(virtual_transitions, plans)
    communication = {
        stage_id: {
            tile.tile_id: 0
            for tile in virtual_submesh(plan).tiles
        }
        for stage_id, plan in plans.items()
    }

    for stage_id, plan in plans.items():
        for virtual_tile in virtual_submesh(plan).tiles:
            tile_id = virtual_tile.tile_id
            l2_bytes = (
                traffic.l2_read_weights[stage_id][tile_id]
                + traffic.l2_write_weights[stage_id][tile_id]
            )
            if l2_bytes:
                communication[stage_id][tile_id] += _ceil_div(
                    l2_bytes,
                    min(virtual_tile.memory.bandwidth, mesh.l2_memory.bandwidth),
                )

    for (source_stage_id, _), matrix in traffic.edge_matrices.items():
        for (source_tile_id, destination_tile_id), bytes_ in matrix.items():
            source_tile = mesh.tile_by_id(source_tile_id)
            destination_tile = mesh.tile_by_id(destination_tile_id)
            communication[source_stage_id][source_tile_id] += _ceil_div(
                bytes_,
                min(source_tile.memory.bandwidth, destination_tile.memory.bandwidth),
            )
    return communication


def selection_objective(metrics: dict[int, float]) -> tuple[float, ...]:
    """Order stage metrics so candidates compare worst bottlenecks first."""

    return tuple(sorted(metrics.values(), reverse=True))


def worst_tile_stage_compute(
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple, ...],
    submesh,
    device_names: tuple[str, ...],
) -> int:
    """Return the greatest accumulated compute cost on any stage tile."""

    return max(
        (
            sum(
                _node_compute_cycles(
                    node,
                    output_layouts,
                    tile,
                    device_names[node_index],
                )
                for node_index, (node, output_layouts) in enumerate(
                    zip(stage_nodes, node_output_layouts)
                )
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def _node_compute_cycles(
    node: Node,
    output_layouts: tuple,
    tile,
    device_name: str,
) -> int:
    """Estimate compute cost for one node on one virtual tile."""

    tile_work = node.payload.build_tile_work(output_layouts=output_layouts, tile=tile)
    cost_model = node.payload.cost_model
    compute_cost = cost_model.cost(
        tile_work,
        tile,
        tile.device_by_name(device_name),
    )
    return int(compute_cost) + int(
        cost_model.placement_cost(node=node, output_layouts=output_layouts)
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return positive integer ceiling division."""

    if denominator <= 0:
        raise ValueError("denominator must be > 0")
    return (numerator + denominator - 1) // denominator


def print_stage_metric_breakdown(
    enabled: bool,
    stage_formation: StageFormation,
    evaluation: SelectionEvaluation,
) -> None:
    """Print the canonical final compute and communication bottlenecks."""

    if not enabled:
        return
    print("[allocation] final_stage_metric_breakdown:")
    for stage_id, stage_nodes in stage_formation.items():
        breakdown = evaluation.stage_breakdowns[stage_id]
        print(
            f"  stage={stage_id} nodes={_stage_label(stage_nodes)} "
            f"compute={breakdown.compute_cycles} "
            f"communication={breakdown.communication_cycles}"
        )
        for label in dict.fromkeys(
            getattr(node.payload.cost_model, "diagnostic_label", None)
            for node in stage_nodes
        ):
            if label is not None:
                print(f"    cost_diagnostic={label}")


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
