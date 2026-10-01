"""L1-feasible seeding and whole-plan virtual allocation search."""

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


def candidate_tile_counts(current: int, budget: int) -> tuple[int, ...]:
    """Probe coarse counts and their neighbors, then refine around the current size.

    A linear sweep spends most analysis on large unused allocations. Powers of
    two cross non-monotone regions, while adjacent counts refine a useful
    allocation and permit one-tile growth or removal. Endpoints retain the
    minimum and maximum budget choices. This is O(log budget) per sweep.
    """

    counts = {
        1, budget, current - 1, current, current + 1,
        current // 2, current * 2,
    }
    count = 1
    while count <= budget:
        counts.update((count - 1, count, count + 1))
        count *= 2
    return tuple(sorted(count for count in counts if 1 <= count <= budget))


def grow_stage_candidates(
    context: AllocationContext,
    mesh: Mesh,
    selected_candidates: dict[int, StageCandidate],
    analyzer: StageCandidateAnalyzer,
    stage_latency_weight: float,
    communication_weight: float,
    debug: bool,
) -> tuple[dict[int, StageCandidate], SelectionEvaluation]:
    """Best-improvement local search over counts, layouts, and tile exchanges.

    Counts are explored with logarithmic coarse probes and local refinement;
    every feasible layout at those counts is considered, including shrinking
    and same-count layout changes. Exchanges transfer a donor's released tiles
    to another stage atomically, so a full mesh can escape an overgrown stage.
    Stage formation stays fixed. Strict objective descent guarantees termination;
    this neighborhood search does not guarantee the global optimum.
    """

    selected_candidates = dict(selected_candidates)
    evaluations: dict[tuple[tuple[int, int, tuple[int, int]], ...], SelectionEvaluation] = {}

    def evaluate(selection: dict[int, StageCandidate]) -> SelectionEvaluation:
        key = tuple(
            (stage, candidate.plan.tile_count, candidate.plan.logical_shape)
            for stage, candidate in selection.items()
        )
        if key not in evaluations:
            evaluations[key] = evaluate_candidate_selection(
                selection, mesh=mesh, graph=context.graph,
                stage_latency_weight=stage_latency_weight,
                communication_weight=communication_weight,
            )
        return evaluations[key]

    def objective(
        selection: dict[int, StageCandidate],
        evaluation: SelectionEvaluation,
    ) -> tuple[tuple[float, ...], int]:
        # Throughput bottleneck first, then other stage service times. On an
        # exact cycle tie prefer fewer tiles, allowing unused capacity.
        return selection_objective(evaluation.metrics), _used_tile_count(selection)

    current_evaluation = evaluate(selected_candidates)
    sweep = 0
    while True:
        sweep += 1
        used_tiles = _used_tile_count(selected_candidates)
        best_selection = selected_candidates
        best_evaluation = current_evaluation
        best_objective = objective(best_selection, best_evaluation)
        _debug(
            debug,
            f"[allocation] sweep={sweep} objective={best_objective} "
            f"used_tiles={used_tiles}/{mesh.num_tiles}",
        )

        def consider(replacements: dict[int, StageCandidate]) -> None:
            nonlocal best_selection, best_evaluation, best_objective
            trial = selected_candidates | replacements
            evaluation = evaluate(trial)
            trial_objective = objective(trial, evaluation)
            if trial_objective < best_objective:
                best_selection, best_evaluation, best_objective = trial, evaluation, trial_objective

        for stage, current in selected_candidates.items():
            budget = mesh.num_tiles - used_tiles + current.plan.tile_count
            counts = candidate_tile_counts(current.plan.tile_count, budget)
            _debug(debug, f"[allocation] sweep={sweep} stage={stage} tile_counts={counts}")
            for count in counts:
                for candidate in analyzer.candidates(stage, count):
                    if candidate is not current:
                        consider({stage: candidate})

        _debug(debug, f"[allocation] sweep={sweep} phase=tile_exchanges")
        for donor, current in selected_candidates.items():
            for count in candidate_tile_counts(current.plan.tile_count, current.plan.tile_count - 1):
                released = current.plan.tile_count - count
                for recipient, recipient_current in selected_candidates.items():
                    if recipient == donor:
                        continue
                    for smaller in analyzer.candidates(donor, count):
                        for larger in analyzer.candidates(
                            recipient, recipient_current.plan.tile_count + released,
                        ):
                            consider({donor: smaller, recipient: larger})

        if best_selection is selected_candidates:
            _debug(debug, "[allocation] no_global_improvement_available")
            return selected_candidates, current_evaluation
        selected_candidates, current_evaluation = best_selection, best_evaluation
        _debug(
            debug,
            f"[allocation] selected_tile_counts={_candidate_tile_counts(selected_candidates)} "
            f"objective={best_objective} "
            f"used_tiles={_used_tile_count(selected_candidates)}/{mesh.num_tiles}",
        )


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
        print(message, flush=True)


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
    """Intrinsic cycles (including collectives) and external transfer service.

    The sum is a conservative service estimate: overlap, runtime scheduling,
    and physical contention are not modeled here. Unit weights compare cycles
    directly; explicit legacy weights remain available for callers.
    """

    stage_latency: int
    communication_cycles: int
    weighted_bottleneck: float


@dataclass(frozen=True)
class SelectionEvaluation:
    """Reusable global evaluation of one complete candidate selection."""

    stage_breakdowns: dict[int, StageMetricBreakdown]

    @property
    def metrics(self) -> dict[int, float]:
        """Return the combined stage service used to order bottlenecks."""

        return {
            stage_id: breakdown.weighted_bottleneck
            for stage_id, breakdown in self.stage_breakdowns.items()
        }


def evaluate_candidate_selection(
    candidates: dict[int, StageCandidate],
    mesh: Mesh,
    stage_latency_weight: float,
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
                stage_latency=candidate.stage_latency,
                communication_cycles=max(
                    virtual_communication[stage_id].values(),
                    default=0,
                ),
                weighted_bottleneck=(
                    stage_latency_weight * candidate.stage_latency
                    + communication_weight
                    * max(virtual_communication[stage_id].values(), default=0)
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
    """Estimate communication service for each virtual tile."""

    runtime = mesh.dma_runtime_cost
    if any((runtime.submission_cycles, runtime.setup_cycles, runtime.publication_cycles)):
        from maps.planning.transitions.dma import virtual_communication_cycles

        return virtual_communication_cycles(
            mesh,
            build_virtual_transitions(graph, plans),
            {
                stage_id: virtual_submesh(plan).tile_ids
                for stage_id, plan in plans.items()
            },
        )

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
    """Print the canonical final Stage Latency and communication bottlenecks."""

    if not enabled:
        return
    print("[allocation] final_stage_metric_breakdown:")
    for stage_id, stage_nodes in stage_formation.items():
        breakdown = evaluation.stage_breakdowns[stage_id]
        print(
            f"  stage={stage_id} nodes={_stage_label(stage_nodes)} "
            f"stage_latency={breakdown.stage_latency} "
            f"communication={breakdown.communication_cycles} "
            f"service_cycles={breakdown.weighted_bottleneck}"
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
    stage_latency_weight: float = 1.0,
    communication_weight: float = 1.0,
    num_token_slots: int = 2,
) -> dict[int, StagePlan]:
    """Choose virtual tile allocations and tensor layouts for all stages.

    Contract:
        ``stage_formation`` must cover every graph node exactly once.
        ``stage_latency_weight`` and ``communication_weight`` weight their respective
        costs when comparing feasible allocations; they do not relax memory
        constraints.

    Behavior:
        The pass validates and classifies the graph, seeds each stage with its
        smallest L1-feasible tile count, searches counts, layouts, and tile exchanges to improve the ordered
        whole-plan service bottleneck. Unused tiles are permitted.

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
        stage_latency_weight=stage_latency_weight,
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
