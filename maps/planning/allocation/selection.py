"""L1-feasible seeding and whole-plan virtual allocation search."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

from maps.hardware import Mesh
from maps.graph import Graph, Node
from maps.planning.stages import (
    StageFormation,
    StagePlan,
    validate_stage_formation,
    virtual_submesh,
)
from maps.planning.allocation.candidates import StageCandidate, StageCandidateAnalyzer
from maps.planning.transitions import VirtualTransition, build_virtual_transitions


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
    prune_candidates: bool = True,
) -> tuple[dict[int, StageCandidate], SelectionEvaluation]:
    """Best-improvement local search over counts, layouts, and tile exchanges.

    Counts are explored with logarithmic coarse probes and local refinement;
    every feasible layout at those counts is considered, including shrinking
    and same-count layout changes. Exchanges transfer a donor's released tiles
    to another stage atomically, so a full mesh can escape an overgrown stage.
    Stage formation stays fixed. Strict objective descent guarantees termination;
    this neighborhood search does not guarantee the global optimum.

    Admissible communication floors stop trials that cannot strictly improve
    the current best objective, including its preference for fewer tiles. The
    candidate neighborhood and exact pricing of accepted trials stay the same.
    """

    selected_candidates = dict(selected_candidates)
    evaluations: dict[tuple[tuple[int, int, tuple[int, int]], ...], SelectionEvaluation] = {}
    communication_cache = AllocationCommunicationCache(
        context.graph, mesh, {stage: candidate.plan for stage, candidate in selected_candidates.items()},
    )

    def evaluate(
        selection: dict[int, StageCandidate],
        objective_limit: tuple[tuple[float, ...], int] | None = None,
    ) -> SelectionEvaluation | None:
        key = tuple(
            (stage, candidate.plan.tile_count, candidate.plan.logical_shape)
            for stage, candidate in selection.items()
        )
        if key not in evaluations:
            result = evaluate_candidate_selection(
                selection, mesh=mesh, graph=context.graph,
                stage_latency_weight=stage_latency_weight,
                communication_weight=communication_weight,
                communication_cache=communication_cache,
                objective_limit=objective_limit if prune_candidates else None,
            )
            if result is None:
                return None  # Partial bounds are never cached as evaluations.
            evaluations[key] = result
        return evaluations[key]

    def objective(
        selection: dict[int, StageCandidate],
        evaluation: SelectionEvaluation,
    ) -> tuple[tuple[float, ...], int]:
        # Throughput bottleneck first, then other stage service times. On an
        # exact cycle tie prefer fewer tiles, allowing unused capacity.
        return selection_objective(evaluation.metrics), _used_tile_count(selection)

    current_evaluation = evaluate(selected_candidates)
    assert current_evaluation is not None
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
            evaluation = evaluate(trial, best_objective)
            if evaluation is None:
                return
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
    communication_cache: AllocationCommunicationCache | None = None,
    objective_limit: tuple[tuple[float, ...], int] | None = None,
) -> SelectionEvaluation | None:
    """Evaluate a selection, or return None when it cannot beat objective_limit.

    Rejection uses lower bounds only with supported integer transfer costs and
    finite non-negative weights. Without a cache or limit, pricing is complete.
    """

    plans = {
        stage_id: candidate.plan
        for stage_id, candidate in candidates.items()
    }
    if communication_cache is not None:
        for candidate in candidates.values():
            if candidate.resident_demands is not None:
                communication_cache.compiler.set_demands(candidate.plan, candidate.resident_demands)
    reject = None
    if (
        objective_limit is not None and communication_cache is not None
        and communication_cache.transfer_bounds.supported
        and all(isfinite(weight) and weight >= 0 for weight in (stage_latency_weight, communication_weight))
    ):
        tile_count = _used_tile_count(candidates)

        def reject(lower_cycles: dict[int, int]) -> bool:
            metrics = {
                stage: stage_latency_weight * candidate.stage_latency
                + communication_weight * lower_cycles[stage]
                for stage, candidate in candidates.items()
            }
            if not all(isfinite(metric) for metric in metrics.values()):
                return False
            return (selection_objective(metrics), tile_count) >= objective_limit

    virtual_communication = (
        _virtual_communication_cycles(graph, mesh, plans)
        if communication_cache is None
        else _virtual_communication_cycles(graph, mesh, plans, communication_cache, reject)
    )
    if virtual_communication is None:
        return None
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
    cache: AllocationCommunicationCache | None = None,
    reject: Callable[[dict[int, int]], bool] | None = None,
) -> dict[int, dict[int, int]] | None:
    """Estimate communication service for each virtual tile."""
    if cache is not None:
        return cache.cycles(plans, reject)
    return _communication_cycles_for_transitions(
        mesh, plans, build_virtual_transitions(graph, plans),
    )


def _communication_cycles_for_transitions(
    mesh: Mesh,
    plans: dict[int, StagePlan],
    transitions: tuple[VirtualTransition, ...],
) -> dict[int, dict[int, int]]:
    """Price a complete boundary or stage edge with the existing rounding rules."""

    runtime = mesh.dma_runtime_cost
    if any((runtime.submission_cycles, runtime.setup_cycles, runtime.publication_cycles)):
        from maps.planning.transitions.dma import virtual_communication_cycles

        return virtual_communication_cycles(
            mesh,
            transitions,
            {
                stage_id: virtual_submesh(plan).tile_ids
                for stage_id, plan in plans.items()
            },
        )

    # Virtual traffic is a pre-placement analysis shared by Allocation estimation
    # and Placement; it does not depend on physical mapping decisions.
    from maps.planning.placement.evaluation import build_virtual_traffic

    traffic = build_virtual_traffic(transitions, plans)
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


class AllocationCommunicationCache:
    """Cache per-tile service for boundaries and candidate pairs within one search.

    Cache whole stage edges rather than individual tensors: bandwidth-only
    scoring sums bytes on each edge before rounding to cycles. Tile costs must
    also be summed before taking the stage maximum. Transfer lists are discarded
    after pricing to keep the cache proportional to tile counts, not tile pairs.
    """

    def __init__(self, graph: Graph, mesh: Mesh, plans: dict[int, StagePlan]) -> None:
        from maps.planning.transitions.compile import VirtualTransitionCompiler

        self.mesh = mesh
        self.compiler = VirtualTransitionCompiler(graph, plans)
        self.boundaries: dict[int, dict[int, dict[int, int]]] = {}
        self.edges: dict[tuple[int, int], dict[int, dict[int, int]]] = {}
        from .bounds import TransferLowerBounds

        self.transfer_bounds = TransferLowerBounds(mesh, self.compiler)
        self.pruned_selections = 0
        self.completed_selections = 0
        self.priced_edges = 0

    def _edge_cycles(self, source: StagePlan, destination: StagePlan):
        key = (id(source), id(destination))
        if key not in self.edges:
            transitions = self.compiler.intermediates(source, destination)
            self.edges[key] = _communication_cycles_for_transitions(
                self.mesh, {source.stage_id: source, destination.stage_id: destination}, transitions,
            )
            self.priced_edges += 1
        return self.edges[key]

    @staticmethod
    def _add(cycles, contribution):
        for stage, tiles in contribution.items():
            for tile, cost in tiles.items():
                cycles[stage][tile] += cost

    def cycles(
        self, plans: dict[int, StagePlan],
        reject: Callable[[dict[int, int]], bool] | None = None,
    ) -> dict[int, dict[int, int]] | None:
        if reject is not None and reject(dict.fromkeys(plans, 0)):
            self.pruned_selections += 1
            return None
        cycles: dict[int, dict[int, int]] = {}
        for stage, plan in plans.items():
            key = id(plan)
            if key not in self.boundaries:
                transitions = self.compiler.inputs(plan) + self.compiler.outputs(plan)
                self.boundaries[key] = _communication_cycles_for_transitions(
                    self.mesh, {stage: plan}, transitions,
                )
            cycles[stage] = dict(self.boundaries[key][stage])
        if reject is None:
            for source, destination in self.compiler.edges:
                self._add(cycles, self._edge_cycles(plans[source], plans[destination]))
            self.completed_selections += 1
            return cycles

        missing = []
        for source, destination in self.compiler.edges:
            key = (id(plans[source]), id(plans[destination]))
            if key in self.edges:
                self._add(cycles, self.edges[key])
            else:
                missing.append((source, destination))
        optimistic = {stage: dict(tiles) for stage, tiles in cycles.items()}
        outgoing_totals = dict.fromkeys(plans, 0)
        floors = {}
        for source, destination in missing:
            sender, receiver = self.transfer_bounds.edge(plans[source], plans[destination])
            floors[source, destination] = sender, receiver
            outgoing_totals[source] += sender
            for tile, cost in receiver.items():
                optimistic[destination][tile] += cost

        def cannot_improve():
            # A stage's true maximum is at least both its known per-tile floor
            # and its total-service floor divided across all of its tiles.
            lower = {
                stage: max(
                    max(tiles.values(), default=0),
                    _ceil_div(sum(tiles.values()) + outgoing_totals[stage], len(tiles)),
                ) for stage, tiles in optimistic.items()
            }
            return reject(lower)

        if cannot_improve():
            self.pruned_selections += 1
            return None
        for source, destination in missing:
            contribution = self._edge_cycles(plans[source], plans[destination])
            self._add(cycles, contribution)
            self._add(optimistic, contribution)
            sender, receiver = floors[source, destination]
            outgoing_totals[source] -= sender
            for tile, cost in receiver.items():
                optimistic[destination][tile] -= cost
            if cannot_improve():
                self.pruned_selections += 1
                return None
        self.completed_selections += 1
        return cycles


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
