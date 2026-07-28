"""L1-feasible seeding and greedy tile-allocation growth."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node
from MAPS.planner.contracts.stages import StagePlan, StageSelection
from MAPS.planner.workload.context import WorkloadContext
from MAPS.planner.workload.metrics import estimate_selection_metrics, selection_objective
from MAPS.planner.workload.plans import best_plan_for_stage, plan_all_stages


def seed_tile_counts(
    context: WorkloadContext,
    mesh: Mesh,
    debug: bool,
    num_token_slots: int = 2,
) -> dict[int, int]:
    """Assign every stage its smallest L1-feasible virtual tile count.

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
    tile_counts = {
        stage_id: initial_tile_count_for_stage(
            mesh=mesh,
            stage_id=stage_id,
            stage_selection=context.stage_selection,
            initializer_tensors=context.initializer_tensors,
            debug=debug,
            num_token_slots=num_token_slots,
        )
        for stage_id in stage_ids
    }
    if sum(tile_counts.values()) > mesh.num_tiles:
        raise ValueError("minimum L1-feasible tile counts exceed available tiles")
    return tile_counts


def grow_tile_counts(
    context: WorkloadContext,
    mesh: Mesh,
    tile_counts: dict[int, int],
    compute_weight: float,
    communication_weight: float,
    debug: bool,
    num_token_slots: int = 2,
) -> tuple[dict[int, int], dict[int, StagePlan]]:
    """Spend remaining tiles while improving the global bottleneck objective.

    On each iteration stages are ordered by their current bottleneck metric.
    Before a stage participates in one growth step, doubling its current tile
    count must improve the objective. A failed doubling probe permanently
    removes that stage from growth consideration.
    The first stage with a feasible allocation growth that lexicographically
    improves all ordered bottlenecks receives the smallest such growth.  Search
    stops when the mesh is full or no globally improving allocation exists.
    The returned plans are the layouts used to evaluate the final tile counts.
    """

    tile_counts = dict(tile_counts) # shallow copy

    # Choose best layouts for current tile counts 
    plans = plan_all_stages(
        context.stage_selection,
        mesh,
        tile_counts,
        initializer_tensors=context.initializer_tensors,
        debug=False,
        num_token_slots=num_token_slots,
    )

    used_tiles = sum(tile_counts.values())
    active_stage_ids = set(context.stage_selection)

    _debug(debug, f"[workload_balancing] start used_tiles={used_tiles}/{mesh.num_tiles}")
    _debug(debug, f"[workload_balancing] initial_tile_counts={tile_counts}")
    _debug(debug, "[workload_balancing] phase=greedy_growth")


    while used_tiles < mesh.num_tiles:
        current_metrics = estimate_selection_metrics(
            plans,
            context.stage_selection,
            mesh=mesh,
            compute_weight=compute_weight,
            communication_weight=communication_weight,
            graph_inputs=context.graph_inputs,
            graph_outputs=context.graph_outputs,
            producer_stage_id_by_tensor=context.producer_stage_id_by_tensor,
            initializer_tensors=context.initializer_tensors,
            graph=context.graph,
        )

        # Order stages by highest workload
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
                f"current_tile_count={tile_counts[stage_id]} "
                f"current_logical_shape={plans[stage_id].logical_shape} "
                f"current_metric={current_metrics[stage_id]}",
            )

            growth_arguments = dict(
                stage_id=stage_id,
                stage_selection=context.stage_selection,
                mesh=mesh,
                tile_counts=tile_counts,
                used_tiles=used_tiles,
                current_metric=current_metrics[stage_id],
                initializer_tensors=context.initializer_tensors,
                debug=debug,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
                graph_inputs=context.graph_inputs,
                graph_outputs=context.graph_outputs,
                producer_stage_id_by_tensor=context.producer_stage_id_by_tensor,
                graph=context.graph,
                current_selection_metrics=current_metrics,
                num_token_slots=num_token_slots,
            )
            doubled_current_count = tile_counts[stage_id] * 2
            doubled_added_tiles = doubled_current_count - tile_counts[stage_id]
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

            # Select stage with possible growth
            if growth is not None:
                chosen_stage_id = stage_id
                chosen_tile_count, plans = growth
                break

            _debug(debug, f"[workload_balancing] stage={stage_id} no_valid_growth")

        # Guard for no-growth stage
        if chosen_stage_id is None or chosen_tile_count is None:
            _debug(debug, "[workload_balancing] no_global_improvement_available")
            break
        
        previous_count = tile_counts[chosen_stage_id]

        _debug(
            debug,
            "[workload_balancing] "
            f"choose worst_stage={chosen_stage_id} new_tile_count={chosen_tile_count}",
        )

        # Update tile counts
        tile_counts[chosen_stage_id] = chosen_tile_count
        used_tiles += chosen_tile_count - previous_count
        
        _debug(debug, f"[workload_balancing] updated_tile_counts={tile_counts}")

    return tile_counts, plans


def initial_tile_count_for_stage(
    mesh: Mesh,
    stage_id: int,
    stage_selection: StageSelection,
    initializer_tensors: frozenset,
    debug: bool = False,
    num_token_slots: int = 2,
) -> int:
    """Return the smallest tile count having an L1-feasible stage layout."""

    stage_nodes = stage_selection[stage_id]
    for tile_count in range(1, mesh.num_tiles + 1):
        try:
            plan = best_plan_for_stage(
                stage_nodes=stage_nodes,
                mesh=mesh,
                stage_id=stage_id,
                tile_count=tile_count,
                initializer_tensors=initializer_tensors,
                debug=False,
                num_token_slots=num_token_slots,
            )
        except ValueError as exc:
            _debug(
                debug,
                "[workload_balancing] "
                f"seed stage={stage_id} tile_count={tile_count} skip={exc}",
            )
            continue
        _debug(
            debug,
            "[workload_balancing] "
            f"seed stage={stage_id} choose tile_count={tile_count} "
            f"logical_shape={plan.logical_shape}",
        )
        return tile_count
    raise ValueError(
        f"stage {stage_id} nodes={tuple(node.name for node in stage_nodes)} "
        f"canonical_node_count={len(stage_nodes)} "
        f"contains_explicit_group={any('stage_group_id' in node.attributes for node in stage_nodes)} "
        f"has no L1-feasible layout on mesh {mesh.shape}; "
        f"attempted_tile_counts=1..{mesh.num_tiles} "
        "layout_families=all_rectangular_factorizations. The caller can lower "
        "stage_selection.max_stage_nodes or select another target."
    )


def grow_tile_count_for_stage(
    stage_id: int,
    stage_selection: StageSelection,
    mesh: Mesh,
    tile_counts: dict[int, int],
    used_tiles: int,
    current_metric: float,
    initializer_tensors: frozenset,
    debug: bool = False,
    compute_weight: float = 1.0,
    communication_weight: float = 1.0,
    graph_inputs: frozenset = frozenset(),
    graph_outputs: frozenset = frozenset(),
    producer_stage_id_by_tensor: dict[object, int] | None = None,
    graph: Graph | None = None,
    current_selection_metrics: dict[int, float] | None = None,
    num_token_slots: int = 2,
) -> int | None:
    """Return the smallest feasible stage growth that improves the objective."""

    growth = _growth_candidate_for_stage(
        stage_id=stage_id,
        stage_selection=stage_selection,
        mesh=mesh,
        tile_counts=tile_counts,
        used_tiles=used_tiles,
        current_metric=current_metric,
        initializer_tensors=initializer_tensors,
        debug=debug,
        compute_weight=compute_weight,
        communication_weight=communication_weight,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        producer_stage_id_by_tensor=producer_stage_id_by_tensor,
        graph=graph,
        current_selection_metrics=current_selection_metrics,
        num_token_slots=num_token_slots,
    )
    return None if growth is None else growth[0]


def _growth_candidate_for_stage(
    stage_id: int,
    stage_selection: StageSelection,
    mesh: Mesh,
    tile_counts: dict[int, int],
    used_tiles: int,
    current_metric: float,
    initializer_tensors: frozenset,
    debug: bool = False,
    compute_weight: float = 1.0,
    communication_weight: float = 1.0,
    graph_inputs: frozenset = frozenset(),
    graph_outputs: frozenset = frozenset(),
    producer_stage_id_by_tensor: dict[object, int] | None = None,
    graph: Graph | None = None,
    current_selection_metrics: dict[int, float] | None = None,
    candidate_counts: tuple[int, ...] | None = None,
    num_token_slots: int = 2,
) -> tuple[int, dict[int, StagePlan]] | None:
    """Return the first improving tile count and its materialized plans."""

    current_tile_count = tile_counts[stage_id]
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
        candidate_tile_counts = dict(tile_counts)
        candidate_tile_counts[stage_id] = candidate_count
        try:
            candidate_plans = plan_all_stages(
                stage_selection,
                mesh,
                candidate_tile_counts,
                initializer_tensors=initializer_tensors,
                debug=False,
                num_token_slots=num_token_slots,
            )
        except ValueError as exc:
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_count={candidate_count} skip={exc}",
            )
            continue

        candidate_metrics = estimate_selection_metrics(
            candidate_plans,
            stage_selection,
            mesh=mesh,
            compute_weight=compute_weight,
            communication_weight=communication_weight,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            producer_stage_id_by_tensor=producer_stage_id_by_tensor or {},
            initializer_tensors=initializer_tensors,
            graph=graph,
        )
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
        return candidate_count, candidate_plans
    return None


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
