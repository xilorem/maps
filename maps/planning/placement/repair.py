"""Ownership-aware local repair of complete Placements."""

from __future__ import annotations

from collections import deque

from maps.hardware import Mesh
from maps.planning.stages import StagePlacement, StagePlan
from maps.planning.placement.evaluation import PlacementEvaluator
from maps.planning.placement.evaluation import (
    PlacementEvaluation,
    RepairCandidate,
    TilePlacementScore,
    VirtualTraffic,
)
from maps.planning.placement.regions import assign_stage_ownerships
from maps.planning.placement.regions import placements_from_regions
from maps.planning.placement.regions import region_anchor_cost, sorted_candidate_tiles
from maps.planning.placement.regions import grow_stage_region, local_stage_order, stage_target_point
from maps.planning.placement.regions import (
    l2_access_point_tile_ids,
    owner_by_tile_id,
    shared_boundary_length,
    shortest_path_between_regions,
)
from maps.planning.transitions import VirtualTransition


def improve_placement(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    traffic: VirtualTraffic,
    virtual_transitions: tuple[VirtualTransition, ...],
    initial_evaluation: PlacementEvaluation,
    debug: bool,
    evaluator: PlacementEvaluator | None = None,
    max_iters: int = 32,
    max_repair_regions: int = 5,
) -> dict[int, StagePlacement]:
    """Apply local region repairs until the exact IO objective stalls.

    Each iteration blames the worst tile, ranks nearby stage unions that could
    relieve its traffic, reconstructs each candidate union, reassigns ownership,
    and evaluates the exact physical objective.  Only strict improvements are
    accepted.  A short tabu queue avoids immediately revisiting the same union.
    """

    if evaluator is None:
        evaluator = PlacementEvaluator(mesh, stage_plans, virtual_transitions)
    current_placements = placements
    current_evaluation = initial_evaluation
    tabu: deque[frozenset[int]] = deque(maxlen=10)
    for iteration in range(max_iters):
        if current_evaluation.worst_tile_id is None:
            break
        worst_tile = current_evaluation.tile_scores[current_evaluation.worst_tile_id]
        if worst_tile.stage_id is None:
            break
        candidates = choose_repair_regions(
            mesh,
            current_placements,
            traffic,
            current_evaluation,
            worst_tile,
        )
        _debug(
            debug,
            "[placement] "
            f"iter={iteration} objective={current_evaluation.objective} "
            f"worst_tile={worst_tile.tile_id} worst_stage={worst_tile.stage_id} "
            f"repair_candidates={[(sorted(c.stages), c.reason) for c in candidates[:max_repair_regions]]}",
        )

        best_trial: PlacementEvaluation | None = None
        best_placements: dict[int, StagePlacement] | None = None
        best_candidate: RepairCandidate | None = None
        for candidate in candidates[:max_repair_regions]:
            if candidate.stages in tabu:
                continue
            trial = repair_region(
                mesh,
                stage_plans,
                current_placements,
                traffic,
                candidate.stages,
                worst_tile.stage_id,
                debug,
            )
            if trial is None:
                continue
            trial = assign_stage_ownerships(
                mesh,
                stage_plans,
                trial,
                traffic,
                stage_ids=candidate.stages,
            )
            evaluation = evaluator.evaluate(
                trial,
                previous=current_evaluation,
                moved_stage_ids=candidate.stages,
            )
            _debug(
                debug,
                "[placement] "
                f"iter={iteration} region={sorted(candidate.stages)} "
                f"reason={candidate.reason} trial_objective={evaluation.objective}",
            )
            if evaluation.objective < current_evaluation.objective and (
                best_trial is None or evaluation.objective < best_trial.objective
            ):
                best_trial = evaluation
                best_placements = trial
                best_candidate = candidate

        if best_trial is None or best_placements is None or best_candidate is None:
            _debug(debug, f"[placement] iter={iteration} no_improving_repair_found")
            break
        current_placements = best_placements
        current_evaluation = best_trial
        tabu.append(best_candidate.stages)
        _debug(
            debug,
            "[placement] "
            f"iter={iteration} accepted_region={sorted(best_candidate.stages)} "
            f"reason={best_candidate.reason} objective={current_evaluation.objective}",
        )
    return current_placements


def repair_region(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    current_placements: dict[int, StagePlacement],
    traffic: VirtualTraffic,
    affected_stages: frozenset[int],
    focus_stage_id: int,
    debug: bool,
) -> dict[int, StagePlacement] | None:
    """Repartition a local stage set inside its existing physical-tile union."""

    affected_tile_ids = {
        tile_id
        for stage_id in affected_stages
        for tile_id in current_placements[stage_id].physical_submesh.tile_ids
    }
    fixed_regions = {
        stage_id: set(placement.physical_submesh.tile_ids)
        for stage_id, placement in current_placements.items()
        if stage_id not in affected_stages
    }
    local_tile_counts = {
        stage_id: stage_plans[stage_id].tile_count
        for stage_id in affected_stages
    }
    ordered_stages = local_stage_order(
        affected_stages,
        local_tile_counts,
        traffic,
        focus_stage_id,
    )

    best_regions: dict[int, set[int]] | None = None
    best_key: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    restart_count = max(1, min(4, len(affected_tile_ids)))
    for restart_idx in range(restart_count):
        free_tile_ids = set(affected_tile_ids)
        placed_regions = dict(fixed_regions)
        local_regions: dict[int, set[int]] = {}
        feasible = True
        for order_idx, stage_id in enumerate(ordered_stages):
            remaining_counts = {
                other_stage_id: local_tile_counts[other_stage_id]
                for other_stage_id in ordered_stages[order_idx + 1:]
            }
            target = stage_target_point(stage_id, mesh, placed_regions, traffic)
            seeds = sorted_candidate_tiles(
                mesh,
                free_tile_ids,
                target,
                stage_id,
                traffic,
                placed_regions,
            )
            if not seeds:
                feasible = False
                break
            preferred_seed = seeds[min(restart_idx, len(seeds) - 1)]
            try:
                region = grow_stage_region(
                    stage_id=stage_id,
                    mesh=mesh,
                    allowed_tile_ids=free_tile_ids,
                    tile_count=local_tile_counts[stage_id],
                    target=target,
                    traffic=traffic,
                    placed_regions=placed_regions,
                    remaining_tile_counts=remaining_counts,
                    preferred_seed=preferred_seed,
                    exhaustive_future_feasibility=False,
                )
            except ValueError:
                feasible = False
                break
            local_regions[stage_id] = region
            placed_regions[stage_id] = region
            free_tile_ids -= region

        if not feasible or set(local_regions) != set(affected_stages):
            continue
        focus_region = tuple(sorted(local_regions[focus_stage_id]))
        key = (
            region_anchor_cost(
                focus_stage_id,
                mesh,
                local_regions[focus_stage_id],
                traffic,
                placed_regions,
            ),
            focus_region,
            tuple(ordered_stages),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_regions = local_regions

    if best_regions is None:
        _debug(debug, f"[placement] repair_failed stages={sorted(affected_stages)}")
        return None
    merged_placements = dict(current_placements)
    merged_placements.update(
        placements_from_regions(mesh, stage_plans, best_regions)
    )
    _debug(debug, f"[placement] repair_regions stages={sorted(affected_stages)}")
    return merged_placements


def choose_repair_regions(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
    traffic: VirtualTraffic,
    evaluation: PlacementEvaluation,
    worst_tile: TilePlacementScore,
) -> list[RepairCandidate]:
    """Rank local repairs using bottleneck blame and physical blockers."""

    del traffic, evaluation
    if worst_tile.stage_id is None:
        return []
    bottleneck_stage_id = worst_tile.stage_id
    candidates: dict[frozenset[int], RepairCandidate] = {}
    consumer_blames = sorted(
        worst_tile.consumer_stage_writes.items(),
        key=lambda item: (-item[1], item[0]),
    )
    for consumer_stage_id, blame in consumer_blames[:2]:
        _record_candidate(
            candidates,
            frozenset({bottleneck_stage_id, consumer_stage_id}),
            float(blame),
            f"direct_{bottleneck_stage_id}_to_{consumer_stage_id}",
        )
        blocker = _first_blocker_on_path(
            mesh,
            placements,
            bottleneck_stage_id,
            consumer_stage_id,
        )
        if blocker is not None:
            _record_candidate(
                candidates,
                frozenset({bottleneck_stage_id, blocker}),
                float(blame) * 0.9,
                f"blocker_{blocker}_toward_{consumer_stage_id}",
            )

    l2_blocker = _first_blocker_to_l2(mesh, placements, bottleneck_stage_id)
    if l2_blocker is not None:
        _record_candidate(
            candidates,
            frozenset({bottleneck_stage_id, l2_blocker}),
            float(worst_tile.l2_reads + worst_tile.l2_writes),
            "l2_blocker",
        )
    for neighbor_stage_id in _neighbor_stage_ids(mesh, placements, bottleneck_stage_id):
        boundary = shared_boundary_length(
            mesh,
            placements[bottleneck_stage_id].physical_submesh.tile_ids,
            placements[neighbor_stage_id].physical_submesh.tile_ids,
        )
        _record_candidate(
            candidates,
            frozenset({bottleneck_stage_id, neighbor_stage_id}),
            float(boundary),
            f"physical_neighbor_{neighbor_stage_id}",
        )

    if len(consumer_blames) >= 2:
        left_stage_id, left_blame = consumer_blames[0]
        right_stage_id, right_blame = consumer_blames[1]
        stronger = max(left_blame, right_blame)
        weaker = min(left_blame, right_blame)
        if weaker > 0 and stronger <= int(1.5 * weaker):
            blockers = {
                _first_blocker_on_path(
                    mesh,
                    placements,
                    bottleneck_stage_id,
                    destination_stage_id,
                )
                for destination_stage_id in (left_stage_id, right_stage_id)
            }
            multi_region = {bottleneck_stage_id} | {
                blocker for blocker in blockers if blocker is not None
            }
            if len(multi_region) >= 2:
                _record_candidate(
                    candidates,
                    frozenset(multi_region),
                    float(left_blame + right_blame),
                    "balanced_multi_source",
                )
    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.priority,
            len(candidate.stages),
            tuple(sorted(candidate.stages)),
        ),
    )


def _record_candidate(
    candidates: dict[frozenset[int], RepairCandidate],
    region: frozenset[int],
    priority: float,
    reason: str,
) -> None:
    """Keep the strongest reason and priority for one repair region."""

    if len(region) < 2:
        return
    existing = candidates.get(region)
    if existing is None or priority > existing.priority:
        candidates[region] = RepairCandidate(region, priority, reason)


def _neighbor_stage_ids(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
    stage_id: int,
) -> set[int]:
    """Return stages sharing a physical boundary with one stage."""

    region = placements[stage_id].physical_submesh.tile_ids
    return {
        other_stage_id
        for other_stage_id, placement in placements.items()
        if other_stage_id != stage_id
        and shared_boundary_length(mesh, region, placement.physical_submesh.tile_ids) > 0
    }


def _first_blocker_on_path(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
    source_stage_id: int,
    destination_stage_id: int,
) -> int | None:
    """Return the first foreign stage on a shortest inter-stage path."""

    owners = owner_by_tile_id(placements)
    path = shortest_path_between_regions(
        mesh,
        placements[source_stage_id].physical_submesh.tile_ids,
        placements[destination_stage_id].physical_submesh.tile_ids,
    )
    for tile_id in path[1:]:
        owner = owners.get(tile_id)
        if owner is not None and owner not in {source_stage_id, destination_stage_id}:
            return owner
    return None


def _first_blocker_to_l2(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
    stage_id: int,
) -> int | None:
    """Return the first foreign stage between a stage and nearest L2 access."""

    l2_points = l2_access_point_tile_ids(mesh)
    if not l2_points:
        return None
    owners = owner_by_tile_id(placements)
    path = shortest_path_between_regions(
        mesh,
        placements[stage_id].physical_submesh.tile_ids,
        l2_points,
    )
    for tile_id in path[1:]:
        owner = owners.get(tile_id)
        if owner is not None and owner != stage_id:
            return owner
    return None


def _debug(enabled: bool, message: str) -> None:
    """Print one repair trace line when enabled."""

    if enabled:
        print(message)
