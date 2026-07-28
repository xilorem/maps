"""High-level construction of connected physical stage regions."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.planner.contracts.stages import StagePlacement, StagePlan
from MAPS.planner.spatial.models import VirtualTraffic
from MAPS.planner.spatial.ownership import stage_order
from MAPS.planner.spatial.region_results import placements_from_regions
from MAPS.planner.spatial.region_scoring import sorted_candidate_tiles
from MAPS.planner.spatial.region_search import beam_connected_region, greedy_connected_region
from MAPS.planner.spatial.topology import l2_access_point_tile_ids, tile_set_center


def build_initial_stage_placements(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    tile_counts: dict[int, int],
    traffic: VirtualTraffic,
    debug: bool,
) -> dict[int, StagePlacement]:
    """Build the first feasible communication-aware stage placement.

    Stages are considered from greatest placement pressure to least.  Each stage
    grows a connected region around a weighted communication/L2 target while a
    feasibility check protects enough connected free space for later stages.
    Regions are disjoint and have exactly the requested tile counts.
    """

    free_tile_ids = set(range(mesh.num_tiles))
    placed_regions: dict[int, set[int]] = {}
    ordered_stage_ids = stage_order(tile_counts, traffic)
    _debug(debug, f"[spatial_mapping] phase=initial_seeding stage_order={ordered_stage_ids}")
    for stage_idx, stage_id in enumerate(ordered_stage_ids):
        remaining_tile_counts = {
            other_stage_id: tile_counts[other_stage_id]
            for other_stage_id in ordered_stage_ids[stage_idx + 1:]
        }
        target = stage_target_point(stage_id, mesh, placed_regions, traffic)
        region = grow_stage_region(
            stage_id=stage_id,
            mesh=mesh,
            allowed_tile_ids=free_tile_ids,
            tile_count=tile_counts[stage_id],
            target=target,
            traffic=traffic,
            placed_regions=placed_regions,
            remaining_tile_counts=remaining_tile_counts,
        )
        placed_regions[stage_id] = region
        free_tile_ids -= region
        _debug(
            debug,
            "[spatial_mapping] "
            f"seeded stage={stage_id} target=({target[0]:.2f},{target[1]:.2f}) "
            f"tiles={sorted(region)}",
        )
    return placements_from_regions(mesh, stage_plans, placed_regions)


def grow_stage_region(
    stage_id: int,
    mesh: Mesh,
    allowed_tile_ids: set[int],
    tile_count: int,
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    remaining_tile_counts: dict[int, int],
    preferred_seed: int | None = None,
    exhaustive_future_feasibility: bool = True,
) -> set[int]:
    """Grow one connected region while protecting future feasibility.

    Up to sixteen communication-ranked seeds are tried with fast greedy growth.
    If all become infeasible, bounded beam search explores a wider set of
    connected shapes.  Failure means no region was found under this heuristic.
    """

    seed_candidates = sorted_candidate_tiles(
        mesh,
        allowed_tile_ids,
        target,
        stage_id,
        traffic,
        placed_regions,
    )
    if preferred_seed is not None and preferred_seed in allowed_tile_ids:
        seed_candidates = [preferred_seed] + [
            tile_id
            for tile_id in seed_candidates
            if tile_id != preferred_seed
        ]
    if not seed_candidates:
        raise ValueError(f"cannot seed stage {stage_id} from an empty free region")

    failures: list[str] = []
    for seed_tile_id in seed_candidates[: min(len(seed_candidates), 16)]:
        try:
            return greedy_connected_region(
                stage_id,
                mesh,
                seed_tile_id,
                allowed_tile_ids,
                tile_count,
                target,
                traffic,
                placed_regions,
                remaining_tile_counts,
                exhaustive_future_feasibility,
            )
        except ValueError as exc:
            failures.append(str(exc))
    region = beam_connected_region(
        stage_id,
        mesh,
        allowed_tile_ids,
        tile_count,
        target,
        traffic,
        placed_regions,
        remaining_tile_counts,
        exhaustive_future_feasibility,
    )
    if region is None:
        raise ValueError(
            "; ".join(failures)
            if failures
            else f"cannot grow region for stage {stage_id}"
        )
    return region


def local_stage_order(
    affected_stages: frozenset[int],
    tile_counts: dict[int, int],
    traffic: VirtualTraffic,
    focus_stage_id: int,
) -> tuple[int, ...]:
    """Bias local repair ordering around the current bottleneck stage."""

    return tuple(
        sorted(
            affected_stages,
            key=lambda stage_id: (
                0 if stage_id == focus_stage_id else 1,
                -traffic.communication_degree.get(stage_id, 0),
                -traffic.bottleneck_risk.get(stage_id, 0),
                -tile_counts[stage_id],
                stage_id,
            ),
        )
    )


def stage_target_point(
    stage_id: int,
    mesh: Mesh,
    placed_regions: dict[int, set[int]],
    traffic: VirtualTraffic,
) -> tuple[float, float]:
    """Return the weighted peer-communication and L2 target for one stage."""

    weighted_points: list[tuple[float, float, float]] = []
    for (source_stage_id, destination_stage_id), weight in traffic.stage_comm.items():
        if weight <= 0:
            continue
        if destination_stage_id == stage_id and source_stage_id in placed_regions:
            x, y = tile_set_center(mesh, placed_regions[source_stage_id])
            weighted_points.append((x, y, float(weight)))
        elif source_stage_id == stage_id and destination_stage_id in placed_regions:
            x, y = tile_set_center(mesh, placed_regions[destination_stage_id])
            weighted_points.append((x, y, float(weight)))

    l2_points = tuple(sorted(l2_access_point_tile_ids(mesh)))
    if l2_points and traffic.l2_pressure.get(stage_id, 0) > 0:
        x, y = tile_set_center(mesh, set(l2_points))
        weighted_points.append((x, y, float(traffic.l2_pressure[stage_id])))
    if not weighted_points:
        return ((mesh.width - 1) / 2.0, (mesh.height - 1) / 2.0)
    total_weight = sum(weight for _, _, weight in weighted_points)
    return (
        sum(x * weight for x, _, weight in weighted_points) / total_weight,
        sum(y * weight for _, y, weight in weighted_points) / total_weight,
    )


def _debug(enabled: bool, message: str) -> None:
    """Print one region-construction trace line when enabled."""

    if enabled:
        print(message)
