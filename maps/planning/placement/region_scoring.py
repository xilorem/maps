"""Communication, compactness, and free-space scores for stage regions."""

from __future__ import annotations

from collections.abc import Iterable

from maps.hardware import Mesh, Tile
from maps.planning.placement.models import VirtualTraffic
from maps.planning.placement.topology import (
    future_space_penalty,
    l2_access_point_tile_ids,
    region_compactness,
    remaining_counts_tuple,
    tile_set_center,
    tile_to_point_distance,
)


def sorted_candidate_tiles(
    mesh: Mesh,
    candidate_tile_ids: Iterable[int],
    target: tuple[float, float],
    stage_id: int,
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
) -> list[int]:
    """Order candidate seeds by communication-aware target score."""

    return sorted(
        candidate_tile_ids,
        key=lambda tile_id: (
            _seed_tile_score(
                stage_id,
                mesh,
                mesh.tile_by_id(tile_id),
                target,
                traffic,
                placed_regions,
            ),
            mesh.tile_by_id(tile_id).y,
            mesh.tile_by_id(tile_id).x,
            tile_id,
        ),
    )


def growth_candidate_score(
    stage_id: int,
    mesh: Mesh,
    tile_id: int,
    chosen: set[int],
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    allowed_tile_ids: set[int],
    remaining_tile_counts: dict[int, int],
) -> tuple[float, float, float, int]:
    """Score one frontier tile for connected-region growth."""

    tile = mesh.tile_by_id(tile_id)
    candidate_region = chosen | {tile_id}
    target_cost = abs(tile.x - target[0]) + abs(tile.y - target[1])
    compactness_cost = region_compactness(mesh, candidate_region)
    anchor_cost = stage_anchor_cost(mesh, stage_id, tile, traffic, placed_regions)
    future_penalty = future_space_penalty(
        mesh,
        allowed_tile_ids - candidate_region,
        remaining_counts_tuple(remaining_tile_counts),
    )
    return (
        target_cost + anchor_cost + compactness_cost + future_penalty,
        future_penalty,
        compactness_cost,
        tile_id,
    )


def region_score(
    stage_id: int,
    mesh: Mesh,
    region: set[int],
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    allowed_tile_ids: set[int],
    remaining_tile_counts: dict[int, int],
) -> tuple[float, float, float, tuple[int, ...]]:
    """Score a complete region by target, anchors, shape, and future space."""

    center = tile_set_center(mesh, region)
    target_cost = abs(center[0] - target[0]) + abs(center[1] - target[1])
    anchor_cost = region_anchor_cost(stage_id, mesh, region, traffic, placed_regions)
    compactness_cost = region_compactness(mesh, region)
    future_penalty = future_space_penalty(
        mesh,
        allowed_tile_ids - region,
        remaining_counts_tuple(remaining_tile_counts),
    )
    return (
        target_cost + anchor_cost + compactness_cost + future_penalty,
        future_penalty,
        compactness_cost,
        tuple(sorted(region)),
    )


def stage_anchor_cost(
    mesh: Mesh,
    stage_id: int,
    tile: Tile,
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
) -> float:
    """Score a tile relative to placed communication and L2 anchors."""

    score = 0.0
    for (source_stage_id, destination_stage_id), weight in traffic.stage_comm.items():
        if weight <= 0:
            continue
        if source_stage_id == stage_id and destination_stage_id in placed_regions:
            center = tile_set_center(mesh, placed_regions[destination_stage_id])
            score += weight * tile_to_point_distance(tile, center)
        elif destination_stage_id == stage_id and source_stage_id in placed_regions:
            center = tile_set_center(mesh, placed_regions[source_stage_id])
            score += weight * tile_to_point_distance(tile, center)

    l2_weight = traffic.l2_pressure.get(stage_id, 0)
    if l2_weight > 0:
        access_points = tuple(
            (mesh.tile_by_id(tile_id).x, mesh.tile_by_id(tile_id).y)
            for tile_id in l2_access_point_tile_ids(mesh)
        )
        if access_points:
            score += l2_weight * min(
                abs(tile.x - x) + abs(tile.y - y)
                for x, y in access_points
            )
    return score / max(1, len(traffic.stage_comm))


def region_anchor_cost(
    stage_id: int,
    mesh: Mesh,
    region: set[int],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
) -> float:
    """Sum communication-anchor costs across a whole region."""

    return sum(
        stage_anchor_cost(
            mesh,
            stage_id,
            mesh.tile_by_id(tile_id),
            traffic,
            placed_regions,
        )
        for tile_id in region
    )


def _seed_tile_score(
    stage_id: int,
    mesh: Mesh,
    tile: Tile,
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
) -> float:
    """Score one seed by target distance and communication anchors."""

    return (
        abs(tile.x - target[0])
        + abs(tile.y - target[1])
        + stage_anchor_cost(mesh, stage_id, tile, traffic, placed_regions)
    )
