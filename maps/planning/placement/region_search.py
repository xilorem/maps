"""Greedy and beam search for connected stage regions."""

from __future__ import annotations

from maps.hardware import Mesh
from maps.planning.placement.models import VirtualTraffic
from maps.planning.placement.region_scoring import growth_candidate_score, region_score
from maps.planning.placement.topology import future_feasible_after_choice, neighbor_ids


def greedy_connected_region(
    stage_id: int,
    mesh: Mesh,
    seed_tile_id: int,
    allowed_tile_ids: set[int],
    tile_count: int,
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    remaining_tile_counts: dict[int, int],
    exhaustive_future_feasibility: bool = True,
) -> set[int]:
    """Grow a connected region from one seed using local best choices."""

    chosen = {seed_tile_id}
    frontier = (neighbor_ids(mesh, seed_tile_id) & allowed_tile_ids) - chosen
    while len(chosen) < tile_count:
        if not frontier:
            raise ValueError(f"stage {stage_id} cannot grow a connected region")
        next_tile_id = None
        candidates = sorted(
            frontier,
            key=lambda tile_id: growth_candidate_score(
                stage_id,
                mesh,
                tile_id,
                chosen,
                target,
                traffic,
                placed_regions,
                allowed_tile_ids,
                remaining_tile_counts,
            ),
        )
        for candidate_tile_id in candidates:
            candidate_region = chosen | {candidate_tile_id}
            if future_feasible_after_choice(
                mesh,
                allowed_tile_ids,
                candidate_region,
                remaining_tile_counts,
                tile_count - len(candidate_region),
                exhaustive=exhaustive_future_feasibility,
            ):
                next_tile_id = candidate_tile_id
                break
        if next_tile_id is None:
            raise ValueError(f"stage {stage_id} fragments the remaining free region")
        chosen.add(next_tile_id)
        frontier.remove(next_tile_id)
        frontier |= (neighbor_ids(mesh, next_tile_id) & allowed_tile_ids) - chosen
    return chosen


def beam_connected_region(
    stage_id: int,
    mesh: Mesh,
    allowed_tile_ids: set[int],
    tile_count: int,
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    remaining_tile_counts: dict[int, int],
    exhaustive_future_feasibility: bool = True,
) -> set[int] | None:
    """Search a wider set of regions when greedy growth gets boxed in."""

    beam_width = 256
    regions = {frozenset({tile_id}) for tile_id in allowed_tile_ids}
    for _ in range(1, tile_count):
        next_regions: set[frozenset[int]] = set()
        for region in regions:
            frontier = set()
            for tile_id in region:
                frontier |= (
                    neighbor_ids(mesh, tile_id) & allowed_tile_ids
                ) - set(region)
            for tile_id in frontier:
                next_regions.add(frozenset((*region, tile_id)))
        if not next_regions:
            return None
        regions = set(
            sorted(
                next_regions,
                key=lambda region: region_score(
                    stage_id,
                    mesh,
                    set(region),
                    target,
                    traffic,
                    placed_regions,
                    allowed_tile_ids,
                    remaining_tile_counts,
                ),
            )[:beam_width]
        )

    feasible_regions = [
        set(region)
        for region in regions
        if future_feasible_after_choice(
            mesh,
            allowed_tile_ids,
            set(region),
            remaining_tile_counts,
            0,
            exhaustive=exhaustive_future_feasibility,
        )
    ]
    if not feasible_regions:
        return None
    return min(
        feasible_regions,
        key=lambda region: region_score(
            stage_id,
            mesh,
            region,
            target,
            traffic,
            placed_regions,
            allowed_tile_ids,
            remaining_tile_counts,
        ),
    )
