"""Mesh topology, connectivity, path, and free-space helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from MAPS.arch import EndpointKind, Mesh, Tile
from MAPS.planner.contracts.stages import StagePlacement


def future_feasible_after_choice(
    mesh: Mesh,
    allowed_tile_ids: set[int],
    chosen_tile_ids: set[int],
    remaining_tile_counts: dict[int, int],
    current_stage_remaining_tiles: int,
    exhaustive: bool = True,
) -> bool:
    """Reject a region choice when it obviously strands later stages."""

    free_after_choice = allowed_tile_ids - chosen_tile_ids
    future_counts = list(remaining_tile_counts.values())
    if current_stage_remaining_tiles > 0:
        future_counts.append(current_stage_remaining_tiles)
    return remaining_counts_fit_free_components(
        mesh=mesh,
        free_tile_ids=free_after_choice,
        remaining_tile_counts=tuple(sorted(future_counts, reverse=True)),
        exhaustive=exhaustive,
    )


def remaining_counts_fit_free_components(
    mesh: Mesh,
    free_tile_ids: set[int],
    remaining_tile_counts: tuple[int, ...],
    exhaustive: bool = True,
) -> bool:
    """Check whether free connected components can host remaining stages."""

    if not remaining_tile_counts:
        return True
    component_sizes = sorted(free_component_sizes(mesh, free_tile_ids), reverse=True)
    requested_sizes = sorted(remaining_tile_counts, reverse=True)
    if sum(component_sizes) < sum(requested_sizes):
        return False
    if not component_sizes or requested_sizes[0] > component_sizes[0]:
        return False
    if (
        exhaustive
        and len(requested_sizes) <= 3
        and sum(requested_sizes) <= 20
    ):
        return _can_partition_connected_regions(
            mesh=mesh,
            free_tile_ids=frozenset(free_tile_ids),
            remaining_tile_counts=requested_sizes,
            memo={},
        )
    return True


def _can_partition_connected_regions(
    mesh: Mesh,
    free_tile_ids: frozenset[int],
    remaining_tile_counts: list[int],
    memo: dict[tuple[frozenset[int], tuple[int, ...]], bool],
) -> bool:
    """Return whether free tiles split into requested connected region sizes."""

    if not remaining_tile_counts:
        return True
    key = (free_tile_ids, tuple(remaining_tile_counts))
    cached = memo.get(key)
    if cached is not None:
        return cached
    tile_count = remaining_tile_counts[0]
    for region in _iter_connected_subsets_of_size(mesh, set(free_tile_ids), tile_count):
        if _can_partition_connected_regions(
            mesh=mesh,
            free_tile_ids=free_tile_ids - region,
            remaining_tile_counts=remaining_tile_counts[1:],
            memo=memo,
        ):
            memo[key] = True
            return True
    memo[key] = False
    return False


def _iter_connected_subsets_of_size(
    mesh: Mesh,
    tile_ids: set[int],
    tile_count: int,
) -> Iterator[frozenset[int]]:
    """Yield connected subsets without enumerating the full powerset."""

    if tile_count <= 0:
        yield frozenset()
        return
    emitted: set[frozenset[int]] = set()
    for seed_tile_id in sorted(tile_ids):
        regions = {frozenset({seed_tile_id})}
        for _ in range(1, tile_count):
            next_regions: set[frozenset[int]] = set()
            for region in regions:
                frontier = set()
                for tile_id in region:
                    frontier |= (
                        (neighbor_ids(mesh, tile_id) & tile_ids) - set(region)
                    )
                for tile_id in frontier:
                    next_regions.add(frozenset((*region, tile_id)))
            regions = next_regions
            if not regions:
                break
        for region in sorted(regions, key=lambda subset: tuple(sorted(subset))):
            if len(region) == tile_count and region not in emitted:
                emitted.add(region)
                yield region


def shortest_path_between_regions(
    mesh: Mesh,
    source_tile_ids: Iterable[int],
    destination_tile_ids: Iterable[int],
) -> tuple[int, ...]:
    """Return one deterministic shortest 4-neighbor path between tile sets."""

    source_set = set(source_tile_ids)
    destination_set = set(destination_tile_ids)
    if source_set & destination_set:
        return (min(source_set & destination_set),)

    queue = deque(sorted(source_set))
    parent: dict[int, int | None] = {tile_id: None for tile_id in source_set}
    while queue:
        tile_id = queue.popleft()
        if tile_id in destination_set:
            break
        for neighbor_id in sorted(neighbor_ids(mesh, tile_id)):
            if neighbor_id in parent:
                continue
            parent[neighbor_id] = tile_id
            queue.append(neighbor_id)

    reached = next(
        (tile_id for tile_id in parent if tile_id in destination_set),
        None,
    )
    if reached is None:
        return ()
    path = []
    cursor: int | None = reached
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return tuple(reversed(path))


def owner_by_tile_id(placements: dict[int, StagePlacement]) -> dict[int, int]:
    """Map each occupied physical tile to its owning stage."""

    return {
        tile_id: stage_id
        for stage_id, placement in placements.items()
        for tile_id in placement.physical_submesh.tile_ids
    }


def shared_boundary_length(
    mesh: Mesh,
    left_tile_ids: Iterable[int],
    right_tile_ids: Iterable[int],
) -> int:
    """Count physical boundary contacts between two tile sets."""

    left = set(left_tile_ids)
    right = set(right_tile_ids)
    if not left or not right:
        return 0
    return sum(
        1
        for tile_id in left
        for neighbor_id in neighbor_ids(mesh, tile_id)
        if neighbor_id in right
    )


def tile_set_center(mesh: Mesh, tile_ids: Iterable[int]) -> tuple[float, float]:
    """Return the geometric center of one tile set."""

    tiles = [mesh.tile_by_id(tile_id) for tile_id in tile_ids]
    if not tiles:
        return (0.0, 0.0)
    return (
        sum(tile.x for tile in tiles) / len(tiles),
        sum(tile.y for tile in tiles) / len(tiles),
    )


def region_compactness(mesh: Mesh, tile_ids: Iterable[int]) -> float:
    """Penalize stretched regions without forcing rectangles."""

    tiles = [mesh.tile_by_id(tile_id) for tile_id in tile_ids]
    if not tiles:
        return 0.0
    xs = [tile.x for tile in tiles]
    ys = [tile.y for tile in tiles]
    bounding_box_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    return float(bounding_box_area - len(tiles))


def future_space_penalty(
    mesh: Mesh,
    free_tile_ids: set[int],
    remaining_tile_counts: tuple[int, ...],
) -> float:
    """Softly penalize fragmentation of space needed by later stages."""

    if not remaining_tile_counts:
        return 0.0
    component_sizes = sorted(free_component_sizes(mesh, free_tile_ids), reverse=True)
    if not component_sizes:
        return 1_000_000.0
    penalty = 0.0
    if sum(component_sizes) < sum(remaining_tile_counts):
        penalty += 1_000_000.0
    if remaining_tile_counts[0] > component_sizes[0]:
        penalty += 1_000_000.0
    penalty += 10.0 * max(0, len(component_sizes) - 1)
    return penalty


def free_component_sizes(mesh: Mesh, free_tile_ids: set[int]) -> tuple[int, ...]:
    """Return sizes of connected components inside the free tile set."""

    seen: set[int] = set()
    sizes: list[int] = []
    for start in sorted(free_tile_ids):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            tile_id = stack.pop()
            size += 1
            for neighbor_id in neighbor_ids(mesh, tile_id):
                if neighbor_id in free_tile_ids and neighbor_id not in seen:
                    seen.add(neighbor_id)
                    stack.append(neighbor_id)
        sizes.append(size)
    return tuple(sizes)


def neighbor_ids(mesh: Mesh, tile_id: int) -> set[int]:
    """Return the existing four-neighbor tile ids of one mesh tile."""

    tile = mesh.tile_by_id(tile_id)
    neighbors: set[int] = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x = tile.x + dx
        y = tile.y + dy
        if mesh.contains_coord(x, y):
            neighbors.add(mesh.tile_id(x, y))
    return neighbors


def l2_access_point_tile_ids(mesh: Mesh) -> set[int]:
    """Return tiles sharing a NoC node with an L2 endpoint."""

    l1_endpoints = tuple(
        endpoint
        for endpoint in mesh.noc.endpoints
        if endpoint.kind is EndpointKind.L1 and endpoint.tile_id is not None
    )
    return {
        endpoint.tile_id
        for l2_endpoint in mesh.noc.endpoints_of_kind(EndpointKind.L2)
        for endpoint in l1_endpoints
        if endpoint.node_id == l2_endpoint.node_id
    }


def remaining_counts_tuple(
    remaining_tile_counts: dict[int, int],
) -> tuple[int, ...]:
    """Normalize remaining tile counts for feasibility scoring."""

    return tuple(sorted(remaining_tile_counts.values(), reverse=True))


def tile_to_point_distance(tile: Tile, point: tuple[float, float]) -> float:
    """Return Manhattan distance from a tile to a floating-point target."""

    return abs(tile.x - point[0]) + abs(tile.y - point[1])
