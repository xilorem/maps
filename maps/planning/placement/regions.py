"""High-level construction of connected physical stage regions."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from maps.hardware import EndpointKind, Mesh, Tile
from maps.planning.mapping import Submesh
from maps.planning.stages import StagePlacement, StagePlan
from maps.planning.stages import virtual_submesh
from maps.planning.placement.evaluation import VirtualTraffic


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
    _debug(debug, f"[placement] phase=initial_seeding stage_order={ordered_stage_ids}")
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
            "[placement] "
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


def placements_from_regions(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    regions: dict[int, set[int]],
) -> dict[int, StagePlacement]:
    """Create placements with stable placeholder ownership for given regions."""

    placements = {}
    for stage_id, region in regions.items():
        virtual = virtual_submesh(stage_plans[stage_id])
        physical = Submesh(
            mesh=mesh,
            submesh_id=stage_id,
            tile_ids=frozenset(region),
        )
        placements[stage_id] = StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual,
            physical_submesh=physical,
            virtual_to_physical=dict(
                zip(
                    (tile.tile_id for tile in virtual.tiles),
                    (tile.tile_id for tile in physical.tiles),
                )
            ),
        )
    return placements


def assign_stage_ownerships(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    traffic: VirtualTraffic,
    stage_ids: frozenset[int] | None = None,
) -> dict[int, StagePlacement]:
    """Choose a bijection from virtual tiles to each stage's physical region.

    Stages and virtual tiles are processed from greatest communication pressure
    to least.  Each virtual tile takes the lowest-cost remaining physical tile,
    considering known peer ownership, unassigned peer-region centers, L2 access,
    and a small compactness bias.  Stable coordinate tie-breaks make the result
    deterministic.
    """

    ordered_stage_ids = stage_order(
        {stage_id: plan.tile_count for stage_id, plan in stage_plans.items()},
        traffic,
    )
    if stage_ids is not None:
        ordered_stage_ids = tuple(
            stage_id for stage_id in ordered_stage_ids if stage_id in stage_ids
        )
    stage_centers = {
        stage_id: tile_set_center(mesh, placement.physical_submesh.tile_ids)
        for stage_id, placement in placements.items()
    }
    assigned = (
        {}
        if stage_ids is None
        else {
            stage_id: placement
            for stage_id, placement in placements.items()
            if stage_id not in stage_ids
        }
    )

    for stage_id in ordered_stage_ids:
        placement = placements[stage_id]
        virtual_tile_ids = tuple(
            tile.tile_id
            for tile in placement.virtual_submesh.tiles
        )
        free_physical_tile_ids = set(placement.physical_submesh.tile_ids)
        virtual_priority = sorted(
            virtual_tile_ids,
            key=lambda virtual_tile_id: (
                -_virtual_priority(stage_id, virtual_tile_id, traffic),
                virtual_tile_id,
            ),
        )
        owner_by_virtual: dict[int, int] = {}
        for virtual_tile_id in virtual_priority:
            physical_tile_id = min(
                free_physical_tile_ids,
                key=lambda candidate_tile_id: (
                    _virtual_assignment_cost(
                        mesh=mesh,
                        stage_id=stage_id,
                        virtual_tile_id=virtual_tile_id,
                        physical_tile_id=candidate_tile_id,
                        placements=placements,
                        assigned=assigned,
                        stage_centers=stage_centers,
                        traffic=traffic,
                    ),
                    mesh.tile_by_id(candidate_tile_id).y,
                    mesh.tile_by_id(candidate_tile_id).x,
                    candidate_tile_id,
                ),
            )
            owner_by_virtual[virtual_tile_id] = physical_tile_id
            free_physical_tile_ids.remove(physical_tile_id)

        assigned[stage_id] = StagePlacement(
            stage_id=placement.stage_id,
            virtual_submesh=placement.virtual_submesh,
            physical_submesh=placement.physical_submesh,
            virtual_to_physical=owner_by_virtual,
        )
    return assigned


def stage_order(
    tile_counts: dict[int, int],
    traffic: VirtualTraffic,
) -> tuple[int, ...]:
    """Order stages by size and communication-aware placement priority."""

    return tuple(
        sorted(
            tile_counts,
            key=lambda stage_id: (
                -tile_counts[stage_id],
                -traffic.communication_degree.get(stage_id, 0),
                -traffic.bottleneck_risk.get(stage_id, 0),
                -traffic.l2_pressure.get(stage_id, 0),
                stage_id,
            ),
        )
    )


def _assignable_reference_points(
    stage_id: int,
    virtual_tile_id: int,
    placements: dict[int, StagePlacement],
    assigned: dict[int, StagePlacement],
    stage_centers: dict[int, tuple[float, float]],
    traffic: VirtualTraffic,
    is_destination: bool,
) -> list[tuple[float, float, int]]:
    """Collect weighted peer locations relevant to one virtual tile."""

    points: list[tuple[float, float, int]] = []
    for (source_stage_id, destination_stage_id), matrix in traffic.edge_matrices.items():
        if is_destination:
            if destination_stage_id != stage_id:
                continue
            for (source_virtual_id, destination_virtual_id), bytes_ in matrix.items():
                if destination_virtual_id != virtual_tile_id or bytes_ <= 0:
                    continue
                if source_stage_id in assigned:
                    source_tile_id = assigned[source_stage_id].physical_tile_id(
                        source_virtual_id
                    )
                    source_tile = placements[
                        source_stage_id
                    ].physical_submesh.mesh.tile_by_id(source_tile_id)
                    points.append((source_tile.x, source_tile.y, bytes_))
                else:
                    x, y = stage_centers[source_stage_id]
                    points.append((x, y, bytes_))
        else:
            if source_stage_id != stage_id:
                continue
            for (source_virtual_id, destination_virtual_id), bytes_ in matrix.items():
                if source_virtual_id != virtual_tile_id or bytes_ <= 0:
                    continue
                if destination_stage_id in assigned:
                    destination_tile_id = assigned[
                        destination_stage_id
                    ].physical_tile_id(destination_virtual_id)
                    destination_tile = placements[
                        destination_stage_id
                    ].physical_submesh.mesh.tile_by_id(destination_tile_id)
                    points.append((destination_tile.x, destination_tile.y, bytes_))
                else:
                    x, y = stage_centers[destination_stage_id]
                    points.append((x, y, bytes_))
    return points


def _virtual_assignment_cost(
    mesh: Mesh,
    stage_id: int,
    virtual_tile_id: int,
    physical_tile_id: int,
    placements: dict[int, StagePlacement],
    assigned: dict[int, StagePlacement],
    stage_centers: dict[int, tuple[float, float]],
    traffic: VirtualTraffic,
) -> float:
    """Score one virtual-to-physical ownership choice."""

    tile = mesh.tile_by_id(physical_tile_id)
    score = 0.0
    for is_destination in (True, False):
        for x, y, bytes_ in _assignable_reference_points(
            stage_id=stage_id,
            virtual_tile_id=virtual_tile_id,
            placements=placements,
            assigned=assigned,
            stage_centers=stage_centers,
            traffic=traffic,
            is_destination=is_destination,
        ):
            score += bytes_ * (abs(tile.x - x) + abs(tile.y - y))

    l2_points = tuple(
        (mesh.tile_by_id(tile_id).x, mesh.tile_by_id(tile_id).y)
        for tile_id in l2_access_point_tile_ids(mesh)
    )
    if l2_points:
        l2_distance = min(
            abs(tile.x - x) + abs(tile.y - y)
            for x, y in l2_points
        )
        score += (
            traffic.l2_read_weights.get(stage_id, {}).get(virtual_tile_id, 0)
            * l2_distance
        )
        score += (
            traffic.l2_write_weights.get(stage_id, {}).get(virtual_tile_id, 0)
            * l2_distance
        )

    center_x, center_y = stage_centers[stage_id]
    score += 0.1 * (abs(tile.x - center_x) + abs(tile.y - center_y))
    return score


def _virtual_priority(
    stage_id: int,
    virtual_tile_id: int,
    traffic: VirtualTraffic,
) -> int:
    """Prioritize the greatest input, output, or L2 pressure."""

    return max(
        traffic.input_weights.get(stage_id, {}).get(virtual_tile_id, 0),
        traffic.output_weights.get(stage_id, {}).get(virtual_tile_id, 0),
        traffic.l2_read_weights.get(stage_id, {}).get(virtual_tile_id, 0)
        + traffic.l2_write_weights.get(stage_id, {}).get(virtual_tile_id, 0),
    )
