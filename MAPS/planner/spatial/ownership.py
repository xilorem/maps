"""Assign virtual stage tiles to tiles inside chosen physical regions."""

from __future__ import annotations

from MAPS.arch import Mesh
from maps.planning.stages import StagePlacement, StagePlan
from MAPS.planner.spatial.models import VirtualTraffic
from MAPS.planner.spatial.topology import l2_access_point_tile_ids, tile_set_center


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
