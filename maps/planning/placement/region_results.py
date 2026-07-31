"""Construct placement contracts from selected physical regions."""

from __future__ import annotations

from maps.hardware import Mesh
from maps.planning.submesh import Submesh
from maps.planning.stages import StagePlacement, StagePlan, virtual_submesh


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
