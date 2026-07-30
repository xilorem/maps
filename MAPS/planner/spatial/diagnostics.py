"""Human-readable spatial-mapping diagnostics."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Node
from MAPS.planner.contracts.stages import StagePlacement, StagePlan
from MAPS.planner.spatial.evaluation import evaluate_mapping
from MAPS.planner.spatial.topology import owner_by_tile_id
from MAPS.transitions import VirtualTransition


def print_spatial_mapping_details(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    virtual_transitions: tuple[VirtualTransition, ...],
    label: str = "mapping",
) -> None:
    """Print physical regions, ownership maps, and exact IO bottlenecks."""

    evaluation = evaluate_mapping(
        mesh,
        stage_plans,
        placements,
        virtual_transitions,
    )
    print(f"\n[spatial_mapping] chosen physical submeshes for {label}:")
    for stage_id in stage_plans:
        placement = placements[stage_id]
        submesh = placement.physical_submesh
        print(
            f"  stage={stage_id} name={_stage_name(stage_plans[stage_id].nodes)} "
            f"bbox=({submesh.x0},{submesh.y0},{submesh.width},{submesh.height}) "
            f"tiles={sorted(submesh.tile_ids)} "
            f"virtual_to_physical={dict(sorted(placement.virtual_to_physical.items()))}"
        )
    print_placement_grid(mesh, placements)
    print(f"[spatial_mapping] stage worst physical-tile IO costs for {label}:")
    for stage_id in stage_plans:
        io_cost = evaluation.stage_breakdowns[stage_id]
        print(
            f"  stage={stage_id} name={_stage_name(stage_plans[stage_id].nodes)} "
            f"tile={io_cost.physical_tile_id} l2_read={io_cost.l2_read} "
            f"l2_write={io_cost.l2_write} l1_write={io_cost.l1_write} "
            f"total={io_cost.total}"
        )
    print(
        f"[spatial_mapping] bottleneck for {label} "
        f"worst_stage_io={max((cost.total for cost in evaluation.stage_breakdowns.values()), default=0)} "
        f"objective={evaluation.objective}"
    )


def print_placement_grid(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
) -> None:
    """Print a compact mesh grid showing physical stage ownership."""

    owners = owner_by_tile_id(placements)
    cell_width = max(1, *(len(str(stage_id)) for stage_id in placements))
    print("Spatial mapping mesh:")
    for y in range(mesh.height):
        cells = []
        for x in range(mesh.width):
            owner = owners.get(mesh.tile_id(x, y))
            cells.append(("." if owner is None else str(owner)).rjust(cell_width))
        print(" ".join(cells))


def _stage_name(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact selected-stage display name."""

    return "+".join(node.name for node in stage_nodes)
