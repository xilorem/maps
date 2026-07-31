"""Generate feasible virtual layout candidates for a stage."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Node
from MAPS.planner.contracts.stages import StagePlan
from MAPS.planner.workload.cost import cost_estimator
from MAPS.planner.workload.memory import permanent_l1_allocation_for_stage
from MAPS.planner.workload.layouts import resolve_stage_layouts, verify_stage_locality
from MAPS.planner.workload.submesh import representative_connected_submesh


class NoL1FeasibleLayoutError(ValueError):
    """No logical shape at one tile count fits permanent L1 residency."""


def best_stage_plan(
    stage_nodes: tuple[Node, ...],
    mesh: Mesh,
    stage_id: int,
    tile_count: int,
    initializer_tensors: frozenset,
    num_token_slots: int = 2,
) -> StagePlan:
    """Choose the lowest-compute L1-feasible layout at one tile count.

    Every factorization of ``tile_count`` is offered to each node payload as a
    logical shape.  A candidate is legal only when the stage peak fits in every
    representative tile's L1.  Compute cost breaks the primary tie, followed by
    logical height for deterministic selection.
    """

    submesh = representative_connected_submesh(mesh, stage_id, tile_count)
    best_plan: StagePlan | None = None
    best_workload = float("inf")
    for logical_shape in logical_shape_options(tile_count):
        layouts = resolve_stage_layouts(stage_nodes, submesh, logical_shape)
        verify_stage_locality(stage_nodes, layouts, submesh)
        allocated_l1_bytes = permanent_l1_allocation_for_stage(
            stage_nodes,
            layouts,
            submesh,
            initializer_tensors,
            num_token_slots,
        )
        if allocated_l1_bytes > min(tile.memory.size for tile in submesh.tiles):
            continue
        plan = StagePlan(
            stage_id=stage_id,
            tile_count=tile_count,
            logical_shape=logical_shape,
            nodes=stage_nodes,
            node_output_layouts=layouts,
        )
        workload = sum(
            cost_estimator(node=node, output_layouts=output_layouts)
            for node, output_layouts in zip(stage_nodes, layouts)
        )
        if best_plan is None or workload < best_workload or (
            workload == best_workload
            and logical_shape[1] < best_plan.logical_shape[1]
        ):
            best_plan, best_workload = plan, workload
    if best_plan is None:
        names = "+".join(node.name for node in stage_nodes)
        raise NoL1FeasibleLayoutError(
            f"stage {names} has no valid logical shape for tile_count={tile_count} "
            "using local stage layouts and permanent L1 allocation"
        )
    return best_plan


def logical_shape_options(tile_count: int) -> tuple[tuple[int, int], ...]:
    """Enumerate rectangular logical shapes whose area equals ``tile_count``."""

    return tuple(
        (tile_count // height, height)
        for height in range(1, tile_count + 1)
        if tile_count % height == 0
    )
