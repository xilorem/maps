"""Materialize stage-plan candidates for fixed tile allocations."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Node
from MAPS.planner.contracts.stages import StagePlan, StageSelection
from MAPS.planner.workload.candidates import best_stage_plan


def plan_all_stages(
    stage_selection: StageSelection,
    mesh: Mesh,
    tile_counts: dict[int, int],
    initializer_tensors: frozenset,
    debug: bool,
    num_token_slots: int = 2,
) -> dict[int, StagePlan]:
    """Choose the best feasible logical layout for every fixed tile count."""

    return {
        stage_id: best_plan_for_stage(
            stage_nodes=stage_nodes,
            mesh=mesh,
            stage_id=stage_id,
            tile_count=tile_counts[stage_id],
            initializer_tensors=initializer_tensors,
            debug=debug,
            num_token_slots=num_token_slots,
        )
        for stage_id, stage_nodes in stage_selection.items()
    }


def best_plan_for_stage(
    stage_nodes: tuple[Node, ...],
    mesh: Mesh,
    stage_id: int,
    tile_count: int,
    initializer_tensors: frozenset,
    debug: bool = False,
    num_token_slots: int = 2,
) -> StagePlan:
    """Return the lowest-cost feasible layout candidate for one stage size."""

    plan = best_stage_plan(
        stage_nodes,
        mesh,
        stage_id,
        tile_count,
        initializer_tensors,
        num_token_slots,
    )
    if debug:
        print(
            "[workload_balancing] "
            f"stage={stage_id} tile_count={tile_count} "
            f"logical_shape={plan.logical_shape}"
        )
    return plan
