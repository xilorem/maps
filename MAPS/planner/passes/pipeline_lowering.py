"""Pipeline-lowering pass facade."""

from __future__ import annotations

from dataclasses import replace

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.pipeline.pipeline import Pipeline
from MAPS.pipeline.execution import ExecutionContract
from MAPS.planner.contracts.stages import StagePlacement, StagePlan
from MAPS.planner.lowering.boundaries import build_finalizations, build_initializations
from MAPS.planner.lowering.context import build_lowering_context
from MAPS.planner.lowering.stages import build_stages
from MAPS.planner.lowering.transitions import build_transitions


def lower_pipeline(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    *,
    execution: ExecutionContract = ExecutionContract(),
) -> Pipeline:
    """Lower complete planner decisions into executable Pipeline IR.

    Contract:
        ``stage_plans`` must cover every selected graph node and contain the
        virtual output layouts chosen by workload balancing.  When supplied,
        ``placements`` must contain exactly one physical binding for every stage
        plan.

    Behavior:
        The pass indexes graph ownership, builds cross-stage transitions, emits
        stages and layer bindings, and finally constructs graph-boundary
        initialization/finalization transfers.  Virtual layout decisions are
        never modified during lowering.

    Returns:
        A complete ``Pipeline`` whose tensors preserve graph ordering and whose
        transfer endpoints refer to physical mesh tile ids.
    """

    if set(placements) != set(stage_plans):
        raise ValueError("placements must contain exactly one entry per stage plan")

    context = build_lowering_context(graph, stage_plans)
    transitions, transition_ids = build_transitions(
        context,
        stage_plans,
        placements,
    )
    stages = build_stages(
        context,
        stage_plans,
        placements,
        transition_ids,
    )
    initializations = build_initializations(context, stage_plans, placements)
    finalizations = build_finalizations(context, stage_plans, placements)

    return Pipeline(
        name=graph.name,
        mesh=mesh,
        execution=execution,
        tensors=tuple(
            replace(tensor, is_initializer=tensor in graph.initializers)
            for tensor in graph.tensors
        ),
        stages=stages,
        transitions=transitions,
        initializations=initializations,
        finalizations=finalizations,
    )


__all__ = ["lower_pipeline"]
