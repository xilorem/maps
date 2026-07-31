"""Validated inputs and graph classifications for workload balancing."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.graph import Graph
from MAPS.planner.contracts.stages import StageSelection
from MAPS.planner.validation.stages import validate_stage_selection


@dataclass(frozen=True)
class WorkloadContext:
    """Graph facts shared by allocation and bottleneck estimation."""

    graph: Graph
    stage_selection: StageSelection
    initializer_tensors: frozenset
    graph_inputs: frozenset
    graph_outputs: frozenset
    producer_stage_id_by_tensor: dict[object, int]


def build_workload_context(
    graph: Graph,
    stage_selection: StageSelection,
) -> WorkloadContext:
    """Validate stage coverage and classify boundary and produced tensors."""

    resolved_selection = validate_stage_selection(graph, stage_selection)
    initializer_tensors = frozenset(graph.initializers)
    return WorkloadContext(
        graph=graph,
        stage_selection=resolved_selection,
        initializer_tensors=initializer_tensors,
        graph_inputs=frozenset(graph.inputs) - initializer_tensors,
        graph_outputs=frozenset(graph.outputs),
        producer_stage_id_by_tensor=producer_stage_id_by_tensor(resolved_selection),
    )


def producer_stage_id_by_tensor(
    stage_selection: StageSelection,
) -> dict[object, int]:
    """Map every produced tensor to the id of its selected producer stage."""

    return {
        tensor: stage_id
        for stage_id, stage_nodes in stage_selection.items()
        for node in stage_nodes
        for tensor in node.outputs
    }
