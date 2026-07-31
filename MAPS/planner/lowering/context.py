"""Indexes used while lowering planner decisions into Execution Plan IR."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.graph import Graph, Node
from MAPS.planner.contracts.stages import StagePlan


@dataclass(frozen=True)
class ExecutionPlanLoweringContext:
    """Precomputed identity indexes required throughout Execution Plan lowering.

    Graph nodes and tensors are immutable domain objects, but several lowering
    decisions depend on object identity rather than value equality.  Building
    these indexes once makes that rule explicit and prevents each lowering
    component from reconstructing subtly different producer or stage maps.
    """

    graph: Graph
    stage_selection: dict[int, tuple[Node, ...]]
    node_stage_ids: dict[int, int]
    node_stage_layer_ids: dict[int, int]
    node_graph_layer_ids: dict[int, int]
    tensor_id_by_tensor: dict[object, int]
    producer_by_tensor: dict[object, Node]


def build_lowering_context(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
) -> ExecutionPlanLoweringContext:
    """Index graph ownership and ordering for a consistent lowering operation."""

    stage_selection = _resolve_stage_selection(graph, stage_plans)
    return ExecutionPlanLoweringContext(
        graph=graph,
        stage_selection=stage_selection,
        node_stage_ids={
            id(node): stage_id
            for stage_id, stage_nodes in stage_selection.items()
            for node in stage_nodes
        },
        node_stage_layer_ids={
            id(node): layer_idx
            for stage_nodes in stage_selection.values()
            for layer_idx, node in enumerate(stage_nodes)
        },
        node_graph_layer_ids={
            id(node): layer_id
            for layer_id, node in enumerate(graph.nodes)
        },
        tensor_id_by_tensor={
            tensor: tensor_id
            for tensor_id, tensor in enumerate(graph.tensors)
        },
        producer_by_tensor={
            tensor: node
            for node in graph.nodes
            for tensor in node.outputs
        },
    )


def _resolve_stage_selection(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
) -> dict[int, tuple[Node, ...]]:
    """Validate and recover the selected nodes carried by stage plans."""

    if any(not plan.nodes for plan in stage_plans.values()):
        raise ValueError("every stage plan must contain its selected nodes")
    stage_selection = {
        stage_id: plan.nodes
        for stage_id, plan in stage_plans.items()
    }
    selected_node_ids = [
        id(node)
        for nodes in stage_selection.values()
        for node in nodes
    ]
    graph_node_ids = {id(node) for node in graph.nodes}
    if len(selected_node_ids) != len(set(selected_node_ids)):
        raise ValueError("stage plans contain a graph node more than once")
    if set(selected_node_ids) != graph_node_ids:
        raise ValueError("stage plans must cover every graph node exactly once")
    return stage_selection
