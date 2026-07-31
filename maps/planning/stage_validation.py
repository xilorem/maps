"""Stage validation shared by formation and Allocation."""

from __future__ import annotations

from dataclasses import dataclass

from maps.graph import Graph, Node
from maps.graph import Tensor
from maps.operations.contracts import find_layout_relation
from maps.planning.stages import StageFormation

STAGE_GROUP_ID_ATTR = "stage_group_id"


@dataclass(frozen=True)
class StageCommunicationEdges:
    """Graph communication facts used to validate Stage boundaries."""

    runtime_inputs: frozenset[Tensor]
    graph_outputs: frozenset[Tensor]
    producer_by_tensor: dict[Tensor, Node]
    consumers_by_tensor: dict[Tensor, tuple[Node, ...]]

    @classmethod
    def from_graph(cls, graph: Graph) -> "StageCommunicationEdges":
        consumers_by_tensor: dict[Tensor, list[Node]] = {}
        for node in graph.nodes:
            for tensor in node.inputs:
                consumers_by_tensor.setdefault(tensor, []).append(node)
        return cls(
            runtime_inputs=frozenset(graph.inputs) - frozenset(graph.initializers),
            graph_outputs=frozenset(graph.outputs),
            producer_by_tensor={
                tensor: node
                for node in graph.nodes
                for tensor in node.outputs
            },
            consumers_by_tensor={
                tensor: tuple(consumers)
                for tensor, consumers in consumers_by_tensor.items()
            },
        )

    def violation(self, nodes: tuple[Node, ...]) -> str | None:
        node_ids = {id(node) for node in nodes}
        for node in nodes[1:]:
            for tensor in node.inputs:
                if tensor in self.runtime_inputs:
                    return (
                        f"the incoming communication edge: Runtime Input "
                        f"{tensor.name} reaches internal Layer {node.name}"
                    )
                producer = self.producer_by_tensor.get(tensor)
                if producer is not None and id(producer) not in node_ids:
                    return (
                        f"the incoming communication edge: cross-stage input "
                        f"{tensor.name} reaches internal Layer {node.name}"
                    )
        for node in nodes[:-1]:
            for tensor in node.outputs:
                if tensor in self.graph_outputs:
                    return (
                        f"the outgoing communication edge: graph output "
                        f"{tensor.name} leaves internal Layer {node.name}"
                    )
                if any(
                    id(consumer) not in node_ids
                    for consumer in self.consumers_by_tensor.get(tensor, ())
                ):
                    return (
                        f"the outgoing communication edge: cross-stage output "
                        f"{tensor.name} leaves internal Layer {node.name}"
                    )
        return None


def validate_stage_formation(
    graph: Graph,
    stage_formation: StageFormation,
) -> StageFormation:
    """Return a complete Stage formation whose boundaries and edges are valid."""

    graph_node_ids = {id(node) for node in graph.nodes}
    selected_node_ids: set[int] = set()
    resolved: StageFormation = {}
    for stage_id, stage_nodes in stage_formation.items():
        if not stage_nodes:
            raise ValueError(f"stage {stage_id} must contain at least one node")
        for node in stage_nodes:
            node_id = id(node)
            if node_id not in graph_node_ids:
                raise ValueError(
                    f"stage {stage_id} contains node {node.name} "
                    f"not present in graph {graph.name}"
                )
            if node_id in selected_node_ids:
                raise ValueError(
                    f"node {node.name} appears in more than one selected stage"
                )
            selected_node_ids.add(node_id)
        resolved[stage_id] = tuple(stage_nodes)

    if selected_node_ids != graph_node_ids:
        missing = tuple(
            node.name
            for node in graph.nodes
            if id(node) not in selected_node_ids
        )
        raise ValueError(f"selected stages do not cover all graph nodes, missing={missing}")

    _validate_explicit_group_ownership(resolved)
    communication_edges = StageCommunicationEdges.from_graph(graph)
    for stage_id, stage_nodes in resolved.items():
        violation = communication_edges.violation(stage_nodes)
        if violation is not None:
            raise ValueError(f"stage {stage_id} violates {violation}")
        incompatible_edge = incompatible_internal_edge(stage_nodes)
        if incompatible_edge is not None:
            producer, consumer = incompatible_edge
            raise ValueError(
                f"stage {stage_id} has incompatible internal dependency "
                f"{producer.name}->{consumer.name}: the consumer has no "
                "slice-containing layout relation"
            )
    return resolved


def _validate_explicit_group_ownership(stage_formation: StageFormation) -> None:
    stage_id_by_group: dict[object, int] = {}
    for stage_id, stage_nodes in stage_formation.items():
        for node in stage_nodes:
            group_key = explicit_stage_group_key(node)
            if group_key is None:
                continue
            if (
                group_key in stage_id_by_group
                and stage_id_by_group[group_key] != stage_id
            ):
                raise ValueError(
                    f"explicit stage group {group_key!r} is split across stages "
                    f"{stage_id_by_group[group_key]} and {stage_id}"
                )
            stage_id_by_group[group_key] = stage_id


def explicit_stage_group_key(node: Node) -> object | None:
    if STAGE_GROUP_ID_ATTR not in node.attributes:
        return None
    group_key = node.attributes[STAGE_GROUP_ID_ATTR]
    try:
        hash(group_key)
    except TypeError as exc:
        raise ValueError(
            f"node {node.name} has an unhashable "
            f"{STAGE_GROUP_ID_ATTR}: {group_key!r}"
        ) from exc
    return group_key


def incompatible_internal_edge(
    stage_nodes: tuple[Node, ...],
) -> tuple[Node, Node] | None:
    """Return the first local dependency lacking the formation contract."""

    producer_by_tensor = {
        tensor: node
        for node in stage_nodes
        for tensor in node.outputs
    }
    for consumer in stage_nodes:
        for input_index, tensor in enumerate(consumer.inputs):
            producer = producer_by_tensor.get(tensor)
            if producer is None:
                continue
            producer_group = explicit_stage_group_key(producer)
            if (
                producer_group is not None
                and producer_group == explicit_stage_group_key(consumer)
            ):
                continue
            if not _has_slice_containing_relation(consumer, input_index):
                return producer, consumer
    return None


def internal_edges_are_compatible(
    edges: tuple[tuple[Node, int, Node, int], ...]
    | list[tuple[Node, int, Node, int]],
) -> bool:
    """Return whether prospective local dependencies meet the Stage contract."""

    if not edges:
        return False
    for _, _, consumer, input_index in edges:
        if not _has_slice_containing_relation(consumer, input_index):
            return False
    return True


def _has_slice_containing_relation(consumer: Node, input_index: int) -> bool:
    relation = find_layout_relation(
        consumer.payload,
        input_index=input_index,
        output_index=0,
    )
    return relation is not None and relation.guarantees_slice_containment
