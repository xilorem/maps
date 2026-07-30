"""Deterministic graph-level stage coalescing."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.graph import Graph, Node
from MAPS.core.tensor import Tensor
from MAPS.ops.common.layout_relation import find_layout_relation
from MAPS.planner.contracts.options import StageSelectionOptions
from MAPS.planner.contracts.stages import StageSelection

STAGE_GROUP_ID_ATTR = "stage_group_id"


@dataclass(frozen=True)
class _Unit:
    nodes: tuple[Node, ...]
    explicit: bool
    key: object | None = None

    @property
    def size(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class _CommunicationEdges:
    runtime_inputs: frozenset[Tensor]
    graph_outputs: frozenset[Tensor]
    producer_by_tensor: dict[Tensor, Node]
    consumers_by_tensor: dict[Tensor, tuple[Node, ...]]

    @classmethod
    def from_graph(cls, graph: Graph) -> "_CommunicationEdges":
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


def form_stages(
    graph: Graph,
    options: StageSelectionOptions | None = None,
) -> StageSelection:
    """Collapse explicit units, then greedily coalesce compatible linear chains."""

    options = options or StageSelectionOptions()
    communication_edges = _CommunicationEdges.from_graph(graph)
    units = _explicit_units(graph)
    if options.max_stage_nodes > 1:
        for unit in units:
            if unit.explicit and unit.size > options.max_stage_nodes:
                raise ValueError(
                    f"explicit stage group containing "
                    f"{tuple(node.name for node in unit.nodes)} has {unit.size} nodes, "
                    f"exceeding max_stage_nodes={options.max_stage_nodes}"
                )
    if options.max_stage_nodes == 1:
        stages = {
            stage_id: unit.nodes
            for stage_id, unit in enumerate(units)
        }
        _validate_explicit_stage_edges(stages, units, communication_edges)
        return stages

    unit_id_by_node = {
        id(node): unit_id
        for unit_id, unit in enumerate(units)
        for node in unit.nodes
    }
    predecessors = {unit_id: set() for unit_id in range(len(units))}
    successors = {unit_id: set() for unit_id in range(len(units))}
    edges_by_units: dict[tuple[int, int], list[tuple[Node, int, Node, int]]] = {}
    for consumer in graph.nodes:
        consumer_unit = unit_id_by_node[id(consumer)]
        for input_index, tensor in enumerate(consumer.inputs):
            producer = communication_edges.producer_by_tensor.get(tensor)
            if producer is None:
                continue
            producer_unit = unit_id_by_node[id(producer)]
            if producer_unit == consumer_unit:
                continue
            predecessors[consumer_unit].add(producer_unit)
            successors[producer_unit].add(consumer_unit)
            output_index = next(
                index
                for index, output in enumerate(producer.outputs)
                if output is tensor
            )
            edges_by_units.setdefault((producer_unit, consumer_unit), []).append(
                (producer, output_index, consumer, input_index)
            )

    stages: list[tuple[Node, ...]] = []
    current = units[0].nodes if units else ()
    current_last_unit = 0
    for next_unit_id in range(1, len(units)):
        next_unit = units[next_unit_id]
        can_merge = (
            successors[current_last_unit] == {next_unit_id}
            and predecessors[next_unit_id] == {current_last_unit}
            and _edges_are_compatible(
                edges_by_units.get((current_last_unit, next_unit_id), ())
            )
            and communication_edges.violation(current + next_unit.nodes) is None
            and (
                options.max_stage_nodes == 0
                or len(current) + next_unit.size <= options.max_stage_nodes
            )
        )
        if can_merge:
            current += next_unit.nodes
        else:
            stages.append(current)
            current = next_unit.nodes
        current_last_unit = next_unit_id
    if current:
        stages.append(current)
    formed_stages = {
        stage_id: nodes
        for stage_id, nodes in enumerate(stages)
    }
    _validate_explicit_stage_edges(formed_stages, units, communication_edges)
    return formed_stages


def _validate_explicit_stage_edges(
    stages: StageSelection,
    units: tuple[_Unit, ...],
    communication_edges: _CommunicationEdges,
) -> None:
    stage_by_node_id = {
        id(node): stage_nodes
        for stage_nodes in stages.values()
        for node in stage_nodes
    }
    for unit in units:
        if not unit.explicit:
            continue
        stage_nodes = stage_by_node_id[id(unit.nodes[0])]
        violation = communication_edges.violation(stage_nodes)
        if violation is not None:
            raise ValueError(f"explicit stage group {unit.key!r} violates {violation}")


def _edges_are_compatible(
    edges: tuple[tuple[Node, int, Node, int], ...]
    | list[tuple[Node, int, Node, int]],
) -> bool:
    if not edges:
        return False
    for _, _, consumer, input_index in edges:
        relation = find_layout_relation(
            consumer.payload,
            input_index=input_index,
            output_index=0,
        )
        if relation is None or not relation.guarantees_slice_containment:
            return False
    return True


def _explicit_units(graph: Graph) -> tuple[_Unit, ...]:
    keys = tuple(_explicit_stage_group_key(node) for node in graph.nodes)
    positions_by_key: dict[object, list[int]] = {}
    for position, key in enumerate(keys):
        if key is not None:
            positions_by_key.setdefault(key, []).append(position)
    for key, positions in positions_by_key.items():
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(
                f"explicit stage group {key!r} is not contiguous in topological order"
            )
        nodes = tuple(graph.nodes[position] for position in positions)
        if len(nodes) > 1 and not _dependency_connected(nodes):
            raise ValueError(f"explicit stage group {key!r} is not dependency-connected")

    units: list[_Unit] = []
    position = 0
    while position < len(graph.nodes):
        key = keys[position]
        if key is None:
            units.append(_Unit((graph.nodes[position],), explicit=False))
            position += 1
            continue
        positions = positions_by_key[key]
        nodes = tuple(graph.nodes[index] for index in positions)
        units.append(_Unit(nodes, explicit=True, key=key))
        position = positions[-1] + 1
    return tuple(units)


def _dependency_connected(nodes: tuple[Node, ...]) -> bool:
    node_ids = {id(node) for node in nodes}
    producer_by_tensor = {
        tensor: node
        for node in nodes
        for tensor in node.outputs
    }
    neighbors = {id(node): set() for node in nodes}
    for node in nodes:
        for tensor in node.inputs:
            producer = producer_by_tensor.get(tensor)
            if producer is not None:
                neighbors[id(node)].add(id(producer))
                neighbors[id(producer)].add(id(node))
    visited = set()
    pending = [id(nodes[0])]
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(neighbors[node_id] - visited)
    return visited == node_ids


def _explicit_stage_group_key(node: Node) -> object | None:
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


__all__ = ["form_stages"]
