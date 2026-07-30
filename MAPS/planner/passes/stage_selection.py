"""Deterministic graph-level stage coalescing."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.graph import Graph, Node
from MAPS.ops.common.layout_relation import find_layout_relation
from MAPS.planner.contracts.options import StageSelectionOptions
from MAPS.planner.contracts.stages import StageSelection

STAGE_GROUP_ID_ATTR = "stage_group_id"


@dataclass(frozen=True)
class _Unit:
    nodes: tuple[Node, ...]
    explicit: bool

    @property
    def size(self) -> int:
        return len(self.nodes)


def form_stages(
    graph: Graph,
    options: StageSelectionOptions | None = None,
) -> StageSelection:
    """Collapse explicit units, then greedily coalesce compatible linear chains."""

    options = options or StageSelectionOptions()
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
        return {
            stage_id: unit.nodes
            for stage_id, unit in enumerate(units)
        }

    unit_id_by_node = {
        id(node): unit_id
        for unit_id, unit in enumerate(units)
        for node in unit.nodes
    }
    producer_by_tensor = {
        tensor: node
        for node in graph.nodes
        for tensor in node.outputs
    }
    predecessors = {unit_id: set() for unit_id in range(len(units))}
    successors = {unit_id: set() for unit_id in range(len(units))}
    edges_by_units: dict[tuple[int, int], list[tuple[Node, int, Node, int]]] = {}
    for consumer in graph.nodes:
        consumer_unit = unit_id_by_node[id(consumer)]
        for input_index, tensor in enumerate(consumer.inputs):
            producer = producer_by_tensor.get(tensor)
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
            and not _has_internal_runtime_input(graph, current + next_unit.nodes)
            and not _has_internal_graph_output(graph, current + next_unit.nodes)
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
    return {stage_id: nodes for stage_id, nodes in enumerate(stages)}


def _has_internal_runtime_input(graph: Graph, nodes: tuple[Node, ...]) -> bool:
    runtime_inputs = frozenset(graph.inputs) - frozenset(graph.initializers)
    return any(
        tensor in runtime_inputs
        for node in nodes[1:]
        for tensor in node.inputs
    )


def _has_internal_graph_output(graph: Graph, nodes: tuple[Node, ...]) -> bool:
    graph_outputs = frozenset(graph.outputs)
    return any(
        tensor in graph_outputs
        for node in nodes[:-1]
        for tensor in node.outputs
    )


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
    producer_by_tensor = {
        tensor: node
        for node in graph.nodes
        for tensor in node.outputs
    }
    consumers_by_tensor: dict[object, list[Node]] = {}
    for node in graph.nodes:
        for tensor in node.inputs:
            consumers_by_tensor.setdefault(tensor, []).append(node)
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
        runtime_inputs = frozenset(graph.inputs) - frozenset(graph.initializers)
        node_ids = {id(node) for node in nodes}
        for node in nodes[1:]:
            for tensor in node.inputs:
                if tensor in runtime_inputs:
                    raise ValueError(
                        f"explicit stage group {key!r} violates the incoming "
                        f"communication edge: Runtime Input {tensor.name} reaches "
                        f"internal Layer {node.name}"
                    )
                producer = producer_by_tensor.get(tensor)
                if producer is not None and id(producer) not in node_ids:
                    raise ValueError(
                        f"explicit stage group {key!r} violates the incoming "
                        f"communication edge: cross-stage input {tensor.name} "
                        f"reaches internal Layer {node.name}"
                    )
        graph_outputs = frozenset(graph.outputs)
        for node in nodes[:-1]:
            for tensor in node.outputs:
                if tensor in graph_outputs:
                    raise ValueError(
                        f"explicit stage group {key!r} violates the outgoing "
                        f"communication edge: graph output {tensor.name} leaves "
                        f"internal Layer {node.name}"
                    )
                if any(
                    id(consumer) not in node_ids
                    for consumer in consumers_by_tensor.get(tensor, ())
                ):
                    raise ValueError(
                        f"explicit stage group {key!r} violates the outgoing "
                        f"communication edge: cross-stage output {tensor.name} "
                        f"leaves internal Layer {node.name}"
                    )

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
        units.append(_Unit(nodes, explicit=True))
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
