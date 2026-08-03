"""Stage formation, Allocation, and Placement contracts owned by Planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from maps.graph import Graph, Node, Tensor
from maps.hardware import Tile
from maps.operations.contracts import find_layout_relation
from maps.planning.mapping import Submesh, TensorSlice

if TYPE_CHECKING:
    from .options import StageFormationOptions


StageFormation = dict[int, tuple[Node, ...]]


@dataclass(frozen=True)
class StagePlan:
    """The Allocation result for one formed Stage.

    ``tile_count`` and ``logical_shape`` describe virtual execution. ``nodes``
    and ``node_output_layouts`` have matching order and preserve the complete
    formed Stage. Physical placement is deliberately absent and is represented
    by the separate ``StagePlacement`` contract produced by Placement.
    """

    stage_id: int
    tile_count: int
    logical_shape: tuple[int, int]
    nodes: tuple[Node, ...]
    node_output_layouts: tuple[tuple, ...]
    device_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.device_names) != len(self.nodes):
            raise ValueError("Stage Plan must retain one Device name per Layer")
        if any(not device_name for device_name in self.device_names):
            raise ValueError("Stage Plan Device names must not be empty")


@dataclass(frozen=True)
class StagePlacement:
    """Bind one virtual stage layout to a connected physical tile region.

    ``virtual_to_physical`` must be a bijection covering both submeshes.  The
    physical region may be non-rectangular; connectivity is established by the
    Placement implementation before this contract is constructed.
    """

    stage_id: int
    virtual_submesh: Submesh
    physical_submesh: Submesh
    virtual_to_physical: dict[int, int]

    def __post_init__(self) -> None:
        """Validate complete bijective coverage of virtual and physical tiles."""

        virtual_tile_ids = {tile.tile_id for tile in self.virtual_submesh.tiles}
        physical_tile_ids = set(self.physical_submesh.tile_ids)
        if set(self.virtual_to_physical) != virtual_tile_ids:
            raise ValueError(
                f"placement for stage {self.stage_id} does not cover all virtual tiles"
            )
        if set(self.virtual_to_physical.values()) != physical_tile_ids:
            raise ValueError(
                f"placement for stage {self.stage_id} does not cover all physical tiles"
            )

    def physical_tile_id(self, virtual_tile_id: int) -> int:
        """Return the physical tile assigned to one virtual tile."""

        return self.virtual_to_physical[virtual_tile_id]


def virtual_submesh(plan: StagePlan):
    """Return the virtual submesh shared by a stage's chosen layouts."""

    for layouts in plan.node_output_layouts:
        if layouts:
            return layouts[0].submesh
    raise ValueError(f"stage {plan.stage_id} has no virtual layouts")
@dataclass(frozen=True)
class StageDependencies:
    """Graph producer facts used during Stage formation."""

    producer_by_tensor: dict[Tensor, Node]

    @classmethod
    def from_graph(cls, graph: Graph) -> "StageDependencies":
        return cls(
            producer_by_tensor={
                tensor: node
                for node in graph.nodes
                for tensor in node.outputs
            },
        )

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

    _validate_source_operation_ownership(resolved)
    for stage_id, stage_nodes in resolved.items():
        incompatible_edge = incompatible_internal_edge(stage_nodes)
        if incompatible_edge is not None:
            producer, consumer = incompatible_edge
            raise ValueError(
                f"stage {stage_id} has incompatible internal dependency "
                f"{producer.name}->{consumer.name}: the consumer has no "
                "slice-containing layout relation"
            )
    return resolved


def _validate_source_operation_ownership(stage_formation: StageFormation) -> None:
    stage_id_by_source_operation: dict[object, int] = {}
    for stage_id, stage_nodes in stage_formation.items():
        for node in stage_nodes:
            source_operation = node.source_operation
            if (
                source_operation in stage_id_by_source_operation
                and stage_id_by_source_operation[source_operation] != stage_id
            ):
                raise ValueError(
                    f"source operation {source_operation!r} is split across stages "
                    f"{stage_id_by_source_operation[source_operation]} and {stage_id}"
                )
            stage_id_by_source_operation[source_operation] = stage_id


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
            if producer.source_operation == consumer.source_operation:
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


@dataclass(frozen=True)
class _FormationUnit:
    nodes: tuple[Node, ...]


def form_stages(
    graph: Graph,
    options: StageFormationOptions | None = None,
) -> StageFormation:
    """Keep source Operations indivisible, then coalesce compatible chains."""

    from .options import StageFormationOptions

    options = options or StageFormationOptions()
    dependencies = StageDependencies.from_graph(graph)
    units = _source_operation_units(graph)
    if options.max_stage_operations == 1:
        operation_stages = {
            stage_id: unit.nodes
            for stage_id, unit in enumerate(units)
        }
        return operation_stages

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
            producer = dependencies.producer_by_tensor.get(tensor)
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
    current_operation_count = 1 if units else 0
    for next_unit_id in range(1, len(units)):
        next_unit = units[next_unit_id]
        can_merge = (
            successors[current_last_unit] == {next_unit_id}
            and predecessors[next_unit_id] == {current_last_unit}
            and internal_edges_are_compatible(
                edges_by_units.get((current_last_unit, next_unit_id), ())
            )
            and (
                options.max_stage_operations == 0
                or current_operation_count + 1 <= options.max_stage_operations
            )
        )
        if can_merge:
            current += next_unit.nodes
            current_operation_count += 1
        else:
            stages.append(current)
            current = next_unit.nodes
            current_operation_count = 1
        current_last_unit = next_unit_id
    if current:
        stages.append(current)
    return {
        stage_id: nodes
        for stage_id, nodes in enumerate(stages)
    }


def _source_operation_units(graph: Graph) -> tuple[_FormationUnit, ...]:
    keys = tuple(node.source_operation for node in graph.nodes)
    positions_by_key: dict[object, list[int]] = {}
    for position, key in enumerate(keys):
        positions_by_key.setdefault(key, []).append(position)
    for key, positions in positions_by_key.items():
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(
                f"source operation {key!r} is not contiguous in topological order"
            )
        nodes = tuple(graph.nodes[position] for position in positions)
        if len(nodes) > 1 and not _dependency_connected(nodes):
            raise ValueError(f"source operation {key!r} is not dependency-connected")

    units: list[_FormationUnit] = []
    position = 0
    while position < len(graph.nodes):
        key = keys[position]
        positions = positions_by_key[key]
        nodes = tuple(graph.nodes[index] for index in positions)
        units.append(_FormationUnit(nodes))
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


def node_output_layouts(plan: StagePlan, node: Node) -> tuple:
    """Return the output layouts chosen for ``node`` within ``plan``."""

    return plan.node_output_layouts[plan_node_index(plan, node)]


def plan_node_index(plan: StagePlan, node: Node) -> int:
    """Return the identity-based position of ``node`` within a stage plan."""

    for node_idx, candidate in enumerate(plan.nodes):
        if candidate is node:
            return node_idx
    raise ValueError(f"node {node.name} is not present in stage plan {plan.stage_id}")


def node_output_index(node: Node, tensor: object) -> int:
    """Return the output position at which ``node`` produces ``tensor``."""

    for output_idx, candidate in enumerate(node.outputs):
        if candidate == tensor:
            return output_idx
    raise ValueError(
        f"tensor {getattr(tensor, 'name', tensor)} is not an output of node {node.name}"
    )


def required_input_slices(
    tensor: object,
    destination_node: Node,
    destination_output_layouts: tuple,
) -> tuple[tuple[Tile, TensorSlice], ...]:
    """Return the slice of ``tensor`` required by every destination tile."""

    required_slices = []
    submesh = destination_output_layouts[0].submesh
    for tile in submesh.tiles:
        tile_work = destination_node.payload.build_tile_work(
            output_layouts=destination_output_layouts,
            tile=tile,
        )
        for reference in tile_work.input_slices:
            if reference.tensor is tensor:
                required_slices.append((tile, reference.tensor_slice))
                break
    return tuple(required_slices)


__all__ = [
    "StageFormation",
    "StagePlacement",
    "StagePlan",
    "form_stages",
    "node_output_index",
    "node_output_layouts",
    "plan_node_index",
    "required_input_slices",
    "validate_stage_formation",
    "virtual_submesh",
]
