"""Helpers for constructing graph connectivity after lowering/transforms."""

from __future__ import annotations

from MAPS.core.graph import Edge, Node
from MAPS.core.tensor import Tensor


def add_generated_tensor(tensor: Tensor, tensors: dict[str, Tensor]) -> None:
    """Register generated Tensor metadata with deterministic collision failure."""

    if tensor.name in tensors:
        raise ValueError(f"generated tensor name collision: '{tensor.name}'")
    tensors[tensor.name] = tensor


def reserve_generated_node_name(name: str, node_names: set[str]) -> None:
    """Reserve a generated Node name with deterministic collision failure."""

    if name in node_names:
        raise ValueError(f"generated node name collision: '{name}'")
    node_names.add(name)


def build_graph_edges_from_nodes(
    nodes: tuple[Node, ...],
    tensors: dict[str, Tensor],
    graph_output_names: tuple[str, ...],
) -> tuple[Edge, ...]:
    """Build explicit graph edges from an already-lowered node sequence."""

    producers: dict[str, Node] = {}
    for node in nodes:
        for tensor in node.outputs:
            if tensor.name in producers:
                raise ValueError(f"tensor '{tensor.name}' has multiple producers")
            producers[tensor.name] = node

    edges: list[Edge] = []

    for node in nodes:
        for tensor in node.inputs:
            edges.append(
                Edge(
                    tensor=tensors[tensor.name],
                    src=producers.get(tensor.name),
                    dst=node,
                )
            )

    for tensor_name in graph_output_names:
        src_node = producers.get(tensor_name)
        if src_node is None:
            continue
        edges.append(Edge(tensor=tensors[tensor_name], src=src_node, dst=None))

    return tuple(edges)
