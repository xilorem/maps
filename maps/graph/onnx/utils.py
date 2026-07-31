"""Shared ONNX importer utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from maps.graph.graph import Edge, Node
from maps.graph.tensor import Tensor
from maps.graph.graph_utils import build_graph_edges_from_nodes

from .node_parser import node_inputs, node_name, node_outputs

if TYPE_CHECKING:
    from onnx import GraphProto


def build_tensor_producer_table(graph: "GraphProto") -> dict[str, tuple[str, int]]:
    """Map each produced tensor name to its producing ONNX node and output index."""

    producers: dict[str, tuple[str, int]] = {}
    for node_idx, node in enumerate(graph.node):
        current_name = node_name(node, node_idx)
        for output_idx, tensor_name in enumerate(node_outputs(node)):
            if tensor_name in producers:
                raise ValueError(f"tensor '{tensor_name}' has multiple producers")
            producers[tensor_name] = (current_name, output_idx)
    return producers


def build_graph_edges(
    graph: "GraphProto",
    nodes: tuple[Node, ...],
    tensors: dict[str, Tensor],
) -> tuple[Edge, ...]:
    """Build explicit graph edges from ONNX connectivity."""

    node_by_name = {node.name: node for node in nodes}
    producers = build_tensor_producer_table(graph)
    graph_output_names = {value.name for value in graph.output}

    edges: list[Edge] = []

    for dst_idx, onnx_node in enumerate(graph.node):
        dst_name = node_name(onnx_node, dst_idx)
        dst_node = node_by_name[dst_name]
        for tensor_name in node_inputs(onnx_node):
            tensor = tensors[tensor_name]
            src_node = None
            if tensor_name in producers:
                src_node = node_by_name[producers[tensor_name][0]]
            edges.append(Edge(tensor=tensor, src=src_node, dst=dst_node))

    for tensor_name in graph_output_names:
        tensor = tensors[tensor_name]
        src_node = None
        if tensor_name in producers:
            src_node = node_by_name[producers[tensor_name][0]]
        if src_node is None:
            continue
        edges.append(Edge(tensor=tensor, src=src_node, dst=None))

    return tuple(edges)


def build_lowered_graph_edges(
    nodes: tuple[Node, ...],
    tensors: dict[str, Tensor],
    graph_output_names: tuple[str, ...],
) -> tuple[Edge, ...]:
    """Build explicit graph edges from an already-lowered node sequence."""
    return build_graph_edges_from_nodes(nodes, tensors, graph_output_names)
