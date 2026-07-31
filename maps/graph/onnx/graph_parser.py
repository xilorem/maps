"""Graph-level ONNX parsing orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from maps.graph.graph import Graph
from maps.graph.graph_utils import build_graph_edges_from_nodes

from .node_parser import parse_node
from .tensor_parser import collect_scheduler_tensors

if TYPE_CHECKING:
    from onnx import GraphProto


def _static_integer_inputs(graph: "GraphProto") -> dict[str, tuple[int, ...]]:
    """Decode small integer initializers used as compile-time op configuration."""

    from onnx import TensorProto, numpy_helper

    values = {}
    for initializer in graph.initializer:
        if initializer.data_type not in (TensorProto.INT32, TensorProto.INT64):
            continue
        array = numpy_helper.to_array(initializer)
        values[initializer.name] = tuple(int(value) for value in array.flat)
    return values


def parse_graph(graph: "GraphProto", *, graph_name: str | None = None) -> Graph:
    """Parse one ONNX graph into the shared scheduler graph IR."""

    tensors = collect_scheduler_tensors(graph)
    static_input_values = _static_integer_inputs(graph)
    nodes = []
    for node_idx, node in enumerate(graph.node):
        nodes.append(
            parse_node(
                node,
                node_idx,
                tensors,
                static_input_values=static_input_values,
            )
        )

    initializer_names = {initializer.name for initializer in graph.initializer}
    graph_input_names = {value.name for value in graph.input if value.name not in initializer_names}
    graph_output_names = tuple(value.name for value in graph.output)
    lowered_nodes = tuple(nodes)
    live_tensor_names = {
        tensor.name
        for node in lowered_nodes
        for tensor in node.inputs + node.outputs
    } | {
        value.name for value in graph.input
        if value.name not in initializer_names
    } | set(graph_output_names)

    return Graph(
        name=graph_name or graph.name,
        tensors=tuple(
            tensor for tensor in tensors.values()
            if tensor.name in live_tensor_names
        ),
        nodes=lowered_nodes,
        edges=build_graph_edges_from_nodes(lowered_nodes, tensors, graph_output_names),
        inputs=tuple(tensors[value.name] for value in graph.input if value.name in graph_input_names),
        outputs=tuple(tensors[value.name] for value in graph.output),
        initializers=tuple(
            tensors[initializer.name]
            for initializer in graph.initializer
            if (
                initializer.name in tensors
                and initializer.name in live_tensor_names
            )
        ),
    )
