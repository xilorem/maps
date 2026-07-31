"""Composite-op decomposition pass."""

from __future__ import annotations

from maps.graph.graph import Graph, Node
from .graph_utils import build_graph_edges_from_nodes


def decompose_graph(graph: Graph) -> Graph:
    """Replace composite nodes with the primitive nodes produced by their op specs."""

    decomposed, _ = decompose_graph_with_sources(graph)
    return decomposed


def decompose_graph_with_sources(
    graph: Graph,
) -> tuple[Graph, tuple[tuple[Node, tuple[Node, ...]], ...]]:
    """Decompose a Graph and retain deterministic source-to-result provenance."""

    tensors = {tensor.name: tensor for tensor in graph.tensors}
    nodes = []
    decompositions = []
    retained_node_names = {
        node.name
        for node in graph.nodes
        if not callable(getattr(node.payload, "decompose", None))
    }
    generated_node_names: set[str] = set()

    for node in graph.nodes:
        if not callable(getattr(node.payload, "decompose", None)):
            nodes.append(node)
            continue

        new_tensors, lowered_nodes = node.payload.decompose(node)
        for tensor in new_tensors:
            if tensor.name in tensors:
                raise ValueError(f"tensor '{tensor.name}' is already present in graph metadata")
            tensors[tensor.name] = tensor
        for lowered_node in lowered_nodes:
            if (
                lowered_node.name in retained_node_names
                or lowered_node.name in generated_node_names
            ):
                raise ValueError(
                    f"generated node name collision: '{lowered_node.name}'"
                )
            generated_node_names.add(lowered_node.name)
        nodes.extend(lowered_nodes)
        decompositions.append((node, lowered_nodes))

    lowered_nodes = tuple(nodes)
    graph_output_names = tuple(tensor.name for tensor in graph.outputs)

    return Graph(
        name=graph.name,
        tensors=tuple(tensors.values()),
        nodes=lowered_nodes,
        edges=build_graph_edges_from_nodes(lowered_nodes, tensors, graph_output_names),
        inputs=graph.inputs,
        outputs=graph.outputs,
        initializers=graph.initializers,
    ), tuple(decompositions)
