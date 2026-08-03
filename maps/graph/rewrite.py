"""Hardware-independent Graph Rewrite contracts and canonical rewrites."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .model import (
    Graph,
    ImportedModel,
    Node,
    Tensor,
    build_graph_edges_from_nodes,
)


@dataclass(frozen=True)
class GraphRewriteEffect:
    """One source Node replaced by a Graph Rewrite."""

    rewrite_name: str
    source_node: Node
    resulting_nodes: tuple[Node, ...]


@dataclass(frozen=True)
class GraphRewriteResult:
    """An Imported Model and the observable replacements that produced it."""

    model: ImportedModel
    effects: tuple[GraphRewriteEffect, ...] = ()


@dataclass(frozen=True)
class GraphRewrite:
    """One named, hardware-independent Imported Model transformation."""

    name: str
    transform: Callable[[ImportedModel], GraphRewriteResult]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Graph Rewrite name must not be empty")

    def apply(self, model: ImportedModel) -> ImportedModel:
        return self.transform(model).model

    def apply_with_effects(self, model: ImportedModel) -> GraphRewriteResult:
        """Apply this rewrite and retain its source-to-result replacements."""

        return self.transform(model)


def _decompose_operations(model: ImportedModel) -> GraphRewriteResult:
    graph, decompositions = decompose_graph_with_sources(model.graph)
    return GraphRewriteResult(
        model=ImportedModel(graph=graph, constants=model.constants),
        effects=tuple(
            GraphRewriteEffect(
                rewrite_name="operation_decomposition",
                source_node=source,
                resulting_nodes=resulting,
            )
            for source, resulting in decompositions
        ),
    )


CANONICAL_GRAPH_REWRITES = (
    GraphRewrite("operation_decomposition", _decompose_operations),
)


def run_graph_rewrites(model: ImportedModel) -> ImportedModel:
    """Apply hardware-independent Graph Rewrites in canonical order."""

    rewritten, _ = run_graph_rewrites_with_effects(model)
    return rewritten


def run_graph_rewrites_with_effects(
    model: ImportedModel,
) -> tuple[ImportedModel, tuple[GraphRewriteEffect, ...]]:
    """Apply canonical Graph Rewrites and retain deterministic replacements."""

    model.validate()
    rewritten = model
    effects: list[GraphRewriteEffect] = []
    for graph_rewrite in CANONICAL_GRAPH_REWRITES:
        result = graph_rewrite.apply_with_effects(rewritten)
        rewritten = result.model
        effects.extend(result.effects)
    rewritten.validate()
    return rewritten, tuple(effects)


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
        if not lowered_nodes:
            raise ValueError(
                f"operation decomposition for '{node.name}' produced no Layers"
            )
        lowered_nodes = tuple(
            replace(
                lowered_node,
                source_operation=node.source_operation,
            )
            for lowered_node in lowered_nodes
        )
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


__all__ = [
    "CANONICAL_GRAPH_REWRITES",
    "GraphRewrite",
    "GraphRewriteEffect",
    "GraphRewriteResult",
    "decompose_graph",
    "run_graph_rewrites",
    "run_graph_rewrites_with_effects",
]
