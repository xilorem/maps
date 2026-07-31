"""Hardware-independent Graph Rewrite contracts and canonical rewrites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .graph import Node
from .model import ImportedModel


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
    from .decompose import decompose_graph_with_sources

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


__all__ = [
    "CANONICAL_GRAPH_REWRITES",
    "GraphRewrite",
    "GraphRewriteEffect",
    "GraphRewriteResult",
    "run_graph_rewrites",
    "run_graph_rewrites_with_effects",
]
