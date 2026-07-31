"""Hardware-independent Graph Rewrite contracts and canonical rewrites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .model import ImportedModel


@dataclass(frozen=True)
class GraphRewrite:
    """One named, hardware-independent Imported Model transformation."""

    name: str
    transform: Callable[[ImportedModel], ImportedModel]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Graph Rewrite name must not be empty")

    def apply(self, model: ImportedModel) -> ImportedModel:
        return self.transform(model)


def _decompose_operations(model: ImportedModel) -> ImportedModel:
    from .decompose import decompose_graph

    return ImportedModel(
        graph=decompose_graph(model.graph),
        constants=model.constants,
    )


CANONICAL_GRAPH_REWRITES = (
    GraphRewrite("operation_decomposition", _decompose_operations),
)


def run_graph_rewrites(model: ImportedModel) -> ImportedModel:
    """Apply hardware-independent Graph Rewrites in canonical order."""

    model.validate()
    rewritten = model
    for graph_rewrite in CANONICAL_GRAPH_REWRITES:
        rewritten = graph_rewrite.apply(rewritten)
    rewritten.validate()
    return rewritten


__all__ = ["CANONICAL_GRAPH_REWRITES", "GraphRewrite", "run_graph_rewrites"]
