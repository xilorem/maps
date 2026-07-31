"""Canonical Imported Model Graph Rewrite Phase and provenance contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from MAPS.arch import WorkSignature
from MAPS.core.constants import validate_constants
from MAPS.importers.model import ImportedModel

from .decompose import decompose_graph_with_sources


@dataclass(frozen=True)
class RewriteEvent:
    """One source Node affected by a named Graph Rewrite."""

    rewrite_name: str
    source_node: str
    original_signature: WorkSignature | None
    resulting_signatures: tuple[WorkSignature, ...]
    converted_initializers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteReport:
    """Deterministic provenance emitted by the Graph Rewrite Phase."""

    events: tuple[RewriteEvent, ...] = ()


@dataclass(frozen=True)
class GraphRewriteResult:
    model: ImportedModel
    events: tuple[RewriteEvent, ...] = ()


@dataclass(frozen=True)
class GraphRewrite:
    """One named Imported Model transformation in the canonical sequence."""

    name: str
    transform: Callable[[ImportedModel], GraphRewriteResult]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Graph Rewrite name must not be empty")

    def apply(self, model: ImportedModel) -> GraphRewriteResult:
        return self.transform(model)


def _signature_or_none(node) -> WorkSignature | None:
    try:
        return WorkSignature.from_node(node)
    except ValueError:
        return None


def _decompose_operations(model: ImportedModel) -> GraphRewriteResult:
    graph, decompositions = decompose_graph_with_sources(model.graph)
    events = tuple(
        RewriteEvent(
            rewrite_name="operation_decomposition",
            source_node=source.name,
            original_signature=_signature_or_none(source),
            resulting_signatures=tuple(
                WorkSignature.from_node(node) for node in resulting_nodes
            ),
        )
        for source, resulting_nodes in decompositions
    )
    return GraphRewriteResult(
        model=ImportedModel(graph=graph, constants=model.constants),
        events=events,
    )


CANONICAL_GRAPH_REWRITES = (
    GraphRewrite("operation_decomposition", _decompose_operations),
)


def run_graph_rewrites(
    model: ImportedModel,
) -> tuple[ImportedModel, RewriteReport]:
    """Apply MAPS-owned rewrites in their one canonical order."""

    validate_constants(model.graph, model.constants)
    rewritten = model
    events = []
    for graph_rewrite in CANONICAL_GRAPH_REWRITES:
        result = graph_rewrite.apply(rewritten)
        rewritten = result.model
        events.extend(result.events)

    require_complete_work_signatures(rewritten)
    validate_constants(rewritten.graph, rewritten.constants)
    return rewritten, RewriteReport(tuple(events))


def require_complete_work_signatures(model: ImportedModel) -> None:
    """Reject incomplete typed work before Stage selection begins."""

    for node in model.graph.nodes:
        WorkSignature.from_node(node)


__all__ = [
    "CANONICAL_GRAPH_REWRITES",
    "GraphRewrite",
    "GraphRewriteResult",
    "RewriteEvent",
    "RewriteReport",
    "require_complete_work_signatures",
    "run_graph_rewrites",
]
