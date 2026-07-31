"""Shared provenance effects emitted by individual Graph Rewrites."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import WorkSignature
from MAPS.importers.model import ImportedModel


@dataclass(frozen=True)
class RewriteEffect:
    """One source Node's provenance before a rewrite name is applied."""

    source_node: str
    original_signature: WorkSignature | None
    resulting_signatures: tuple[WorkSignature, ...]
    converted_initializers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteTransformResult:
    """One rewritten Imported Model and its unstamped provenance effects."""

    model: ImportedModel
    effects: tuple[RewriteEffect, ...] = ()


__all__ = ["RewriteEffect", "RewriteTransformResult"]
