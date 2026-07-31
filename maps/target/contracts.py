"""Uniform contracts shared by concrete target packages."""

from __future__ import annotations

from dataclasses import dataclass

from maps.graph import ImportedModel
from maps.hardware import WorkSignature


@dataclass(frozen=True)
class SpecializationOptions:
    """Optional target specialization behavior requested by the caller."""

    enable_precision_lowering: bool = False


@dataclass(frozen=True)
class RewriteEvent:
    """One source Node changed during Target Specialization."""

    rewrite_name: str
    source_node: str
    original_signature: WorkSignature | None
    resulting_signatures: tuple[WorkSignature, ...]
    converted_initializers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteReport:
    """Deterministic provenance for Target Specialization."""

    events: tuple[RewriteEvent, ...] = ()


@dataclass(frozen=True)
class SpecializationResult:
    """A target-suited Imported Model and its provenance."""

    model: ImportedModel
    report: RewriteReport = RewriteReport()


__all__ = [
    "RewriteEvent",
    "RewriteReport",
    "SpecializationOptions",
    "SpecializationResult",
]
