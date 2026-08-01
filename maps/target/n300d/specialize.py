"""N300D Target Specialization."""

from __future__ import annotations

from maps.graph import ImportedModel
from maps.hardware import Mesh
from maps.target.contracts import (
    RewriteReport,
    SpecializationOptions,
    SpecializationResult,
)


def specialize(
    model: ImportedModel,
    mesh: Mesh,
    options: SpecializationOptions | None = None,
) -> SpecializationResult:
    """Return an equivalent model because N300D needs no target rewrites."""

    model.validate()
    return SpecializationResult(model, RewriteReport())


__all__ = ["specialize"]
