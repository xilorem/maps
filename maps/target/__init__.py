"""Concrete planner-ready hardware targets."""

from . import magia, n300d
from .contracts import (
    PrecisionLoweringRecipe,
    RewriteEvent,
    RewriteReport,
    SpecializationOptions,
    SpecializationResult,
)

__all__ = [
    "PrecisionLoweringRecipe",
    "RewriteEvent",
    "RewriteReport",
    "SpecializationOptions",
    "SpecializationResult",
    "magia",
    "n300d",
]
