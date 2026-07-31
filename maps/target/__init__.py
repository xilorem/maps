"""Concrete planner-ready hardware targets."""

from .contracts import (
    PrecisionLoweringRecipe,
    RewriteEvent,
    RewriteReport,
    SpecializationOptions,
    SpecializationResult,
)
from . import magia, n300d

__all__ = [
    "PrecisionLoweringRecipe",
    "RewriteEvent",
    "RewriteReport",
    "SpecializationOptions",
    "SpecializationResult",
    "magia",
    "n300d",
]
