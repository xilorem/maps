"""Concrete planner-ready hardware targets."""

from .contracts import (
    PrecisionLoweringRecipe,
    RewriteEvent,
    RewriteReport,
    SpecializationOptions,
    SpecializationResult,
)
from . import magia, magia_v3, n300d

__all__ = [
    "PrecisionLoweringRecipe",
    "RewriteEvent",
    "RewriteReport",
    "SpecializationOptions",
    "SpecializationResult",
    "magia",
    "magia_v3",
    "n300d",
]
