"""Concrete planner-ready hardware targets."""

from . import magia, n300d
from .contracts import (
    RewriteEvent,
    RewriteReport,
    SpecializationOptions,
    SpecializationResult,
)

__all__ = [
    "RewriteEvent",
    "RewriteReport",
    "SpecializationOptions",
    "SpecializationResult",
    "magia",
    "n300d",
]
