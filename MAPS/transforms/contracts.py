"""Temporary target-specialization contracts pending Target migration."""

from enum import Enum, auto
from typing import Protocol, runtime_checkable

from maps.target.contracts import PrecisionLoweringRecipe


class GraphRewriteKind(Enum):
    """Target-requested, type-preserving Graph Rewrites."""

    CONV_TO_GEMM = auto()


@runtime_checkable
class TargetSpecializationPolicy(Protocol):
    """Target-owned Graph specialization required by legacy planning."""

    @property
    def precision_lowering_recipes(self) -> tuple[PrecisionLoweringRecipe, ...]: ...

    @property
    def required_graph_rewrites(self) -> tuple[GraphRewriteKind, ...]: ...


__all__ = [
    "GraphRewriteKind",
    "PrecisionLoweringRecipe",
    "TargetSpecializationPolicy",
]
