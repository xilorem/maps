"""Temporary target-specialization contracts pending Target migration."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable

from maps.hardware import WorkSignature


class GraphRewriteKind(Enum):
    """Target-requested, type-preserving Graph Rewrites."""

    CONV_TO_GEMM = auto()


@dataclass(frozen=True)
class PrecisionLoweringRecipe:
    """One target-approved typed operation precision conversion."""

    source_signature: WorkSignature
    target_signature: WorkSignature
    device_name: str

    def __post_init__(self) -> None:
        if not self.device_name:
            raise ValueError("precision lowering device name must not be empty")


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
