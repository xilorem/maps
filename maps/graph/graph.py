"""Shared graph IR for importer output and scheduler-side graph reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from .tensor import Tensor

if TYPE_CHECKING:
    from MAPS.ops.common import OperationPayload


class OpKind(IntEnum):
    GEMM = 0
    ELEMENTWISE = 1
    REDUCTION = 2
    CONV = 3
    TRANSFORM = 4
    CUSTOM = 255


@dataclass(frozen=True)
class Node:
    """One logical compute node in the graph."""

    name: str
    kind: OpKind
    inputs: tuple[Tensor, ...] = field(default_factory=tuple)
    outputs: tuple[Tensor, ...] = field(default_factory=tuple)
    payload: "OperationPayload | None" = None
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("node name must not be empty")


@dataclass(frozen=True)
class Edge:
    """One tensor dependency between two graph nodes."""

    tensor: Tensor
    src: Node | None
    dst: Node | None

    def __post_init__(self) -> None:
        if self.src is None and self.dst is None:
            raise ValueError("edge must connect at least one endpoint")


@dataclass(frozen=True)
class Graph:
    """One logical graph ready to be consumed by the scheduler."""

    name: str
    tensors: tuple[Tensor, ...] = field(default_factory=tuple)
    nodes: tuple[Node, ...] = field(default_factory=tuple)
    edges: tuple[Edge, ...] = field(default_factory=tuple)
    inputs: tuple[Tensor, ...] = field(default_factory=tuple)
    outputs: tuple[Tensor, ...] = field(default_factory=tuple)
    initializers: tuple[Tensor, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("graph name must not be empty")
