"""Immutable virtual and physical communication contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from maps.graph import Tensor
from maps.planning.mapping import TensorSlice, TensorSubSlice


@dataclass(frozen=True)
class VirtualInputDestination:
    virtual_tile_id: int
    tensor_slice: TensorSlice


@dataclass(frozen=True)
class InputDestination:
    tile_id: int
    tensor_slice: TensorSlice


@dataclass(frozen=True)
class VirtualTransfer:
    source_virtual_tile_id: int
    destination_virtual_tile_id: int
    source_subslice: TensorSubSlice
    destination_subslice: TensorSubSlice


@dataclass(frozen=True)
class Transfer:
    source_tile_id: int
    destination_tile_id: int
    source_subslice: TensorSubSlice
    destination_subslice: TensorSubSlice


@dataclass(frozen=True)
class VirtualOutputSource:
    virtual_tile_id: int
    tensor_slice: TensorSlice


@dataclass(frozen=True)
class OutputSource:
    tile_id: int
    tensor_slice: TensorSlice


@dataclass(frozen=True)
class VirtualInputTransition:
    tensor: Tensor
    tensor_id: int
    destination_stage_id: int
    destinations: tuple[VirtualInputDestination, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class InputTransition:
    tensor_id: int
    destination_stage_id: int
    destinations: tuple[InputDestination, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VirtualIntermediateTransition:
    tensor: Tensor
    tensor_id: int
    source_stage_id: int
    destination_stage_id: int
    transfers: tuple[VirtualTransfer, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IntermediateTransition:
    tensor_id: int
    source_stage_id: int
    destination_stage_id: int
    transfers: tuple[Transfer, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VirtualOutputTransition:
    tensor: Tensor
    tensor_id: int
    source_stage_id: int
    sources: tuple[VirtualOutputSource, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OutputTransition:
    tensor_id: int
    source_stage_id: int
    sources: tuple[OutputSource, ...] = field(default_factory=tuple)


VirtualTransition: TypeAlias = (
    VirtualInputTransition
    | VirtualIntermediateTransition
    | VirtualOutputTransition
)
Transition: TypeAlias = InputTransition | IntermediateTransition | OutputTransition
