"""Physical Execution Plan models owned by Planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from maps.planning.layouts import TensorLayout
from maps.planning.submesh import Submesh
from maps.planning.transitions.contracts import InputDestination, Transition

if TYPE_CHECKING:
    from maps.graph import Node, Tensor
    from maps.hardware import Mesh


@dataclass(frozen=True)
class ExecutionContract:
    """Execution settings that affect planning and backend allocation."""

    num_token_slots: int = 2

    def __post_init__(self) -> None:
        if self.num_token_slots <= 0:
            raise ValueError("num_token_slots must be > 0")


@dataclass(frozen=True)
class InitializerInput:
    """Per-tile residency for an immutable input Tensor."""

    destinations: tuple[InputDestination, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TransitionSource:
    """Layer input supplied by an Input or Intermediate Transition."""

    transition_id: int

    def __post_init__(self) -> None:
        if self.transition_id < 0:
            raise ValueError("transition sources require transition_id >= 0")


@dataclass(frozen=True)
class LocalInput:
    """Layer input read from a previous Layer output in the same Stage."""

    layer_idx: int
    tensor_id: int

    def __post_init__(self) -> None:
        if self.layer_idx < 0 or self.tensor_id < 0:
            raise ValueError("layer_idx and tensor_id must be >= 0")


LayerInputSource = InitializerInput | TransitionSource | LocalInput


@dataclass(frozen=True)
class LayerInput:
    """One input of a Layer."""

    tensor_id: int
    source: LayerInputSource

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")

    @classmethod
    def initializer(
        cls,
        tensor_id: int,
        destinations: tuple[InputDestination, ...],
    ) -> "LayerInput":
        return cls(
            tensor_id=tensor_id,
            source=InitializerInput(destinations=destinations),
        )

    @classmethod
    def transition_source(
        cls,
        tensor_id: int,
        transition_id: int,
    ) -> "LayerInput":
        return cls(
            tensor_id=tensor_id,
            source=TransitionSource(transition_id=transition_id),
        )

    @classmethod
    def local(cls, tensor_id: int, layer_idx: int) -> "LayerInput":
        return cls(
            tensor_id=tensor_id,
            source=LocalInput(layer_idx=layer_idx, tensor_id=tensor_id),
        )


@dataclass(frozen=True)
class LayerOutput:
    """One output of a Layer."""

    tensor_id: int
    layout: TensorLayout

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")


@dataclass(frozen=True)
class Layer:
    """One scheduled Graph Node inside a Stage."""

    node: Node
    inputs: tuple[LayerInput, ...] = field(default_factory=tuple)
    outputs: tuple[LayerOutput, ...] = field(default_factory=tuple)
    device_name: str | None = None

    def validate_tensors(self, tensors: tuple[Tensor, ...]) -> None:
        """Validate bound Tensor ids and output layout compatibility."""

        for layer_input in self.inputs:
            if layer_input.tensor_id >= len(tensors):
                raise ValueError(f"input tensor_id out of range: {layer_input.tensor_id}")
        for layer_output in self.outputs:
            if layer_output.tensor_id >= len(tensors):
                raise ValueError(
                    f"output tensor_id out of range: {layer_output.tensor_id}"
                )
            layer_output.layout.validate_for(tensors[layer_output.tensor_id])


@dataclass(frozen=True)
class Stage:
    """One scheduled execution unit on a physical Submesh."""

    name: str
    submesh: Submesh
    layers: tuple[Layer, ...] = field(default_factory=tuple)
    virtual_to_physical: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if not self.layers:
            raise ValueError("stages must contain at least one layer")

    @property
    def physical_to_virtual(self) -> dict[int, int]:
        """Return physical tile ids keyed to their virtual tile ids."""

        return {
            physical_tile_id: virtual_tile_id
            for virtual_tile_id, physical_tile_id in self.virtual_to_physical.items()
        }

    def validate_tensors(self, tensors: tuple[Tensor, ...]) -> None:
        """Validate Layer Tensor ids and output layout compatibility."""

        for layer in self.layers:
            layer.validate_tensors(tensors)


@dataclass(frozen=True)
class ExecutionPlan:
    """One complete physical execution decision."""

    name: str
    mesh: Mesh
    tensors: tuple[Tensor, ...] = field(default_factory=tuple)
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    transitions: tuple[Transition, ...] = field(default_factory=tuple)
    execution: ExecutionContract = field(default_factory=ExecutionContract)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("execution plan name must not be empty")


__all__ = [
    "ExecutionContract",
    "ExecutionPlan",
    "InitializerInput",
    "Layer",
    "LayerInput",
    "LayerInputSource",
    "LayerOutput",
    "LocalInput",
    "Stage",
    "TransitionSource",
]
