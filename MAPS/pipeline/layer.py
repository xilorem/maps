"""Layer IR for scheduled graph nodes inside a pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from MAPS.core.layout import TensorLayout
from MAPS.transitions.contracts import InputDestination

if TYPE_CHECKING:
    from MAPS.core.graph import Node
    from MAPS.core.tensor import Tensor


@dataclass(frozen=True)
class ExternalInput:
    """Layer input read from an external base address."""

    base_addr: int

    def __post_init__(self) -> None:
        if self.base_addr <= 0:
            raise ValueError("external inputs require base_addr > 0")


@dataclass(frozen=True)
class TransitionInput:
    """Layer input produced by an explicit inter-stage transition."""

    transition_id: int

    def __post_init__(self) -> None:
        if self.transition_id < 0:
            raise ValueError("transition inputs require transition_id >= 0")


@dataclass(frozen=True)
class InitializerInput:
    """Per-tile residency for an immutable input tensor."""

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
    """Layer input read from a previous layer output in the same stage."""

    layer_idx: int
    tensor_id: int

    def __post_init__(self) -> None:
        if self.layer_idx < 0 or self.tensor_id < 0:
            raise ValueError("layer_idx and tensor_id must be >= 0")


LayerInputSource = (
    ExternalInput
    | InitializerInput
    | TransitionInput
    | TransitionSource
    | LocalInput
)


@dataclass(frozen=True)
class LayerInput:
    """One input of a layer."""

    tensor_id: int
    source: LayerInputSource

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")

    @classmethod
    def external(cls, tensor_id: int, base_addr: int) -> "LayerInput":
        return cls(tensor_id=tensor_id, source=ExternalInput(base_addr=base_addr))

    @classmethod
    def transition(cls, tensor_id: int, transition_id: int) -> "LayerInput":
        return cls(
            tensor_id=tensor_id,
            source=TransitionInput(transition_id=transition_id),
        )

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
    """One output of a layer."""

    tensor_id: int
    layout: TensorLayout

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")


@dataclass(frozen=True)
class Layer:
    """One scheduled graph node inside a stage."""

    node: "Node"
    inputs: tuple[LayerInput, ...] = field(default_factory=tuple)
    outputs: tuple[LayerOutput, ...] = field(default_factory=tuple)

    def validate_tensors(self, tensors: tuple["Tensor", ...]) -> None:
        """Validate bound tensor ids and output layout compatibility."""

        for layer_input in self.inputs:
            if layer_input.tensor_id >= len(tensors):
                raise ValueError(f"input tensor_id out of range: {layer_input.tensor_id}")
        for layer_output in self.outputs:
            if layer_output.tensor_id >= len(tensors):
                raise ValueError(f"output tensor_id out of range: {layer_output.tensor_id}")
            layer_output.layout.validate_for(tensors[layer_output.tensor_id])
