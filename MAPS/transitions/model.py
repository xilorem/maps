"""Inter-layer transition IR matching the runtime-side `transition_t`."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from MAPS.core.layout import TensorLayout, TensorSubSlice
from MAPS.core.tensor import Tensor


class TransitionMode(IntEnum):
    DIRECT_REMAP = 1
    PERMUTED_REMAP = 2


@dataclass(frozen=True)
class TransitionFragment:
    """One fragment of an inter-layer transition plan."""

    src_hartid: int
    dst_hartid: int
    src_subslice: TensorSubSlice
    dst_subslice: TensorSubSlice

    def __post_init__(self) -> None:
        if self.src_hartid < 0 or self.dst_hartid < 0:
            raise ValueError("hart ids must be >= 0")


@dataclass(frozen=True)
class Transition:
    """One inter-layer edge in the scheduled pipeline."""

    name: str
    tensor_id: int
    src_layer_id: int
    src_output_idx: int
    dst_layer_id: int
    dst_input_idx: int
    mode: TransitionMode
    src_layout: TensorLayout
    dst_layout: TensorLayout
    fragments: tuple[TransitionFragment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("transition name must not be empty")
        for value in (
            self.tensor_id,
            self.src_layer_id,
            self.src_output_idx,
            self.dst_layer_id,
            self.dst_input_idx,
        ):
            if value < 0:
                raise ValueError("transition ids and indices must be >= 0")

    def validate_for(self, tensor: Tensor) -> None:
        self.src_layout.validate_for(tensor)
        self.dst_layout.validate_for(tensor)
