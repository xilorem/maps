"""Helpers to build transition objects from producer ownership and consumer demand."""

from MAPS.core.layout import TensorLayout
from MAPS.core.layout import TensorSlice
from MAPS.core.tensor import Tensor
from MAPS.arch import Tile
from MAPS.transitions.model import Transition, TransitionMode

from .remap import build_direct_remap_fragments


def build_transition(
    name: str,
    tensor: Tensor,
    tensor_id: int,
    src_layer_id: int,
    src_output_idx: int,
    dst_layer_id: int,
    dst_input_idx: int,
    src_layout: TensorLayout,
    dst_layout: TensorLayout,
    dst_required_slices: tuple[tuple[Tile, TensorSlice], ...],
    mode: TransitionMode = TransitionMode.DIRECT_REMAP,
) -> Transition:
    """Build one concrete transition instance."""

    src_layout.validate_for(tensor)
    dst_layout.validate_for(tensor)

    fragments = build_direct_remap_fragments(
        tensor=tensor,
        src_layout=src_layout,
        dst_required_slices=dst_required_slices,
    )
    return Transition(
        name=name,
        tensor_id=tensor_id,
        src_layer_id=src_layer_id,
        src_output_idx=src_output_idx,
        dst_layer_id=dst_layer_id,
        dst_input_idx=dst_input_idx,
        mode=mode,
        src_layout=src_layout,
        dst_layout=dst_layout,
        fragments=fragments,
    )
