"""Tensor ownership helpers used by canonical Transition compilation."""

from maps.graph import Tensor
from maps.hardware import Tile
from maps.planning.layouts import TensorLayout, TensorSlice, tile_tensor_slice


def tile_owned_slices(tensor: Tensor, layout: TensorLayout) -> tuple[tuple[Tile, TensorSlice], ...]:
    """Return the concrete slice owned by each tile in one submesh."""

    owned: list[tuple[Tile, TensorSlice]] = []
    for tile in layout.submesh.tiles:
        owned.append(
            (
                tile,
                tile_tensor_slice(
                    tensor=tensor,
                    layout=layout,
                    tile=tile,
                ),
            )
        )
    return tuple(owned)
