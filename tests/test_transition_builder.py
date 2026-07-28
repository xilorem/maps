import pytest

from MAPS.transitions import build_transition
from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
    tile_tensor_slice,
)
from MAPS.hw.chips import magia_mesh
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.transitions.model import TransitionMode
from MAPS.ops.defs.rearrange import TransposePayload


def test_build_transition_allows_empty_demand() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    tensor = Tensor(name="x", rank=2, dims=(8, 8), elem_bytes=2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )

    transition = build_transition(
        name="reuse",
        tensor=tensor,
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=0,
        src_layout=layout,
        dst_layout=layout,
        dst_required_slices=(),
    )

    assert transition.fragments == ()


def test_build_transition_builds_direct_remap_fragments() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    tensor = Tensor(name="x", rank=2, dims=(8, 8), elem_bytes=2)
    src_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )
    dst_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )

    transition = build_transition(
        name="remap",
        tensor=tensor,
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=0,
        src_layout=src_layout,
        dst_layout=dst_layout,
        dst_required_slices=tuple(
            (tile, tile_tensor_slice(tensor, dst_layout, tile))
            for tile in dst_layout.submesh.tiles
        ),
    )

    assert transition.mode is TransitionMode.DIRECT_REMAP
    # Destination mesh_x is replicated, so both tiles in each destination row
    # need the same row slice. That duplicates the top-row and bottom-row
    # transfers across two consumers each.
    assert len(transition.fragments) == 8
    assert {(fragment.src_hartid, fragment.dst_hartid) for fragment in transition.fragments} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (8, 8),
        (8, 9),
        (9, 8),
        (9, 9),
    }
    assert {fragment.src_subslice.parent for fragment in transition.fragments} == {
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=0, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=4, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=4, length=4),
                TensorRange(start=0, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=4, length=4),
                TensorRange(start=4, length=4),
            ),
        ),
    }
    assert {
        fragment.dst_subslice.dims
        for fragment in transition.fragments
    } == {
        (
            TensorRange(start=0, length=4),
            TensorRange(start=0, length=4),
        ),
        (
            TensorRange(start=0, length=4),
            TensorRange(start=4, length=4),
        ),
    }


def test_transpose_builds_permuted_all_to_all_ownership_exchange() -> None:
    mesh = magia_mesh(width=2, height=1)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    x = Tensor(name="x", rank=2, dims=(4, 4), elem_bytes=2)
    output = Tensor(name="output", rank=2, dims=(4, 4), elem_bytes=2)
    payload = TransposePayload(x, output, (1, 0))
    src_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )
    output_layout = payload.output_layouts(submesh)[0]
    input_layout = payload.layout_relations[0].input_layout_from_output_layout(
        output_layout
    )
    required = tuple(
        (
            tile,
            payload.build_tile_work((output_layout,), tile).input_slice,
        )
        for tile in submesh.tiles
    )

    transition = build_transition(
        name="transpose_exchange",
        tensor=x,
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=0,
        src_layout=src_layout,
        dst_layout=input_layout,
        dst_required_slices=required,
        mode=TransitionMode.PERMUTED_REMAP,
    )

    assert transition.mode is TransitionMode.PERMUTED_REMAP
    assert input_layout.mesh_x.tensor_axis == 0
    assert len(transition.fragments) == 4
    assert {
        (fragment.src_hartid, fragment.dst_hartid)
        for fragment in transition.fragments
    } == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_build_transition_builds_direct_remap_between_different_submeshes() -> None:
    mesh = magia_mesh()
    src_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    dst_submesh = Submesh(mesh=mesh, submesh_id=1, x0=2, y0=2, width=2, height=2)
    tensor = Tensor(name="x", rank=2, dims=(8, 8), elem_bytes=2)
    src_layout = TensorLayout(
        submesh=src_submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )
    dst_layout = TensorLayout(
        submesh=dst_submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )

    transition = build_transition(
        name="cross_submesh_remap",
        tensor=tensor,
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=0,
        src_layout=src_layout,
        dst_layout=dst_layout,
        dst_required_slices=tuple(
            (tile, tile_tensor_slice(tensor, dst_layout, tile))
            for tile in dst_layout.submesh.tiles
        ),
    )

    assert transition.mode is TransitionMode.DIRECT_REMAP
    assert len(transition.fragments) == 4
    assert {(fragment.src_hartid, fragment.dst_hartid) for fragment in transition.fragments} == {
        (0, 18),
        (1, 19),
        (8, 26),
        (9, 27),
    }
    assert {fragment.src_subslice.parent for fragment in transition.fragments} == {
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=0, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=4, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=4, length=4),
                TensorRange(start=0, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=4, length=4),
                TensorRange(start=4, length=4),
            ),
        ),
    }
