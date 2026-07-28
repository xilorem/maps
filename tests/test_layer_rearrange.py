from MAPS.arch import WorkKind
from MAPS.core.layout import TensorRange
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.hw.chips import magia_mesh
from MAPS.ops.defs.rearrange import ReshapePayload, TransposePayload
from MAPS.transitions.model import TransitionMode


def test_reshape_preserves_rectangular_channel_ownership() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    x = Tensor("x", 4, (1, 4, 4, 4), 2)
    output = Tensor("output", 6, (1, 4, 2, 2, 2, 2), 2)
    payload = ReshapePayload(x, output)

    tile_work = payload.build_tile_work(
        payload.output_layouts(submesh),
        submesh.tiles[1],
    )

    assert tile_work.work_kind is WorkKind.RESHAPE
    assert tile_work.input_slice.dims[1] == TensorRange(2, 2)
    assert tile_work.output_slice.dims[1] == TensorRange(2, 2)
    assert tile_work.input_slice.num_elements == tile_work.output_slice.num_elements


def test_transpose_inverse_permutates_required_input_ownership() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    x = Tensor("x", 3, (2, 4, 6), 2)
    output = Tensor("output", 3, (6, 2, 4), 2)
    payload = TransposePayload(x, output, (2, 0, 1))

    tile_work = payload.build_tile_work(
        payload.output_layouts(submesh),
        submesh.tiles[1],
    )

    assert payload.input_transition_mode is TransitionMode.PERMUTED_REMAP
    assert tile_work.output_slice.dims == (
        TensorRange(0, 6),
        TensorRange(0, 2),
        TensorRange(2, 2),
    )
    assert tile_work.input_slice.dims == (
        TensorRange(0, 2),
        TensorRange(2, 2),
        TensorRange(0, 6),
    )
