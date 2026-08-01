from maps.hardware import L1Memory, Tile, WorkKind
from maps.target.magia import build_mesh as magia_mesh
from maps.target.magia import CORE_DEVICE as MAGIA_CORE_DEVICE
from tests.noc_utils import TEST_SCALAR_DEVICE
from maps.planning.mapping import TensorRange
from maps.planning.mapping import Submesh
from maps.graph import Tensor
from maps.operations.convolution import Conv2DPayload
from maps.operations.depthwise_convolution import DepthwiseConvPayload


def test_direct_conv_shards_output_channels_and_height_with_clamped_halo() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    x = Tensor(name="x", rank=4, dims=(1, 3, 8, 9), elem_bytes=2)
    w = Tensor(name="w", rank=4, dims=(7, 3, 3, 2), elem_bytes=2)
    b = Tensor(name="b", rank=1, dims=(7,), elem_bytes=2)
    output = Tensor(name="out", rank=4, dims=(1, 7, 4, 6), elem_bytes=2)
    op = Conv2DPayload(
        x=x,
        w=w,
        b=b,
        output=output,
        strides=(2, 2),
        pads=(2, 1, 1, 2),
        dilations=(2, 1),
    )

    work = op.build_tile_work(op.output_layouts(submesh), submesh.tiles[3])

    assert work.work_kind is WorkKind.CONV2D
    assert work.output_slice.dims == (
        TensorRange(0, 1),
        TensorRange(4, 3),
        TensorRange(2, 2),
        TensorRange(0, 6),
    )
    assert work.input_slice.dims == (
        TensorRange(0, 1),
        TensorRange(0, 3),
        TensorRange(2, 6),
        TensorRange(0, 9),
    )
    assert work.weight_slice.dims[0] == TensorRange(4, 3)
    assert work.bias_slice is not None
    assert work.bias_slice.dims[0] == TensorRange(4, 3)
    assert work.local_padding == (0, 1, 1, 2)
    assert work.strides == (2, 2)
    assert work.dilations == (2, 1)
    assert work.operation_count() == 648
    generic_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=1),
        devices=(TEST_SCALAR_DEVICE,),
    )
    assert op.cost_model.cost(work, generic_tile, TEST_SCALAR_DEVICE) == 648


def test_depthwise_conv_shards_matching_input_weight_and_bias_channels() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    x = Tensor(name="x", rank=4, dims=(1, 4, 5, 5), elem_bytes=2)
    w = Tensor(name="w", rank=4, dims=(4, 1, 3, 3), elem_bytes=2)
    b = Tensor(name="b", rank=1, dims=(4,), elem_bytes=2)
    output = Tensor(name="out", rank=4, dims=(1, 4, 3, 3), elem_bytes=2)
    op = DepthwiseConvPayload(x=x, w=w, b=b, output=output)

    tile_work = op.build_tile_work(
        output_layouts=op.output_layouts(submesh),
        tile=submesh.tiles[1],
    )

    assert tile_work.work_kind is WorkKind.DEPTHWISE_CONV
    assert tile_work.output_slice.dims[1] == TensorRange(start=2, length=2)
    assert tile_work.input_slice.dims[1] == TensorRange(start=2, length=2)
    assert tile_work.weight_slice.dims[0] == TensorRange(start=2, length=2)
    assert tile_work.bias_slice is not None
    assert tile_work.bias_slice.dims[0] == TensorRange(start=2, length=2)
    assert tile_work.operation_count() == 162
    assert op.cost_model.cost(
        tile_work,
        submesh.tiles[1],
        MAGIA_CORE_DEVICE,
    ) == 162
