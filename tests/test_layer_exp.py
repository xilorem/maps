from MAPS.arch import WorkKind
from MAPS.hw.chips import magia_mesh
from MAPS.hw.devices.spatz import SPATZ_DEVICE
from MAPS.core.dtype import TensorDType
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.costs.elementwise_cost import ElementwiseCostModel
from MAPS.ops.defs.elementwise import (
    BINARY_ELEMENTWISE_OPS,
    UNARY_ELEMENTWISE_OPS,
    BinaryElementwisePayload,
    UnaryElementwisePayload,
)


def _make_exp_op() -> UnaryElementwisePayload:
    return UnaryElementwisePayload(
        op_name="Exp",
        x=Tensor(
            name="x", rank=2, dims=(4, 8), elem_bytes=2,
            dtype=TensorDType.FLOAT16,
        ),
        output=Tensor(
            name="out", rank=2, dims=(4, 8), elem_bytes=2,
            dtype=TensorDType.FLOAT16,
        ),
        work_kind=WorkKind.EXP,
    )


def test_exp_tile_work_uses_output_slice_as_input_slice() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    op = _make_exp_op()
    output_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )

    tile_work = op.build_tile_work(
        output_layouts=(output_layout,),
        tile=submesh.tiles[0],
    )

    assert tile_work.input_tile_slices[0] == tile_work.output_slice
    assert tuple((dim.start, dim.length) for dim in tile_work.output_slice.dims) == (
        (0, 4),
        (0, 4),
    )
    assert tile_work.l1_bytes == sum(
        ref.num_bytes for ref in tile_work.input_slices + tile_work.output_slices
    )
    assert tile_work.fits_l1(submesh.tiles[0])


def test_exp_cost_uses_exp_capable_device() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    op = _make_exp_op()
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )
    tile_work = op.build_tile_work(
        output_layouts=(layout,),
        tile=submesh.tiles[0],
    )

    assert ElementwiseCostModel(work_kind=WorkKind.EXP).cost(
        tile_work,
        submesh.tiles[0],
        SPATZ_DEVICE,
    ) == SPATZ_DEVICE.cycles(tile_work)


def test_elementwise_work_kinds_preserve_operation_identity() -> None:
    assert UNARY_ELEMENTWISE_OPS["Sigmoid"] is WorkKind.SIGMOID
    assert UNARY_ELEMENTWISE_OPS["Sqrt"] is WorkKind.SQRT
    assert UNARY_ELEMENTWISE_OPS["Log"] is WorkKind.LOG
    assert BINARY_ELEMENTWISE_OPS["Add"] is WorkKind.ADD
    assert BINARY_ELEMENTWISE_OPS["Div"] is WorkKind.DIV


def test_binary_elementwise_tile_work_supports_broadcasting() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    lhs = Tensor(name="lhs", rank=2, dims=(4, 8), elem_bytes=2)
    rhs = Tensor(name="rhs", rank=1, dims=(8,), elem_bytes=2)
    output = Tensor(name="out", rank=2, dims=(4, 8), elem_bytes=2)
    op = BinaryElementwisePayload(op_name="Add", lhs=lhs, rhs=rhs, output=output)
    output_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )

    tile_work = op.build_tile_work(
        output_layouts=(output_layout,),
        tile=submesh.tiles[1],
    )

    assert tuple((dim.start, dim.length) for dim in tile_work.output_slice.dims) == (
        (0, 4),
        (4, 4),
    )
    assert tuple((dim.start, dim.length) for dim in tile_work.input_tile_slices[0].dims) == (
        (0, 4),
        (4, 4),
    )
    assert tuple((dim.start, dim.length) for dim in tile_work.input_tile_slices[1].dims) == (
        (4, 4),
    )
