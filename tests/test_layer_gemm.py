from MAPS.core.graph import Node, OpKind
from maps.planning import Layer
from maps.planning.layouts import LayoutAxis, LayoutAxisMode, TensorLayout, TensorRange, TensorSlice
from MAPS.hw.chips import magia_mesh
from maps.planning import Stage
from maps.planning.submesh import Submesh
from MAPS.core.tensor import Tensor
from maps.operations.gemm import GemmPayload


def _make_layout(submesh: Submesh) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )

def test_stage_requires_at_least_one_layer() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)

    try:
        Stage(name="empty", submesh=submesh, layers=())
    except ValueError as exc:
        assert "at least one layer" in str(exc)
    else:
        raise AssertionError("expected empty stage construction to fail")


def test_stage_can_group_multiple_nodes() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    x = Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2)
    w0 = Tensor(name="w0", rank=2, dims=(8, 16), elem_bytes=2)
    y0 = Tensor(name="y0", rank=2, dims=(4, 16), elem_bytes=2)
    w1 = Tensor(name="w1", rank=2, dims=(16, 12), elem_bytes=2)
    y1 = Tensor(name="y1", rank=2, dims=(4, 12), elem_bytes=2)

    node0 = Node(
        name="gemm0",
        kind=OpKind.GEMM,
        inputs=(x, w0),
        outputs=(y0,),
        payload=GemmPayload(x=x, w=w0, y=None, output=y0),
    )
    node1 = Node(
        name="gemm1",
        kind=OpKind.GEMM,
        inputs=(y0, w1),
        outputs=(y1,),
        payload=GemmPayload(x=y0, w=w1, y=None, output=y1),
    )

    stage = Stage(
        name="stage0",
        submesh=submesh,
        layers=(
            Layer(node=node0),
            Layer(node=node1),
        ),
    )

    assert tuple(layer.node for layer in stage.layers) == (node0, node1)


def test_gemm_payload_accepts_valid_tensor_shapes() -> None:
    tensors = (
        Tensor(name="x", rank=3, dims=(2, 4, 8), elem_bytes=2),
        Tensor(name="w", rank=3, dims=(2, 8, 16), elem_bytes=2),
        Tensor(name="out", rank=3, dims=(2, 4, 16), elem_bytes=2),
        Tensor(name="y", rank=3, dims=(2, 4, 16), elem_bytes=2),
    )

    GemmPayload(
        x=tensors[0],
        w=tensors[1],
        y=tensors[3],
        output=tensors[2],
    )


def test_gemm_payload_rejects_incompatible_tensor_element_sizes() -> None:
    tensors = (
        Tensor(name="x", rank=3, dims=(2, 4, 8), elem_bytes=2),
        Tensor(name="w", rank=3, dims=(2, 8, 16), elem_bytes=4),
        Tensor(name="out", rank=3, dims=(2, 4, 16), elem_bytes=2),
    )

    try:
        GemmPayload(
            x=tensors[0],
            w=tensors[1],
            y=None,
            output=tensors[2],
        )
    except ValueError as exc:
        assert "element size" in str(exc)
    else:
        raise AssertionError("expected incompatible GEMM tensors to fail")


def test_gemm_payload_rejects_incompatible_k_dimension() -> None:
    tensors = (
        Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        Tensor(name="w", rank=2, dims=(7, 16), elem_bytes=2),
        Tensor(name="out", rank=2, dims=(4, 16), elem_bytes=2),
    )

    try:
        GemmPayload(
            x=tensors[0],
            w=tensors[1],
            y=None,
            output=tensors[2],
        )
    except ValueError as exc:
        assert "K dimension" in str(exc)
    else:
        raise AssertionError("expected incompatible GEMM K dimension to fail")


def test_build_gemm_tile_work_derives_required_operand_slices() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    tile = mesh.tile(1, 0)
    op = GemmPayload(
        x=Tensor(name="x", rank=2, dims=(8, 16), elem_bytes=2),
        w=Tensor(name="w", rank=2, dims=(16, 12), elem_bytes=2),
        y=Tensor(name="y", rank=2, dims=(8, 12), elem_bytes=2),
        output=Tensor(name="out", rank=2, dims=(8, 12), elem_bytes=2),
    )
    output_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )

    work = op.build_tile_work((output_layout,), tile)

    assert work.output_slice == TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=4),
            TensorRange(start=6, length=6),
        ),
    )
    assert work.x_slice == TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=4),
            TensorRange(start=0, length=16),
        ),
    )
    assert work.w_slice == TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=16),
            TensorRange(start=6, length=6),
        ),
    )
    assert work.y_slice == work.output_slice
    assert work.l1_bytes == sum(ref.num_bytes for ref in work.input_slices + work.output_slices)
    assert work.fits_l1(tile)


def test_gemm_output_layouts_accept_logical_shape() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=6, height=1)
    op = GemmPayload(
        x=Tensor(name="x", rank=2, dims=(8, 16), elem_bytes=2),
        w=Tensor(name="w", rank=2, dims=(16, 12), elem_bytes=2),
        y=None,
        output=Tensor(name="out", rank=2, dims=(8, 12), elem_bytes=2),
    )

    output_layouts = op.output_layouts(submesh, logical_shape=(3, 2))

    assert all(layout.logical_width == 3 for layout in output_layouts)
    assert all(layout.logical_height == 2 for layout in output_layouts)
