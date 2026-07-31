from MAPS.arch import WorkKind
from MAPS.hw.chips import magia_mesh
from MAPS.hw.devices.spatz import SPATZ_DEVICE
from MAPS.core.dtype import TensorDType
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.costs.reduction_cost import ReductionCostModel
from MAPS.ops.defs.reduction import ReductionPayload


def _make_reduce_sum_op() -> ReductionPayload:
    return ReductionPayload(
        op_name="ReduceSum",
        x=Tensor(
            name="x", rank=2, dims=(4, 8), elem_bytes=2,
            dtype=TensorDType.FLOAT16,
        ),
        output=Tensor(
            name="out", rank=2, dims=(4, 1), elem_bytes=2,
            dtype=TensorDType.FLOAT16,
        ),
        axis=1,
        work_kind=WorkKind.REDUCE_SUM,
    )


def test_reduce_op_replicates_output_along_reduced_mesh_axis() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    op = _make_reduce_sum_op()

    output_layout = op.output_layouts(submesh)[0]
    tile0_work = op.build_tile_work(
        output_layouts=(output_layout,),
        tile=submesh.tiles[0],
    )
    tile1_work = op.build_tile_work(
        output_layouts=(output_layout,),
        tile=submesh.tiles[1],
    )

    assert output_layout.mesh_x.mode.name == "REPLICATE"
    assert tuple((dim.start, dim.length) for dim in tile0_work.input_slice.dims) == (
        (0, 4),
        (0, 4),
    )
    assert tuple((dim.start, dim.length) for dim in tile1_work.input_slice.dims) == (
        (0, 4),
        (4, 4),
    )
    assert tuple((dim.start, dim.length) for dim in tile0_work.output_slice.dims) == (
        (0, 4),
        (0, 1),
    )
    assert tile1_work.output_slice == tile0_work.output_slice


def test_reduce_cost_counts_input_elements() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    op = _make_reduce_sum_op()
    output_layout = op.output_layouts(submesh)[0]
    tile_work = op.build_tile_work(
        output_layouts=(output_layout,),
        tile=submesh.tiles[0],
    )

    assert ReductionCostModel(work_kind=WorkKind.REDUCE_SUM).cost(
        tile_work,
        submesh.tiles[0],
        SPATZ_DEVICE,
    ) == SPATZ_DEVICE.cycles(tile_work)
