from MAPS.hw.chips import magia_mesh
from MAPS.core.dtype import TensorDType
from MAPS.core.graph import Node, OpKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from maps.planning.allocation.cost import cost_estimator, placement_cost_estimator
from maps.operations.collective import AllReducePayload


def _make_allreduce_sum_node() -> Node:
    x = Tensor(
        name="x", rank=2, dims=(4, 1), elem_bytes=2, dtype=TensorDType.FLOAT16
    )
    out = Tensor(
        name="out", rank=2, dims=(4, 1), elem_bytes=2, dtype=TensorDType.FLOAT16
    )
    op = AllReducePayload(
        op_name="AllReduceSum",
        x=x,
        output=out,
        reduction="sum",
        collective_axis="x",
    )
    return Node(
        name="allreduce_sum",
        kind=OpKind.CUSTOM,
        inputs=(x,),
        outputs=(out,),
        payload=op,
    )


def test_allreduce_op_replicates_collective_axis_layout() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    node = _make_allreduce_sum_node()
    op = node.payload

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
    assert tile0_work.input_slice == tile1_work.input_slice
    assert tile0_work.output_slice == tile1_work.output_slice


def test_allreduce_uses_placement_sensitive_cost() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    node = _make_allreduce_sum_node()
    output_layouts = node.payload.output_layouts(submesh)

    assert placement_cost_estimator(node, output_layouts) > 0
    assert cost_estimator(node, output_layouts) == placement_cost_estimator(
        node,
        output_layouts,
    )
