from maps.target.magia import build_mesh as magia_mesh
from maps.graph import TensorDType
from maps.graph import Node, OpKind
from maps.planning.mapping import LayoutAxis, LayoutAxisMode, Submesh, TensorLayout
from maps.graph import Tensor
from maps.planning.allocation.candidates import cost_estimator, placement_cost_estimator
from maps.operations import LayoutRelation
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
    )
    return Node(
        name="allreduce_sum",
        kind=OpKind.CUSTOM,
        inputs=(x,),
        outputs=(out,),
        payload=op,
    )


def test_allreduce_resolves_every_partial_layout_axis() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    node = _make_allreduce_sum_node()
    op = node.payload

    input_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(LayoutAxisMode.PARTIAL, tensor_axis=0),
        mesh_y=LayoutAxis(LayoutAxisMode.REPLICATE),
    )
    output_layout = op.layout_relations[0].output_layout_from_input_layout(input_layout)
    tile0_work = op.build_tile_work(
        output_layouts=(output_layout,),
        tile=submesh.tiles[0],
    )
    tile1_work = op.build_tile_work(
        output_layouts=(output_layout,),
        tile=submesh.tiles[1],
    )

    assert output_layout.mesh_x.mode is LayoutAxisMode.REPLICATE
    assert tile0_work.input_slice == tile1_work.input_slice
    assert tile0_work.output_slice == tile1_work.output_slice


def test_relation_drops_shard_granularity_when_output_is_replicated() -> None:
    tensor = Tensor("partial", rank=1, dims=(9,), elem_bytes=2)
    mesh = magia_mesh(width=2, height=1)
    submesh = Submesh(mesh, 0, frozenset((0, 1)))
    input_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(
            LayoutAxisMode.SHARD,
            tensor_axis=0,
            shard_granularity=3,
        ),
        mesh_y=LayoutAxis(LayoutAxisMode.REPLICATE),
    )
    relation = LayoutRelation(
        input_index=0,
        output_index=0,
        input_axis_for_output_axis=(0,),
        guarantees_slice_containment=True,
        replicated_output_axes=frozenset({0}),
    )

    output_layout = relation.output_layout_from_input_layout(input_layout)

    assert output_layout.mesh_x == LayoutAxis(LayoutAxisMode.REPLICATE)


def test_allreduce_protocol_cost_is_not_prescribed_by_the_generic_operation() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    node = _make_allreduce_sum_node()
    output_layouts = node.payload.output_layouts(submesh)

    assert placement_cost_estimator(node, output_layouts) == 0
    assert cost_estimator(node, output_layouts) == 0


def test_allreduce_sum_and_max_have_distinct_work_kinds() -> None:
    sum_node = _make_allreduce_sum_node()
    max_payload = AllReducePayload(
        op_name="AllReduceMax",
        x=sum_node.inputs[0],
        output=sum_node.outputs[0],
        reduction="max",
    )

    assert sum_node.payload.work_kind.name == "ALL_REDUCE_SUM"
    assert max_payload.work_kind.name == "ALL_REDUCE_MAX"
    assert sum_node.payload.work_kind is not max_payload.work_kind
