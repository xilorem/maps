import pytest

from MAPS.arch import WorkKind
from MAPS.core import (
    Graph,
    LayoutAxisMode,
    Node,
    OpKind,
    Submesh,
    Tensor,
    TensorDType,
)
from MAPS.core.graph import Edge
from MAPS.hw.chips import magia_mesh
from MAPS.ops.defs.cast import CastPayload
from MAPS.planner.contracts.options import PlannerOptions, SpatialMappingOptions
from MAPS.planner.plan import plan_graph
from MAPS.utils.execution_plan_json import execution_plan_json_payload


def _tensor(name: str, dtype: TensorDType, dims: tuple[int, ...]) -> Tensor:
    elem_bytes = 2 if dtype is TensorDType.FLOAT16 else 4
    return Tensor(
        name=name,
        rank=len(dims),
        dims=dims,
        elem_bytes=elem_bytes,
        dtype=dtype,
    )


def test_cast_preserves_shape_and_builds_typed_tile_work() -> None:
    x = _tensor("x", TensorDType.FLOAT16, (2, 4))
    output = _tensor("output", TensorDType.FLOAT32, (2, 4))
    payload = CastPayload(x=x, output=output)
    mesh = magia_mesh(width=1, height=1)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids={0})

    tile_work = payload.build_tile_work(payload.output_layouts(submesh), mesh.tiles[0])

    assert payload.work_kind is WorkKind.CAST
    assert tile_work.work_kind is WorkKind.CAST
    assert tile_work.input_slices[0].tensor.dtype is TensorDType.FLOAT16
    assert tile_work.input_slices[0].num_bytes == 16
    assert tile_work.output_slices[0].tensor.dtype is TensorDType.FLOAT32
    assert tile_work.output_slices[0].num_bytes == 32
    assert tile_work.operation_count() == 8
    assert tile_work.l1_bytes == 48


def test_cast_shards_output_and_preserves_the_exact_input_layout() -> None:
    payload = CastPayload(
        x=_tensor("x", TensorDType.FLOAT16, (4, 8)),
        output=_tensor("output", TensorDType.FLOAT32, (4, 8)),
    )
    mesh = magia_mesh(width=2, height=2)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids={0, 1, 2, 3})

    output_layout = payload.output_layouts(submesh, logical_shape=(2, 2))[0]
    input_layout = payload.layout_relations[0].input_layout_from_output_layout(
        output_layout
    )

    assert output_layout.mesh_x.mode is LayoutAxisMode.SHARD
    assert output_layout.mesh_x.tensor_axis == 1
    assert output_layout.mesh_y.mode is LayoutAxisMode.SHARD
    assert output_layout.mesh_y.tensor_axis == 0
    assert input_layout == output_layout


def test_cast_requires_shape_preservation() -> None:
    with pytest.raises(ValueError, match="Cast input and output shapes must match"):
        CastPayload(
            x=_tensor("x", TensorDType.FLOAT16, (2, 4)),
            output=_tensor("output", TensorDType.FLOAT32, (4, 2)),
        )


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype", "expected_cycles"),
    (
        (TensorDType.FLOAT16, TensorDType.FLOAT32, 12),
        (TensorDType.FLOAT32, TensorDType.FLOAT16, 11),
    ),
)
def test_spatz_costs_both_cast_directions(
    input_dtype: TensorDType,
    output_dtype: TensorDType,
    expected_cycles: int,
) -> None:
    payload = CastPayload(
        x=_tensor("x", input_dtype, (2, 4)),
        output=_tensor("output", output_dtype, (2, 4)),
    )
    mesh = magia_mesh(width=1, height=1)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids={0})
    tile_work = payload.build_tile_work(payload.output_layouts(submesh), mesh.tiles[0])

    assert payload.cost_model.cost(
        tile_work,
        mesh.tiles[0],
        mesh.tiles[0].device_by_name("spatz"),
    ) == expected_cycles


def _cast_graph(input_dtype: TensorDType, output_dtype: TensorDType) -> Graph:
    x = _tensor("x", input_dtype, (2, 4))
    output = _tensor("output", output_dtype, (2, 4))
    node = Node(
        name="cast",
        kind=OpKind.TRANSFORM,
        inputs=(x,),
        outputs=(output,),
        payload=CastPayload(x=x, output=output),
    )
    return Graph(
        name="typed_cast",
        tensors=(x, output),
        nodes=(node,),
        edges=(Edge(x, None, node), Edge(output, node, None)),
        inputs=(x,),
        outputs=(output,),
    )


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    (
        (TensorDType.FLOAT16, TensorDType.FLOAT32),
        (TensorDType.FLOAT32, TensorDType.FLOAT16),
    ),
)
def test_magia_plans_and_serializes_explicit_cast_on_spatz(
    input_dtype: TensorDType,
    output_dtype: TensorDType,
) -> None:
    execution_plan = plan_graph(
        _cast_graph(input_dtype, output_dtype),
        magia_mesh(width=1, height=1),
        PlannerOptions(
            spatial_mapping=SpatialMappingOptions(print_mapping=False),
            print_execution_plan_cost=False,
        ),
    )

    layer = execution_plan.stages[0].layers[0]
    payload = execution_plan_json_payload(execution_plan)
    layer_payload = payload["stages"][0]["layers"][0]

    assert layer.device_name == "spatz"
    assert isinstance(layer.node.payload, CastPayload)
    assert layer.inputs[0].tensor_id == 0
    assert layer.outputs[0].tensor_id == 1
    assert tuple(layer.outputs[0].layout.submesh.tile_ids) == (0,)
    assert payload["tensors"][0]["dtype"] == input_dtype.value
    assert payload["tensors"][0]["elem_bytes"] == _tensor(
        "expected-input", input_dtype, (2, 4)
    ).elem_bytes
    assert payload["tensors"][1]["dtype"] == output_dtype.value
    assert payload["tensors"][1]["elem_bytes"] == _tensor(
        "expected-output", output_dtype, (2, 4)
    ).elem_bytes
    assert layer_payload["device_name"] == "spatz"
    assert layer_payload["node"]["payload"]["work_kind"] == "CAST"
    assert layer_payload["inputs"][0]["tensor_id"] == 0
    assert layer_payload["outputs"][0]["tensor_id"] == 1


def test_unsupported_cast_dtype_fails_with_work_signature_diagnostic() -> None:
    with pytest.raises(ValueError) as error:
        plan_graph(
            _cast_graph(TensorDType.INT32, TensorDType.FLOAT32),
            magia_mesh(width=1, height=1),
            PlannerOptions(
                spatial_mapping=SpatialMappingOptions(print_mapping=False),
                print_execution_plan_cost=False,
            ),
        )

    message = str(error.value)
    assert "node cast" in message
    assert "WorkSignature" in message
    assert "TensorDType.INT32" in message
    assert "work_kind=<WorkKind.CAST" in message
    assert "tile 0" in message
    assert "configured assignment=None" in message
    assert "considered devices: idma_read, idma_write, core, spatz, redmule" in message
