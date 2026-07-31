from dataclasses import replace

import pytest

from MAPS.arch import FixedDeviceAssignment
from MAPS.core import ConstantStore, Graph, Node, OpKind, Tensor, TensorDType
from MAPS.core.graph import Edge
from MAPS.hw.chips import magia_mesh
from MAPS.importers.model import ImportedModel
from maps.operations.elementwise import UnaryElementwisePayload
from maps.operations.gemm import GemmPayload
from MAPS.planner.contracts.options import PlannerOptions, SpatialMappingOptions
from maps.planning import PlanningConstraints, validate_execution_plan
from MAPS.planner.plan import plan_graph, plan_model
from MAPS.utils.execution_plan_json import execution_plan_json_payload


def _typed_tensor(
    name: str,
    dtype: TensorDType,
    dims: tuple[int, ...],
    *,
    initializer: bool = False,
) -> Tensor:
    return Tensor(
        name=name,
        rank=len(dims),
        dims=dims,
        elem_bytes=2 if dtype is TensorDType.FLOAT16 else 4,
        is_initializer=initializer,
        dtype=dtype,
    )


def _gemm_graph(dtype: TensorDType, *, with_bias: bool) -> Graph:
    x = _typed_tensor("x", dtype, (2, 3))
    weight = _typed_tensor("weight", dtype, (3, 4), initializer=True)
    output = _typed_tensor("output", dtype, (2, 4))
    bias = (
        _typed_tensor("bias", dtype, (4,), initializer=True)
        if with_bias
        else None
    )
    inputs = (x, weight) + ((bias,) if bias is not None else ())
    node = Node(
        name="gemm",
        kind=OpKind.GEMM,
        inputs=inputs,
        outputs=(output,),
        payload=GemmPayload(x=x, w=weight, y=bias, output=output),
    )
    initializers = (weight,) + ((bias,) if bias is not None else ())
    return Graph(
        name="typed_gemm",
        tensors=inputs + (output,),
        nodes=(node,),
        edges=tuple(Edge(tensor, None, node) for tensor in inputs)
        + (Edge(output, node, None),),
        inputs=(x,),
        outputs=(output,),
        initializers=initializers,
    )


def _unary_model(op_name: str) -> ImportedModel:
    x = _typed_tensor("x", TensorDType.FLOAT32, (8,))
    output = _typed_tensor("output", TensorDType.FLOAT32, (8,))
    node = Node(
        name=op_name.lower(),
        kind=OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(output,),
        payload=UnaryElementwisePayload(
            op_name=op_name,
            x=x,
            output=output,
        ),
    )
    return ImportedModel(
        graph=Graph(
            name=f"typed_{op_name.lower()}",
            tensors=(x, output),
            nodes=(node,),
            edges=(Edge(x, None, node), Edge(output, node, None)),
            inputs=(x,),
            outputs=(output,),
        ),
        constants=ConstantStore(()),
    )


def _quiet_options() -> PlannerOptions:
    return PlannerOptions(
        spatial_mapping=SpatialMappingOptions(print_mapping=False),
        print_execution_plan_cost=False,
    )


@pytest.mark.parametrize(
    ("dtype", "with_bias", "expected_device"),
    (
        (TensorDType.FLOAT16, False, "redmule"),
        (TensorDType.FLOAT16, True, "redmule"),
        (TensorDType.FLOAT32, False, "core"),
    ),
)
def test_magia_plans_typed_gemm_with_one_hardware_owned_device_name(
    dtype: TensorDType,
    with_bias: bool,
    expected_device: str,
) -> None:
    execution_plan = plan_graph(
        _gemm_graph(dtype, with_bias=with_bias),
        magia_mesh(width=1, height=1),
        _quiet_options(),
    )

    layer = execution_plan.stages[0].layers[0]
    assert layer.device_name == expected_device
    layer_payload = execution_plan_json_payload(execution_plan)["stages"][0][
        "layers"
    ][0]
    assert layer_payload["device_name"] == expected_device
    assert "capabilities" not in layer_payload
    assert "device_assignment" not in layer_payload
    assert "throughput" not in layer_payload


@pytest.mark.parametrize(
    ("op_name", "expected_device"),
    (("Relu", "spatz"), ("Log", "core")),
)
def test_magia_plans_elementwise_work_with_explicit_device_assignment(
    op_name: str,
    expected_device: str,
) -> None:
    bundle = plan_model(
        _unary_model(op_name),
        magia_mesh(width=1, height=1),
        _quiet_options(),
    )

    assert bundle.execution_plan.stages[0].layers[0].device_name == expected_device


def test_execution_plan_validation_rejects_incapable_layer_device_name() -> None:
    execution_plan = plan_graph(
        _gemm_graph(TensorDType.FLOAT16, with_bias=False),
        magia_mesh(width=1, height=1),
        _quiet_options(),
    )
    layer = execution_plan.stages[0].layers[0]
    invalid_layer = replace(layer, device_name="core")
    invalid_stage = replace(execution_plan.stages[0], layers=(invalid_layer,))
    invalid_plan = replace(execution_plan, stages=(invalid_stage,))

    report = validate_execution_plan(invalid_plan, PlanningConstraints())

    assert not report.is_valid
    assert report.violations[0].kind == "layer_device_assignment_invalid"
    assert "gemm" in report.violations[0].message
    assert "core" in report.violations[0].message


def test_execution_plan_validation_rejects_missing_layer_device_name() -> None:
    execution_plan = plan_graph(
        _gemm_graph(TensorDType.FLOAT16, with_bias=False),
        magia_mesh(width=1, height=1),
        _quiet_options(),
    )
    layer = replace(execution_plan.stages[0].layers[0], device_name=None)
    stage = replace(execution_plan.stages[0], layers=(layer,))

    report = validate_execution_plan(
        replace(execution_plan, stages=(stage,)),
        PlanningConstraints(),
    )

    assert not report.is_valid
    assert report.violations[0].kind == "layer_device_assignment_invalid"
    assert "has no retained Device name" in report.violations[0].message


def test_gemm_planning_error_names_node_signature_tile_and_considered_devices() -> None:
    mesh = magia_mesh(width=1, height=1)
    unassigned_tile = replace(
        mesh.tiles[0],
        device_assignment=FixedDeviceAssignment(),
    )
    unassigned_mesh = replace(mesh, tiles=(unassigned_tile,))

    with pytest.raises(ValueError) as error:
        plan_graph(
            _gemm_graph(TensorDType.FLOAT16, with_bias=False),
            unassigned_mesh,
            _quiet_options(),
        )

    message = str(error.value)
    assert "node gemm" in message
    assert "WorkSignature" in message
    assert "tile 0" in message
    assert "configured assignment=None" in message
    assert "considered devices: idma_read, idma_write, core, spatz, redmule" in message
