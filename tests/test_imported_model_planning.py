import json

import pytest

from MAPS.arch import L1Memory, L2Memory, Mesh
from maps.graph import ConstantStore, Graph, ImportedModel, Node, OpKind, Tensor
from MAPS.deployment import write_execution_plan_bundle
from maps.graph import import_onnx_model
from maps.operations.elementwise import UnaryElementwisePayload
from maps.operations.softmax import SoftmaxPayload
from MAPS.planner.contracts.options import PlannerOptions, SpatialMappingOptions
from MAPS.planner.plan import plan_model
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _mesh(width: int = 2) -> Mesh:
    return Mesh(
        width=width,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, 1),
        tiles=rectangular_test_tiles(
            width,
            1,
            memory=L1Memory(size=4096, bandwidth=1),
        ),
    )


def _quiet_options() -> PlannerOptions:
    return PlannerOptions(
        spatial_mapping=SpatialMappingOptions(print_mapping=False),
        print_execution_plan_cost=False,
    )


def test_plan_model_runs_mandatory_decomposition_and_serializes_provenance(
    tmp_path,
) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    source = helper.make_model(
        helper.make_graph(
            [helper.make_node("Softmax", ("x",), ("output",), name="softmax")],
            "softmax_model",
            (x,),
            (output,),
        )
    )
    model_path = tmp_path / "softmax.onnx"
    onnx.save(source, model_path)

    imported = import_onnx_model(model_path)
    assert len(imported.graph.nodes) == 1
    assert isinstance(imported.graph.nodes[0].payload, SoftmaxPayload)

    bundle = plan_model(imported, _mesh(), _quiet_options())
    independently_planned = plan_model(imported, _mesh(), _quiet_options())

    assert bundle.constants is imported.constants
    assert len(bundle.graph.nodes) > 1
    assert bundle.execution_plan.name == "softmax_model"
    assert len(bundle.rewrite_report.events) == 1
    event = bundle.rewrite_report.events[0]
    assert event.rewrite_name == "operation_decomposition"
    assert event.source_node == "softmax"
    assert event.original_signature is None
    assert event.resulting_signatures
    assert event.converted_initializers == ()
    assert independently_planned.rewrite_report == bundle.rewrite_report

    json_path, _ = write_execution_plan_bundle(
        bundle,
        tmp_path / "bundle.json",
        tmp_path / "weights.bin",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["rewrite_report"][0]["rewrite_name"] == (
        "operation_decomposition"
    )
    assert payload["provenance"]["rewrite_report"][0]["source_node"] == "softmax"


def test_plan_model_rejects_missing_tensor_types_before_stage_selection() -> None:
    x = Tensor("x", 1, (4,), 4)
    output = Tensor("output", 1, (4,), 4)
    node = Node(
        "relu",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Relu", x, output),
    )
    model = ImportedModel(
        Graph(
            "untyped",
            tensors=(x, output),
            nodes=(node,),
            inputs=(x,),
            outputs=(output,),
        ),
        ConstantStore(()),
    )

    with pytest.raises(
        ValueError,
        match=r"node relu has untyped tensors: x, output",
    ):
        plan_model(model, _mesh(width=1), _quiet_options())


def test_plan_model_rejects_generated_rewrite_name_collisions(tmp_path) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    intermediate = helper.make_tensor_value_info(
        "intermediate", TensorProto.FLOAT, [2, 4]
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    source = helper.make_model(
        helper.make_graph(
            (
                helper.make_node(
                    "Softmax", ("x",), ("intermediate",), name="softmax"
                ),
                helper.make_node(
                    "Relu",
                    ("intermediate",),
                    ("output",),
                    name="softmax__exp",
                ),
            ),
            "collision",
            (x,),
            (output,),
            value_info=(intermediate,),
        )
    )
    model_path = tmp_path / "collision.onnx"
    onnx.save(source, model_path)

    with pytest.raises(
        ValueError,
        match="generated node name collision: 'softmax__exp'",
    ):
        plan_model(import_onnx_model(model_path), _mesh(), _quiet_options())
