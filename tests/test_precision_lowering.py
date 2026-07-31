from dataclasses import replace
import json

import numpy as np
import pytest

from MAPS.arch import FixedDeviceAssignment, WorkKind, WorkSignature
from MAPS.core import Constant, ConstantStore, Graph, Node, OpKind, Tensor, TensorDType
from MAPS.core.graph import Edge
from MAPS.deployment import write_execution_plan_bundle
from MAPS.hw.chips import magia_mesh, magia_planner_options
from MAPS.importers.model import ImportedModel
from maps.operations.cast import CastPayload
from maps.operations.elementwise import UnaryElementwisePayload
from maps.operations.gemm import GemmPayload
from MAPS.planner.contracts.options import PlannerOptions, SpatialMappingOptions
from MAPS.planner.plan import plan_model


def _fp32_tensor(
    name: str,
    dims: tuple[int, ...],
    *,
    initializer: bool = False,
) -> Tensor:
    return Tensor(
        name=name,
        rank=len(dims),
        dims=dims,
        elem_bytes=4,
        is_initializer=initializer,
        dtype=TensorDType.FLOAT32,
    )


def _gemm_model(*, with_bias: bool = False) -> ImportedModel:
    x = _fp32_tensor("x", (2, 3))
    weight = _fp32_tensor("weight", (3, 4), initializer=True)
    output = _fp32_tensor("output", (2, 4))
    bias = _fp32_tensor("bias", (4,), initializer=True) if with_bias else None
    inputs = (x, weight) + ((bias,) if bias is not None else ())
    gemm = Node(
        name="gemm",
        kind=OpKind.GEMM,
        inputs=inputs,
        outputs=(output,),
        payload=GemmPayload(x=x, w=weight, y=bias, output=output),
    )
    initializers = (weight,) + ((bias,) if bias is not None else ())
    graph = Graph(
        name="fp32_gemm",
        tensors=inputs + (output,),
        nodes=(gemm,),
        edges=tuple(Edge(tensor, None, gemm) for tensor in inputs)
        + (Edge(output, gemm, None),),
        inputs=(x,),
        outputs=(output,),
        initializers=initializers,
    )
    values = np.arange(12, dtype="<f4").reshape(3, 4)
    constants = [
        Constant("weight", TensorDType.FLOAT32, (3, 4), values.tobytes())
    ]
    if bias is not None:
        constants.append(
            Constant(
                "bias",
                TensorDType.FLOAT32,
                (4,),
                np.arange(4, dtype="<f4").tobytes(),
            )
        )
    return ImportedModel(
        graph=graph,
        constants=ConstantStore(tuple(constants)),
    )


def _quiet_magia_options(*, enable_precision_lowering: bool = True):
    return replace(
        magia_planner_options(
            enable_precision_lowering=enable_precision_lowering,
        ),
        spatial_mapping=SpatialMappingOptions(print_mapping=False),
        print_execution_plan_cost=False,
    )


def _without_assignment(mesh, signature: WorkSignature):
    tiles = tuple(
        replace(
            tile,
            device_assignment=FixedDeviceAssignment(
                {
                    assigned_signature: device_name
                    for assigned_signature, device_name in (
                        tile.device_assignment.assignments.items()
                    )
                    if assigned_signature != signature
                }
            ),
        )
        for tile in mesh.tiles
    )
    return replace(mesh, tiles=tiles)


def _consecutive_gemm_model() -> ImportedModel:
    x = _fp32_tensor("x", (2, 3))
    first_weight = _fp32_tensor("first_weight", (3, 4), initializer=True)
    intermediate = _fp32_tensor("intermediate", (2, 4))
    second_weight = _fp32_tensor("second_weight", (4, 5), initializer=True)
    output = _fp32_tensor("output", (2, 5))
    first = Node(
        "first",
        OpKind.GEMM,
        (x, first_weight),
        (intermediate,),
        GemmPayload(x, first_weight, None, intermediate),
    )
    second = Node(
        "second",
        OpKind.GEMM,
        (intermediate, second_weight),
        (output,),
        GemmPayload(intermediate, second_weight, None, output),
    )
    graph = Graph(
        "consecutive_gemms",
        tensors=(x, first_weight, intermediate, second_weight, output),
        nodes=(first, second),
        edges=(
            Edge(x, None, first),
            Edge(first_weight, None, first),
            Edge(intermediate, first, second),
            Edge(second_weight, None, second),
            Edge(output, second, None),
        ),
        inputs=(x,),
        outputs=(output,),
        initializers=(first_weight, second_weight),
    )
    return ImportedModel(
        graph,
        ConstantStore(
            (
                Constant(
                    "first_weight",
                    TensorDType.FLOAT32,
                    (3, 4),
                    np.arange(12, dtype="<f4").tobytes(),
                ),
                Constant(
                    "second_weight",
                    TensorDType.FLOAT32,
                    (4, 5),
                    np.arange(20, dtype="<f4").tobytes(),
                ),
            )
        ),
    )


def _fanout_model() -> ImportedModel:
    x = _fp32_tensor("x", (2, 3))
    weight = _fp32_tensor("weight", (3, 4), initializer=True)
    gemm_output = _fp32_tensor("gemm_output", (2, 4))
    relu_output = _fp32_tensor("relu_output", (2, 3))
    gemm = Node(
        "gemm",
        OpKind.GEMM,
        (x, weight),
        (gemm_output,),
        GemmPayload(x, weight, None, gemm_output),
    )
    relu = Node(
        "relu",
        OpKind.ELEMENTWISE,
        (x,),
        (relu_output,),
        UnaryElementwisePayload("Relu", x, relu_output),
    )
    graph = Graph(
        "fanout",
        tensors=(x, weight, gemm_output, relu_output),
        nodes=(gemm, relu),
        edges=(
            Edge(x, None, gemm),
            Edge(weight, None, gemm),
            Edge(gemm_output, gemm, None),
            Edge(x, None, relu),
            Edge(relu_output, relu, None),
        ),
        inputs=(x,),
        outputs=(gemm_output, relu_output),
        initializers=(weight,),
    )
    return ImportedModel(
        graph,
        ConstantStore(
            (
                Constant(
                    "weight",
                    TensorDType.FLOAT32,
                    (3, 4),
                    np.arange(12, dtype="<f4").tobytes(),
                ),
            )
        ),
    )


def _fp16_gemm_model() -> ImportedModel:
    x = Tensor("x", 2, (2, 3), 2, dtype=TensorDType.FLOAT16)
    weight = Tensor(
        "weight",
        2,
        (3, 4),
        2,
        is_initializer=True,
        dtype=TensorDType.FLOAT16,
    )
    output = Tensor("output", 2, (2, 4), 2, dtype=TensorDType.FLOAT16)
    gemm = Node(
        "gemm",
        OpKind.GEMM,
        (x, weight),
        (output,),
        GemmPayload(x, weight, None, output),
    )
    return ImportedModel(
        Graph(
            "fp16_gemm",
            tensors=(x, weight, output),
            nodes=(gemm,),
            edges=(
                Edge(x, None, gemm),
                Edge(weight, None, gemm),
                Edge(output, gemm, None),
            ),
            inputs=(x,),
            outputs=(output,),
            initializers=(weight,),
        ),
        ConstantStore(
            (
                Constant(
                    "weight",
                    TensorDType.FLOAT16,
                    (3, 4),
                    np.arange(12, dtype="<f2").tobytes(),
                ),
            )
        ),
    )


def _initializer_fanout_model() -> ImportedModel:
    imported = _gemm_model()
    weight = imported.graph.initializers[0]
    relu_output = _fp32_tensor("relu_output", weight.dims)
    relu = Node(
        "relu",
        OpKind.ELEMENTWISE,
        (weight,),
        (relu_output,),
        UnaryElementwisePayload("Relu", weight, relu_output),
    )
    graph = replace(
        imported.graph,
        name="initializer_fanout",
        tensors=imported.graph.tensors + (relu_output,),
        nodes=imported.graph.nodes + (relu,),
        edges=imported.graph.edges
        + (
            Edge(weight, None, relu),
            Edge(relu_output, relu, None),
        ),
        outputs=imported.graph.outputs + (relu_output,),
    )
    return replace(imported, graph=graph)


def _shared_lowered_initializer_model() -> ImportedModel:
    imported = _gemm_model()
    x, weight, first_output = imported.graph.tensors
    second_output = _fp32_tensor("second_output", first_output.dims)
    first = imported.graph.nodes[0]
    second = Node(
        "second",
        OpKind.GEMM,
        (x, weight),
        (second_output,),
        GemmPayload(x, weight, None, second_output),
    )
    graph = replace(
        imported.graph,
        name="shared_lowered_initializer",
        tensors=imported.graph.tensors + (second_output,),
        nodes=(first, second),
        edges=(
            Edge(x, None, first),
            Edge(weight, None, first),
            Edge(first_output, first, None),
            Edge(x, None, second),
            Edge(weight, None, second),
            Edge(second_output, second, None),
        ),
        outputs=(first_output, second_output),
    )
    return replace(imported, graph=graph)


def test_magia_precision_lowers_fp32_gemm_and_restores_its_output() -> None:
    bundle = plan_model(
        _gemm_model(),
        magia_mesh(width=3, height=1),
        _quiet_magia_options(),
    )

    assert [node.name for node in bundle.graph.nodes] == [
        "gemm__input_0_cast_float16",
        "gemm",
        "gemm__output_0_cast_float32",
    ]
    input_cast, lowered_gemm, output_cast = bundle.graph.nodes
    assert isinstance(input_cast.payload, CastPayload)
    assert isinstance(lowered_gemm.payload, GemmPayload)
    assert isinstance(output_cast.payload, CastPayload)
    assert [tensor.dtype for tensor in lowered_gemm.inputs] == [
        TensorDType.FLOAT16,
        TensorDType.FLOAT16,
    ]
    assert lowered_gemm.outputs[0].dtype is TensorDType.FLOAT16
    assert output_cast.outputs == bundle.graph.outputs
    assert bundle.graph.outputs[0].dtype is TensorDType.FLOAT32

    weight = next(tensor for tensor in bundle.graph.initializers if tensor.name == "weight")
    constant = bundle.constants.get("weight")
    assert weight.dtype is TensorDType.FLOAT16
    assert weight.elem_bytes == 2
    assert constant.dtype is TensorDType.FLOAT16
    assert np.frombuffer(constant.data, dtype="<f2").tolist() == list(
        np.arange(12, dtype="<f2")
    )

    layers = tuple(
        layer
        for stage in bundle.execution_plan.stages
        for layer in stage.layers
    )
    assert [layer.node.name for layer in layers] == [node.name for node in bundle.graph.nodes]
    assert [layer.device_name for layer in layers] == ["spatz", "redmule", "spatz"]

    assert len(bundle.rewrite_report.events) == 1
    event = bundle.rewrite_report.events[0]
    assert event.rewrite_name == "precision_lowering"
    assert event.source_node == "gemm"
    assert event.original_signature == WorkSignature(
        WorkKind.GEMM,
        (TensorDType.FLOAT32, TensorDType.FLOAT32),
        (TensorDType.FLOAT32,),
    )
    assert event.resulting_signatures == tuple(
        WorkSignature.from_node(node) for node in bundle.graph.nodes
    )
    assert event.converted_initializers == ("weight",)


def test_disabling_magia_precision_lowering_retains_fp32_core_execution() -> None:
    imported = _gemm_model()

    bundle = plan_model(
        imported,
        magia_mesh(width=1, height=1),
        _quiet_magia_options(enable_precision_lowering=False),
    )

    assert [node.name for node in bundle.graph.nodes] == ["gemm"]
    assert all(tensor.dtype is TensorDType.FLOAT32 for tensor in bundle.graph.tensors)
    assert bundle.constants == imported.constants
    assert bundle.execution_plan.stages[0].layers[0].device_name == "core"
    assert bundle.rewrite_report.events == ()


def test_magia_precision_lowers_initializer_bias_without_a_runtime_cast() -> None:
    bundle = plan_model(
        _gemm_model(with_bias=True),
        magia_mesh(width=3, height=1),
        _quiet_magia_options(),
    )

    assert [node.name for node in bundle.graph.nodes] == [
        "gemm__input_0_cast_float16",
        "gemm",
        "gemm__output_0_cast_float32",
    ]
    gemm = bundle.graph.nodes[1]
    assert [tensor.dtype for tensor in gemm.inputs] == [TensorDType.FLOAT16] * 3
    assert bundle.constants.get("weight").dtype is TensorDType.FLOAT16
    assert bundle.constants.get("bias").dtype is TensorDType.FLOAT16
    assert bundle.rewrite_report.events[0].converted_initializers == (
        "weight",
        "bias",
    )
    layers = tuple(
        layer for stage in bundle.execution_plan.stages for layer in stage.layers
    )
    assert [layer.device_name for layer in layers] == ["spatz", "redmule", "spatz"]


def test_consecutive_lowered_gemms_restore_fp32_between_operations() -> None:
    bundle = plan_model(
        _consecutive_gemm_model(),
        magia_mesh(),
        _quiet_magia_options(),
    )

    assert [node.name for node in bundle.graph.nodes] == [
        "first__input_0_cast_float16",
        "first",
        "first__output_0_cast_float32",
        "second__input_0_cast_float16",
        "second",
        "second__output_0_cast_float32",
    ]
    first_restore = bundle.graph.nodes[2]
    second_lower = bundle.graph.nodes[3]
    assert first_restore.outputs[0].name == "intermediate"
    assert first_restore.outputs[0].dtype is TensorDType.FLOAT32
    assert second_lower.inputs == first_restore.outputs
    assert second_lower.outputs[0].dtype is TensorDType.FLOAT16
    assert len(bundle.rewrite_report.events) == 2
    assert [event.source_node for event in bundle.rewrite_report.events] == [
        "first",
        "second",
    ]


def test_precision_lowering_casts_only_the_rewritten_fanout_path() -> None:
    bundle = plan_model(
        _fanout_model(),
        magia_mesh(),
        _quiet_magia_options(),
    )

    relu = next(node for node in bundle.graph.nodes if node.name == "relu")
    gemm_input_cast = bundle.graph.nodes[0]
    assert relu.inputs[0].name == "x"
    assert relu.inputs[0].dtype is TensorDType.FLOAT32
    assert gemm_input_cast.inputs == relu.inputs
    assert gemm_input_cast.outputs[0].dtype is TensorDType.FLOAT16
    assert sum(
        isinstance(node.payload, CastPayload) for node in bundle.graph.nodes
    ) == 2


def test_precision_lowering_option_defaults_are_target_specific() -> None:
    assert not PlannerOptions().graph_rewrites.enable_precision_lowering
    assert magia_planner_options().graph_rewrites.enable_precision_lowering
    assert not magia_planner_options(
        enable_precision_lowering=False
    ).graph_rewrites.enable_precision_lowering


def test_enabled_precision_lowering_does_not_infer_a_missing_recipe() -> None:
    mesh = replace(
        magia_mesh(width=1, height=1),
        precision_lowering_recipes=(),
    )

    bundle = plan_model(_gemm_model(), mesh, _quiet_magia_options())

    assert [node.name for node in bundle.graph.nodes] == ["gemm"]
    assert bundle.execution_plan.stages[0].layers[0].device_name == "core"
    assert bundle.rewrite_report.events == ()


def test_magia_preset_leaves_fp16_gemm_native_to_redmule() -> None:
    imported = _fp16_gemm_model()

    bundle = plan_model(
        imported,
        magia_mesh(width=1, height=1),
        _quiet_magia_options(),
    )

    assert bundle.graph.nodes == imported.graph.nodes
    assert bundle.constants == imported.constants
    assert bundle.execution_plan.stages[0].layers[0].device_name == "redmule"
    assert bundle.rewrite_report.events == ()


@pytest.mark.parametrize(
    "missing_signature",
    (
        WorkSignature(
            WorkKind.CAST,
            (TensorDType.FLOAT32,),
            (TensorDType.FLOAT16,),
        ),
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT16, TensorDType.FLOAT16),
            (TensorDType.FLOAT16,),
        ),
    ),
)
def test_precision_lowering_requires_cast_and_target_assignments(
    missing_signature: WorkSignature,
) -> None:
    mesh = _without_assignment(
        magia_mesh(width=3, height=1),
        missing_signature,
    )

    with pytest.raises(ValueError) as error:
        plan_model(_gemm_model(), mesh, _quiet_magia_options())

    message = str(error.value)
    assert "node gemm cannot apply Precision Lowering" in message
    assert repr(missing_signature) in message
    assert "configured assignment=None" in message
    assert "tile 0" in message


def test_precision_lowering_rejects_inconsistent_initializer_bytes() -> None:
    imported = _gemm_model()
    malformed = replace(
        imported,
        constants=ConstantStore(
            (
                replace(
                    imported.constants.get("weight"),
                    data=bytes(4),
                ),
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"constant 'weight' has 4 bytes; expected 48",
    ):
        plan_model(
            malformed,
            magia_mesh(width=3, height=1),
            _quiet_magia_options(),
        )


def test_precision_lowering_serializes_deterministic_artifacts_and_provenance(
    tmp_path,
) -> None:
    first = plan_model(
        _gemm_model(),
        magia_mesh(width=3, height=1),
        _quiet_magia_options(),
    )
    second = plan_model(
        _gemm_model(),
        magia_mesh(width=3, height=1),
        _quiet_magia_options(),
    )

    first_json, first_weights = write_execution_plan_bundle(
        first,
        tmp_path / "first" / "model.json",
        tmp_path / "first" / "weights.bin",
    )
    second_json, second_weights = write_execution_plan_bundle(
        second,
        tmp_path / "second" / "model.json",
        tmp_path / "second" / "weights.bin",
    )

    assert first.graph == second.graph
    assert first.rewrite_report == second.rewrite_report
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_weights.read_bytes() == second_weights.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    event = payload["provenance"]["rewrite_report"][0]
    assert event == {
        "converted_initializers": ["weight"],
        "original_signature": {
            "input_dtypes": ["float32", "float32"],
            "output_dtypes": ["float32"],
            "work_kind": "GEMM",
        },
        "resulting_signatures": [
            {
                "input_dtypes": ["float32"],
                "output_dtypes": ["float16"],
                "work_kind": "CAST",
            },
            {
                "input_dtypes": ["float16", "float16"],
                "output_dtypes": ["float16"],
                "work_kind": "GEMM",
            },
            {
                "input_dtypes": ["float16"],
                "output_dtypes": ["float32"],
                "work_kind": "CAST",
            },
        ],
        "rewrite_name": "precision_lowering",
        "source_node": "gemm",
    }


def test_precision_lowering_rejects_generated_name_collisions() -> None:
    imported = _gemm_model()
    collision = Tensor(
        "gemm__input_0_float16",
        1,
        (1,),
        2,
        dtype=TensorDType.FLOAT16,
    )
    imported = replace(
        imported,
        graph=replace(
            imported.graph,
            tensors=imported.graph.tensors + (collision,),
        ),
    )

    with pytest.raises(
        ValueError,
        match="generated tensor name collision: 'gemm__input_0_float16'",
    ):
        plan_model(
            imported,
            magia_mesh(width=3, height=1),
            _quiet_magia_options(),
        )


def test_shared_initializer_is_cloned_for_only_the_lowered_fanout_path() -> None:
    bundle = plan_model(
        _initializer_fanout_model(),
        magia_mesh(),
        _quiet_magia_options(),
    )

    gemm = next(node for node in bundle.graph.nodes if node.name == "gemm")
    relu = next(node for node in bundle.graph.nodes if node.name == "relu")
    assert gemm.inputs[1].name == "gemm__input_1_float16"
    assert gemm.inputs[1].dtype is TensorDType.FLOAT16
    assert relu.inputs[0].name == "weight"
    assert relu.inputs[0].dtype is TensorDType.FLOAT32
    assert [tensor.name for tensor in bundle.graph.initializers] == [
        "weight",
        "gemm__input_1_float16",
    ]
    assert bundle.constants.get("weight").dtype is TensorDType.FLOAT32
    assert bundle.constants.get("gemm__input_1_float16").dtype is TensorDType.FLOAT16
    assert bundle.rewrite_report.events[0].converted_initializers == (
        "gemm__input_1_float16",
    )


def test_shared_initializer_is_converted_once_for_all_lowered_consumers() -> None:
    bundle = plan_model(
        _shared_lowered_initializer_model(),
        magia_mesh(),
        _quiet_magia_options(),
    )

    gemms = tuple(
        node for node in bundle.graph.nodes if isinstance(node.payload, GemmPayload)
    )
    assert len(gemms) == 2
    assert gemms[0].inputs[1] is gemms[1].inputs[1]
    assert gemms[0].inputs[1].name == "weight"
    assert gemms[0].inputs[1].dtype is TensorDType.FLOAT16
    assert [constant.name for constant in bundle.constants.constants] == ["weight"]
    assert bundle.constants.get("weight").dtype is TensorDType.FLOAT16
    assert [
        event.converted_initializers for event in bundle.rewrite_report.events
    ] == [("weight",), ()]
