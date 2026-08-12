from dataclasses import replace
import json

import numpy as np
import pytest

from maps.deployment import write_deployment_bundle
from maps.graph import (
    Constant,
    ConstantStore,
    Edge,
    Graph,
    ImportedModel,
    Node,
    OpKind,
    Tensor,
    TensorDType,
)
from maps.hardware import WorkKind, WorkSignature
from maps.operations.convolution import Conv2DPayload
from maps.operations.convolution_transforms import Im2ColPayload, OutputReformatPayload
from maps.operations.gemm import GemmPayload
from maps.planning.allocation.candidates import (
    permanent_l1_allocation_for_stage,
    resolve_stage_layouts,
)
from maps.planning.execution_plan import LocalInput
from maps.target.magia import build_mesh as magia_mesh
from tests.target.workflows import magia_workflow_options, plan_magia_model


def _conv_tensor(
    name: str,
    dims: tuple[int, ...],
    dtype: TensorDType,
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


def _conv_model(
    dtype: TensorDType,
    *,
    batch: int = 1,
) -> tuple[ImportedModel, np.ndarray]:
    x = _conv_tensor("x", (batch, 2, 3, 3), dtype)
    weight = _conv_tensor(
        "weight", (4, 2, 2, 2), dtype, initializer=True
    )
    bias = _conv_tensor("bias", (4,), dtype, initializer=True)
    output = _conv_tensor("output", (batch, 4, 2, 2), dtype)
    conv = Node(
        name="conv",
        kind=OpKind.CONV,
        inputs=(x, weight, bias),
        outputs=(output,),
        payload=Conv2DPayload(
            x=x,
            w=weight,
            b=bias,
            output=output,
        ),
    )
    graph = Graph(
        name=f"{dtype.value}_conv",
        tensors=(x, weight, bias, output),
        nodes=(conv,),
        edges=(
            Edge(x, None, conv),
            Edge(weight, None, conv),
            Edge(bias, None, conv),
            Edge(output, conv, None),
        ),
        inputs=(x,),
        outputs=(output,),
        initializers=(weight, bias),
    )
    numpy_dtype = {
        TensorDType.FLOAT16: np.dtype("<f2"),
        TensorDType.FLOAT32: np.dtype("<f4"),
    }[dtype]
    weight_values = np.arange(32, dtype=numpy_dtype).reshape(4, 2, 2, 2)
    return ImportedModel(
        graph,
        ConstantStore(
            (
                Constant(
                    "weight",
                    dtype,
                    (4, 2, 2, 2),
                    weight_values.tobytes(),
                ),
                Constant(
                    "bias",
                    dtype,
                    (4,),
                    np.arange(4, dtype=numpy_dtype).tobytes(),
                ),
            )
        ),
    ), weight_values


def _quiet_magia_options():
    return magia_workflow_options(
        enable_precision_lowering=False,
    )


def test_conv_to_gemm_rejects_multiple_samples_at_specialization() -> None:
    model, _ = _conv_model(TensorDType.FLOAT16, batch=2)

    with pytest.raises(
        ValueError,
        match="node conv Conv-to-GEMM supports only batch size 1, got 2",
    ):
        plan_magia_model(
            model,
            magia_mesh(width=1, height=1),
            _quiet_magia_options(),
        )


def test_magia_lowers_fp16_conv_to_auditable_redmule_execution(
    tmp_path,
    capsys,
) -> None:
    model, original_weight = _conv_model(TensorDType.FLOAT16)
    options = _quiet_magia_options()
    options = replace(
        options,
        allocation=replace(options.allocation, print_progress=True),
    )

    bundle = plan_magia_model(
        model,
        magia_mesh(width=1, height=1),
        options,
    )
    diagnostics = capsys.readouterr().out
    assert "provisional Conv-to-GEMM transform estimate" in diagnostics
    assert "bytes-read/bytes-written/core-L1" in diagnostics

    assert [node.name for node in bundle.graph.nodes] == [
        "conv__input_0_im2col_float16",
        "conv__output_0_gemm_float16",
        "conv__output_0_reformat_float16",
    ]
    im2col, gemm, output_reformat = bundle.graph.nodes
    assert isinstance(im2col.payload, Im2ColPayload)
    assert isinstance(gemm.payload, GemmPayload)
    assert isinstance(output_reformat.payload, OutputReformatPayload)
    assert im2col.outputs[0].dims == (4, 8)
    assert gemm.inputs[1].dims == (8, 4)
    assert gemm.inputs[2].name == "bias"
    assert gemm.outputs[0].dims == (4, 4)
    assert output_reformat.outputs == bundle.graph.outputs
    assert all("weight_pack" not in node.name for node in bundle.graph.nodes)
    assert all(node.payload.work_kind is not WorkKind.ADD for node in bundle.graph.nodes)

    packed_weight = bundle.constants.get("weight")
    expected_weight = original_weight.transpose(1, 2, 3, 0).reshape(8, 4)
    assert packed_weight.shape == (8, 4)
    np.testing.assert_array_equal(
        np.frombuffer(packed_weight.data, dtype="<f2").reshape(8, 4),
        expected_weight,
    )
    assert next(
        tensor for tensor in bundle.graph.initializers if tensor.name == "weight"
    ).dims == (8, 4)

    layers = tuple(
        layer
        for stage in bundle.execution_plan.stages
        for layer in stage.layers
    )
    assert [layer.device_name for layer in layers] == ["core", "redmule", "core"]
    assert [layer.node.name for layer in layers] == [
        "conv__input_0_im2col_float16",
        "conv__output_0_gemm_float16",
        "conv__output_0_reformat_float16",
    ]

    tile = bundle.execution_plan.mesh.tile(0, 0)
    transform_costs = []
    for node in (im2col, output_reformat):
        layouts = node.payload.output_layouts(
            bundle.execution_plan.stages[0].submesh,
        )
        tile_work = node.payload.build_tile_work(layouts, tile)
        cost_model = node.payload.cost_model
        transform_costs.append(
            cost_model.cost(tile_work, tile, tile.device_by_name("core"))
        )
        assert "provisional" in cost_model.diagnostic_label
        assert "bytes-read/bytes-written/core-L1" in cost_model.diagnostic_label
    assert all(cost >= 0 for cost in transform_costs)

    event = next(
        event
        for event in bundle.rewrite_report.events
        if event.rewrite_name == "conv_to_gemm"
    )
    assert event.source_node == "conv"
    assert event.original_signature == WorkSignature(
        WorkKind.CONV2D,
        (TensorDType.FLOAT16,) * 3,
        (TensorDType.FLOAT16,),
    )
    assert [signature.work_kind for signature in event.resulting_signatures] == [
        WorkKind.IM2COL,
        WorkKind.GEMM,
        WorkKind.OUTPUT_REFORMAT,
    ]
    assert event.converted_initializers == ("weight",)

    output_json = tmp_path / "conv.json"
    output_weights = tmp_path / "conv.weights.bin"
    write_deployment_bundle(bundle, output_json, output_weights)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert [
        item["rewrite_name"]
        for item in payload["provenance"]["rewrite_report"]
    ] == ["conv_to_gemm"]


def test_magia_composes_fp32_conv_lowering_with_precision_lowering(
    tmp_path,
) -> None:
    model, original_weight = _conv_model(TensorDType.FLOAT32)
    options = magia_workflow_options()

    first = plan_magia_model(model, magia_mesh(width=1, height=1), options)
    second = plan_magia_model(model, magia_mesh(width=1, height=1), options)

    assert [node.payload.work_kind for node in first.graph.nodes] == [
        WorkKind.IM2COL,
        WorkKind.CAST,
        WorkKind.GEMM,
        WorkKind.CAST,
        WorkKind.OUTPUT_REFORMAT,
    ]
    assert [node.name for node in first.graph.nodes] == [
        "conv__input_0_im2col_float32",
        "conv__output_0_gemm_float32__input_0_cast_float16",
        "conv__output_0_gemm_float32",
        "conv__output_0_gemm_float32__output_0_cast_float32",
        "conv__output_0_reformat_float32",
    ]
    im2col, activation_cast, gemm, output_cast, output_reformat = first.graph.nodes
    assert im2col.inputs[0].dtype is TensorDType.FLOAT32
    assert activation_cast.outputs[0].dtype is TensorDType.FLOAT16
    assert isinstance(gemm.payload, GemmPayload)
    assert [tensor.dtype for tensor in gemm.inputs] == [TensorDType.FLOAT16] * 3
    assert gemm.outputs[0].dtype is TensorDType.FLOAT16
    assert output_cast.outputs[0].dtype is TensorDType.FLOAT32
    assert output_reformat.outputs == first.graph.outputs
    assert first.graph.outputs[0].dtype is TensorDType.FLOAT32
    assert activation_cast.attributes == {}
    assert output_cast.attributes == {}
    assert im2col.source_operation == "conv"
    assert gemm.source_operation == "conv"
    assert output_reformat.source_operation == "conv"

    packed_weight = first.constants.get("weight")
    assert packed_weight.dtype is TensorDType.FLOAT16
    assert packed_weight.shape == (8, 4)
    np.testing.assert_array_equal(
        np.frombuffer(packed_weight.data, dtype="<f2").reshape(8, 4),
        original_weight.transpose(1, 2, 3, 0).reshape(8, 4),
    )
    assert first.constants.get("bias").dtype is TensorDType.FLOAT16
    assert all("weight_pack" not in node.name for node in first.graph.nodes)

    layers = tuple(
        layer for stage in first.execution_plan.stages for layer in stage.layers
    )
    assert [layer.device_name for layer in layers] == [
        "core",
        "spatz",
        "redmule",
        "spatz",
        "core",
    ]
    assert [event.rewrite_name for event in first.rewrite_report.events] == [
        "conv_to_gemm",
        "precision_lowering",
    ]
    conv_event, precision_event = first.rewrite_report.events
    assert conv_event.converted_initializers == ("weight",)
    assert precision_event.source_node == "conv__output_0_gemm_float32"
    assert precision_event.original_signature == WorkSignature(
        WorkKind.GEMM,
        (TensorDType.FLOAT32,) * 3,
        (TensorDType.FLOAT32,),
    )
    assert precision_event.resulting_signatures == (
        WorkSignature(
            WorkKind.CAST,
            (TensorDType.FLOAT32,),
            (TensorDType.FLOAT16,),
        ),
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT16,) * 3,
            (TensorDType.FLOAT16,),
        ),
        WorkSignature(
            WorkKind.CAST,
            (TensorDType.FLOAT16,),
            (TensorDType.FLOAT32,),
        ),
    )
    assert precision_event.converted_initializers == ("weight", "bias")
    assert first.rewrite_report == second.rewrite_report
    assert first.graph == second.graph
    assert first.constants == second.constants

    first_json, first_weights = write_deployment_bundle(
        first,
        tmp_path / "first" / "model.json",
        tmp_path / "first" / "model.weights.bin",
    )
    second_json, second_weights = write_deployment_bundle(
        second,
        tmp_path / "second" / "model.json",
        tmp_path / "second" / "model.weights.bin",
    )
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_weights.read_bytes() == second_weights.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert [
        event["rewrite_name"]
        for event in payload["provenance"]["rewrite_report"]
    ] == ["conv_to_gemm", "precision_lowering"]
    assert payload["provenance"]["rewrite_report"][1][
        "converted_initializers"
    ] == ["weight", "bias"]
    assert payload["provenance"]["rewrite_report"][1][
        "resulting_signatures"
    ] == [
        {
            "input_dtypes": ["float32"],
            "output_dtypes": ["float16"],
            "work_kind": "CAST",
        },
        {
            "input_dtypes": ["float16", "float16", "float16"],
            "output_dtypes": ["float16"],
            "work_kind": "GEMM",
        },
        {
            "input_dtypes": ["float16"],
            "output_dtypes": ["float32"],
            "work_kind": "CAST",
        },
    ]


@pytest.mark.parametrize(
    ("dtype", "enable_precision_lowering", "matrix_layer_indexes"),
    (
        (TensorDType.FLOAT16, False, (0, 1)),
        (TensorDType.FLOAT32, True, (0, 1, 2, 3)),
        (TensorDType.FLOAT32, False, (0, 1)),
    ),
)
def test_all_conv_to_gemm_precision_paths_preserve_spatial_ownership(
    dtype: TensorDType,
    enable_precision_lowering: bool,
    matrix_layer_indexes: tuple[int, ...],
) -> None:
    model, _ = _conv_model(dtype)
    bundle = plan_magia_model(
        model,
        magia_mesh(width=2, height=2),
        magia_workflow_options(
            enable_precision_lowering=enable_precision_lowering,
        ),
    )

    assert len(bundle.execution_plan.stages) == 1
    stage = bundle.execution_plan.stages[0]
    assert len(stage.submesh.tile_ids) > 1
    matrix_layouts = tuple(
        stage.layers[index].outputs[0].layout
        for index in matrix_layer_indexes
    )
    assert all(layout.mesh_y.tensor_axis == 0 for layout in matrix_layouts)
    assert all(layout.mesh_y.shard_granularity == 2 for layout in matrix_layouts)
    final_layout = stage.layers[-1].outputs[0].layout
    assert final_layout.mesh_x.tensor_axis == 1
    assert final_layout.mesh_y.tensor_axis == 2
    assert final_layout.mesh_y.shard_granularity == 1
    assert all(layer.collective_groups == () for layer in stage.layers)
    assert all(
        isinstance(layer_input.source, LocalInput)
        for layer in stage.layers[1:]
        for layer_input in layer.inputs
        if layer_input.tensor_id
        in {
            output.tensor_id
            for producer in stage.layers
            for output in producer.outputs
        }
    )


def test_public_planning_selects_spatial_rows_and_lowers_per_tile_l1() -> None:
    model, _ = _conv_model(TensorDType.FLOAT32)
    bundle = plan_magia_model(
        model,
        magia_mesh(width=2, height=2),
        _quiet_magia_options(),
    )

    stage = bundle.execution_plan.stages[0]
    layouts = tuple(
        tuple(output.layout for output in layer.outputs)
        for layer in stage.layers
    )
    assert layouts[-1][0].effective_logical_height == 2
    spatial_l1 = permanent_l1_allocation_for_stage(
        tuple(layer.node for layer in stage.layers),
        layouts,
        stage.submesh,
        frozenset(bundle.graph.initializers),
    )
    channel_only_layouts = resolve_stage_layouts(
        tuple(layer.node for layer in stage.layers),
        stage.submesh,
        (len(stage.submesh.tile_ids), 1),
    )
    channel_only_l1 = permanent_l1_allocation_for_stage(
        tuple(layer.node for layer in stage.layers),
        channel_only_layouts,
        stage.submesh,
        frozenset(bundle.graph.initializers),
    )

    assert spatial_l1 < channel_only_l1


def test_magia_keeps_lowered_fp32_conv_on_core_when_precision_is_disabled() -> None:
    model, original_weight = _conv_model(TensorDType.FLOAT32)

    bundle = plan_magia_model(
        model,
        magia_mesh(width=1, height=1),
        _quiet_magia_options(),
    )

    assert [node.payload.work_kind for node in bundle.graph.nodes] == [
        WorkKind.IM2COL,
        WorkKind.GEMM,
        WorkKind.OUTPUT_REFORMAT,
    ]
    assert all(
        tensor.dtype is TensorDType.FLOAT32
        for node in bundle.graph.nodes
        for tensor in node.inputs + node.outputs
    )
    assert bundle.graph.outputs[0].dtype is TensorDType.FLOAT32
    packed_weight = bundle.constants.get("weight")
    assert packed_weight.dtype is TensorDType.FLOAT32
    assert packed_weight.shape == (8, 4)
    np.testing.assert_array_equal(
        np.frombuffer(packed_weight.data, dtype="<f4").reshape(8, 4),
        original_weight.transpose(1, 2, 3, 0).reshape(8, 4),
    )
    layers = tuple(
        layer
        for stage in bundle.execution_plan.stages
        for layer in stage.layers
    )
    assert [layer.device_name for layer in layers] == ["core", "core", "core"]
    assert [event.rewrite_name for event in bundle.rewrite_report.events] == [
        "conv_to_gemm"
    ]


def test_composed_conv_rewrite_rejects_generated_name_collisions() -> None:
    model, _ = _conv_model(TensorDType.FLOAT32)
    collision = Tensor(
        "conv__input_0_im2col_output_float32",
        1,
        (1,),
        4,
        dtype=TensorDType.FLOAT32,
    )
    model = replace(
        model,
        graph=replace(model.graph, tensors=model.graph.tensors + (collision,)),
    )

    with pytest.raises(
        ValueError,
        match=(
            "generated tensor name collision: "
            "'conv__input_0_im2col_output_float32'"
        ),
    ):
        plan_magia_model(
            model,
            magia_mesh(width=1, height=1),
            magia_workflow_options(),
        )


def test_conv_data_transforms_require_exact_magia_core_capabilities() -> None:
    tile = magia_mesh(width=1, height=1).tile(0, 0)
    fp16_im2col = WorkSignature(
        WorkKind.IM2COL,
        (TensorDType.FLOAT16,),
        (TensorDType.FLOAT16,),
    )
    fp16_output_reformat = WorkSignature(
        WorkKind.OUTPUT_REFORMAT,
        (TensorDType.FLOAT16,),
        (TensorDType.FLOAT16,),
    )

    assert tile.assigned_device(fp16_im2col).name == "core"
    assert tile.assigned_device(fp16_output_reformat).name == "core"
