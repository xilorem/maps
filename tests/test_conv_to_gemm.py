from dataclasses import replace
import json

import numpy as np

from MAPS.arch import WorkKind, WorkSignature
from MAPS.core import Constant, ConstantStore, Graph, Node, OpKind, Tensor, TensorDType
from MAPS.core.graph import Edge
from MAPS.deployment import write_execution_plan_bundle
from MAPS.hw.chips import magia_mesh, magia_planner_options
from MAPS.importers.model import ImportedModel
from MAPS.ops.defs.conv_transforms import Im2ColPayload, OutputReformatPayload
from MAPS.ops.defs.direct_conv import Conv2DPayload
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.planner.contracts.options import SpatialMappingOptions
from MAPS.planner.plan import plan_model


def _fp16_tensor(
    name: str,
    dims: tuple[int, ...],
    *,
    initializer: bool = False,
) -> Tensor:
    return Tensor(
        name=name,
        rank=len(dims),
        dims=dims,
        elem_bytes=2,
        is_initializer=initializer,
        dtype=TensorDType.FLOAT16,
    )


def _fp16_conv_model() -> tuple[ImportedModel, np.ndarray]:
    x = _fp16_tensor("x", (1, 2, 3, 3))
    weight = _fp16_tensor("weight", (4, 2, 2, 2), initializer=True)
    bias = _fp16_tensor("bias", (4,), initializer=True)
    output = _fp16_tensor("output", (1, 4, 2, 2))
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
        name="fp16_conv",
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
    weight_values = np.arange(32, dtype="<f2").reshape(4, 2, 2, 2)
    return ImportedModel(
        graph,
        ConstantStore(
            (
                Constant(
                    "weight",
                    TensorDType.FLOAT16,
                    (4, 2, 2, 2),
                    weight_values.tobytes(),
                ),
                Constant(
                    "bias",
                    TensorDType.FLOAT16,
                    (4,),
                    np.arange(4, dtype="<f2").tobytes(),
                ),
            )
        ),
    ), weight_values


def _quiet_magia_options():
    return replace(
        magia_planner_options(enable_precision_lowering=False),
        spatial_mapping=SpatialMappingOptions(print_mapping=False),
        print_execution_plan_cost=False,
    )


def test_magia_lowers_fp16_conv_to_auditable_redmule_execution(
    tmp_path,
    capsys,
) -> None:
    model, original_weight = _fp16_conv_model()
    options = _quiet_magia_options()
    options = replace(
        options,
        workload=replace(options.workload, print_progress=True),
    )

    bundle = plan_model(
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
    write_execution_plan_bundle(bundle, output_json, output_weights)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert [
        item["rewrite_name"]
        for item in payload["provenance"]["rewrite_report"]
    ] == ["conv_to_gemm"]


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

    assert WorkKind.IM2COL.fallback_kind is WorkKind.IM2COL
    assert WorkKind.OUTPUT_REFORMAT.fallback_kind is WorkKind.OUTPUT_REFORMAT
    assert tile.assigned_device(fp16_im2col).name == "core"
    assert tile.assigned_device(fp16_output_reformat).name == "core"
