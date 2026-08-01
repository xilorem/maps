"""Tests for ONNX lowering into the shared graph IR."""

import pytest

from maps.hardware import WorkKind
from maps.graph import OpKind, TensorDType, decompose_graph
from maps.graph.onnx.parser import onnx_dtype_elem_bytes, parse_graph
from maps.graph.onnx.operations import ONNX_OPERATION_CONVERTERS
from maps.operations import SoftmaxPayload
from maps.operations.collective import AllReducePayload
from maps.operations.convolution import ConvPayload
from maps.operations.convolution import Conv2DPayload
from maps.operations.depthwise_convolution import DepthwiseConvPayload
from maps.operations.elementwise import BinaryElementwisePayload, UnaryElementwisePayload
from maps.operations.gemm import GemmPayload
from maps.operations.cast import CastPayload
from maps.operations.normalization import GroupNormalizationPayload
from maps.operations.reduction import (
    GlobalAveragePoolPayload,
    ReduceSumPayload,
    ReductionPayload,
    ScalarMultiplyPayload,
)
from maps.operations.rearrangement import ReshapePayload, TransposePayload
from maps.operations.split import SplitPayload, StaticSlicePayload


def _make_tiny_matmul_graph():
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 16])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [16, 12])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 12])
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"], name="matmul_0")
    return helper.make_graph([node], "tiny_matmul", [x, w], [y])


def test_parse_graph_lowers_matmul_to_graph_node() -> None:
    try:
        graph = _make_tiny_matmul_graph()
    except ImportError:
        return

    lowered_graph = parse_graph(graph)

    assert lowered_graph.name == "tiny_matmul"
    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.GEMM
    assert isinstance(lowered_graph.nodes[0].payload, GemmPayload)
    assert lowered_graph.nodes[0].inputs[0].name == "x"
    assert lowered_graph.nodes[0].inputs[1].name == "w"
    assert lowered_graph.nodes[0].outputs[0].name == "y"
    assert len(lowered_graph.edges) == 3
    input_edges = [edge for edge in lowered_graph.edges if edge.dst is not None]
    assert len(input_edges) == 2
    assert all(edge.src is None for edge in input_edges)


def test_parse_graph_supports_gemm_bias() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 16])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [16, 12])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [8, 12])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 12])
    node = helper.make_node("Gemm", inputs=["x", "w", "b"], outputs=["y"], name="gemm_0")
    graph = helper.make_graph([node], "tiny_gemm", [x, w, b], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert isinstance(lowered_graph.nodes[0].payload, GemmPayload)
    assert lowered_graph.nodes[0].payload.y is not None
    assert lowered_graph.nodes[0].payload.y.name == "b"


def test_parse_graph_explicitly_converts_cast_to_maps_operation() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 4])
    node = helper.make_node(
        "Cast",
        inputs=["x"],
        outputs=["y"],
        name="cast_0",
        to=TensorProto.FLOAT16,
    )

    lowered = parse_graph(helper.make_graph([node], "tiny_cast", [x], [y])).nodes[0]

    assert lowered.kind is OpKind.TRANSFORM
    assert isinstance(lowered.payload, CastPayload)
    assert lowered.payload.x.dtype is TensorDType.FLOAT32
    assert lowered.payload.output.dtype is TensorDType.FLOAT16


def test_parse_graph_lowers_conv_to_graph_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [8, 3, 3, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 4, 4])
    node = helper.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["y"],
        name="conv_0",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([node], "tiny_conv", [x, w, b], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.CONV
    assert isinstance(lowered_graph.nodes[0].payload, ConvPayload)
    assert lowered_graph.nodes[0].payload.strides == (2, 2)
    assert lowered_graph.nodes[0].payload.pads == (1, 1, 1, 1)
    assert lowered_graph.nodes[0].payload.b is not None
    assert lowered_graph.nodes[0].attributes["strides"] == (2, 2)


def test_decompose_graph_lowers_dense_conv_to_one_direct_primitive() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [8, 3, 3, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 4, 4])
    node = helper.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["y"],
        name="conv_0",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([node], "tiny_conv", [x, w, b], [y])

    lowered_graph = decompose_graph(parse_graph(graph))

    assert tuple(node.name for node in lowered_graph.nodes) == ("conv_0",)
    lowered = lowered_graph.nodes[0]
    assert lowered.kind is OpKind.CONV
    assert isinstance(lowered.payload, Conv2DPayload)
    assert lowered.payload.strides == (2, 2)
    assert lowered.payload.pads == (1, 1, 1, 1)
    assert "stage_group_id" not in lowered.attributes
    assert tuple(tensor.name for tensor in lowered.inputs) == ("x", "w", "b")
    assert not any(
        tensor.name.startswith(
            (
                "conv_0__patches",
                "conv_0__packed_w",
                "conv_0__matmul",
                "conv_0__biased",
            )
        )
        for tensor in lowered_graph.tensors
    )
    assert tuple(tensor.name for tensor in lowered_graph.outputs) == ("y",)


def test_decompose_graph_lowers_depthwise_conv_to_tile_local_operation() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [4, 1, 3, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 4])
    node = helper.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["y"],
        name="depthwise_0",
        group=4,
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([node], "tiny_depthwise", [x, w, b], [y])

    lowered_graph = decompose_graph(parse_graph(graph))

    assert tuple(node.name for node in lowered_graph.nodes) == (
        "depthwise_0__depthwise",
    )
    lowered = lowered_graph.nodes[0]
    assert lowered.kind is OpKind.CONV
    assert isinstance(lowered.payload, DepthwiseConvPayload)
    assert lowered.payload.strides == (2, 2)
    assert lowered.payload.pads == (1, 1, 1, 1)
    assert lowered.attributes["stage_group_id"] == "depthwise_0::depthwise_conv"
    assert lowered.attributes["conv_step"] == "depthwise_conv"
    assert tuple(tensor.name for tensor in lowered_graph.outputs) == ("y",)


def test_parse_graph_lowers_exp_to_graph_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Exp", inputs=["x"], outputs=["y"], name="exp_0")
    graph = helper.make_graph([node], "tiny_exp", [x], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.ELEMENTWISE
    assert isinstance(lowered_graph.nodes[0].payload, UnaryElementwisePayload)
    assert lowered_graph.nodes[0].payload.op_name == "exp"
    assert lowered_graph.nodes[0].payload.x.name == "x"
    assert lowered_graph.nodes[0].payload.output.name == "y"


def test_parse_graph_lowers_log_to_graph_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Log", inputs=["x"], outputs=["y"], name="log_0")
    graph = helper.make_graph([node], "tiny_log", [x], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.ELEMENTWISE
    assert isinstance(lowered_graph.nodes[0].payload, UnaryElementwisePayload)
    assert lowered_graph.nodes[0].payload.op_name == "log"
    assert lowered_graph.nodes[0].payload.x.name == "x"
    assert lowered_graph.nodes[0].payload.output.name == "y"


def test_parse_graph_lowers_sigmoid_to_graph_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node(
        "Sigmoid",
        inputs=["x"],
        outputs=["y"],
        name="sigmoid_0",
    )
    graph = helper.make_graph([node], "tiny_sigmoid", [x], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.ELEMENTWISE
    assert isinstance(lowered_graph.nodes[0].payload, UnaryElementwisePayload)
    assert lowered_graph.nodes[0].payload.op_name == "sigmoid"
    assert lowered_graph.nodes[0].payload.work_kind is WorkKind.SIGMOID
    assert lowered_graph.nodes[0].payload.x.name == "x"
    assert lowered_graph.nodes[0].payload.output.name == "y"


def test_parse_graph_lowers_relu_to_elementwise_operation() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"], name="relu_0")],
        "relu",
        [x],
        [y],
    )

    lowered = parse_graph(graph).nodes[0].payload

    assert isinstance(lowered, UnaryElementwisePayload)
    assert lowered.op_name == "relu"
    assert lowered.work_kind is WorkKind.RELU


def test_onnx_reduce_sum_consumes_static_axis_and_adds_collective() -> None:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4, 1])
    axes = numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name="axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=1)],
        "reduce_sum",
        [x],
        [y],
        initializer=[axes],
    )

    parsed = parse_graph(graph)
    assert isinstance(parsed.nodes[0].payload, ReduceSumPayload)
    assert parsed.nodes[0].payload.axis == 2
    assert parsed.nodes[0].inputs == (parsed.inputs[0],)
    assert parsed.initializers == ()

    lowered = decompose_graph(parsed)
    assert tuple(node.name for node in lowered.nodes) == (
        "ReduceSum_0__local",
        "ReduceSum_0__allreduce",
    )
    assert isinstance(lowered.nodes[0].payload, ReductionPayload)
    assert isinstance(lowered.nodes[1].payload, AllReducePayload)
    assert lowered.nodes[1].payload.collective_axis == "x"


def test_onnx_reduce_sum_rejects_rank_reducing_form() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceSum",
                ["x"],
                ["y"],
                axes=[1],
                keepdims=0,
            )
        ],
        "reduce_sum",
        [x],
        [y],
    )

    with pytest.raises(NotImplementedError, match="keepdims=1"):
        parse_graph(graph)


def test_onnx_reduce_sum_validates_static_axis_value() -> None:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 1])
    axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=1)],
        "reduce_sum_wrong_axis",
        [x],
        [y],
        initializer=[axes],
    )

    with pytest.raises(ValueError, match="axes do not match output shape"):
        parse_graph(graph)


def test_global_average_pool_decomposes_to_spatial_sums_and_scale() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8, 4, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 8, 1, 1])
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["x"], ["y"], name="pool")],
        "global_average_pool",
        [x],
        [y],
    )

    parsed = parse_graph(graph)
    assert isinstance(parsed.nodes[0].payload, GlobalAveragePoolPayload)

    lowered = decompose_graph(parsed)
    assert tuple(node.name for node in lowered.nodes) == (
        "pool__reduce_width",
        "pool__allreduce_width",
        "pool__reduce_height",
        "pool__allreduce_height",
        "pool__scale",
    )
    assert isinstance(lowered.nodes[-1].payload, ScalarMultiplyPayload)
    assert lowered.nodes[-1].payload.factor == pytest.approx(1.0 / 24)


def test_flatten_lowers_to_static_row_major_reshape() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [6, 20])
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["x"], ["y"], axis=2)],
        "flatten",
        [x],
        [y],
    )

    payload = parse_graph(graph).nodes[0].payload

    assert isinstance(payload, ReshapePayload)
    assert payload.x.dims == (2, 3, 4, 5)
    assert payload.output.dims == (6, 20)


def test_parse_graph_consumes_static_reshape_shape_as_configuration() -> None:
    try:
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2, 2, 2])
    shape = numpy_helper.from_array(
        np.array([1, 4, 2, 2, 2, 2], dtype=np.int64),
        name="shape",
    )
    node = helper.make_node("Reshape", inputs=["x", "shape"], outputs=["y"])
    graph = helper.make_graph([node], "reshape", [x], [y], initializer=[shape])

    lowered_graph = parse_graph(graph)

    assert isinstance(lowered_graph.nodes[0].payload, ReshapePayload)
    assert tuple(tensor.name for tensor in lowered_graph.nodes[0].inputs) == ("x",)
    assert "shape" not in {tensor.name for tensor in lowered_graph.tensors}
    assert lowered_graph.initializers == ()


def test_parse_graph_rejects_dynamic_reshape_shape() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 2])
    node = helper.make_node("Reshape", inputs=["x", "shape"], outputs=["y"])
    graph = helper.make_graph([node], "reshape", [x, shape], [y])

    with pytest.raises(NotImplementedError, match="requires a static shape initializer"):
        parse_graph(graph)


def test_parse_graph_lowers_transpose_with_semantic_permutation() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [6, 2, 4])
    node = helper.make_node(
        "Transpose",
        inputs=["x"],
        outputs=["y"],
        perm=[2, 0, 1],
    )
    graph = helper.make_graph([node], "transpose", [x], [y])

    lowered_graph = parse_graph(graph)

    payload = lowered_graph.nodes[0].payload
    assert isinstance(payload, TransposePayload)
    assert payload.permutation == (2, 0, 1)


def test_parse_graph_keeps_group_normalization_semantic() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, 2])
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])
    node = helper.make_node(
        "GroupNormalization",
        inputs=["x", "scale", "bias"],
        outputs=["y"],
        num_groups=2,
        epsilon=1e-5,
    )
    graph = helper.make_graph([node], "group_norm", [x, scale, bias], [y])

    lowered_graph = parse_graph(graph)

    payload = lowered_graph.nodes[0].payload
    assert isinstance(payload, GroupNormalizationPayload)
    assert payload.num_groups == 2
    assert payload.epsilon == pytest.approx(1e-5)


def test_parse_graph_lowers_binary_elementwise_to_graph_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Add", inputs=["x", "b"], outputs=["y"], name="add_0")
    graph = helper.make_graph([node], "tiny_add", [x, b], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.ELEMENTWISE
    assert isinstance(lowered_graph.nodes[0].payload, BinaryElementwisePayload)
    assert lowered_graph.nodes[0].payload.op_name == "add"


def test_parse_graph_keeps_softmax_as_high_level_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], name="softmax_0", axis=-1)
    graph = helper.make_graph([node], "tiny_softmax", [x], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].name == "softmax_0"
    assert lowered_graph.nodes[0].kind is OpKind.CUSTOM
    assert isinstance(lowered_graph.nodes[0].payload, SoftmaxPayload)
    assert lowered_graph.nodes[0].payload.axis == 1


def test_decompose_graph_lowers_softmax_to_grouped_internal_nodes() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], name="softmax_0", axis=-1)
    graph = helper.make_graph([node], "tiny_softmax", [x], [y])

    lowered_graph = decompose_graph(parse_graph(graph))

    assert tuple(node.name for node in lowered_graph.nodes) == (
        "softmax_0__reduce_max",
        "softmax_0__allreduce_max",
        "softmax_0__sub",
        "softmax_0__exp",
        "softmax_0__reduce_sum",
        "softmax_0__allreduce_sum",
        "softmax_0__div",
    )
    assert isinstance(lowered_graph.nodes[0].payload, ReductionPayload)
    assert isinstance(lowered_graph.nodes[1].payload, AllReducePayload)
    assert isinstance(lowered_graph.nodes[2].payload, BinaryElementwisePayload)
    assert isinstance(lowered_graph.nodes[3].payload, UnaryElementwisePayload)
    assert isinstance(lowered_graph.nodes[4].payload, ReductionPayload)
    assert isinstance(lowered_graph.nodes[5].payload, AllReducePayload)
    assert isinstance(lowered_graph.nodes[6].payload, BinaryElementwisePayload)
    assert tuple(
        node.attributes["stage_group_id"]
        for node in lowered_graph.nodes
    ) == (
        "softmax_0::softmax:max",
        "softmax_0::softmax:max",
        "softmax_0::softmax:normalize",
        "softmax_0::softmax:normalize",
        "softmax_0::softmax:normalize",
        "softmax_0::softmax:normalize",
        "softmax_0::softmax:normalize",
    )
    assert tuple(tensor.name for tensor in lowered_graph.outputs) == ("y",)
    assert {
        "softmax_0__max_local",
        "softmax_0__max_global",
        "softmax_0__shifted",
        "softmax_0__exp",
        "softmax_0__sum_local",
        "softmax_0__sum_global",
    }.issubset({tensor.name for tensor in lowered_graph.tensors})

    edges_by_dst = {
        node.name: {edge.tensor.name for edge in lowered_graph.edges if edge.dst == node}
        for node in lowered_graph.nodes
    }
    assert edges_by_dst["softmax_0__reduce_max"] == {"x"}
    assert edges_by_dst["softmax_0__allreduce_max"] == {"softmax_0__max_local"}
    assert edges_by_dst["softmax_0__sub"] == {"x", "softmax_0__max_global"}
    assert edges_by_dst["softmax_0__exp"] == {"softmax_0__shifted"}
    assert edges_by_dst["softmax_0__reduce_sum"] == {"softmax_0__exp"}
    assert edges_by_dst["softmax_0__allreduce_sum"] == {"softmax_0__sum_local"}
    assert edges_by_dst["softmax_0__div"] == {"softmax_0__exp", "softmax_0__sum_global"}

    output_edges = [edge for edge in lowered_graph.edges if edge.dst is None]
    assert len(output_edges) == 1
    assert output_edges[0].src == lowered_graph.nodes[-1]
    assert output_edges[0].tensor.name == "y"


def test_decompose_graph_lowers_softmax_without_collectives_outside_default_mesh_axes() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 8])
    node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], name="softmax_0", axis=0)
    graph = helper.make_graph([node], "tiny_softmax_no_collective", [x], [y])

    lowered_graph = decompose_graph(parse_graph(graph))

    assert tuple(node.name for node in lowered_graph.nodes) == (
        "softmax_0__reduce_max",
        "softmax_0__sub",
        "softmax_0__exp",
        "softmax_0__reduce_sum",
        "softmax_0__div",
    )
    assert all(not isinstance(node.payload, AllReducePayload) for node in lowered_graph.nodes)


def test_onnx_split_constant_sizes_decompose_to_offset_static_slices() -> None:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 7, 4])
    outputs = (
        helper.make_tensor_value_info("q", TensorProto.FLOAT16, [1, 1, 4]),
        helper.make_tensor_value_info("k", TensorProto.FLOAT16, [1, 3, 4]),
        helper.make_tensor_value_info("v", TensorProto.FLOAT16, [1, 3, 4]),
    )
    sizes = numpy_helper.from_array(
        np.asarray((1, 3, 3), dtype=np.int64),
        name="sizes",
    )
    node = helper.make_node(
        "Split",
        inputs=["x", "sizes"],
        outputs=[output.name for output in outputs],
        name="split_0",
        axis=-2,
    )
    graph = helper.make_graph(
        [node],
        "tiny_split",
        [x],
        list(outputs),
        initializer=[sizes],
    )

    parsed = parse_graph(graph)
    assert isinstance(parsed.nodes[0].payload, SplitPayload)
    assert parsed.nodes[0].inputs == (parsed.inputs[0],)
    assert parsed.initializers == ()

    lowered = decompose_graph(parsed)

    assert tuple(node.name for node in lowered.nodes) == (
        "split_0__slice_0",
        "split_0__slice_1",
        "split_0__slice_2",
    )
    assert all(isinstance(node.payload, StaticSlicePayload) for node in lowered.nodes)
    assert tuple(node.payload.offsets for node in lowered.nodes) == (
        (0, 0, 0),
        (0, 1, 0),
        (0, 4, 0),
    )
    assert all(node.inputs == (parsed.inputs[0],) for node in lowered.nodes)
    assert {
        edge.dst.name
        for edge in lowered.edges
        if edge.src is None and edge.tensor.name == "x"
    } == {
        "split_0__slice_0",
        "split_0__slice_1",
        "split_0__slice_2",
    }


def test_onnx_split_num_outputs_uses_default_axis_and_smaller_final_chunk() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [10, 2])
    outputs = (
        helper.make_tensor_value_info("y0", TensorProto.FLOAT, [4, 2]),
        helper.make_tensor_value_info("y1", TensorProto.FLOAT, [4, 2]),
        helper.make_tensor_value_info("y2", TensorProto.FLOAT, [2, 2]),
    )
    node = helper.make_node(
        "Split",
        inputs=["x"],
        outputs=[output.name for output in outputs],
        name="split_0",
        num_outputs=3,
    )
    graph = helper.make_graph([node], "tiny_split", [x], list(outputs))

    parsed = parse_graph(graph)
    payload = parsed.nodes[0].payload

    assert isinstance(payload, SplitPayload)
    assert payload.axis == 0
    assert payload.sizes == (4, 4, 2)
    assert tuple(
        node.payload.offsets
        for node in decompose_graph(parsed).nodes
    ) == ((0, 0), (4, 0), (8, 0))


def test_onnx_split_rejects_dynamic_or_ambiguous_size_configuration() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6])
    sizes = helper.make_tensor_value_info("sizes", TensorProto.INT64, [2])
    outputs = (
        helper.make_tensor_value_info("y0", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("y1", TensorProto.FLOAT, [3]),
    )

    dynamic_node = helper.make_node(
        "Split",
        inputs=["x", "sizes"],
        outputs=[output.name for output in outputs],
        name="dynamic_split",
    )
    dynamic_graph = helper.make_graph(
        [dynamic_node],
        "dynamic_split",
        [x, sizes],
        list(outputs),
    )
    with pytest.raises(NotImplementedError, match="static split initializer"):
        parse_graph(dynamic_graph)

    both_node = helper.make_node(
        "Split",
        inputs=["x", "sizes"],
        outputs=[output.name for output in outputs],
        name="ambiguous_split",
        num_outputs=2,
    )
    both_graph = helper.make_graph(
        [both_node],
        "ambiguous_split",
        [x, sizes],
        list(outputs),
    )
    with pytest.raises(ValueError, match="exactly one"):
        parse_graph(both_graph)

    neither_node = helper.make_node(
        "Split",
        inputs=["x"],
        outputs=[output.name for output in outputs],
        name="unconfigured_split",
    )
    neither_graph = helper.make_graph(
        [neither_node],
        "unconfigured_split",
        [x],
        list(outputs),
    )
    with pytest.raises(ValueError, match="exactly one"):
        parse_graph(neither_graph)


def test_onnx_split_rejects_invalid_output_geometry() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    outputs = (
        helper.make_tensor_value_info("y0", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1]),
    )
    node = helper.make_node(
        "Split",
        inputs=["x"],
        outputs=[output.name for output in outputs],
        name="zero_output_split",
        num_outputs=3,
    )
    graph = helper.make_graph([node], "zero_output_split", [x], list(outputs))

    with pytest.raises(ValueError, match="zero-sized output"):
        parse_graph(graph)


def test_onnx_split_rejects_invalid_static_initializer_metadata() -> None:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6])
    outputs = (
        helper.make_tensor_value_info("y0", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("y1", TensorProto.FLOAT, [3]),
    )
    sizes = numpy_helper.from_array(
        np.asarray((3, 3), dtype=np.int32),
        name="sizes",
    )
    node = helper.make_node(
        "Split",
        inputs=["x", "sizes"],
        outputs=[output.name for output in outputs],
        name="invalid_sizes",
    )
    graph = helper.make_graph(
        [node],
        "invalid_sizes",
        [x],
        list(outputs),
        initializer=[sizes],
    )

    with pytest.raises(ValueError, match="rank-one INT64"):
        parse_graph(graph)


def test_explicit_onnx_mapping_reports_supported_ops() -> None:
    assert set(ONNX_OPERATION_CONVERTERS) >= {
        "MatMul",
        "Gemm",
        "Conv",
        "Exp",
        "Log",
        "Softmax",
        "Split",
        "ReduceSum",
        "Relu",
        "GlobalAveragePool",
        "Flatten",
    }


def test_parse_graph_rejects_unmapped_external_operation() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("FakeIdentityTestOp", inputs=["x"], outputs=["y"], name="fake_0")
    graph = helper.make_graph([node], "tiny_fake_identity", [x], [y])

    with pytest.raises(NotImplementedError, match="FakeIdentityTestOp"):
        parse_graph(graph)


def test_parse_graph_builds_node_to_node_and_initializer_edges() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 16])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [8, 10])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [8, 12])
    w0 = helper.make_tensor(
        "w0",
        TensorProto.FLOAT,
        [16, 12],
        [0.0] * (16 * 12),
    )
    w1 = helper.make_tensor(
        "w1",
        TensorProto.FLOAT,
        [12, 10],
        [0.0] * (12 * 10),
    )
    node0 = helper.make_node("MatMul", inputs=["x", "w0"], outputs=["y0"], name="matmul_0")
    node1 = helper.make_node("MatMul", inputs=["y0", "w1"], outputs=["y1"], name="matmul_1")
    graph = helper.make_graph(
        [node0, node1],
        "toy",
        [x],
        [y1],
        initializer=[w0, w1],
        value_info=[y0],
    )

    lowered_graph = parse_graph(graph)

    first_node = lowered_graph.nodes[0]
    second_node = lowered_graph.nodes[1]
    incoming_first = [edge for edge in lowered_graph.edges if edge.dst == first_node]
    incoming_second = [edge for edge in lowered_graph.edges if edge.dst == second_node]
    output_edges = [edge for edge in lowered_graph.edges if edge.dst is None]

    assert any(edge.tensor.name == "x" and edge.src is None for edge in incoming_first)
    assert any(edge.tensor.name == "w0" and edge.src is None for edge in incoming_first)
    assert any(edge.tensor.name == "y0" and edge.src == first_node for edge in incoming_second)
    assert any(edge.tensor.name == "w1" and edge.src is None for edge in incoming_second)
    assert any(edge.tensor.name == "y1" and edge.src == second_node for edge in output_edges)
    assert tuple(tensor.name for tensor in lowered_graph.initializers) == ("w0", "w1")
    assert not lowered_graph.tensors[0].is_initializer
    assert all(tensor.is_initializer for tensor in lowered_graph.initializers)


def test_onnx_dtype_elem_bytes_maps_common_float32() -> None:
    assert onnx_dtype_elem_bytes(1) == 4
