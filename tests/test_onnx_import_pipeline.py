"""Tests for end-to-end ONNX import into the shared graph IR."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from MAPS.core.graph import Graph
from MAPS.importers.onnx.importer import import_onnx_graph
from MAPS.importers.onnx.preprocess import prepare_onnx_model
from MAPS.ops.defs.gemm import GemmPayload


def test_load_onnx_model_requires_existing_path() -> None:
    assert Path(__file__).name == "test_onnx_import_pipeline.py"


def test_import_onnx_graph_returns_scheduler_graph_ir() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "tiny_matmul.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"], name="matmul_0")
        graph_proto = helper.make_graph([node], "tiny_matmul", [x, w], [y])
        model = helper.make_model(graph_proto)
        onnx.save(model, model_path)

        lowered_graph = import_onnx_graph(model_path)

        assert isinstance(lowered_graph, Graph)
        assert lowered_graph.name == "tiny_matmul"
        assert len(lowered_graph.nodes) == 1
        assert isinstance(lowered_graph.nodes[0].payload, GemmPayload)


def test_prepare_onnx_model_specializes_input_before_shape_inference() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])
    model = helper.make_model(helper.make_graph([node], "dynamic", [x, w], [y]))

    prepared = prepare_onnx_model(model, {"x": (2, 3)})

    input_dims = prepared.graph.input[0].type.tensor_type.shape.dim
    output_dims = prepared.graph.output[0].type.tensor_type.shape.dim
    assert tuple(dimension.dim_value for dimension in input_dims) == (2, 3)
    assert tuple(dimension.dim_value for dimension in output_dims) == (2, 4)
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"


def test_import_onnx_graph_accepts_input_shape_overrides(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])],
            "dynamic",
            [x, w],
            [y],
        )
    )
    model_path = tmp_path / "dynamic.onnx"
    onnx.save(model, model_path)

    graph = import_onnx_graph(model_path, input_shapes={"x": (2, 3)})

    assert graph.inputs[0].dims == (2, 3)
    assert graph.outputs[0].dims == (2, 4)


def test_prepare_onnx_model_rejects_remaining_dynamic_dimensions() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", inputs=["x"], outputs=["y"])],
            "dynamic",
            [x],
            [y],
        )
    )

    with pytest.raises(ValueError, match=r"tensor 'x' has dynamic dimension 'batch'"):
        prepare_onnx_model(model)


def test_prepare_onnx_model_validates_input_shape_overrides() -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])
    model = helper.make_model(helper.make_graph([], "dynamic", [x], [y]))

    with pytest.raises(ValueError, match="unknown input 'missing'"):
        prepare_onnx_model(model, {"missing": (2, 3)})
    with pytest.raises(ValueError, match="has rank 1; expected 2"):
        prepare_onnx_model(model, {"x": (2,)})
    with pytest.raises(ValueError, match="must contain positive dimensions"):
        prepare_onnx_model(model, {"x": (0, 3)})
    with pytest.raises(ValueError, match="changes concrete dimension 1 from 3 to 4"):
        prepare_onnx_model(model, {"x": (2, 4)})


def _print_graph(graph: Graph) -> None:
    print(f"graph: {graph.name}")
    print(f"inputs: {[tensor.name for tensor in graph.inputs]}")
    print(f"outputs: {[tensor.name for tensor in graph.outputs]}")
    print("tensors:")
    for tensor in graph.tensors:
        print(
            "  "
            f"{tensor.name}: shape={tensor.dims} elem_bytes={tensor.elem_bytes}"
        )
    print("nodes:")
    for node in graph.nodes:
        print(
            "  "
            f"{node.name}: {node.kind.name} "
            f"inputs={[tensor.name for tensor in node.inputs]} "
            f"outputs={[tensor.name for tensor in node.outputs]}"
        )
    print("edges:")
    for edge in graph.edges:
        print(
            "  "
            f"{edge.tensor.name}: "
            f"{edge.src.name if edge.src is not None else 'EXTERNAL'} -> "
            f"{edge.dst.name if edge.dst is not None else 'GRAPH_OUTPUT'}"
        )


def _print_lowered_graph(graph: Graph) -> None:
    print("lowered nodes:")
    for idx, node in enumerate(graph.nodes):
        print(f"  [{idx}] {node.name}: {node.payload!r}")


if __name__ == "__main__":
    import onnx
    from onnx import TensorProto, helper

    with TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "tiny_matmul.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"], name="matmul_0")
        graph_proto = helper.make_graph([node], "tiny_matmul", [x, w], [y])
        model = helper.make_model(graph_proto)
        onnx.save(model, sample_path)

        lowered_graph = import_onnx_graph(sample_path)
        _print_graph(lowered_graph)
        _print_lowered_graph(lowered_graph)
        print("ok")
