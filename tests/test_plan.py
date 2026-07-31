import json
from pathlib import Path
from tempfile import TemporaryDirectory

from MAPS.arch import L1Memory, L2Memory, Mesh
from MAPS.core.graph import OpKind
from MAPS.pipeline import (
    ExecutionPlan,
    InitializerInput,
    LocalInput,
    TransitionSource,
)
import MAPS.planner.plan as plan_module
from MAPS.planner.passes.execution_plan_validation import validate_execution_plan
from MAPS.planner.plan import build_execution_plan
from MAPS.planner.validation.contracts import PlannerConstraints
from MAPS.transitions import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
)
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _mesh_with_l1(width: int, height: int, l1_size: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(
            width,
            height,
            memory=L1Memory(size=l1_size, bandwidth=1),
        ),
    )


def test_build_execution_plan_returns_a_valid_execution_plan() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "two_matmuls.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w0 = helper.make_tensor(
            "w0",
            TensorProto.FLOAT,
            [3, 4],
            [0.0] * 12,
        )
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        w1 = helper.make_tensor(
            "w1",
            TensorProto.FLOAT,
            [4, 5],
            [0.0] * 20,
        )
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 5])
        node0 = helper.make_node("MatMul", inputs=["x", "w0"], outputs=["y"], name="matmul_0")
        node1 = helper.make_node("MatMul", inputs=["y", "w1"], outputs=["z"], name="matmul_1")
        graph = helper.make_graph(
            [node0, node1],
            "two_matmuls",
            [x],
            [z],
            value_info=[y],
            initializer=[w0, w1],
        )
        model = helper.make_model(graph)
        onnx.save(model, model_path)

        execution_plan = build_execution_plan(
            model_path,
            _mesh_with_l1(2, 2, l1_size=4096),
        )

    assert isinstance(execution_plan, ExecutionPlan)
    assert execution_plan.name == "two_matmuls"
    assert len(execution_plan.stages) == 2
    assert tuple(type(item) for item in execution_plan.transitions) == (
        InputTransition,
        IntermediateTransition,
        OutputTransition,
    )
    transition = execution_plan.transitions[1]
    assert transition.source_stage_id == 0
    assert transition.destination_stage_id == 1
    assert transition.transfers
    for transfer in transition.transfers:
        assert transfer.source_subslice.rank == execution_plan.tensors[transition.tensor_id].rank
        assert (
            transfer.destination_subslice.rank
            == execution_plan.tensors[transition.tensor_id].rank
        )
        for src_dim, dst_dim in zip(
            transfer.source_subslice.dims,
            transfer.destination_subslice.dims,
        ):
            assert src_dim.length > 0
            assert dst_dim.length > 0
    assert isinstance(
        execution_plan.stages[0].layers[0].inputs[0].source,
        TransitionSource,
    )
    assert execution_plan.stages[0].layers[0].inputs[0].source.transition_id == 0
    assert isinstance(
        execution_plan.stages[1].layers[0].inputs[0].source,
        TransitionSource,
    )
    assert execution_plan.stages[1].layers[0].inputs[0].source.transition_id == 1

    report = validate_execution_plan(execution_plan, PlannerConstraints())
    assert report.is_valid, report.violations


def test_build_execution_plan_lowers_softmax_into_edge_communicating_stages() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "softmax.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
        node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], name="softmax_0", axis=-1)
        graph = helper.make_graph([node], "tiny_softmax", [x], [y])
        model = helper.make_model(graph)
        onnx.save(model, model_path)

        execution_plan = build_execution_plan(
            model_path,
            _mesh_with_l1(2, 2, l1_size=4096),
        )

    assert execution_plan.name == "tiny_softmax"
    assert len(execution_plan.stages) == 2
    assert tuple(type(item) for item in execution_plan.transitions) == (
        InputTransition,
        InputTransition,
        IntermediateTransition,
        OutputTransition,
    )
    assert tuple(layer.node.name for layer in execution_plan.stages[0].layers) == (
        "softmax_0__reduce_max",
        "softmax_0__allreduce_max",
    )
    assert tuple(layer.node.name for layer in execution_plan.stages[1].layers) == (
        "softmax_0__sub",
        "softmax_0__exp",
        "softmax_0__reduce_sum",
        "softmax_0__allreduce_sum",
        "softmax_0__div",
    )
    assert isinstance(execution_plan.stages[0].layers[1].inputs[0].source, LocalInput)
    assert execution_plan.stages[0].layers[1].inputs[0].source.layer_idx == 0
    assert isinstance(
        execution_plan.stages[1].layers[0].inputs[1].source,
        TransitionSource,
    )
    assert execution_plan.stages[1].layers[0].inputs[1].source.transition_id == 2
    assert isinstance(execution_plan.stages[1].layers[4].inputs[1].source, LocalInput)
    assert execution_plan.stages[1].layers[4].inputs[1].source.layer_idx == 3

    report = validate_execution_plan(
        execution_plan,
        PlannerConstraints(max_stage_nodes=5),
    )
    assert report.is_valid, report.violations


def test_build_execution_plan_exports_direct_conv_semantics(tmp_path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    model_path = tmp_path / "conv.onnx"
    json_path = tmp_path / "conv.execution-plan.json"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    w = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [4, 1, 3, 3],
        [0.0] * 36,
    )
    b = helper.make_tensor("b", TensorProto.FLOAT, [4], [0.0] * 4)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 4])
    node = helper.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["y"],
        name="conv_0",
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [node],
        "tiny_conv",
        [x],
        [y],
        initializer=[w, b],
    )
    onnx.save(helper.make_model(graph), model_path)

    execution_plan = build_execution_plan(
        model_path,
        _mesh_with_l1(2, 1, l1_size=4096),
        output_json_path=json_path,
    )

    assert len(execution_plan.stages) == 1
    assert tuple(layer.node.name for layer in execution_plan.stages[0].layers) == (
        "conv_0",
    )
    assert tuple(type(item) for item in execution_plan.transitions) == (
        InputTransition,
        OutputTransition,
    )
    assert all(
        isinstance(layer_input.source, (TransitionSource, InitializerInput))
        for layer_input in execution_plan.stages[0].layers[0].inputs
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [item["kind"] for item in payload["transitions"]] == [
        "INPUT",
        "OUTPUT",
    ]
    assert "initializations" not in payload
    assert "finalizations" not in payload
    second_json_path = tmp_path / "conv.execution-plan-again.json"
    independently_planned = build_execution_plan(
        model_path,
        _mesh_with_l1(2, 1, l1_size=4096),
        output_json_path=second_json_path,
    )
    assert independently_planned is not execution_plan
    assert json_path.read_bytes() == second_json_path.read_bytes()
    layer = payload["stages"][0]["layers"][0]
    assert layer["node"]["kind"] == int(OpKind.CONV)
    assert layer["node"]["payload"]["strides"] == [1, 1]
    assert layer["node"]["payload"]["pads"] == [1, 1, 1, 1]
    assert layer["node"]["payload"]["dilations"] == [1, 1]
    assert layer["node"]["payload"]["work_kind"] == "CONV2D"


def test_build_execution_plan_disables_mapping_progress_by_default(monkeypatch) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    seen = {}
    def fake_map_spatially(mesh, stage_plans, virtual_transitions, **kwargs):
        del mesh, stage_plans
        seen["show_progress"] = kwargs["show_progress"]
        seen["mapped_transitions"] = virtual_transitions
        return {}

    monkeypatch.setattr(plan_module, "map_spatially", fake_map_spatially)

    def fake_lower(
        graph,
        mesh,
        plans,
        placements,
        virtual_transitions,
        **kwargs,
    ):
        del graph, plans, placements, kwargs
        assert virtual_transitions is seen["mapped_transitions"]
        seen["execution_plan"] = ExecutionPlan("built", mesh)
        return seen["execution_plan"]

    monkeypatch.setattr(plan_module, "lower_execution_plan", fake_lower)
    monkeypatch.setattr(
        plan_module,
        "print_execution_plan_stage_cost",
        lambda mesh, plans, placements, virtual_transitions: None,
    )

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "two_matmuls.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w0 = helper.make_tensor_value_info("w0", TensorProto.FLOAT, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        w1 = helper.make_tensor_value_info("w1", TensorProto.FLOAT, [4, 5])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 5])
        node0 = helper.make_node("MatMul", inputs=["x", "w0"], outputs=["y"], name="matmul_0")
        node1 = helper.make_node("MatMul", inputs=["y", "w1"], outputs=["z"], name="matmul_1")
        graph = helper.make_graph(
            [node0, node1],
            "two_matmuls",
            [x, w0, w1],
            [z],
            value_info=[y],
        )
        model = helper.make_model(graph)
        onnx.save(model, model_path)

        execution_plan = build_execution_plan(
            model_path,
            _mesh_with_l1(2, 2, l1_size=4096),
        )

    assert execution_plan is seen["execution_plan"]
    assert seen["show_progress"] is False


def test_build_execution_plan_can_enable_mapping_progress(monkeypatch) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    seen = {}
    def fake_map_spatially(mesh, stage_plans, virtual_transitions, **kwargs):
        del mesh, stage_plans, virtual_transitions
        seen["show_progress"] = kwargs["show_progress"]
        return {}

    monkeypatch.setattr(plan_module, "map_spatially", fake_map_spatially)

    def fake_lower(
        graph,
        mesh,
        plans,
        placements,
        virtual_transitions,
        **kwargs,
    ):
        del graph, plans, placements, virtual_transitions, kwargs
        seen["execution_plan"] = ExecutionPlan("built", mesh)
        return seen["execution_plan"]

    monkeypatch.setattr(plan_module, "lower_execution_plan", fake_lower)
    monkeypatch.setattr(
        plan_module,
        "print_execution_plan_stage_cost",
        lambda mesh, plans, placements, virtual_transitions: None,
    )

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "two_matmuls.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w0 = helper.make_tensor_value_info("w0", TensorProto.FLOAT, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        w1 = helper.make_tensor_value_info("w1", TensorProto.FLOAT, [4, 5])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 5])
        node0 = helper.make_node("MatMul", inputs=["x", "w0"], outputs=["y"], name="matmul_0")
        node1 = helper.make_node("MatMul", inputs=["y", "w1"], outputs=["z"], name="matmul_1")
        graph = helper.make_graph(
            [node0, node1],
            "two_matmuls",
            [x, w0, w1],
            [z],
            value_info=[y],
        )
        model = helper.make_model(graph)
        onnx.save(model, model_path)

        execution_plan = build_execution_plan(
            model_path,
            _mesh_with_l1(2, 2, l1_size=4096),
            print_spatial_mapping_progress=True,
        )

    assert execution_plan is seen["execution_plan"]
    assert seen["show_progress"] is True
