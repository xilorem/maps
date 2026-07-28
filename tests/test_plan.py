import json
from pathlib import Path
from tempfile import TemporaryDirectory

from MAPS.arch import L1Memory, L2Memory, Mesh
from MAPS.hw.chips import magia_mesh
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.layout import TensorRange, TensorSlice, tile_tensor_slice
from MAPS.pipeline import Layer, Pipeline, Stage
from MAPS.pipeline.execution import ExecutionContract
from MAPS.pipeline.layer import ExternalInput, LocalInput, TransitionInput
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.transitions.model import TransitionMode
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.planner.contracts.stages import StagePlacement, StagePlan, virtual_submesh
import MAPS.planner.plan as plan_module
from MAPS.planner.passes.pipeline_lowering import lower_pipeline
from MAPS.planner.passes.validation import validate_constraints
from MAPS.planner.plan import build_pipeline
from MAPS.planner.validation.contracts import PlannerConstraints
from MAPS.utils.pipeline_json import pipeline_json_payload, write_pipeline_json
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _mesh_with_l1(width: int, height: int, l1_size: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(width, height, memory=L1Memory(size=l1_size, bandwidth=1)),
    )


def _identity_placements(
    plans: dict[int, StagePlan],
) -> dict[int, StagePlacement]:
    """Place test plans on the same tile ids used by their virtual layouts."""

    return {
        stage_id: StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual_submesh(plan),
            physical_submesh=virtual_submesh(plan),
            virtual_to_physical={
                tile.tile_id: tile.tile_id
                for tile in virtual_submesh(plan).tiles
            },
        )
        for stage_id, plan in plans.items()
    }


def test_pipeline_json_omits_redundant_nested_mesh_state() -> None:
    mesh = magia_mesh(width=4, height=4)
    stage = Stage(
        name="stage",
        submesh=Submesh(mesh=mesh, submesh_id=0, tile_ids={0}),
        layers=(Layer(Node("node", OpKind.CUSTOM)),),
    )

    payload = pipeline_json_payload(Pipeline("pipeline", mesh, stages=(stage,)))

    assert payload["execution"] == {"num_token_slots": 2}
    assert set(payload["mesh"]) == {"width", "height", "l2_memory", "tiles"}
    assert set(payload["mesh"]["l2_memory"]) == {"size"}
    assert set(payload["mesh"]["tiles"][0]) == {"tile_id", "x", "y"}
    assert payload["stages"][0]["submesh"] == {
        "submesh_id": 0,
        "tile_ids": [0],
    }


def test_pipeline_json_carries_planner_selected_token_slots() -> None:
    mesh = magia_mesh(width=1, height=1)
    pipeline = Pipeline(
        "pipeline",
        mesh,
        execution=ExecutionContract(num_token_slots=3),
    )

    assert pipeline_json_payload(pipeline)["execution"] == {
        "num_token_slots": 3,
    }


def test_lower_pipeline_assembles_stages_transitions_and_bindings(tmp_path: Path) -> None:
    mesh = magia_mesh()
    src_submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids=frozenset((0, 1)))
    dst_submesh = Submesh(mesh=mesh, submesh_id=1, tile_ids=frozenset((8, 16)))

    x = Tensor(name="x", rank=2, dims=(4, 4), elem_bytes=2)
    w0 = Tensor(name="w0", rank=2, dims=(4, 8), elem_bytes=2)
    y = Tensor(name="y", rank=2, dims=(4, 8), elem_bytes=2)
    w1 = Tensor(name="w1", rank=2, dims=(8, 6), elem_bytes=2)
    z = Tensor(name="z", rank=2, dims=(4, 6), elem_bytes=2)

    gemm0 = GemmPayload(x=x, w=w0, y=None, output=y)
    gemm1 = GemmPayload(x=y, w=w1, y=None, output=z)
    node0 = Node(
        name="gemm_0",
        kind=OpKind.GEMM,
        inputs=(x, w0),
        outputs=(y,),
        payload=gemm0,
    )
    node1 = Node(
        name="gemm_1",
        kind=OpKind.GEMM,
        inputs=(y, w1),
        outputs=(z,),
        payload=gemm1,
    )
    graph = Graph(
        name="direct_two_gemms",
        tensors=(x, w0, y, w1, z),
        nodes=(node0, node1),
        edges=(
            Edge(tensor=x, src=None, dst=node0),
            Edge(tensor=w0, src=None, dst=node0),
            Edge(tensor=y, src=node0, dst=node1),
            Edge(tensor=w1, src=None, dst=node1),
            Edge(tensor=z, src=node1, dst=None),
        ),
        inputs=(x,),
        outputs=(z,),
        initializers=(w0, w1),
    )
    plan0 = StagePlan(
        stage_id=0,
        tile_count=2,
        logical_shape=(2, 1),
        nodes=(node0,),
        node_output_layouts=(
            gemm0.output_layouts(src_submesh, logical_shape=(2, 1)),
        ),
    )
    plan1 = StagePlan(
        stage_id=1,
        tile_count=2,
        logical_shape=(2, 1),
        nodes=(node1,),
        node_output_layouts=(
            gemm1.output_layouts(dst_submesh, logical_shape=(2, 1)),
        ),
    )

    plans = {0: plan0, 1: plan1}
    pipeline = lower_pipeline(graph, mesh, plans, _identity_placements(plans))

    assert pipeline.name == "direct_two_gemms"
    assert len(pipeline.stages) == 2
    assert len(pipeline.transitions) == 1
    assert tuple(initialization.name for initialization in pipeline.initializations) == (
        "init_x",
        "init_w0",
        "init_w1",
    )
    assert tuple(finalization.name for finalization in pipeline.finalizations) == (
        "output_z",
    )
    assert tuple(tensor.is_initializer for tensor in pipeline.tensors) == (
        False,
        True,
        False,
        True,
        False,
    )
    assert pipeline.stages[0].layers[0].node == node0
    assert pipeline.stages[1].layers[0].node == node1
    assert isinstance(pipeline.stages[0].layers[0].inputs[0].source, ExternalInput)
    assert isinstance(pipeline.stages[1].layers[0].inputs[0].source, TransitionInput)
    assert pipeline.stages[1].layers[0].inputs[0].source.transition_id == 0

    transition = pipeline.transitions[0]
    assert transition.mode is TransitionMode.DIRECT_REMAP
    assert transition.tensor_id == 2
    assert transition.src_layer_id == 0
    assert transition.dst_layer_id == 1
    assert transition.src_layout == pipeline.stages[0].layers[-1].outputs[0].layout
    assert transition.dst_layout == plan1.node_output_layouts[-1][0]
    assert len(transition.fragments) == 4
    assert {
        (fragment.src_hartid, fragment.dst_hartid)
        for fragment in transition.fragments
    } == {(0, 8), (1, 8), (0, 16), (1, 16)}
    assert {
        fragment.src_subslice.parent
        for fragment in transition.fragments
    } == {
        tile_tensor_slice(y, transition.src_layout, tile)
        for tile in transition.src_layout.submesh.tiles
    }
    assert {
        fragment.dst_subslice.dims
        for fragment in transition.fragments
    } == {
        (
            TensorRange(start=0, length=4),
            TensorRange(start=0, length=4),
        ),
        (
            TensorRange(start=0, length=4),
            TensorRange(start=4, length=4),
        ),
    }
    assert {
        fragment.dst_subslice.parent
        for fragment in transition.fragments
    } == {
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=0, length=8),
            ),
        )
    }
    weight_initialization = pipeline.initializations[1]
    assert weight_initialization.tensor_id == 1
    assert weight_initialization.dst_layer_id == 0
    assert weight_initialization.dst_input_idx == 1
    assert {
        (fragment.src_hartid, fragment.dst_hartid)
        for fragment in weight_initialization.fragments
    } == {(-1, 0), (-1, 1)}
    assert {
        fragment.src_slice.dims[-1]
        for fragment in weight_initialization.fragments
    } == {
        tile_tensor_slice(y, transition.src_layout, tile).dims[-1]
        for tile in transition.src_layout.submesh.tiles
    }
    finalization = pipeline.finalizations[0]
    assert finalization.tensor_id == 4
    assert finalization.src_layer_id == 1
    assert finalization.src_output_idx == 0
    assert {
        (fragment.src_hartid, fragment.dst_hartid)
        for fragment in finalization.fragments
    } == {(8, -1), (16, -1)}
    assert {
        fragment.src_slice
        for fragment in finalization.fragments
    } == {
        tile_tensor_slice(z, plan1.node_output_layouts[-1][0], tile)
        for tile in plan1.node_output_layouts[-1][0].submesh.tiles
    }
    json_path = write_pipeline_json(pipeline, tmp_path / "pipeline.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [tensor["is_initializer"] for tensor in payload["tensors"]] == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert "is_initializer" not in payload["stages"][0]["layers"][0]["node"]["inputs"][1]
    assert payload["initializations"][1]["name"] == "init_w0"
    assert payload["finalizations"][0]["name"] == "output_z"

    report = validate_constraints(pipeline, PlannerConstraints())
    assert report.is_valid, report.violations


def test_lower_pipeline_builds_local_inputs_for_grouped_stage_nodes() -> None:
    mesh = magia_mesh()
    stage0_submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids=frozenset((0, 1)))
    stage1_submesh = Submesh(mesh=mesh, submesh_id=1, tile_ids=frozenset((8, 9)))

    x = Tensor(name="x", rank=2, dims=(4, 4), elem_bytes=2)
    w0 = Tensor(name="w0", rank=2, dims=(4, 8), elem_bytes=2)
    y0 = Tensor(name="y0", rank=2, dims=(4, 8), elem_bytes=2)
    w1 = Tensor(name="w1", rank=2, dims=(8, 6), elem_bytes=2)
    y1 = Tensor(name="y1", rank=2, dims=(4, 6), elem_bytes=2)
    w2 = Tensor(name="w2", rank=2, dims=(6, 5), elem_bytes=2)
    z = Tensor(name="z", rank=2, dims=(4, 5), elem_bytes=2)

    gemm0 = GemmPayload(x=x, w=w0, y=None, output=y0)
    gemm1 = GemmPayload(x=y0, w=w1, y=None, output=y1)
    gemm2 = GemmPayload(x=y1, w=w2, y=None, output=z)
    node0 = Node(
        name="gemm_0",
        kind=OpKind.GEMM,
        inputs=(x, w0),
        outputs=(y0,),
        payload=gemm0,
    )
    node1 = Node(
        name="gemm_1",
        kind=OpKind.GEMM,
        inputs=(y0, w1),
        outputs=(y1,),
        payload=gemm1,
    )
    node2 = Node(
        name="gemm_2",
        kind=OpKind.GEMM,
        inputs=(y1, w2),
        outputs=(z,),
        payload=gemm2,
    )
    graph = Graph(
        name="grouped_two_gemms",
        tensors=(x, w0, y0, w1, y1, w2, z),
        nodes=(node0, node1, node2),
        edges=(
            Edge(tensor=x, src=None, dst=node0),
            Edge(tensor=w0, src=None, dst=node0),
            Edge(tensor=y0, src=node0, dst=node1),
            Edge(tensor=w1, src=None, dst=node1),
            Edge(tensor=y1, src=node1, dst=node2),
            Edge(tensor=w2, src=None, dst=node2),
            Edge(tensor=z, src=node2, dst=None),
        ),
        inputs=(x,),
        outputs=(z,),
        initializers=(w0, w1, w2),
    )
    plan0 = StagePlan(
        stage_id=0,
        tile_count=2,
        logical_shape=(2, 1),
        nodes=(node0, node1),
        node_output_layouts=(
            gemm0.output_layouts(stage0_submesh, logical_shape=(2, 1)),
            gemm1.output_layouts(stage0_submesh, logical_shape=(2, 1)),
        ),
    )
    plan1 = StagePlan(
        stage_id=1,
        tile_count=2,
        logical_shape=(2, 1),
        nodes=(node2,),
        node_output_layouts=(gemm2.output_layouts(stage1_submesh, logical_shape=(2, 1)),),
    )

    plans = {0: plan0, 1: plan1}
    pipeline = lower_pipeline(graph, mesh, plans, _identity_placements(plans))

    assert len(pipeline.stages) == 2
    assert len(pipeline.stages[0].layers) == 2
    assert pipeline.stages[0].layers[0].node == node0
    assert pipeline.stages[0].layers[1].node == node1
    assert isinstance(pipeline.stages[0].layers[1].inputs[0].source, LocalInput)
    assert pipeline.stages[0].layers[1].inputs[0].source.layer_idx == 0
    assert len(pipeline.transitions) == 1
    assert tuple(initialization.name for initialization in pipeline.initializations) == (
        "init_x",
        "init_w0",
        "init_w1",
        "init_w2",
    )
    assert pipeline.initializations[-1].dst_layer_id == 2
    assert pipeline.finalizations[0].src_layer_id == 2
    assert pipeline.transitions[0].src_layer_id == 0
    assert pipeline.transitions[0].dst_layer_id == 1
    assert isinstance(pipeline.stages[1].layers[0].inputs[0].source, TransitionInput)
    assert pipeline.stages[1].layers[0].inputs[0].source.transition_id == 0

    report = validate_constraints(pipeline, PlannerConstraints(max_stage_nodes=2))
    assert report.is_valid, report.violations


def test_build_pipeline_parses_balances_maps_and_builds_transitions() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

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

        pipeline = build_pipeline(model_path, _mesh_with_l1(2, 2, l1_size=4096))

    assert pipeline.name == "two_matmuls"
    assert len(pipeline.stages) == 2
    if pipeline.transitions:
        assert len(pipeline.transitions) == 1
        transition = pipeline.transitions[0]
        assert transition.src_layer_id == 0
        assert transition.dst_layer_id == 1
        assert transition.src_layout == pipeline.stages[0].layers[-1].outputs[0].layout
        assert transition.dst_layout == pipeline.stages[1].layers[0].outputs[0].layout
        assert transition.mode is TransitionMode.DIRECT_REMAP
        assert transition.fragments
        for fragment in transition.fragments:
            assert fragment.src_subslice.rank == pipeline.tensors[transition.tensor_id].rank
            assert fragment.dst_subslice.rank == pipeline.tensors[transition.tensor_id].rank
            for src_dim, dst_dim in zip(fragment.src_subslice.dims, fragment.dst_subslice.dims):
                assert src_dim.length > 0
                assert dst_dim.length > 0
        assert isinstance(pipeline.stages[1].layers[0].inputs[0].source, TransitionInput)
        assert pipeline.stages[1].layers[0].inputs[0].source.transition_id == 0
    assert isinstance(pipeline.stages[0].layers[0].inputs[0].source, ExternalInput)

    report = validate_constraints(pipeline, PlannerConstraints())
    assert report.is_valid, report.violations


def test_build_pipeline_lowers_softmax_into_one_grouped_stage() -> None:
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

        pipeline = build_pipeline(model_path, _mesh_with_l1(2, 2, l1_size=4096))

    assert pipeline.name == "tiny_softmax"
    assert len(pipeline.stages) == 1
    assert len(pipeline.transitions) == 0
    assert tuple(layer.node.name for layer in pipeline.stages[0].layers) == (
        "softmax_0__reduce_max",
        "softmax_0__allreduce_max",
        "softmax_0__sub",
        "softmax_0__exp",
        "softmax_0__reduce_sum",
        "softmax_0__allreduce_sum",
        "softmax_0__div",
    )
    assert isinstance(pipeline.stages[0].layers[1].inputs[0].source, LocalInput)
    assert pipeline.stages[0].layers[1].inputs[0].source.layer_idx == 0
    assert isinstance(pipeline.stages[0].layers[2].inputs[1].source, LocalInput)
    assert pipeline.stages[0].layers[2].inputs[1].source.layer_idx == 1
    assert isinstance(pipeline.stages[0].layers[6].inputs[1].source, LocalInput)
    assert pipeline.stages[0].layers[6].inputs[1].source.layer_idx == 5

    report = validate_constraints(pipeline, PlannerConstraints(max_stage_nodes=7))
    assert report.is_valid, report.violations


def test_build_pipeline_exports_direct_conv_semantics(tmp_path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    model_path = tmp_path / "conv.onnx"
    json_path = tmp_path / "conv.pipeline.json"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [4, 1, 3, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 4])
    node = helper.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["y"],
        name="conv_0",
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([node], "tiny_conv", [x, w, b], [y])
    onnx.save(helper.make_model(graph), model_path)

    pipeline = build_pipeline(
        model_path,
        _mesh_with_l1(2, 1, l1_size=4096),
        output_json_path=json_path,
    )

    assert len(pipeline.stages) == 1
    assert tuple(layer.node.name for layer in pipeline.stages[0].layers) == ("conv_0",)
    assert len(pipeline.transitions) == 0
    assert {init.name for init in pipeline.initializations} == {"init_x", "init_w", "init_b"}
    assert pipeline.finalizations[0].name == "output_y"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    layer = payload["stages"][0]["layers"][0]
    assert layer["node"]["kind"] == int(OpKind.CONV)
    assert layer["node"]["payload"]["strides"] == [1, 1]
    assert layer["node"]["payload"]["pads"] == [1, 1, 1, 1]
    assert layer["node"]["payload"]["dilations"] == [1, 1]
    assert layer["node"]["payload"]["work_kind"] == "CONV2D"


def test_build_pipeline_disables_spatial_mapping_progress_by_default(monkeypatch) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    seen = {}
    built_pipeline = object()

    def fake_map_spatially(graph, mesh, stage_plans, **kwargs):
        seen["show_progress"] = kwargs["show_progress"]
        return {}

    monkeypatch.setattr(plan_module, "map_spatially", fake_map_spatially)
    monkeypatch.setattr(
        plan_module,
        "lower_pipeline",
        lambda graph, mesh, plans, placements, **kwargs: built_pipeline,
    )
    monkeypatch.setattr(
        plan_module,
        "print_pipeline_stage_cost",
        lambda graph, mesh, plans, placements: None,
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

        pipeline = build_pipeline(model_path, _mesh_with_l1(2, 2, l1_size=4096))

    assert pipeline is built_pipeline
    assert seen["show_progress"] is False


def test_build_pipeline_can_enable_spatial_mapping_progress(monkeypatch) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    seen = {}
    built_pipeline = object()

    def fake_map_spatially(graph, mesh, stage_plans, **kwargs):
        seen["show_progress"] = kwargs["show_progress"]
        return {}

    monkeypatch.setattr(plan_module, "map_spatially", fake_map_spatially)
    monkeypatch.setattr(
        plan_module,
        "lower_pipeline",
        lambda graph, mesh, plans, placements, **kwargs: built_pipeline,
    )
    monkeypatch.setattr(
        plan_module,
        "print_pipeline_stage_cost",
        lambda graph, mesh, plans, placements: None,
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

        pipeline = build_pipeline(
            model_path,
            _mesh_with_l1(2, 2, l1_size=4096),
            print_spatial_mapping_progress=True,
        )

    assert pipeline is built_pipeline
    assert seen["show_progress"] is True
