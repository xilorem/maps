import pytest

from maps.deployment import build_deployment_bundle
from maps.deployment.serialization import execution_plan_json_payload
from maps.graph import (
    ConstantStore,
    Edge,
    Graph,
    ImportedModel,
    Node,
    OpKind,
    Tensor,
    TensorDType,
)
from maps.graph.onnx.parser import parse_graph
from maps.hardware import WorkKind, WorkSignature
from maps.operations.split import SplitPayload
from maps.planning import (
    PlacementOptions,
    PlanningOptions,
    StageFormationOptions,
    plan,
)
from maps.planning.mapping import TensorRange
from maps.planning.stages import form_stages
from maps.planning.transitions import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
)
from maps.target import SpecializationOptions, magia, n300d


def _split_model(
    dtype: TensorDType = TensorDType.FLOAT16,
    sizes: tuple[int, ...] = (1, 3, 3),
) -> ImportedModel:
    elem_bytes = 2 if dtype is TensorDType.FLOAT16 else 4
    x = Tensor("x", 2, (2, sum(sizes)), elem_bytes, dtype=dtype)
    outputs = tuple(
        Tensor(f"output{index}", 2, (2, size), elem_bytes, dtype=dtype)
        for index, size in enumerate(sizes)
    )
    split = Node(
        "split",
        OpKind.TRANSFORM,
        inputs=(x,),
        outputs=outputs,
        payload=SplitPayload(x, outputs, axis=1, sizes=sizes),
    )
    return ImportedModel(
        Graph(
            "multi_output_split",
            tensors=(x, *outputs),
            nodes=(split,),
            edges=(
                Edge(x, None, split),
                *(Edge(output, split, None) for output in outputs),
            ),
            inputs=(x,),
            outputs=outputs,
        ),
        ConstantStore(()),
    )


def _onnx_split_model() -> ImportedModel:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 7])
    split_outputs = (
        helper.make_tensor_value_info("q", TensorProto.FLOAT16, [2, 1]),
        helper.make_tensor_value_info("k", TensorProto.FLOAT16, [2, 3]),
        helper.make_tensor_value_info("v", TensorProto.FLOAT16, [2, 3]),
    )
    graph_outputs = tuple(
        helper.make_tensor_value_info(
            f"{output.name}_relu",
            TensorProto.FLOAT16,
            [2, size],
        )
        for output, size in zip(split_outputs, (1, 3, 3))
    )
    sizes = numpy_helper.from_array(
        np.asarray((1, 3, 3), dtype=np.int64),
        name="sizes",
    )
    split = helper.make_node(
        "Split",
        inputs=("x", "sizes"),
        outputs=tuple(output.name for output in split_outputs),
        name="split",
        axis=1,
    )
    consumers = tuple(
        helper.make_node(
            "Relu",
            inputs=(split_output.name,),
            outputs=(graph_output.name,),
            name=f"{split_output.name}_consumer",
        )
        for split_output, graph_output in zip(split_outputs, graph_outputs)
    )
    onnx_graph = helper.make_graph(
        (split, *consumers),
        "branched_split",
        (x,),
        graph_outputs,
        initializer=(sizes,),
        value_info=split_outputs,
    )
    return ImportedModel(parse_graph(onnx_graph), ConstantStore(()))


@pytest.mark.parametrize(
    ("target", "device_name"),
    ((magia, "core"), (n300d, "tensix_vector")),
)
@pytest.mark.parametrize("dtype", (TensorDType.FLOAT16, TensorDType.FLOAT32))
def test_targets_assign_exact_three_output_split_signatures(
    target,
    device_name: str,
    dtype: TensorDType,
) -> None:
    signature = WorkSignature(
        WorkKind.SPLIT,
        (dtype,),
        (dtype, dtype, dtype),
    )

    assert target.build_mesh().tiles[0].assigned_device(signature).name == device_name


def test_public_planning_retains_one_split_layer_and_all_runtime_outputs() -> None:
    model = _split_model()
    mesh = magia.build_mesh(width=1, height=1)
    specialization = magia.specialize(
        model,
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )

    execution_plan = plan(
        specialization.model.graph,
        mesh,
        PlanningOptions(
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )
    bundle = build_deployment_bundle(specialization, execution_plan)

    assert specialization.model.graph == model.graph
    assert specialization.report.events == ()
    assert len(execution_plan.stages) == 1
    assert len(execution_plan.stages[0].layers) == 1
    layer = execution_plan.stages[0].layers[0]
    assert isinstance(layer.node.payload, SplitPayload)
    assert layer.device_name == "core"
    assert tuple(
        execution_plan.tensors[output.tensor_id]
        for output in layer.outputs
    ) == model.graph.outputs
    input_transition = next(
        transition
        for transition in execution_plan.transitions
        if isinstance(transition, InputTransition)
    )
    assert input_transition.destinations[0].tensor_slice.dims == (
        TensorRange(0, 2),
        TensorRange(0, 7),
    )
    assert tuple(
        execution_plan.tensors[transition.tensor_id]
        for transition in execution_plan.transitions
        if isinstance(transition, OutputTransition)
    ) == model.graph.outputs
    serialized_plan = execution_plan_json_payload(execution_plan)
    assert execution_plan_json_payload(bundle.execution_plan) == serialized_plan


def test_imported_split_plans_three_consumer_branches_deterministically() -> None:
    model = _onnx_split_model()
    mesh = magia.build_mesh(width=4, height=1)
    specialization = magia.specialize(
        model,
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )

    options = PlanningOptions(
        placement=PlacementOptions(print_placement=False),
        print_execution_plan_cost=False,
    )
    first = plan(specialization.model.graph, mesh, options)
    second = plan(specialization.model.graph, mesh, options)
    bundle = build_deployment_bundle(specialization, first)

    split, *consumers = specialization.model.graph.nodes
    assert isinstance(split.payload, SplitPayload)
    assert specialization.report.events == ()
    stage_id_by_node = {
        id(layer.node): stage_id
        for stage_id, stage in enumerate(first.stages)
        for layer in stage.layers
    }
    branch_stage_ids = (
        stage_id_by_node[id(split)],
        *(stage_id_by_node[id(node)] for node in consumers),
    )
    assert len(set(branch_stage_ids)) == 4
    assert tuple(
        first.tensors[transition.tensor_id]
        for transition in first.transitions
        if isinstance(transition, IntermediateTransition)
    ) == split.outputs
    assert first == second
    assert execution_plan_json_payload(first) == execution_plan_json_payload(second)
    assert execution_plan_json_payload(
        bundle.execution_plan
    ) == execution_plan_json_payload(first)

    stages = form_stages(
        specialization.model.graph,
        StageFormationOptions(max_stage_operations=1),
    )
    assert tuple(stages.values()) == tuple(
        (node,) for node in specialization.model.graph.nodes
    )


@pytest.mark.parametrize(
    "model",
    (
        _split_model(sizes=(2, 2)),
        _split_model(dtype=TensorDType.INT32),
    ),
)
def test_public_planning_rejects_an_undeclared_split_signature(
    model: ImportedModel,
) -> None:
    mesh = magia.build_mesh(width=1, height=1)

    with pytest.raises(ValueError, match="no fixed assignment"):
        plan(
            model.graph,
            mesh,
            PlanningOptions(
                placement=PlacementOptions(print_placement=False),
                print_execution_plan_cost=False,
            ),
        )
