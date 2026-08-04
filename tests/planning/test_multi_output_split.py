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
from maps.operations.elementwise import UnaryElementwisePayload
from maps.operations.split import SplitPayload
from maps.planning import (
    LocalInput,
    PlacementOptions,
    PlanningOptions,
    StageFormationOptions,
    TransitionSource,
    plan,
)
from maps.planning.allocation import allocate
from maps.planning.execution_plan import construct_execution_plan
from maps.planning.mapping import TensorRange
from maps.planning.placement import place
from maps.planning.stages import form_stages
from maps.planning.transitions import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
    VirtualIntermediateTransition,
    build_virtual_transitions,
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


def _onnx_split_model(
    *,
    batch_size: int = 2,
    sizes: tuple[int, int, int] = (1, 3, 3),
) -> ImportedModel:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info(
        "x",
        TensorProto.FLOAT16,
        [batch_size, sum(sizes)],
    )
    split_outputs = tuple(
        helper.make_tensor_value_info(
            name,
            TensorProto.FLOAT16,
            [batch_size, size],
        )
        for name, size in zip(("q", "k", "v"), sizes)
    )
    graph_outputs = tuple(
        helper.make_tensor_value_info(
            f"{output.name}_relu",
            TensorProto.FLOAT16,
            [batch_size, size],
        )
        for output, size in zip(split_outputs, sizes)
    )
    sizes = numpy_helper.from_array(
        np.asarray(sizes, dtype=np.int64),
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
    model = _onnx_split_model(batch_size=128, sizes=(1, 10, 100))
    mesh = magia.build_mesh(width=8, height=1)
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
    branch_stages = tuple(first.stages[stage_id] for stage_id in branch_stage_ids[1:])
    assert tuple(stage.submesh.num_tiles for stage in branch_stages) == (1, 1, 2)
    assert tuple(
        (
            stage.layers[0].outputs[0].layout.effective_logical_width,
            stage.layers[0].outputs[0].layout.effective_logical_height,
        )
        for stage in branch_stages
    ) == ((1, 1), (1, 1), (2, 1))
    assert len({stage.submesh.tile_ids for stage in branch_stages}) == 3
    assert tuple(
        first.tensors[transition.tensor_id]
        for transition in first.transitions
        if isinstance(transition, IntermediateTransition)
    ) == split.outputs
    assert all(
        transition.transfers
        for transition in first.transitions
        if isinstance(transition, IntermediateTransition)
    )
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


def test_split_fuses_with_one_local_consumer_without_an_intermediate_transition() -> None:
    split_model = _split_model()
    split = split_model.graph.nodes[0]
    local_output = Tensor(
        "local_output",
        split.outputs[0].rank,
        split.outputs[0].dims,
        split.outputs[0].elem_bytes,
        dtype=split.outputs[0].dtype,
    )
    consumer = Node(
        "local_consumer",
        OpKind.CUSTOM,
        inputs=(split.outputs[0],),
        outputs=(local_output,),
        payload=UnaryElementwisePayload(
            "relu",
            split.outputs[0],
            local_output,
        ),
    )
    graph = Graph(
        "split_with_local_consumer",
        tensors=(*split_model.graph.tensors, local_output),
        nodes=(split, consumer),
        edges=(
            Edge(split.inputs[0], None, split),
            Edge(split.outputs[0], split, consumer),
            Edge(local_output, consumer, None),
            Edge(split.outputs[1], split, None),
            Edge(split.outputs[2], split, None),
        ),
        inputs=split_model.graph.inputs,
        outputs=(local_output, split.outputs[1], split.outputs[2]),
    )

    execution_plan = plan(
        graph,
        magia.build_mesh(width=1, height=1),
        PlanningOptions(
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )

    assert len(execution_plan.stages) == 1
    assert tuple(layer.node for layer in execution_plan.stages[0].layers) == (
        split,
        consumer,
    )
    assert isinstance(execution_plan.stages[0].layers[1].inputs[0].source, LocalInput)
    assert not any(
        isinstance(transition, IntermediateTransition)
        for transition in execution_plan.transitions
    )
    assert tuple(
        execution_plan.tensors[transition.tensor_id]
        for transition in execution_plan.transitions
        if isinstance(transition, OutputTransition)
    ) == graph.outputs


def test_split_branches_share_destination_residency_and_fan_out_per_stage() -> None:
    split_model = _split_model()
    split = split_model.graph.nodes[0]

    def relu(name: str, tensor: Tensor) -> tuple[Node, Tensor]:
        output = Tensor(
            f"{name}_output",
            tensor.rank,
            tensor.dims,
            tensor.elem_bytes,
            dtype=tensor.dtype,
        )
        return (
            Node(
                name,
                OpKind.CUSTOM,
                inputs=(tensor,),
                outputs=(output,),
                payload=UnaryElementwisePayload("relu", tensor, output),
            ),
            output,
        )

    local_consumer, local_output = relu("local_consumer", split.outputs[1])
    shared_consumer0, shared_output0 = relu("shared_consumer0", split.outputs[0])
    shared_consumer1, shared_output1 = relu("shared_consumer1", split.outputs[0])
    remote_consumer, remote_output = relu("remote_consumer", split.outputs[0])
    consumers = (
        local_consumer,
        shared_consumer0,
        shared_consumer1,
        remote_consumer,
    )
    consumer_outputs = (
        local_output,
        shared_output0,
        shared_output1,
        remote_output,
    )
    graph = Graph(
        "split_mixed_branches",
        tensors=(
            *split_model.graph.tensors,
            local_output,
            shared_output0,
            shared_output1,
            remote_output,
        ),
        nodes=(split, *consumers),
        edges=(
            Edge(split.inputs[0], None, split),
            *(Edge(node.inputs[0], split, node) for node in consumers),
            *(
                Edge(output, node, None)
                for node, output in zip(consumers, consumer_outputs)
            ),
            Edge(split.outputs[2], split, None),
        ),
        inputs=split_model.graph.inputs,
        outputs=(
            local_output,
            shared_output0,
            shared_output1,
            remote_output,
            split.outputs[2],
        ),
    )
    stage_formation = {
        0: (split, local_consumer),
        1: (shared_consumer0, shared_consumer1),
        2: (remote_consumer,),
    }
    mesh = magia.build_mesh(width=4, height=1)
    stage_plans = allocate(graph, mesh, stage_formation)

    virtual_transitions = build_virtual_transitions(graph, stage_plans)

    split_fanout = tuple(
        transition
        for transition in virtual_transitions
        if isinstance(transition, VirtualIntermediateTransition)
        and transition.tensor is split.outputs[0]
    )
    assert tuple(
        transition.destination_stage_id for transition in split_fanout
    ) == (1, 2)
    assert all(transition.transfers for transition in split_fanout)
    assert not any(
        isinstance(transition, VirtualIntermediateTransition)
        and transition.tensor is split.outputs[1]
        for transition in virtual_transitions
    )
    assert build_virtual_transitions(graph, stage_plans) == virtual_transitions

    placements = place(
        mesh,
        stage_plans,
        virtual_transitions,
        print_placement=False,
    )
    execution_plan = construct_execution_plan(
        graph,
        mesh,
        stage_plans,
        placements,
        virtual_transitions,
    )
    shared_inputs = tuple(
        layer.inputs[0].source
        for layer in execution_plan.stages[1].layers
    )
    assert all(isinstance(source, TransitionSource) for source in shared_inputs)
    assert shared_inputs[0] == shared_inputs[1]
    assert isinstance(execution_plan.stages[0].layers[1].inputs[0].source, LocalInput)
    assert tuple(
        execution_plan.tensors[transition.tensor_id]
        for transition in execution_plan.transitions
        if isinstance(transition, OutputTransition)
    ) == graph.outputs


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
