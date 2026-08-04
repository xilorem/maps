import json
from pathlib import Path

import pytest

from maps.deployment import build_deployment_bundle, write_execution_plan_bundle
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
    run_graph_rewrites_with_effects,
)
from maps.graph.onnx.parser import parse_graph
from maps.hardware import WorkKind, WorkSignature
from maps.operations.elementwise import BinaryElementwisePayload, UnaryElementwisePayload
from maps.operations.split import SplitPayload, StaticSlicePayload
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


def _tensor_like(name: str, tensor: Tensor) -> Tensor:
    return Tensor(
        name,
        tensor.rank,
        tensor.dims,
        tensor.elem_bytes,
        dtype=tensor.dtype,
    )


def _relu(name: str, tensor: Tensor) -> tuple[Node, Tensor]:
    output = _tensor_like(f"{name}_output", tensor)
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


def _onnx_split_model(
    *,
    batch_size: int = 2,
    sizes: tuple[int, ...] = (1, 3, 3),
    dtype: TensorDType = TensorDType.FLOAT16,
) -> ImportedModel:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    onnx_dtype = {
        TensorDType.FLOAT16: TensorProto.FLOAT16,
        TensorDType.INT32: TensorProto.INT32,
    }[dtype]
    x = helper.make_tensor_value_info(
        "x",
        onnx_dtype,
        [batch_size, sum(sizes)],
    )
    output_names = tuple(
        ("q", "k", "v")[index] if index < 3 else f"output{index}"
        for index in range(len(sizes))
    )
    split_outputs = tuple(
        helper.make_tensor_value_info(
            name,
            onnx_dtype,
            [batch_size, size],
        )
        for name, size in zip(output_names, sizes)
    )
    graph_outputs = tuple(
        helper.make_tensor_value_info(
            f"{output.name}_relu",
            onnx_dtype,
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


def test_imported_split_deploys_three_consumer_branches_deterministically(
    tmp_path: Path,
) -> None:
    model = _onnx_split_model(batch_size=128, sizes=(1, 10, 100))
    mesh = magia.build_mesh(width=8, height=1)
    options = PlanningOptions(
        placement=PlacementOptions(print_placement=False),
        print_execution_plan_cost=False,
    )

    bundles = []
    serialized_payloads = []
    serialized_bytes = []
    for build_index in range(2):
        rewritten_model, rewrite_effects = run_graph_rewrites_with_effects(model)
        specialization = magia.specialize(
            rewritten_model,
            mesh,
            SpecializationOptions(enable_precision_lowering=False),
        )
        execution_plan = plan(specialization.model.graph, mesh, options)
        bundle = build_deployment_bundle(
            specialization,
            execution_plan,
            graph_rewrite_effects=rewrite_effects,
        )
        output_dir = tmp_path / f"build_{build_index}"
        output_json, _ = write_execution_plan_bundle(
            bundle,
            output_dir / "execution_plan.json",
            output_dir / "weights.bin",
        )
        bundles.append(bundle)
        serialized_payloads.append(
            json.loads(output_json.read_text(encoding="utf-8"))
        )
        serialized_bytes.append(output_json.read_bytes())

    first_bundle, second_bundle = bundles
    first_execution_plan = first_bundle.execution_plan
    second_execution_plan = second_bundle.execution_plan
    specialization_graph = first_bundle.graph

    split, *consumers = specialization_graph.nodes
    assert isinstance(split.payload, SplitPayload)
    assert tuple(
        node
        for node in specialization_graph.nodes
        if isinstance(node.payload, SplitPayload)
    ) == (split,)
    assert not any(
        isinstance(node.payload, StaticSlicePayload)
        for node in specialization_graph.nodes
    )
    assert first_bundle.rewrite_report.events == ()
    stage_id_by_node = {
        id(layer.node): stage_id
        for stage_id, stage in enumerate(first_execution_plan.stages)
        for layer in stage.layers
    }
    branch_stage_ids = (
        stage_id_by_node[id(split)],
        *(stage_id_by_node[id(node)] for node in consumers),
    )
    assert len(set(branch_stage_ids)) == 4
    split_stage = first_execution_plan.stages[branch_stage_ids[0]]
    split_layer = split_stage.layers[0]
    assert tuple(
        layer
        for stage in first_execution_plan.stages
        for layer in stage.layers
        if isinstance(layer.node.payload, SplitPayload)
    ) == (split_layer,)
    assert not any(
        isinstance(layer.node.payload, StaticSlicePayload)
        for stage in first_execution_plan.stages
        for layer in stage.layers
    )
    branch_stages = tuple(
        first_execution_plan.stages[stage_id]
        for stage_id in branch_stage_ids[1:]
    )
    branch_output_layouts = tuple(
        stage.layers[0].outputs[0].layout
        for stage in branch_stages
    )
    assert tuple(stage.submesh.num_tiles for stage in branch_stages) == (1, 1, 2)
    assert tuple(
        (
            layout.effective_logical_width,
            layout.effective_logical_height,
        )
        for layout in branch_output_layouts
    ) == ((1, 1), (1, 1), (2, 1))
    assert len({stage.submesh.tile_ids for stage in branch_stages}) == 3
    assert tuple(
        first_execution_plan.tensors[transition.tensor_id]
        for transition in first_execution_plan.transitions
        if isinstance(transition, IntermediateTransition)
    ) == split.outputs
    transition_id_by_tensor = {
        first_execution_plan.tensors[transition.tensor_id]: transition_id
        for transition_id, transition in enumerate(first_execution_plan.transitions)
        if isinstance(transition, IntermediateTransition)
    }
    for output, consumer in zip(split.outputs, consumers):
        transition_id = transition_id_by_tensor[output]
        transition = first_execution_plan.transitions[transition_id]
        consumer_stage_id = stage_id_by_node[id(consumer)]
        consumer_layer = first_execution_plan.stages[consumer_stage_id].layers[0]
        assert isinstance(transition, IntermediateTransition)
        assert transition.destination_stage_id == consumer_stage_id
        assert consumer_layer.inputs[0].source == TransitionSource(transition_id)
    assert split_layer.node == split
    assert split_layer.source_operation == "split"
    assert split_layer.device_name == "core"
    assert tuple(
        first_execution_plan.tensors[output.tensor_id]
        for output in split_layer.outputs
    ) == split.outputs
    split_output_layouts = tuple(output.layout for output in split_layer.outputs)
    assert all(
        layout.submesh == split_output_layouts[0].submesh
        and (
            layout.effective_logical_width,
            layout.effective_logical_height,
        )
        == (
            split_output_layouts[0].effective_logical_width,
            split_output_layouts[0].effective_logical_height,
        )
        for layout in split_output_layouts
    )
    assert set(split_output_layouts[0].submesh.tile_ids) == set(
        split_stage.virtual_to_physical
    )
    remapped_transition = next(
        transition
        for transition in first_execution_plan.transitions
        if isinstance(transition, IntermediateTransition)
        and first_execution_plan.tensors[transition.tensor_id] == split.outputs[2]
    )
    assert tuple(
        (
            transfer.source_tile_id,
            transfer.destination_tile_id,
            transfer.source_subslice.dims,
            transfer.destination_subslice.dims,
        )
        for transfer in remapped_transition.transfers
    ) == tuple(
        (
            source_tile_id,
            destination_tile_id,
            (
                TensorRange(0, 32),
                TensorRange(destination_column * 50, 50),
            ),
            (
                TensorRange(source_row * 32, 32),
                TensorRange(0, 50),
            ),
        )
        for source_tile_id, source_row in ((1, 0), (2, 1), (0, 2), (3, 3))
        for destination_tile_id, destination_column in ((4, 0), (5, 1))
    )
    assert tuple(
        first_execution_plan.tensors[transition.tensor_id]
        for transition in first_execution_plan.transitions
        if isinstance(transition, OutputTransition)
    ) == specialization_graph.outputs
    assert first_bundle.graph == second_bundle.graph
    assert first_bundle.rewrite_report == second_bundle.rewrite_report
    assert first_execution_plan == second_execution_plan
    assert first_execution_plan.transitions == second_execution_plan.transitions
    assert serialized_payloads[0] == serialized_payloads[1]
    assert serialized_bytes[0] == serialized_bytes[1]

    serialized_plan = serialized_payloads[0]
    serialized_split_layers = tuple(
        layer
        for stage in serialized_plan["stages"]
        for layer in stage["layers"]
        if layer["node"]["name"] == "split"
    )
    assert len(serialized_split_layers) == 1
    serialized_split_layer = serialized_split_layers[0]
    assert serialized_split_layer["node"]["payload"]["work_kind"] == "SPLIT"
    assert serialized_split_layer["node"]["payload"]["axis"] == 1
    assert serialized_split_layer["node"]["payload"]["sizes"] == [1, 10, 100]
    assert serialized_split_layer["source_operation"] == "split"
    assert serialized_split_layer["device_name"] == "core"
    assert [
        output["tensor_id"] for output in serialized_split_layer["outputs"]
    ] == [specialization_graph.tensors.index(output) for output in split.outputs]
    for serialized_output, planned_output in zip(
        serialized_split_layer["outputs"],
        split_layer.outputs,
    ):
        layout = planned_output.layout
        serialized_layout = serialized_output["layout"]
        assert (
            serialized_layout["submesh"]["submesh_id"]
            == layout.submesh.submesh_id
        )
        assert serialized_layout["submesh"]["tile_ids"] == sorted(
            layout.submesh.tile_ids
        )
        assert serialized_layout["mesh_x"] == {
            "mode": layout.mesh_x.mode,
            "tensor_axis": layout.mesh_x.tensor_axis,
        }
        assert serialized_layout["mesh_y"] == {
            "mode": layout.mesh_y.mode,
            "tensor_axis": layout.mesh_y.tensor_axis,
        }
        assert serialized_layout["logical_width"] == layout.logical_width
        assert serialized_layout["logical_height"] == layout.logical_height
    assert [
        first_execution_plan.tensors[transition["tensor_id"]].name
        for transition in serialized_plan["transitions"]
        if transition["kind"] == "INTERMEDIATE"
    ] == ["q", "k", "v"]
    assert [
        first_execution_plan.tensors[transition["tensor_id"]].name
        for transition in serialized_plan["transitions"]
        if transition["kind"] == "OUTPUT"
    ] == ["q_relu", "k_relu", "v_relu"]
    assert all(
        "transfers" in transition
        for transition in serialized_plan["transitions"]
        if transition["kind"] == "INTERMEDIATE"
    )
    assert all(
        "sources" in transition and "transfers" not in transition
        for transition in serialized_plan["transitions"]
        if transition["kind"] == "OUTPUT"
    )
    assert serialized_plan["provenance"]["rewrite_report"] == []

    stages = form_stages(
        specialization_graph,
        StageFormationOptions(max_stage_operations=1),
    )
    assert tuple(stages.values()) == tuple(
        (node,) for node in specialization_graph.nodes
    )


def test_split_fuses_with_one_local_consumer_without_an_intermediate_transition() -> None:
    split_model = _split_model()
    split = split_model.graph.nodes[0]
    consumer, local_output = _relu("local_consumer", split.outputs[0])
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
    local_consumer, local_output = _relu("local_consumer", split.outputs[1])
    shared_consumer0, shared_output0 = _relu(
        "shared_consumer0",
        split.outputs[0],
    )
    shared_output1 = _tensor_like(
        "shared_consumer1_output",
        split.outputs[0],
    )
    shared_consumer1 = Node(
        "shared_consumer1",
        OpKind.CUSTOM,
        inputs=(split.outputs[0], shared_output0),
        outputs=(shared_output1,),
        payload=BinaryElementwisePayload(
            "add",
            split.outputs[0],
            shared_output0,
            shared_output1,
        ),
    )
    remote_consumer, remote_output = _relu(
        "remote_consumer",
        split.outputs[0],
    )
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
            Edge(split.outputs[1], split, local_consumer),
            Edge(split.outputs[0], split, shared_consumer0),
            Edge(split.outputs[0], split, shared_consumer1),
            Edge(shared_output0, shared_consumer0, shared_consumer1),
            Edge(split.outputs[0], split, remote_consumer),
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


@pytest.mark.parametrize(
    ("sizes", "dtype"),
    (
        ((2, 2), TensorDType.FLOAT16),
        ((1, 3, 3), TensorDType.INT32),
    ),
)
def test_imported_split_rejects_an_undeclared_signature_with_node_diagnostic(
    sizes: tuple[int, ...],
    dtype: TensorDType,
) -> None:
    model = _onnx_split_model(sizes=sizes, dtype=dtype)
    mesh = magia.build_mesh(width=1, height=1)
    rewritten_model, _ = run_graph_rewrites_with_effects(model)
    specialization = magia.specialize(
        rewritten_model,
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )

    with pytest.raises(ValueError) as error:
        plan(
            specialization.model.graph,
            mesh,
            PlanningOptions(
                placement=PlacementOptions(print_placement=False),
                print_execution_plan_cost=False,
            ),
        )

    message = str(error.value)
    expected_signature = WorkSignature(
        WorkKind.SPLIT,
        (dtype,),
        (dtype,) * len(sizes),
    )
    assert "node split" in message
    assert str(expected_signature) in message
    assert "no fixed assignment" in message
