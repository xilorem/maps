from dataclasses import replace
from pathlib import Path

from maps.graph import TensorDType
from maps.graph import Edge, Graph, Node, OpKind
from maps.planning.mapping import Submesh
from maps.graph import Tensor
from maps.target.magia import build_mesh as magia_mesh
from maps.operations.gemm import GemmPayload
from maps.operations.elementwise import BinaryElementwisePayload, UnaryElementwisePayload
from maps.planning import (
    ExecutionPlan,
    InitializerInput,
    Layer,
    LayerInput,
    LayerOutput,
    LocalInput,
    PlanningConstraints,
    Stage,
    TransitionSource,
    validate_execution_plan,
)
from maps.planning.execution_plan import construct_execution_plan
from maps.planning.execution_plan import estimate_stage_l1_memory_for_tile
from maps.planning.execution_plan import estimate_stage_l2_memory
from maps.planning.stages import StagePlacement, StagePlan
from maps.planning.allocation.candidates import permanent_l1_allocation_for_tile
from maps.planning.transitions import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
    build_virtual_transitions,
)
from maps.deployment.serialization import (
    execution_plan_json_payload,
    write_execution_plan_json,
)


def _placement(
    stage_id: int,
    virtual_submesh: Submesh,
    physical_submesh: Submesh,
) -> StagePlacement:
    return StagePlacement(
        stage_id=stage_id,
        virtual_submesh=virtual_submesh,
        physical_submesh=physical_submesh,
        virtual_to_physical={
            virtual.tile_id: physical.tile_id
            for virtual, physical in zip(
                virtual_submesh.tiles,
                physical_submesh.tiles,
            )
        },
    )


def test_execution_plan_serializes_shard_granularity_for_every_layout_axis() -> None:
    from maps.planning.mapping import LayoutAxis, LayoutAxisMode, TensorLayout

    mesh = magia_mesh(width=2, height=1)
    submesh = Submesh(mesh, 0, frozenset((0, 1)))
    tensor = Tensor("flattened_rows", 1, (9,), 2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(
            LayoutAxisMode.SHARD,
            tensor_axis=0,
            shard_granularity=3,
        ),
        mesh_y=LayoutAxis(LayoutAxisMode.REPLICATE),
        logical_width=2,
        logical_height=1,
    )
    stage = Stage(
        "granular",
        submesh,
        layers=(
            Layer(
                Node("source", OpKind.CUSTOM, outputs=(tensor,)),
                outputs=(LayerOutput(tensor_id=0, layout=layout),),
            ),
        ),
    )
    execution_plan = ExecutionPlan(
        "granular",
        mesh,
        tensors=(tensor,),
        stages=(stage,),
    )

    serialized_layout = execution_plan_json_payload(execution_plan)["stages"][0][
        "layers"
    ][0]["outputs"][0]["layout"]

    assert serialized_layout["mesh_x"]["shard_granularity"] == 3
    assert serialized_layout["mesh_y"]["shard_granularity"] == 1


def test_stage_tensor_residency_is_shared_and_internal_outputs_can_leave() -> None:
    mesh = magia_mesh(width=1, height=1)
    virtual = Submesh(mesh=mesh, submesh_id=0, tile_ids=frozenset((0,)))
    x = Tensor("x", 1, (4,), 2, dtype=TensorDType.FLOAT16)
    middle = Tensor("middle", 1, (4,), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 1, (4,), 2, dtype=TensorDType.FLOAT16)
    first_payload = UnaryElementwisePayload("Relu", x, middle)
    second_payload = BinaryElementwisePayload("Add", middle, x, output)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(middle,),
        payload=first_payload,
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(middle, x),
        outputs=(output,),
        payload=second_payload,
    )
    graph = Graph(
        "resident",
        tensors=(x, middle, output),
        nodes=(first, second),
        inputs=(x,),
        outputs=(middle, output),
    )
    plan = StagePlan(
        stage_id=0,
        tile_count=1,
        logical_shape=(1, 1),
        nodes=(first, second),
        node_output_layouts=(
            first_payload.output_layouts(virtual, (1, 1)),
            second_payload.output_layouts(virtual, (1, 1)),
        ),
        device_names=("spatz", "spatz"),
    )
    transitions = build_virtual_transitions(graph, {0: plan})

    execution_plan = construct_execution_plan(
        graph,
        mesh,
        {0: plan},
        {0: _placement(0, virtual, virtual)},
        transitions,
    )

    input_sources = (
        execution_plan.stages[0].layers[0].inputs[0].source,
        execution_plan.stages[0].layers[1].inputs[1].source,
    )
    assert all(isinstance(source, TransitionSource) for source in input_sources)
    assert {source.transition_id for source in input_sources} == {0}
    assert tuple(type(transition) for transition in execution_plan.transitions) == (
        InputTransition,
        OutputTransition,
        OutputTransition,
    )
    assert execution_plan.transitions[1].tensor_id == 1
    assert estimate_stage_l1_memory_for_tile(
        execution_plan.stages[0],
        execution_plan,
        mesh.tiles[0],
    ) == 48
    assert permanent_l1_allocation_for_tile(
        plan.nodes,
        plan.node_output_layouts,
        mesh.tiles[0],
        frozenset(),
    ) == 48
    assert estimate_stage_l2_memory(
        execution_plan.stages[0],
        execution_plan,
    ) == 8
    report = validate_execution_plan(execution_plan, PlanningConstraints())
    assert report.is_valid, report.violations


def test_construct_execution_plan_unifies_communication_and_initializer_residency(
    tmp_path: Path,
) -> None:
    mesh = magia_mesh()
    virtual0 = Submesh(mesh=mesh, submesh_id=0, tile_ids=frozenset((0, 1)))
    virtual1 = Submesh(mesh=mesh, submesh_id=1, tile_ids=frozenset((0, 1)))
    physical0 = Submesh(mesh=mesh, submesh_id=2, tile_ids=frozenset((8, 9)))
    physical1 = Submesh(mesh=mesh, submesh_id=3, tile_ids=frozenset((16, 17)))

    x = Tensor("x", 2, (4, 4), 2, dtype=TensorDType.FLOAT16)
    w0 = Tensor("w0", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    y0 = Tensor("y0", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    w1 = Tensor("w1", 2, (8, 8), 2, dtype=TensorDType.FLOAT16)
    y1 = Tensor("y1", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    w2 = Tensor("w2", 2, (8, 6), 2, dtype=TensorDType.FLOAT16)
    z = Tensor("z", 2, (4, 6), 2, dtype=TensorDType.FLOAT16)

    payload0 = GemmPayload(x=x, w=w0, y=None, output=y0)
    payload1 = GemmPayload(x=y0, w=w1, y=None, output=y1)
    payload2 = GemmPayload(x=y1, w=w2, y=None, output=z)
    node0 = Node(
        "gemm_0",
        OpKind.GEMM,
        inputs=(x, w0),
        outputs=(y0,),
        payload=payload0,
    )
    node1 = Node(
        "gemm_1",
        OpKind.GEMM,
        inputs=(y0, w1),
        outputs=(y1,),
        payload=payload1,
    )
    node2 = Node(
        "gemm_2",
        OpKind.GEMM,
        inputs=(y1, w2),
        outputs=(z,),
        payload=payload2,
    )
    graph = Graph(
        name="unified",
        tensors=(x, w0, y0, w1, y1, w2, z),
        nodes=(node0, node1, node2),
        edges=(
            Edge(x, None, node0),
            Edge(w0, None, node0),
            Edge(y0, node0, node1),
            Edge(w1, None, node1),
            Edge(y1, node1, node2),
            Edge(w2, None, node2),
            Edge(z, node2, None),
        ),
        inputs=(x,),
        outputs=(z,),
        initializers=(w0, w1, w2),
    )
    plans = {
        0: StagePlan(
            stage_id=0,
            tile_count=2,
            logical_shape=(2, 1),
            nodes=(node0, node1),
            node_output_layouts=(
                payload0.output_layouts(virtual0, logical_shape=(2, 1)),
                payload1.output_layouts(virtual0, logical_shape=(2, 1)),
            ),
            device_names=("redmule", "redmule"),
        ),
        1: StagePlan(
            stage_id=1,
            tile_count=2,
            logical_shape=(2, 1),
            nodes=(node2,),
            node_output_layouts=(
                payload2.output_layouts(virtual1, logical_shape=(2, 1)),
            ),
            device_names=("redmule",),
        ),
    }
    placements = {
        0: _placement(0, virtual0, physical0),
        1: _placement(1, virtual1, physical1),
    }
    virtual_submeshes = {0: virtual0, 1: virtual1}
    virtual_transitions = build_virtual_transitions(graph, plans)

    execution_plan = construct_execution_plan(
        graph,
        mesh,
        plans,
        placements,
        virtual_transitions,
    )

    assert execution_plan.name == "unified"
    assert tuple(type(item) for item in execution_plan.transitions) == (
        InputTransition,
        IntermediateTransition,
        OutputTransition,
    )
    assert isinstance(
        execution_plan.stages[0].layers[0].inputs[0].source,
        TransitionSource,
    )
    assert execution_plan.stages[0].layers[0].inputs[0].source.transition_id == 0
    assert isinstance(
        execution_plan.stages[0].layers[0].inputs[1].source,
        InitializerInput,
    )
    assert {
        destination.tile_id
        for destination in execution_plan.stages[0].layers[0].inputs[1].source.destinations
    } == {8, 9}
    assert isinstance(
        execution_plan.stages[0].layers[1].inputs[0].source,
        LocalInput,
    )
    assert isinstance(
        execution_plan.stages[0].layers[1].inputs[1].source,
        InitializerInput,
    )
    assert isinstance(
        execution_plan.stages[1].layers[0].inputs[0].source,
        TransitionSource,
    )
    assert execution_plan.stages[1].layers[0].inputs[0].source.transition_id == 1
    assert isinstance(
        execution_plan.stages[1].layers[0].inputs[1].source,
        InitializerInput,
    )
    for stage_id, plan in plans.items():
        for virtual_tile in virtual_submeshes[stage_id].tiles:
            physical_tile = mesh.tile_by_id(
                placements[stage_id].physical_tile_id(virtual_tile.tile_id)
            )
            assert estimate_stage_l1_memory_for_tile(
                execution_plan.stages[stage_id],
                execution_plan,
                physical_tile,
            ) == permanent_l1_allocation_for_tile(
                plan.nodes,
                plan.node_output_layouts,
                virtual_tile,
                frozenset(graph.initializers),
            )

    report = validate_execution_plan(execution_plan, PlanningConstraints())
    assert report.is_valid, report.violations

    payload = execution_plan_json_payload(execution_plan)
    assert execution_plan_json_payload(execution_plan) == payload
    written_payload = write_execution_plan_json(
        execution_plan,
        tmp_path / "execution-plan.json",
    ).read_text(encoding="utf-8")
    assert '"kind": "INTERMEDIATE"' in written_payload
    assert set(payload) == {
        "name",
        "mesh",
        "tensors",
        "stages",
        "transitions",
        "execution",
    }
    assert [transition["kind"] for transition in payload["transitions"]] == [
        "INPUT",
        "INTERMEDIATE",
        "OUTPUT",
    ]
    assert set(payload["transitions"][0]) == {
        "kind",
        "tensor_id",
        "destination_stage_id",
        "destinations",
    }
    assert set(payload["transitions"][1]) == {
        "kind",
        "tensor_id",
        "source_stage_id",
        "destination_stage_id",
        "transfers",
    }
    assert set(payload["transitions"][2]) == {
        "kind",
        "tensor_id",
        "source_stage_id",
        "sources",
    }
    serialized = str(payload["transitions"])
    for legacy_name in (
        "initializations",
        "finalizations",
        "src_hartid",
        "dst_hartid",
        "fragments",
        "src_layout",
        "dst_layout",
        "mode",
    ):
        assert legacy_name not in serialized


def test_execution_plan_validation_rejects_transition_endpoint_mismatches() -> None:
    from maps.hardware import L2Memory, Mesh
    from maps.planning.mapping import TensorRange, TensorSlice
    from maps.planning.transitions import InputDestination
    from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles

    mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=1024, bandwidth=1),
        noc=rectangular_test_noc(1, 1),
        tiles=rectangular_test_tiles(1, 1),
    )
    tensor = Tensor("x", 1, (4,), 1)
    node = Node("node", OpKind.CUSTOM, inputs=(tensor,))
    stage = Stage(
        "stage",
        Submesh(mesh, 0, frozenset((0,))),
        layers=(Layer(node, inputs=(LayerInput.transition_source(0, 0),)),),
    )
    transition = InputTransition(
        tensor_id=0,
        destination_stage_id=0,
        destinations=(),
    )
    execution_plan = ExecutionPlan(
        "invalid",
        mesh,
        tensors=(tensor,),
        stages=(stage,),
        transitions=(transition,),
    )

    invalid_transition = replace(
        transition,
        destinations=(
            InputDestination(
                tile_id=1,
                tensor_slice=TensorSlice(
                    rank=1,
                    dims=(TensorRange(start=3, length=2),),
                ),
            ),
        ),
    )
    invalid_plan = replace(execution_plan, transitions=(invalid_transition,))

    report = validate_execution_plan(invalid_plan, PlanningConstraints())

    assert {violation.kind for violation in report.violations} >= {
        "transition_destination_tile_out_of_mesh",
        "transition_slice_invalid",
    }

    invalid_input = replace(stage.layers[0].inputs[0], tensor_id=1)
    invalid_layer = replace(stage.layers[0], inputs=(invalid_input,))
    invalid_stage = replace(stage, layers=(invalid_layer,))
    invalid_binding_plan = replace(execution_plan, stages=(invalid_stage,))

    binding_report = validate_execution_plan(
        invalid_binding_plan,
        PlanningConstraints(),
    )

    assert "stage_tensor_binding_invalid" in {
        violation.kind
        for violation in binding_report.violations
    }

    initializer_tensor = replace(tensor, is_initializer=True)
    initializer_input = LayerInput.initializer(tensor_id=0, destinations=())
    initializer_layer = replace(stage.layers[0], inputs=(initializer_input,))
    initializer_stage = replace(stage, layers=(initializer_layer,))
    missing_residency_plan = replace(
        execution_plan,
        tensors=(initializer_tensor,),
        stages=(initializer_stage,),
        transitions=(),
    )

    residency_report = validate_execution_plan(
        missing_residency_plan,
        PlanningConstraints(),
    )

    assert "initializer_residency_tiles_mismatch" in {
        violation.kind
        for violation in residency_report.violations
    }

    initializer_transition_plan = replace(
        execution_plan,
        tensors=(initializer_tensor,),
    )
    initializer_transition_report = validate_execution_plan(
        initializer_transition_plan,
        PlanningConstraints(),
    )
    assert "initializer_transition" in {
        violation.kind
        for violation in initializer_transition_report.violations
    }

    negative_reference_cases = (
        (
            replace(transition, tensor_id=-1),
            "transition_tensor_out_of_range",
        ),
        (
            replace(transition, destination_stage_id=-1),
            "transition_destination_stage_out_of_range",
        ),
    )
    for invalid_reference, expected_violation in negative_reference_cases:
        negative_report = validate_execution_plan(
            replace(execution_plan, transitions=(invalid_reference,)),
            PlanningConstraints(),
        )
        assert expected_violation in {
            violation.kind
            for violation in negative_report.violations
        }


def test_execution_plan_validation_rejects_mismatched_transfer_regions() -> None:
    from maps.hardware import L2Memory, Mesh
    from maps.planning.mapping import (
        LayoutAxis,
        LayoutAxisMode,
        TensorLayout,
        TensorRange,
        TensorSlice,
        TensorSubSlice,
    )
    from maps.planning.transitions import IntermediateTransition, Transfer
    from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles

    mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=1024, bandwidth=1),
        noc=rectangular_test_noc(1, 1),
        tiles=rectangular_test_tiles(1, 1),
    )
    submesh = Submesh(mesh, 0, frozenset((0,)))
    tensor = Tensor("value", 1, (4,), 1)
    layout = TensorLayout(
        submesh,
        LayoutAxis(LayoutAxisMode.REPLICATE),
        LayoutAxis(LayoutAxisMode.REPLICATE),
    )
    source_node = Node("source", OpKind.CUSTOM, outputs=(tensor,))
    destination_node = Node("destination", OpKind.CUSTOM, inputs=(tensor,))
    source_stage = Stage(
        "source",
        submesh,
        layers=(
            Layer(
                source_node,
                outputs=(LayerOutput(tensor_id=0, layout=layout),),
            ),
        ),
    )
    destination_stage = Stage(
        "destination",
        submesh,
        layers=(
            Layer(
                destination_node,
                inputs=(LayerInput.transition_source(0, 0),),
            ),
        ),
    )
    parent = TensorSlice(1, (TensorRange(start=0, length=4),))
    transition = IntermediateTransition(
        tensor_id=0,
        source_stage_id=0,
        destination_stage_id=1,
        transfers=(
            Transfer(
                source_tile_id=0,
                destination_tile_id=0,
                source_subslice=TensorSubSlice(
                    parent,
                    (TensorRange(start=0, length=2),),
                ),
                destination_subslice=TensorSubSlice(
                    parent,
                    (TensorRange(start=1, length=2),),
                ),
            ),
        ),
    )
    execution_plan = ExecutionPlan(
        "invalid-transfer",
        mesh,
        tensors=(tensor,),
        stages=(source_stage, destination_stage),
        transitions=(transition,),
    )

    report = validate_execution_plan(
        execution_plan,
        PlanningConstraints(
            enforce_l1_capacity=False,
            enforce_l2_capacity=False,
        ),
    )

    assert "transfer_subslice_region_mismatch" in {
        violation.kind
        for violation in report.violations
    }

    negative_source_report = validate_execution_plan(
        replace(
            execution_plan,
            transitions=(
                replace(
                    transition,
                    source_stage_id=-1,
                ),
            ),
        ),
        PlanningConstraints(
            enforce_l1_capacity=False,
            enforce_l2_capacity=False,
        ),
    )
    assert {
        "transition_source_stage_out_of_range",
    } <= {
        violation.kind
        for violation in negative_source_report.violations
    }

    same_stage_report = validate_execution_plan(
        replace(
            execution_plan,
            transitions=(
                replace(transition, destination_stage_id=0),
            ),
        ),
        PlanningConstraints(
            enforce_l1_capacity=False,
            enforce_l2_capacity=False,
        ),
    )
    assert "intermediate_transition_within_stage" in {
        violation.kind
        for violation in same_stage_report.violations
    }
