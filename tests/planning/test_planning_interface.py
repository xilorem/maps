import ast
from dataclasses import replace
from pathlib import Path

import pytest

from maps.graph import (
    ConstantStore,
    Edge,
    Graph,
    GraphRewriteEffect,
    ImportedModel,
    Node,
    OpKind,
    Tensor,
    TensorDType,
    run_graph_rewrites_with_effects,
)
from maps.hardware import FixedDeviceAssignment, WorkKind
from maps.operations.elementwise import UnaryElementwisePayload
from maps.operations.collective import AllReducePayload
from maps.operations.gemm import GemmPayload
from maps.operations.normalization import GroupNormalizationPayload
from maps.operations.reduction import GlobalAveragePoolPayload, ReduceSumPayload
from maps.operations.softmax import SoftmaxPayload
from maps.planning import (
    AllocationOptions,
    CollectiveGroup,
    ConstraintReport,
    ConstraintViolation,
    ExecutionContract,
    ExecutionPlan,
    InitializerInput,
    Layer,
    LayerInput,
    LayerOutput,
    LocalInput,
    PlacementOptions,
    PlanningConstraints,
    PlanningOptions,
    Stage,
    StageFormationOptions,
    TransitionSource,
    plan,
    validate_execution_plan,
)
from maps.planning.execution_plan import construct_execution_plan
from maps.planning.execution_plan import estimate_stage_l1_memory_for_tile
from maps.planning.execution_plan import print_execution_plan_stage_cost
from maps.planning.allocation import allocate
from maps.planning.mapping import LayoutAxis, LayoutAxisMode, TensorLayout
from maps.planning.placement import place
from maps.planning.mapping import Submesh
from maps.planning.stages import form_stages
from maps.planning.stages import StagePlan
from maps.planning.transitions import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
    build_virtual_transitions,
)
from maps.deployment import build_deployment_bundle
from maps.deployment.serialization import execution_plan_json_payload
from maps.target import SpecializationOptions, SpecializationResult, magia, n300d

from tests.target.test_precision_lowering import _gemm_model


class _CollidingAllReducePayload(AllReducePayload):
    @property
    def work_kind(self) -> WorkKind:
        return WorkKind.GROUP_REDUCE


def test_planning_owns_stage_formation_and_allocation_contracts() -> None:
    assert StageFormationOptions.__module__ == "maps.planning.options"
    assert AllocationOptions.__module__ == "maps.planning.options"
    assert form_stages.__module__ == "maps.planning.stages"
    assert allocate.__module__ == "maps.planning.allocation.selection"
    assert StagePlan.__module__ == "maps.planning.stages"


def test_planning_owns_transitions_placement_and_layout_contracts() -> None:
    assert build_virtual_transitions.__module__ == "maps.planning.transitions.compile"
    assert place.__module__ == "maps.planning.placement"
    assert TensorLayout.__module__ == "maps.planning.mapping"
    assert Submesh.__module__ == "maps.planning.mapping"


def test_planning_owns_execution_plan_construction_and_validation() -> None:
    assert ExecutionContract.__module__ == "maps.planning.execution_plan"
    assert ExecutionPlan.__module__ == "maps.planning.execution_plan"
    assert Stage.__module__ == "maps.planning.execution_plan"
    assert Layer.__module__ == "maps.planning.execution_plan"
    assert LayerInput.__module__ == "maps.planning.execution_plan"
    assert LayerOutput.__module__ == "maps.planning.execution_plan"
    assert InitializerInput.__module__ == "maps.planning.execution_plan"
    assert TransitionSource.__module__ == "maps.planning.execution_plan"
    assert LocalInput.__module__ == "maps.planning.execution_plan"
    assert construct_execution_plan.__module__ == "maps.planning.execution_plan"
    assert PlanningConstraints.__module__ == "maps.planning.validation"
    assert ConstraintViolation.__module__ == "maps.planning.validation"
    assert ConstraintReport.__module__ == "maps.planning.validation"
    assert validate_execution_plan.__module__ == "maps.planning.validation"
    assert estimate_stage_l1_memory_for_tile.__module__ == "maps.planning.execution_plan"
    assert print_execution_plan_stage_cost.__module__ == "maps.planning.execution_plan"


def test_planning_has_no_downstream_deployment_dependencies() -> None:
    planning_package = Path(__file__).parents[2] / "maps" / "planning"
    imports = set()
    for source_path in planning_package.rglob("*.py"):
        for node in ast.walk(ast.parse(source_path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert not {
        name
        for name in imports
        if name == "maps.deployment" or name.startswith("maps.deployment.")
    }


@pytest.fixture
def fused_softmax_planning() -> tuple[
    SpecializationResult,
    tuple[GraphRewriteEffect, ...],
    dict[int, StagePlan],
    ExecutionPlan,
]:
    x = Tensor("x", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    produced = Tensor("produced", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    normalized = Tensor("normalized", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    producer = Node(
        "producer",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(produced,),
        payload=UnaryElementwisePayload("Relu", x, produced),
    )
    softmax = Node(
        "softmax",
        OpKind.CUSTOM,
        inputs=(produced,),
        outputs=(normalized,),
        payload=SoftmaxPayload(produced, normalized, axis=1),
    )
    consumer = Node(
        "consumer",
        OpKind.ELEMENTWISE,
        inputs=(normalized,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Neg", normalized, output),
    )
    model, rewrite_effects = run_graph_rewrites_with_effects(
        ImportedModel(
            Graph(
                "fused_softmax",
                tensors=(x, produced, normalized, output),
                nodes=(producer, softmax, consumer),
                inputs=(x,),
                outputs=(output,),
            ),
            ConstantStore(()),
        )
    )
    mesh = magia.build_mesh(width=2, height=1)
    specialized = magia.specialize(
        model,
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )

    stage_formation = form_stages(specialized.model.graph)
    virtual_plans = allocate(
        specialized.model.graph,
        mesh,
        stage_formation=stage_formation,
    )
    execution_plan = plan(
        specialized.model.graph,
        mesh,
        PlanningOptions(
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )
    return specialized, rewrite_effects, virtual_plans, execution_plan


def test_magia_fuses_operations_around_decomposed_softmax_with_provenance(
    fused_softmax_planning,
) -> None:
    specialized, rewrite_effects, virtual_plans, execution_plan = (
        fused_softmax_planning
    )
    virtual_collective_groups = tuple(
        groups
        for stage_plan in virtual_plans.values()
        for node, groups in zip(
            stage_plan.nodes,
            stage_plan.virtual_collective_groups,
        )
        if isinstance(node.payload, AllReducePayload)
    )
    assert len(virtual_collective_groups) == 2
    assert all(groups for groups in virtual_collective_groups)

    bundle = build_deployment_bundle(
        specialized,
        execution_plan,
        graph_rewrite_effects=rewrite_effects,
    )

    assert bundle.execution_plan is execution_plan
    assert tuple(
        event.source_node for event in bundle.rewrite_report.events
    ) == ("softmax",)
    assert len(execution_plan.stages) == 1
    assert tuple(
        layer.source_operation
        for layer in execution_plan.stages[0].layers
    ) == ("producer", *("softmax",) * 7, "consumer")
    payload = execution_plan_json_payload(execution_plan)
    assert execution_plan_json_payload(execution_plan) == payload
    assert tuple(
        layer["source_operation"]
        for layer in payload["stages"][0]["layers"]
    ) == ("producer", *("softmax",) * 7, "consumer")
    assert all(
        "destination_input_index" not in transition
        and "source_output_index" not in transition
        for transition in payload["transitions"]
    )
    collective_layers = tuple(
        layer
        for layer in execution_plan.stages[0].layers
        if isinstance(layer.node.payload, AllReducePayload)
    )
    assert len(collective_layers) == 2
    for layer in collective_layers:
        participants = tuple(
            tile_id
            for group in layer.collective_groups
            for tile_id in group.tile_ids
        )
        assert tuple(sorted(participants)) == tuple(
            sorted(execution_plan.stages[0].submesh.tile_ids)
        )
    serialized_collectives = tuple(
        layer["collective_groups"]
        for layer in payload["stages"][0]["layers"]
        if layer["collective_groups"]
    )
    assert len(serialized_collectives) == 2
    assert validate_execution_plan(
        execution_plan,
        PlanningConstraints(max_stage_operations=3),
    ).is_valid
    limited = validate_execution_plan(
        execution_plan,
        PlanningConstraints(max_stage_operations=2),
    )
    assert tuple(violation.kind for violation in limited.violations) == (
        "stage_operation_limit_exceeded",
    )


def test_gemm_softmax_preserves_locality_when_batch_axis_has_one_element() -> None:
    x = Tensor("x", 2, (1, 4), 2, dtype=TensorDType.FLOAT16)
    weight = Tensor("weight", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    logits = Tensor("logits", 2, (1, 8), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 2, (1, 8), 2, dtype=TensorDType.FLOAT16)
    gemm = Node(
        "gemm",
        OpKind.GEMM,
        inputs=(x, weight),
        outputs=(logits,),
        payload=GemmPayload(x, weight, None, logits),
    )
    softmax = Node(
        "softmax",
        OpKind.CUSTOM,
        inputs=(logits,),
        outputs=(output,),
        payload=SoftmaxPayload(logits, output, axis=1),
    )
    model, _ = run_graph_rewrites_with_effects(
        ImportedModel(
            Graph(
                "gemm_softmax",
                tensors=(x, weight, logits, output),
                nodes=(gemm, softmax),
                inputs=(x, weight),
                outputs=(output,),
            ),
            ConstantStore(()),
        )
    )
    mesh = magia.build_mesh(width=2, height=1)
    specialized = magia.specialize(
        model,
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )

    execution_plan = plan(
        specialized.model.graph,
        mesh,
        PlanningOptions(
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )

    assert len(execution_plan.stages) == 1
    assert tuple(
        layer.source_operation for layer in execution_plan.stages[0].layers
    ) == ("gemm", *("softmax",) * 7)


def test_validation_rejects_overlapping_collective_groups(
    fused_softmax_planning,
) -> None:
    execution_plan = fused_softmax_planning[-1]
    collective_index = next(
        index
        for index, layer in enumerate(execution_plan.stages[0].layers)
        if isinstance(layer.node.payload, AllReducePayload)
    )
    collective = execution_plan.stages[0].layers[collective_index]
    repeated_tile = min(execution_plan.stages[0].submesh.tile_ids)
    malformed_layers = list(execution_plan.stages[0].layers)
    malformed_layers[collective_index] = replace(
        collective,
        collective_groups=(
            CollectiveGroup((repeated_tile,)),
            CollectiveGroup((repeated_tile,)),
        ),
    )
    malformed_plan = replace(
        execution_plan,
        stages=(
            replace(execution_plan.stages[0], layers=tuple(malformed_layers)),
        ),
    )
    assert "collective_groups_overlap" in {
        violation.kind
        for violation in validate_execution_plan(
            malformed_plan,
            PlanningConstraints(),
        ).violations
    }


def test_validation_rejects_inconsistent_collective_group_binding(
    fused_softmax_planning,
) -> None:
    execution_plan = fused_softmax_planning[-1]
    collective_index = next(
        index
        for index, layer in enumerate(execution_plan.stages[0].layers)
        if isinstance(layer.node.payload, AllReducePayload)
    )
    collective = execution_plan.stages[0].layers[collective_index]
    tile_ids = tuple(sorted(execution_plan.stages[0].submesh.tile_ids))
    regrouped = (
        tuple(CollectiveGroup((tile_id,)) for tile_id in tile_ids)
        if len(collective.collective_groups) == 1
        else (CollectiveGroup(tile_ids),)
    )
    layers = list(execution_plan.stages[0].layers)
    layers[collective_index] = replace(collective, collective_groups=regrouped)
    invalid_plan = replace(
        execution_plan,
        stages=(replace(execution_plan.stages[0], layers=tuple(layers)),),
    )

    assert "collective_group_binding_invalid" in {
        violation.kind
        for violation in validate_execution_plan(
            invalid_plan,
            PlanningConstraints(),
        ).violations
    }


def test_validation_accepts_nonidentity_physical_collective_binding(
    fused_softmax_planning,
) -> None:
    specialized = fused_softmax_planning[0]
    mesh = magia.build_mesh(width=3, height=1)
    execution_plan = plan(
        specialized.model.graph,
        mesh,
        PlanningOptions(
            stage_formation=StageFormationOptions(max_stage_operations=1),
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )
    collective_stage = next(
        stage
        for stage in execution_plan.stages
        if any(
            isinstance(layer.node.payload, AllReducePayload)
            for layer in stage.layers
        )
    )

    assert any(
        virtual_tile_id != physical_tile_id
        for virtual_tile_id, physical_tile_id in (
            collective_stage.virtual_to_physical.items()
        )
    )
    assert validate_execution_plan(
        execution_plan,
        PlanningConstraints(),
    ).is_valid


def test_validation_reports_incomplete_and_nonbijective_collective_bindings(
    fused_softmax_planning,
) -> None:
    execution_plan = fused_softmax_planning[-1]
    stage = execution_plan.stages[0]
    virtual_tile_ids = tuple(sorted(stage.virtual_to_physical))
    incomplete = dict(stage.virtual_to_physical)
    incomplete.pop(virtual_tile_ids[-1])
    duplicate = {
        virtual_tile_id: min(stage.submesh.tile_ids)
        for virtual_tile_id in virtual_tile_ids
    }

    for binding in (incomplete, duplicate):
        invalid_plan = replace(
            execution_plan,
            stages=(replace(stage, virtual_to_physical=binding),),
        )
        assert "collective_group_binding_invalid" in {
            violation.kind
            for violation in validate_execution_plan(
                invalid_plan,
                PlanningConstraints(),
            ).violations
        }


def test_validation_rejects_unresolved_partial_collective_output(
    fused_softmax_planning,
) -> None:
    execution_plan = fused_softmax_planning[-1]
    collective_index = next(
        index
        for index, layer in enumerate(execution_plan.stages[0].layers)
        if isinstance(layer.node.payload, AllReducePayload)
    )
    collective = execution_plan.stages[0].layers[collective_index]
    partial_layers = list(execution_plan.stages[0].layers)
    partial_output = collective.outputs[0]
    partial_layers[collective_index] = replace(
        collective,
        outputs=(
            replace(
                partial_output,
                layout=replace(
                    partial_output.layout,
                    mesh_x=LayoutAxis(LayoutAxisMode.PARTIAL, tensor_axis=0),
                ),
            ),
        ),
    )
    partial_plan = replace(
        execution_plan,
        stages=(replace(execution_plan.stages[0], layers=tuple(partial_layers)),),
    )
    partial_report = validate_execution_plan(partial_plan, PlanningConstraints())
    assert {
        "collective_output_remains_partial",
        "partial_value_consumed_by_ordinary_layer",
    } <= {violation.kind for violation in partial_report.violations}


def test_validation_rejects_partial_value_transition_escape(
    fused_softmax_planning,
) -> None:
    execution_plan = fused_softmax_planning[-1]
    escaping_layers = list(execution_plan.stages[0].layers)
    final_layer = escaping_layers[-1]
    final_output = final_layer.outputs[0]
    escaping_layers[-1] = replace(
        final_layer,
        outputs=(
            replace(
                final_output,
                layout=replace(
                    final_output.layout,
                    mesh_x=LayoutAxis(LayoutAxisMode.PARTIAL, tensor_axis=0),
                ),
            ),
        ),
    )
    escaping_plan = replace(
        execution_plan,
        stages=(replace(execution_plan.stages[0], layers=tuple(escaping_layers)),),
    )
    assert "partial_value_transition_escape" in {
        violation.kind
        for violation in validate_execution_plan(
            escaping_plan,
            PlanningConstraints(),
        ).violations
    }


def test_validation_rejects_collective_work_kind_collision(
    fused_softmax_planning,
) -> None:
    execution_plan = fused_softmax_planning[-1]
    layers = list(execution_plan.stages[0].layers)
    collective_index = next(
        index
        for index, layer in enumerate(layers)
        if isinstance(layer.node.payload, AllReducePayload)
    )
    collective = layers[collective_index]
    payload = collective.node.payload
    layers[collective_index] = replace(
        collective,
        node=replace(
            collective.node,
            payload=_CollidingAllReducePayload(
                payload.op_name,
                payload.x,
                payload.output,
                payload.reduction,
            ),
        ),
    )
    invalid_plan = replace(
        execution_plan,
        stages=(replace(execution_plan.stages[0], layers=tuple(layers)),),
    )

    assert "collective_work_kind_collision" in {
        violation.kind
        for violation in validate_execution_plan(
            invalid_plan,
            PlanningConstraints(),
        ).violations
    }


def _plan_collective_graph(graph: Graph) -> ExecutionPlan:
    model, _ = run_graph_rewrites_with_effects(
        ImportedModel(graph, ConstantStore(()))
    )
    mesh = magia.build_mesh(width=2, height=1)
    specialized = magia.specialize(
        model,
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )
    return plan(
        specialized.model.graph,
        mesh,
        PlanningOptions(
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )


def _reduce_sum_graph() -> Graph:
    x = Tensor("x", 2, (4, 8), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 2, (4, 1), 2, dtype=TensorDType.FLOAT16)
    node = Node(
        "reduce_sum",
        OpKind.REDUCTION,
        (x,),
        (output,),
        ReduceSumPayload(x, output, axis=1),
    )
    return Graph(
        "reduce_sum",
        tensors=(x, output),
        nodes=(node,),
        inputs=(x,),
        outputs=(output,),
    )


def _global_average_pool_graph() -> Graph:
    x = Tensor("x", 4, (1, 2, 4, 8), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 4, (1, 2, 1, 1), 2, dtype=TensorDType.FLOAT16)
    node = Node(
        "global_average_pool",
        OpKind.REDUCTION,
        (x,),
        (output,),
        GlobalAveragePoolPayload(x, output),
    )
    return Graph(
        "global_average_pool",
        tensors=(x, output),
        nodes=(node,),
        inputs=(x,),
        outputs=(output,),
    )


def _group_normalization_graph() -> Graph:
    x = Tensor("x", 4, (1, 4, 2, 2), 2, dtype=TensorDType.FLOAT16)
    scale = Tensor("scale", 1, (4,), 2, dtype=TensorDType.FLOAT16)
    bias = Tensor("bias", 1, (4,), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 4, x.dims, 2, dtype=TensorDType.FLOAT16)
    node = Node(
        "group_normalization",
        OpKind.CUSTOM,
        (x, scale, bias),
        (output,),
        GroupNormalizationPayload(x, scale, bias, output, num_groups=2),
    )
    return Graph(
        "group_normalization",
        tensors=(x, scale, bias, output),
        nodes=(node,),
        inputs=(x, scale, bias),
        outputs=(output,),
    )


@pytest.mark.parametrize(
    "graph",
    (_reduce_sum_graph(), _global_average_pool_graph(), _group_normalization_graph()),
    ids=("reduce_sum", "global_average_pool", "group_normalization"),
)
def test_composite_reductions_use_complete_collective_groups_through_planning(
    graph: Graph,
) -> None:
    execution_plan = _plan_collective_graph(graph)
    collective_layers = tuple(
        layer
        for stage in execution_plan.stages
        for layer in stage.layers
        if isinstance(layer.node.payload, AllReducePayload)
    )

    assert collective_layers
    assert all(layer.collective_groups for layer in collective_layers)
    assert all(
        output.layout.mesh_x.mode is not LayoutAxisMode.PARTIAL
        and output.layout.mesh_y.mode is not LayoutAxisMode.PARTIAL
        for layer in collective_layers
        for output in layer.outputs
    )
    assert validate_execution_plan(
        execution_plan,
        PlanningConstraints(),
    ).is_valid


@pytest.mark.parametrize(
    (
        "target",
        "specialization_options",
        "expected_device",
        "expected_stage_placements",
        "expected_transition_summary",
    ),
    (
        (
            magia,
            SpecializationOptions(enable_precision_lowering=True),
            "redmule",
            (
                ((16,), ((0, 16),)),
                ((24, 32), ((0, 24), (1, 32))),
            ),
            (
                ("input", 0, 0, ((16, ((0, 2), (0, 3))),)),
                (
                    "intermediate",
                    3,
                    0,
                    1,
                    (
                        (16, 24, ((0, 2), (0, 3)), ((0, 2), (0, 3))),
                        (16, 32, ((0, 2), (0, 3)), ((0, 2), (0, 3))),
                    ),
                ),
                (
                    "output",
                    2,
                    1,
                    (
                        (24, ((0, 2), (0, 2))),
                        (32, ((0, 2), (2, 2))),
                    ),
                ),
            ),
        ),
        (
            n300d,
            SpecializationOptions(),
            "tensix_matrix",
            (((27,), ((0, 27),)),),
            (
                ("input", 0, 0, ((27, ((0, 2), (0, 3))),)),
                ("output", 2, 0, ((27, ((0, 2), (0, 4))),)),
            ),
        ),
    ),
)
def test_specialized_target_graphs_plan_through_one_public_interface(
    target,
    specialization_options: SpecializationOptions,
    expected_device: str,
    expected_stage_placements: tuple,
    expected_transition_summary: tuple,
    tmp_path,
    monkeypatch,
) -> None:
    mesh = target.build_mesh()
    specialized = target.specialize(
        _gemm_model(),
        mesh,
        specialization_options,
    )
    monkeypatch.chdir(tmp_path)

    execution_plan = plan(
        specialized.model.graph,
        mesh,
        PlanningOptions(
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )

    assert isinstance(execution_plan, ExecutionPlan)
    assert execution_plan.mesh is mesh
    assert execution_plan.stages
    assert expected_device in {
        layer.device_name
        for stage in execution_plan.stages
        for layer in stage.layers
    }
    assert tuple(
        (
            tuple(sorted(stage.submesh.tile_ids)),
            tuple(sorted(stage.virtual_to_physical.items())),
        )
        for stage in execution_plan.stages
    ) == expected_stage_placements
    assert tuple(
        _physical_transition_summary(transition)
        for transition in execution_plan.transitions
    ) == expected_transition_summary
    assert tuple(tmp_path.iterdir()) == ()


def _slice_dims(tensor_slice) -> tuple[tuple[int, int], ...]:
    return tuple((dimension.start, dimension.length) for dimension in tensor_slice.dims)


def _physical_transition_summary(transition) -> tuple:
    if isinstance(transition, InputTransition):
        return (
            "input",
            transition.tensor_id,
            transition.destination_stage_id,
            tuple(
                (destination.tile_id, _slice_dims(destination.tensor_slice))
                for destination in transition.destinations
            ),
        )
    if isinstance(transition, IntermediateTransition):
        return (
            "intermediate",
            transition.tensor_id,
            transition.source_stage_id,
            transition.destination_stage_id,
            tuple(
                (
                    transfer.source_tile_id,
                    transfer.destination_tile_id,
                    _slice_dims(transfer.source_subslice),
                    _slice_dims(transfer.destination_subslice),
                )
                for transfer in transition.transfers
            ),
        )
    assert isinstance(transition, OutputTransition)
    return (
        "output",
        transition.tensor_id,
        transition.source_stage_id,
        tuple(
            (source.tile_id, _slice_dims(source.tensor_slice))
            for source in transition.sources
        ),
    )


def test_unsupported_device_signature_has_actionable_planning_diagnostic() -> None:
    mesh = magia.build_mesh(width=1, height=1)
    unsupported_mesh = replace(
        mesh,
        tiles=(
            replace(
                mesh.tiles[0],
                device_assignment=FixedDeviceAssignment(),
            ),
        ),
    )

    with pytest.raises(ValueError) as error:
        plan(
            magia.specialize(_gemm_model(), mesh).model.graph,
            unsupported_mesh,
            PlanningOptions(
                placement=PlacementOptions(print_placement=False),
                print_execution_plan_cost=False,
            ),
        )

    message = str(error.value)
    assert "node gemm" in message
    assert "WorkSignature" in message
    assert "tile 0" in message
    assert "configured assignment=None" in message
    assert "considered devices: idma_read, idma_write, core, spatz, redmule" in message


@pytest.mark.parametrize(
    ("device_name", "message"),
    (
        (None, "has no retained Device name"),
        ("redmule", "Device capability match is False"),
    ),
)
def test_execution_plan_validation_rejects_invalid_retained_device_names(
    device_name: str | None,
    message: str,
) -> None:
    mesh = magia.build_mesh(width=1, height=1)
    specialized = magia.specialize(
        _gemm_model(),
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )
    execution_plan = plan(
        specialized.model.graph,
        mesh,
        PlanningOptions(
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )
    layer = replace(execution_plan.stages[0].layers[0], device_name=device_name)
    stage = replace(execution_plan.stages[0], layers=(layer,))

    report = validate_execution_plan(
        replace(execution_plan, stages=(stage,)),
        PlanningConstraints(),
    )

    assert not report.is_valid
    assert report.violations[0].kind == "layer_device_assignment_invalid"
    assert message in report.violations[0].message


def test_planning_rejects_untyped_operation_inputs() -> None:
    x = Tensor("x", 1, (4,), 4)
    output = Tensor("output", 1, (4,), 4)
    node = Node(
        "relu",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Relu", x, output),
    )
    graph = Graph(
        "untyped",
        tensors=(x, output),
        nodes=(node,),
        edges=(Edge(x, None, node), Edge(output, node, None)),
        inputs=(x,),
        outputs=(output,),
    )

    with pytest.raises(
        ValueError,
        match=r"node relu has untyped tensors: x, output",
    ):
        plan(
            graph,
            magia.build_mesh(width=1, height=1),
            PlanningOptions(
                placement=PlacementOptions(print_placement=False),
                print_execution_plan_cost=False,
            ),
        )
