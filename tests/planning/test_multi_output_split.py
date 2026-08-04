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
from maps.hardware import WorkKind, WorkSignature
from maps.operations.split import SplitPayload
from maps.planning import PlacementOptions, PlanningOptions, plan
from maps.planning.mapping import TensorRange
from maps.planning.transitions import InputTransition, OutputTransition
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


def test_public_planning_rejects_an_undeclared_split_arity() -> None:
    model = _split_model(sizes=(2, 2))
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
