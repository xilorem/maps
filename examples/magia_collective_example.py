"""Build a producer-to-softmax-to-consumer MAGIA deployment bundle."""

from __future__ import annotations

from pathlib import Path

from maps.deployment import (
    DeploymentBundle,
    build_deployment_bundle,
    write_execution_plan_bundle,
)
from maps.graph import (
    ConstantStore,
    Graph,
    ImportedModel,
    Node,
    OpKind,
    Tensor,
    TensorDType,
    run_graph_rewrites_with_effects,
)
from maps.operations.elementwise import UnaryElementwisePayload
from maps.operations.softmax import SoftmaxPayload
from maps.planning import ExecutionContract, PlacementOptions, PlanningOptions, plan
from maps.target import SpecializationOptions, magia


def build_collective_bundle() -> DeploymentBundle:
    x = Tensor("x", 2, (2, 128), 2, dtype=TensorDType.FLOAT16)
    produced = Tensor("produced", 2, x.dims, 2, dtype=TensorDType.FLOAT16)
    normalized = Tensor("normalized", 2, x.dims, 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 2, x.dims, 2, dtype=TensorDType.FLOAT16)
    producer = Node(
        "producer",
        OpKind.ELEMENTWISE,
        (x,),
        (produced,),
        UnaryElementwisePayload("Relu", x, produced),
    )
    softmax = Node(
        "softmax",
        OpKind.CUSTOM,
        (produced,),
        (normalized,),
        SoftmaxPayload(produced, normalized, axis=1),
    )
    consumer = Node(
        "consumer",
        OpKind.ELEMENTWISE,
        (normalized,),
        (output,),
        UnaryElementwisePayload("Neg", normalized, output),
    )
    imported = ImportedModel(
        Graph(
            "magia_collective",
            tensors=(x, produced, normalized, output),
            nodes=(producer, softmax, consumer),
            inputs=(x,),
            outputs=(output,),
        ),
        ConstantStore(()),
    )
    rewritten, effects = run_graph_rewrites_with_effects(imported)
    mesh = magia.build_mesh(width=4, height=1)
    specialization = magia.specialize(
        rewritten,
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )
    execution_plan = plan(
        specialization.model.graph,
        mesh,
        PlanningOptions(
            execution=ExecutionContract(num_token_slots=2),
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )
    return build_deployment_bundle(
        specialization,
        execution_plan,
        graph_rewrite_effects=effects,
    )


def write_collective_bundle(output_path: Path) -> tuple[Path, Path]:
    weights_path = output_path.with_suffix(".weights.bin")
    return write_execution_plan_bundle(
        build_collective_bundle(),
        output_path,
        weights_path,
    )


if __name__ == "__main__":
    write_collective_bundle(
        Path(__file__).resolve().parents[1]
        / "generated"
        / "magia_collective.execution-plan.json"
    )
