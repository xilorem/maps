"""Run a small ONNX network through the MAGIA planning flow."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maps.deployment import build_deployment_bundle, write_execution_plan_bundle
from maps.graph import import_onnx_model, run_graph_rewrites_with_effects
from maps.target import SpecializationOptions, magia
from maps.planning import (
    AllocationOptions,
    ExecutionContract,
    PlacementOptions,
    PlanningOptions,
    PlanningConstraints,
    StageFormationOptions,
    plan,
    validate_execution_plan,
)
from maps.planning.execution_plan import print_submeshes

DEFAULT_MODEL_PATH = PROJECT_ROOT / "examples" / "simple_three_stage.onnx"


def main():
    mesh = magia.build_mesh(width=4, height=4)
    output_path = (
        PROJECT_ROOT / "generated" / "magia_example.execution-plan.json"
    )
    weights_path = output_path.with_suffix(".weights.bin")
    imported = import_onnx_model(DEFAULT_MODEL_PATH)
    rewritten, graph_rewrite_effects = run_graph_rewrites_with_effects(imported)
    specialization = magia.specialize(
        rewritten,
        mesh,
        SpecializationOptions(enable_precision_lowering=True),
    )
    execution_plan = plan(
        specialization.model.graph,
        mesh,
        PlanningOptions(
            execution=ExecutionContract(num_token_slots=2),
            stage_formation=StageFormationOptions(max_stage_operations=1),
            allocation=AllocationOptions(
                stage_latency_weight=1.0,
                communication_weight=10.0,
                print_progress=True,
            ),
            placement=PlacementOptions(
                print_progress=True,
                print_placement=False,
                print_costs=True,
            ),
        ),
    )
    bundle = build_deployment_bundle(
        specialization,
        execution_plan,
        graph_rewrite_effects=graph_rewrite_effects,
    )
    report = validate_execution_plan(execution_plan, PlanningConstraints())

    print(f"Model: {execution_plan.name}")
    print(f"Mesh: {mesh.width}x{mesh.height}")
    print(f"Stages: {len(execution_plan.stages)}")
    print(f"Transitions: {len(execution_plan.transitions)}")
    print(f"Constraint valid: {report.is_valid}")
    print_submeshes(execution_plan)
    if report.violations:
        print("Constraint violations:")
        for violation in report.violations:
            print(f"  {violation.kind}: {violation.message}")
    execution_plan_path, packed_weights_path = write_execution_plan_bundle(
        bundle,
        output_path,
        weights_path,
    )
    print(f"Execution Plan bundle: {execution_plan_path}")
    print(f"Packed weights: {packed_weights_path}")


if __name__ == "__main__":
    main()
