"""Run a small ONNX network through the MAGIA planning flow."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MAPS.deployment import write_execution_plan_bundle
from maps.target.magia import build_mesh
from MAPS.pipeline import ExecutionContract
from MAPS.planner.contracts.options import (
    PlannerOptions,
    SpatialMappingOptions,
    StageFormationOptions,
    AllocationOptions,
)
from MAPS.planner.passes.execution_plan_validation import validate_execution_plan
from MAPS.planner.validation.contracts import PlannerConstraints
from MAPS.planner.plan import build_execution_plan_bundle
from MAPS.utils.print_submeshes import print_submeshes

DEFAULT_MODEL_PATH = PROJECT_ROOT / "examples" / "simple_three_stage.onnx"


def main():
    mesh = build_mesh(width=4, height=4)
    output_path = (
        PROJECT_ROOT / "generated" / "magia_example.execution-plan.json"
    )
    weights_path = output_path.with_suffix(".weights.bin")
    bundle = build_execution_plan_bundle(
        DEFAULT_MODEL_PATH,
        mesh,
        PlannerOptions(
            execution=ExecutionContract(num_token_slots=2),
            stage_formation=StageFormationOptions(max_stage_nodes=1),
            allocation=AllocationOptions(
                compute_weight=1.0,
                communication_weight=10.0,
                print_progress=True,
            ),
            spatial_mapping=SpatialMappingOptions(
                print_progress=True,
                print_mapping=False,
                print_costs=True,
            ),
        ),
    )
    execution_plan = bundle.execution_plan
    report = validate_execution_plan(execution_plan, PlannerConstraints())

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
