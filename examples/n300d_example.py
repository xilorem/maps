"""Run a small ONNX network through the Wormhole n300d planning flow."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maps.target.n300d import build_mesh
from MAPS.planner.passes.execution_plan_validation import validate_execution_plan
from MAPS.planner.validation.contracts import PlannerConstraints
from MAPS.planner.plan import build_execution_plan
from MAPS.utils.execution_plan_json import write_execution_plan_json
from MAPS.utils.print_submeshes import print_submeshes

DEFAULT_MODEL_PATH = PROJECT_ROOT / "examples" / "simple_three_stage.onnx"


def main():
    mesh = build_mesh()
    output_path = (
        PROJECT_ROOT / "generated" / "n300d_example.execution-plan.json"
    )
    execution_plan = build_execution_plan(
        DEFAULT_MODEL_PATH,
        mesh,
        print_allocation=True,
        print_spatial_mapping=True,
        print_spatial_mapping_progress=True,
    )
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
    print(
        "Execution Plan JSON: "
        f"{write_execution_plan_json(execution_plan, output_path)}"
    )


if __name__ == "__main__":
    main()
