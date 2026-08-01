"""Run a small ONNX network through the Wormhole n300d planning flow."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maps.graph import import_onnx_model, run_graph_rewrites
from maps.target import SpecializationOptions, n300d
from maps.planning import (
    AllocationOptions,
    PlacementOptions,
    PlanningConstraints,
    PlanningOptions,
    plan,
    validate_execution_plan,
)
from maps.deployment.serialization import write_execution_plan_json
from maps.planning.execution_plan import print_submeshes

DEFAULT_MODEL_PATH = PROJECT_ROOT / "examples" / "simple_three_stage.onnx"


def main():
    mesh = n300d.build_mesh()
    output_path = (
        PROJECT_ROOT / "generated" / "n300d_example.execution-plan.json"
    )
    imported = import_onnx_model(DEFAULT_MODEL_PATH)
    rewritten = run_graph_rewrites(imported)
    specialization = n300d.specialize(
        rewritten,
        mesh,
        SpecializationOptions(),
    )
    execution_plan = plan(
        specialization.model.graph,
        mesh,
        PlanningOptions(
            allocation=AllocationOptions(print_progress=True),
            placement=PlacementOptions(
                print_progress=True,
                print_placement=True,
            ),
        ),
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
    print(
        "Execution Plan JSON: "
        f"{write_execution_plan_json(execution_plan, output_path)}"
    )


if __name__ == "__main__":
    main()
