"""Run a small ONNX network through the MAGIA planning flow."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MAPS.deployment import write_pipeline_bundle
from MAPS.hw.chips import magia_mesh
from MAPS.pipeline import ExecutionContract
from MAPS.planner.contracts.options import (
    PlannerOptions,
    SpatialMappingOptions,
    WorkloadBalancingOptions,
)
from MAPS.planner.passes.validation import validate_constraints
from MAPS.planner.validation.contracts import PlannerConstraints
from MAPS.planner.plan import build_pipeline_bundle
from MAPS.utils.print_submeshes import print_submeshes

DEFAULT_MODEL_PATH = PROJECT_ROOT / "examples" / "simple_three_stage.onnx"


def main():
    mesh = magia_mesh(width=4, height=4)
    output_path = PROJECT_ROOT / "generated" / "magia_example.pipeline.json"
    weights_path = output_path.with_suffix(".weights.bin")
    bundle = build_pipeline_bundle(
        DEFAULT_MODEL_PATH,
        mesh,
        PlannerOptions(
            execution=ExecutionContract(num_token_slots=2),
            workload=WorkloadBalancingOptions(
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
    pipeline = bundle.pipeline
    report = validate_constraints(pipeline, PlannerConstraints())

    print(f"Model: {pipeline.name}")
    print(f"Mesh: {mesh.width}x{mesh.height}")
    print(f"Stages: {len(pipeline.stages)}")
    print(f"Transitions: {len(pipeline.transitions)}")
    print(f"Constraint valid: {report.is_valid}")
    print_submeshes(pipeline)
    if report.violations:
        print("Constraint violations:")
        for violation in report.violations:
            print(f"  {violation.kind}: {violation.message}")
    pipeline_path, packed_weights_path = write_pipeline_bundle(
        bundle,
        output_path,
        weights_path,
    )
    print(f"Pipeline bundle: {pipeline_path}")
    print(f"Packed weights: {packed_weights_path}")


if __name__ == "__main__":
    main()
