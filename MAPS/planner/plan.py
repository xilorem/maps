"""Top-level planner flow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.importers.onnx.importer import import_onnx_graph, import_onnx_model
from MAPS.importers.onnx.preprocess import InputShapes
from MAPS.pipeline.execution import ExecutionContract
from MAPS.pipeline.execution_plan import ExecutionPlan
from MAPS.pipeline.pipeline import Pipeline
from MAPS.planner.contracts.options import (
    PlannerOptions,
    SpatialMappingOptions,
    StageSelectionOptions,
    WorkloadBalancingOptions,
)
from MAPS.planner.passes.execution_plan_lowering import lower_execution_plan
from MAPS.planner.passes.execution_plan_validation import validate_execution_plan
from MAPS.planner.passes.pipeline_lowering import lower_pipeline
from MAPS.planner.passes.spatial_mapping import map_spatially
from MAPS.planner.passes.stage_selection import form_stages
from MAPS.planner.passes.workload_balancing import balance_workload
from MAPS.planner.reporting.pipeline import print_pipeline_stage_cost
from MAPS.planner.validation.contracts import PlannerConstraints
from MAPS.transitions import build_virtual_transitions
from MAPS.utils.execution_plan_json import write_execution_plan_json

if TYPE_CHECKING:
    from MAPS.deployment.bundle import DeploymentBundle


def plan_graph(
    graph: Graph,
    mesh: Mesh,
    options: PlannerOptions,
) -> ExecutionPlan:
    """Plan an imported graph for a homogeneous multi-tile mesh.

    Contract:
        ``graph`` must be in planner-supported Graph IR and ``mesh`` must fully
        describe its tiles, memories, and NoC.  This function performs no model
        import and writes no files.  All configurable search and diagnostic
        behavior is supplied through ``PlannerOptions``.

    Pass order:
        1. Select graph nodes that execute together as stages.
        2. Allocate virtual tiles and choose stage-local tensor layouts.
        3. Map those virtual stages onto disjoint connected physical regions.
        4. Bind retained transitions and lower an executable Execution Plan.
        5. Validate the completed physical plan.

    Returns:
        A validated physical ``ExecutionPlan``. A failure to find or validate a
        legal decision is reported as ``ValueError``.
    """
    stage_plans, virtual_transitions, placements = _plan_decisions(
        graph,
        mesh,
        options,
    )

    execution_plan = lower_execution_plan(
        graph,
        mesh,
        stage_plans,
        placements,
        virtual_transitions,
        execution=options.execution,
    )
    validation = validate_execution_plan(
        execution_plan,
        PlannerConstraints(max_stage_nodes=options.stage_selection.max_stage_nodes),
    )
    if not validation.is_valid:
        details = "; ".join(
            f"{violation.kind}: {violation.message}"
            for violation in validation.violations
        )
        raise ValueError(f"planner produced an invalid Execution Plan: {details}")

    if options.print_pipeline_cost:
        print_pipeline_stage_cost(
            execution_plan,
            stage_plans,
            placements,
            virtual_transitions,
        )

    return execution_plan


def _plan_decisions(
    graph: Graph,
    mesh: Mesh,
    options: PlannerOptions,
):
    stage_selection = form_stages(graph, options.stage_selection)

    stage_plans = balance_workload(
        graph,
        mesh,
        stage_selection=stage_selection,
        debug=options.workload.print_progress,
        compute_weight=options.workload.compute_weight,
        communication_weight=options.workload.communication_weight,
        num_token_slots=options.execution.num_token_slots,
    )
    virtual_transitions = build_virtual_transitions(graph, stage_plans)

    placements = map_spatially(
        mesh,
        stage_plans,
        virtual_transitions,
        show_progress=options.spatial_mapping.print_progress,
        print_mapping=options.spatial_mapping.print_mapping,
        print_costs=options.spatial_mapping.print_costs,
    )

    return stage_plans, virtual_transitions, placements


def build_pipeline(
    model_path: str | Path,
    mesh: Mesh,
    print_workload_balancing: bool = False,
    print_spatial_mapping: bool = False,
    print_spatial_mapping_progress: bool = False,
    output_json_path: str | Path | None = None,
    *,
    input_shapes: InputShapes | None = None,
    max_stage_nodes: int = 0,
    num_token_slots: int = 2,
) -> ExecutionPlan:
    """Main planning entry point"""

    graph = import_onnx_graph(model_path, input_shapes=input_shapes)
    execution_plan = plan_graph(
        graph,
        mesh,
        PlannerOptions(
            execution=ExecutionContract(num_token_slots=num_token_slots),
            stage_selection=StageSelectionOptions(max_stage_nodes=max_stage_nodes),
            workload=WorkloadBalancingOptions(
                compute_weight=1.0,
                communication_weight=10.0,
                print_progress=print_workload_balancing,
            ),
            spatial_mapping=SpatialMappingOptions(
                print_progress=print_spatial_mapping_progress,
                print_mapping=not print_spatial_mapping,
                print_costs=print_spatial_mapping,
            ),
        ),
    )
    if output_json_path is not None:
        write_execution_plan_json(execution_plan, output_json_path)
    return execution_plan


def build_pipeline_bundle(
    model_path: str | Path,
    arch: Mesh,
    options: PlannerOptions,
    *,
    input_shapes: InputShapes | None = None,
) -> DeploymentBundle:
    """Import constants alongside a model and produce its deployment bundle."""

    from MAPS.deployment.bundle import DeploymentBundle

    model = import_onnx_model(model_path, input_shapes=input_shapes)
    stage_plans, virtual_transitions, placements = _plan_decisions(
        model.graph,
        arch,
        options,
    )
    pipeline = lower_pipeline(
        model.graph,
        arch,
        stage_plans,
        placements,
        execution=options.execution,
    )
    if options.print_pipeline_cost:
        print_pipeline_stage_cost(
            pipeline,
            stage_plans,
            placements,
            virtual_transitions,
        )
    return DeploymentBundle(
        pipeline=pipeline,
        graph=model.graph,
        constants=model.constants,
    )


__all__ = ["build_pipeline", "build_pipeline_bundle", "plan_graph"]
