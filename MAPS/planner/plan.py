"""Top-level planner flow."""

from __future__ import annotations

from pathlib import Path

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.deployment.bundle import DeploymentBundle
from MAPS.importers.onnx.importer import import_onnx_graph, import_onnx_model
from MAPS.importers.onnx.preprocess import InputShapes
from MAPS.pipeline.pipeline import Pipeline
from MAPS.pipeline.execution import ExecutionContract
from MAPS.planner.contracts.options import (
    PlannerOptions,
    SpatialMappingOptions,
    StageSelectionOptions,
    WorkloadBalancingOptions,
)
from MAPS.planner.passes.pipeline_lowering import lower_pipeline
from MAPS.planner.passes.spatial_mapping import map_spatially
from MAPS.planner.passes.stage_selection import form_stages
from MAPS.planner.passes.workload_balancing import balance_workload
from MAPS.planner.reporting.pipeline import print_pipeline_stage_cost
from MAPS.transitions import build_virtual_transitions
from MAPS.utils.pipeline_json import write_pipeline_json


def plan_graph(
    graph: Graph,
    mesh: Mesh,
    options: PlannerOptions,
) -> Pipeline:
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
        4. Lower the decisions into executable Pipeline IR.

    Returns:
        A complete physical ``Pipeline``.  A failure to find a legal decision is
        reported as ``ValueError`` by the pass that discovered the infeasibility.
    """


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

    pipeline = lower_pipeline(
        graph,
        mesh,
        stage_plans,
        placements,
        execution=options.execution,
    )

    if options.print_pipeline_cost:
        print_pipeline_stage_cost(
            mesh,
            stage_plans,
            placements,
            virtual_transitions,
        )

    return pipeline


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
) -> Pipeline:
    """Main planning entry point"""

    graph = import_onnx_graph(model_path, input_shapes=input_shapes)
    pipeline = plan_graph(
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
        write_pipeline_json(pipeline, output_json_path)
    return pipeline


def build_pipeline_bundle(
    model_path: str | Path,
    arch: Mesh,
    options: PlannerOptions,
    *,
    input_shapes: InputShapes | None = None,
) -> DeploymentBundle:
    """Import constants alongside a model and produce its deployment bundle."""

    model = import_onnx_model(model_path, input_shapes=input_shapes)
    pipeline = plan_graph(model.graph, arch, options)
    return DeploymentBundle(
        pipeline=pipeline,
        graph=model.graph,
        constants=model.constants,
    )


__all__ = ["build_pipeline", "build_pipeline_bundle", "plan_graph"]
