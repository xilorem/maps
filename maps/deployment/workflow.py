"""Shared Imported Model to Deployment Bundle workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from maps.graph import import_onnx_model, run_graph_rewrites_with_effects
from maps.planning import (
    AllocationOptions,
    ExecutionContract,
    PlacementOptions,
    PlanningOptions,
    plan,
)
from maps.target import SpecializationOptions, magia

from .bundle import DeploymentBundle, build_deployment_bundle


def build_magia_deployment_bundle(
    model: Path,
    *,
    mesh_width: int,
    mesh_height: int,
    num_token_slots: int,
    progress: Callable[[str], None] | None,
) -> DeploymentBundle:
    """Compose rewriting, Target Specialization, Planning, and bundling."""

    mesh = magia.build_mesh(width=mesh_width, height=mesh_height)
    rewritten, graph_rewrite_effects = run_graph_rewrites_with_effects(
        import_onnx_model(model)
    )
    specialization = magia.specialize(
        rewritten,
        mesh,
        SpecializationOptions(enable_precision_lowering=False),
    )
    execution_plan = plan(
        specialization.model.graph,
        mesh,
        PlanningOptions(
            execution=ExecutionContract(num_token_slots=num_token_slots),
            allocation=AllocationOptions(print_progress=progress is not None),
            placement=PlacementOptions(
                print_progress=progress is not None,
                print_placement=False,
                print_costs=False,
            ),
            print_execution_plan_cost=False,
        ),
    )
    return build_deployment_bundle(
        specialization,
        execution_plan,
        graph_rewrite_effects=graph_rewrite_effects,
    )


__all__ = ["build_magia_deployment_bundle"]
