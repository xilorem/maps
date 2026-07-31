from dataclasses import dataclass, field

from maps.deployment import DeploymentBundle, build_deployment_bundle
from maps.graph import ImportedModel
from maps.hardware import Mesh
from maps.planning import AllocationOptions, PlacementOptions, PlanningOptions, plan
from maps.target import SpecializationOptions, magia


@dataclass(frozen=True)
class MagiaWorkflowOptions:
    enable_precision_lowering: bool = True
    allocation: AllocationOptions = field(default_factory=AllocationOptions)


def magia_workflow_options(
    *,
    enable_precision_lowering: bool = True,
) -> MagiaWorkflowOptions:
    return MagiaWorkflowOptions(enable_precision_lowering=enable_precision_lowering)


def plan_magia_model(
    model: ImportedModel,
    mesh: Mesh,
    options: MagiaWorkflowOptions,
) -> DeploymentBundle:
    specialization = magia.specialize(
        model,
        mesh,
        SpecializationOptions(
            enable_precision_lowering=options.enable_precision_lowering,
        ),
    )
    execution_plan = plan(
        specialization.model.graph,
        mesh,
        PlanningOptions(
            allocation=options.allocation,
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )
    return build_deployment_bundle(specialization, execution_plan)
