from dataclasses import replace

import pytest

from maps.hardware import FixedDeviceAssignment
from maps.planning import ExecutionPlan, PlacementOptions, PlanningOptions, plan
from maps.planning.allocation import allocate
from maps.planning.stage_formation import form_stages
from maps.planning.stages import StagePlan
from maps.target import SpecializationOptions, magia, n300d

from tests.test_precision_lowering import _gemm_model


def test_planning_owns_stage_formation_and_allocation_contracts() -> None:
    assert form_stages.__module__ == "maps.planning.stage_formation"
    assert allocate.__module__ == "maps.planning.allocation"
    assert StagePlan.__module__ == "maps.planning.stages"


@pytest.mark.parametrize(
    ("target", "specialization_options", "expected_device"),
    (
        (
            magia,
            SpecializationOptions(enable_precision_lowering=True),
            "redmule",
        ),
        (n300d, SpecializationOptions(), "tensix_matrix"),
    ),
)
def test_specialized_target_graphs_plan_through_one_public_interface(
    target,
    specialization_options: SpecializationOptions,
    expected_device: str,
    tmp_path,
    monkeypatch,
) -> None:
    mesh = target.build_mesh()
    specialized = target.specialize(
        _gemm_model(),
        mesh,
        specialization_options,
    )
    monkeypatch.chdir(tmp_path)

    execution_plan = plan(
        specialized.model.graph,
        mesh,
        PlanningOptions(
            placement=PlacementOptions(print_mapping=False),
            print_execution_plan_cost=False,
        ),
    )

    assert isinstance(execution_plan, ExecutionPlan)
    assert execution_plan.mesh is mesh
    assert execution_plan.stages
    assert expected_device in {
        layer.device_name
        for stage in execution_plan.stages
        for layer in stage.layers
    }
    assert tuple(tmp_path.iterdir()) == ()


def test_unsupported_device_signature_has_actionable_planning_diagnostic() -> None:
    mesh = magia.build_mesh(width=1, height=1)
    unsupported_mesh = replace(
        mesh,
        tiles=(
            replace(
                mesh.tiles[0],
                device_assignment=FixedDeviceAssignment(),
            ),
        ),
    )

    with pytest.raises(ValueError) as error:
        plan(
            magia.specialize(_gemm_model(), mesh).model.graph,
            unsupported_mesh,
            PlanningOptions(
                placement=PlacementOptions(print_mapping=False),
                print_execution_plan_cost=False,
            ),
        )

    message = str(error.value)
    assert "node gemm" in message
    assert "WorkSignature" in message
    assert "tile 0" in message
    assert "configured assignment=None" in message
    assert "considered devices: idma_read, idma_write, core, spatz, redmule" in message
