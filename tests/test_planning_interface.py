from dataclasses import replace

import pytest

from maps.hardware import FixedDeviceAssignment
from maps.planning import (
    AllocationOptions,
    ExecutionPlan,
    PlacementOptions,
    PlanningOptions,
    StageFormationOptions,
    plan,
)
from maps.planning.allocation import allocate
from maps.planning.layouts import TensorLayout
from maps.planning.placement import place
from maps.planning.submesh import Submesh
from maps.planning.stage_formation import form_stages
from maps.planning.stages import StagePlan
from maps.planning.transitions import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
    build_virtual_transitions,
)
from maps.target import SpecializationOptions, magia, n300d

from tests.test_precision_lowering import _gemm_model


def test_planning_owns_stage_formation_and_allocation_contracts() -> None:
    assert StageFormationOptions.__module__ == "maps.planning.options"
    assert AllocationOptions.__module__ == "maps.planning.options"
    assert form_stages.__module__ == "maps.planning.stage_formation"
    assert allocate.__module__ == "maps.planning.allocation"
    assert StagePlan.__module__ == "maps.planning.stages"


def test_planning_owns_transitions_placement_and_layout_contracts() -> None:
    assert build_virtual_transitions.__module__ == "maps.planning.transitions.compile"
    assert place.__module__ == "maps.planning.placement"
    assert TensorLayout.__module__ == "maps.planning.layouts"
    assert Submesh.__module__ == "maps.planning.submesh"


@pytest.mark.parametrize(
    (
        "target",
        "specialization_options",
        "expected_device",
        "expected_stage_placements",
        "expected_transition_summary",
    ),
    (
        (
            magia,
            SpecializationOptions(enable_precision_lowering=True),
            "redmule",
            (
                ((16,), ((0, 16),)),
                ((24, 32), ((0, 24), (1, 32))),
            ),
            (
                ("input", 0, 0, 0, ((16, ((0, 2), (0, 3))),)),
                (
                    "intermediate",
                    3,
                    0,
                    0,
                    1,
                    0,
                    (
                        (16, 24, ((0, 2), (0, 3)), ((0, 2), (0, 3))),
                        (16, 32, ((0, 2), (0, 3)), ((0, 2), (0, 3))),
                    ),
                ),
                (
                    "output",
                    2,
                    1,
                    0,
                    (
                        (24, ((0, 2), (0, 2))),
                        (32, ((0, 2), (2, 2))),
                    ),
                ),
            ),
        ),
        (
            n300d,
            SpecializationOptions(),
            "tensix_matrix",
            (((27,), ((0, 27),)),),
            (
                ("input", 0, 0, 0, ((27, ((0, 2), (0, 3))),)),
                ("output", 2, 0, 0, ((27, ((0, 2), (0, 4))),)),
            ),
        ),
    ),
)
def test_specialized_target_graphs_plan_through_one_public_interface(
    target,
    specialization_options: SpecializationOptions,
    expected_device: str,
    expected_stage_placements: tuple,
    expected_transition_summary: tuple,
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
            placement=PlacementOptions(print_placement=False),
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
    assert tuple(
        (
            tuple(sorted(stage.submesh.tile_ids)),
            tuple(sorted(stage.virtual_to_physical.items())),
        )
        for stage in execution_plan.stages
    ) == expected_stage_placements
    assert tuple(
        _physical_transition_summary(transition)
        for transition in execution_plan.transitions
    ) == expected_transition_summary
    assert tuple(tmp_path.iterdir()) == ()


def _slice_dims(tensor_slice) -> tuple[tuple[int, int], ...]:
    return tuple((dimension.start, dimension.length) for dimension in tensor_slice.dims)


def _physical_transition_summary(transition) -> tuple:
    if isinstance(transition, InputTransition):
        return (
            "input",
            transition.tensor_id,
            transition.destination_stage_id,
            transition.destination_input_index,
            tuple(
                (destination.tile_id, _slice_dims(destination.tensor_slice))
                for destination in transition.destinations
            ),
        )
    if isinstance(transition, IntermediateTransition):
        return (
            "intermediate",
            transition.tensor_id,
            transition.source_stage_id,
            transition.source_output_index,
            transition.destination_stage_id,
            transition.destination_input_index,
            tuple(
                (
                    transfer.source_tile_id,
                    transfer.destination_tile_id,
                    _slice_dims(transfer.source_subslice),
                    _slice_dims(transfer.destination_subslice),
                )
                for transfer in transition.transfers
            ),
        )
    assert isinstance(transition, OutputTransition)
    return (
        "output",
        transition.tensor_id,
        transition.source_stage_id,
        transition.source_output_index,
        tuple(
            (source.tile_id, _slice_dims(source.tensor_slice))
            for source in transition.sources
        ),
    )


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
                placement=PlacementOptions(print_placement=False),
                print_execution_plan_cost=False,
            ),
        )

    message = str(error.value)
    assert "node gemm" in message
    assert "WorkSignature" in message
    assert "tile 0" in message
    assert "configured assignment=None" in message
    assert "considered devices: idma_read, idma_write, core, spatz, redmule" in message
