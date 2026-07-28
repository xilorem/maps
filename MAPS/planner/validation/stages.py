"""Validation of stages, layer bindings, and per-tile L1 capacity."""

from __future__ import annotations

from MAPS.pipeline.layer import LocalInput, TransitionInput
from MAPS.pipeline.pipeline import Pipeline
from MAPS.pipeline.stage import Stage
from MAPS.planner.validation.contracts import (
    ConstraintReport,
    ConstraintViolation,
    PlannerConstraints,
    append_violation,
)
from MAPS.planner.validation.memory import estimate_stage_l1_memory_for_tile


def validate_stage(
    stage: Stage,
    stage_id: int,
    pipeline: Pipeline,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    """Validate one stage's mesh, size, bindings, and L1 requirements.

    All independent problems are accumulated rather than raised.  Transition
    inputs are checked against the pipeline transition table, local inputs must
    reference an earlier layer output in the same stage, and optional capacity
    enforcement evaluates every physical stage tile.
    """

    violations: list[ConstraintViolation] = []
    if stage.submesh.mesh != pipeline.mesh:
        append_violation(
            violations,
            "stage_mesh_mismatch",
            f"stage {stage_id} submesh belongs to a different mesh",
        )
    if (
        constraints.max_stage_nodes > 0
        and len(stage.layers) > constraints.max_stage_nodes
    ):
        append_violation(
            violations,
            "stage_node_limit_exceeded",
            f"stage {stage_id} has {len(stage.layers)} layers, exceeding "
            f"max_stage_nodes={constraints.max_stage_nodes}",
        )
    try:
        stage.validate_tensors(pipeline.tensors)
    except ValueError as exc:
        append_violation(
            violations,
            "stage_tensor_binding_invalid",
            f"stage {stage_id}: {exc}",
        )

    for layer_idx, layer in enumerate(stage.layers):
        for binding_idx, binding in enumerate(layer.inputs):
            if isinstance(binding.source, TransitionInput):
                _validate_transition_input(
                    violations,
                    stage_id,
                    layer_idx,
                    binding_idx,
                    binding,
                    pipeline,
                )
            elif isinstance(binding.source, LocalInput):
                _validate_local_input(
                    violations,
                    stage_id,
                    layer_idx,
                    binding_idx,
                    binding,
                    stage,
                )

    if constraints.enforce_l1_capacity:
        for tile in stage.submesh.tiles:
            required_memory = estimate_stage_l1_memory_for_tile(stage, pipeline, tile)
            if required_memory > tile.memory.size:
                append_violation(
                    violations,
                    "tile_l1_capacity_exceeded",
                    f"stage {stage_id} requires {required_memory} L1 memory "
                    f"but tile {tile.tile_id} only provides {tile.memory.size}",
                )
    return ConstraintReport(tuple(violations))


def _validate_transition_input(
    violations: list[ConstraintViolation],
    stage_id: int,
    layer_idx: int,
    binding_idx: int,
    binding,
    pipeline: Pipeline,
) -> None:
    """Validate one layer binding that claims a transition source."""

    if not isinstance(binding.source, TransitionInput):
        append_violation(
            violations,
            "transition_input_source_invalid",
            f"stage {stage_id} layer {layer_idx} input {binding_idx} "
            "is not a transition input",
        )
        return
    transition_id = binding.source.transition_id
    if transition_id >= len(pipeline.transitions):
        append_violation(
            violations,
            "transition_reference_out_of_range",
            f"stage {stage_id} layer {layer_idx} input {binding_idx} references "
            f"missing transition {transition_id}",
        )
        return
    transition = pipeline.transitions[transition_id]
    if transition.dst_layer_id != stage_id:
        append_violation(
            violations,
            "transition_destination_mismatch",
            f"stage {stage_id} layer {layer_idx} input {binding_idx} references "
            f"transition {transition_id} targeting stage {transition.dst_layer_id}",
        )
    if transition.tensor_id != binding.tensor_id:
        append_violation(
            violations,
            "transition_tensor_mismatch",
            f"stage {stage_id} layer {layer_idx} input {binding_idx} "
            f"tensor_id {binding.tensor_id} does not match transition "
            f"tensor_id {transition.tensor_id}",
        )


def _validate_local_input(
    violations: list[ConstraintViolation],
    stage_id: int,
    layer_idx: int,
    binding_idx: int,
    binding,
    stage: Stage,
) -> None:
    """Validate one binding to a previous layer in the same stage."""

    if not isinstance(binding.source, LocalInput):
        append_violation(
            violations,
            "local_input_source_invalid",
            f"stage {stage_id} layer {layer_idx} input {binding_idx} "
            "is not a local input",
        )
        return
    local_output = binding.source
    if local_output.tensor_id != binding.tensor_id:
        append_violation(
            violations,
            "local_input_tensor_mismatch",
            f"stage {stage_id} layer {layer_idx} input {binding_idx} "
            f"tensor_id {binding.tensor_id} does not match local source "
            f"tensor_id {local_output.tensor_id}",
        )
    if local_output.layer_idx >= layer_idx:
        append_violation(
            violations,
            "local_output_layer_not_previous",
            f"stage {stage_id} layer {layer_idx} input {binding_idx} references "
            f"non-previous local layer {local_output.layer_idx}",
        )
        return
    local_layer = stage.layers[local_output.layer_idx]
    if not any(
        output.tensor_id == local_output.tensor_id
        for output in local_layer.outputs
    ):
        append_violation(
            violations,
            "local_output_tensor_missing",
            f"stage {stage_id} layer {layer_idx} input {binding_idx} references "
            f"tensor_id {local_output.tensor_id} missing from layer "
            f"{local_output.layer_idx} outputs",
        )
