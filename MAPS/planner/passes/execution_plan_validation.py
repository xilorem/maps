"""Planner-side validation for unified Execution Plans."""

from __future__ import annotations

from MAPS.core.layout import TensorSlice, TensorSubSlice
from MAPS.arch import WorkSignature
from MAPS.pipeline.execution_plan import ExecutionPlan
from MAPS.pipeline.layer import InitializerInput, LocalInput, TransitionSource
from MAPS.planner.validation.contracts import (
    ConstraintReport,
    ConstraintViolation,
    PlannerConstraints,
    append_violation,
)
from MAPS.planner.validation.memory import (
    estimate_stage_l1_memory_for_tile,
    estimate_stage_l2_memory,
)
from MAPS.transitions.contracts import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
    Transition,
)


def validate_execution_plan(
    execution_plan: ExecutionPlan,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    """Validate all physical references and variant-specific invariants."""

    violations: list[ConstraintViolation] = []
    required_l2_memory = 0
    for stage_id, stage in enumerate(execution_plan.stages):
        tensor_bindings_valid = _validate_stage(
            violations,
            stage_id,
            execution_plan,
            constraints,
        )
        if constraints.enforce_l2_capacity and tensor_bindings_valid:
            required_l2_memory += estimate_stage_l2_memory(stage, execution_plan)

    if (
        constraints.enforce_l2_capacity
        and required_l2_memory > execution_plan.mesh.l2_memory.size
    ):
        append_violation(
            violations,
            "mesh_l2_capacity_exceeded",
            f"execution plan requires {required_l2_memory} L2 memory but mesh "
            f"only provides {execution_plan.mesh.l2_memory.size}",
        )

    for transition_id, transition in enumerate(execution_plan.transitions):
        _validate_transition(
            violations,
            transition_id,
            transition,
            execution_plan,
        )
    return ConstraintReport(tuple(violations))


def require_valid_execution_plan(
    execution_plan: ExecutionPlan,
    constraints: PlannerConstraints,
    *,
    error_prefix: str,
) -> None:
    """Raise one detailed error when an Execution Plan violates constraints."""

    report = validate_execution_plan(execution_plan, constraints)
    if report.is_valid:
        return
    details = "; ".join(
        f"{violation.kind}: {violation.message}"
        for violation in report.violations
    )
    raise ValueError(f"{error_prefix}: {details}")


def _validate_stage(
    violations: list[ConstraintViolation],
    stage_id: int,
    execution_plan: ExecutionPlan,
    constraints: PlannerConstraints,
) -> bool:
    stage = execution_plan.stages[stage_id]
    if stage.submesh.mesh != execution_plan.mesh:
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
            f"stage {stage_id} exceeds max_stage_nodes={constraints.max_stage_nodes}",
        )
    tensor_bindings_valid = True
    try:
        stage.validate_tensors(execution_plan.tensors)
    except ValueError as exc:
        tensor_bindings_valid = False
        append_violation(
            violations,
            "stage_tensor_binding_invalid",
            f"stage {stage_id}: {exc}",
        )

    for layer_index, layer in enumerate(stage.layers):
        _validate_layer_device(
            violations,
            stage_id,
            layer_index,
            execution_plan,
        )
        for input_index, layer_input in enumerate(layer.inputs):
            source = layer_input.source
            if isinstance(source, InitializerInput):
                _validate_initializer_input(
                    violations,
                    stage_id,
                    layer_index,
                    input_index,
                    source,
                    execution_plan,
                )
            elif isinstance(source, TransitionSource):
                _validate_transition_source(
                    violations,
                    stage_id,
                    layer_index,
                    input_index,
                    source,
                    execution_plan,
                )
            elif isinstance(source, LocalInput):
                _validate_local_input(
                    violations,
                    stage_id,
                    layer_index,
                    input_index,
                    source,
                    execution_plan,
                )
            else:
                append_violation(
                    violations,
                    "legacy_layer_input_source",
                    f"stage {stage_id} layer {layer_index} input {input_index} "
                    "uses a source outside the Execution Plan contract",
                )

    if constraints.enforce_l1_capacity and tensor_bindings_valid:
        for tile in stage.submesh.tiles:
            required_memory = estimate_stage_l1_memory_for_tile(
                stage,
                execution_plan,
                tile,
            )
            if required_memory > tile.memory.size:
                append_violation(
                    violations,
                    "tile_l1_capacity_exceeded",
                    f"stage {stage_id} requires {required_memory} L1 memory but "
                    f"tile {tile.tile_id} only provides {tile.memory.size}",
                )
    return tensor_bindings_valid


def _validate_layer_device(
    violations: list[ConstraintViolation],
    stage_id: int,
    layer_index: int,
    execution_plan: ExecutionPlan,
) -> None:
    stage = execution_plan.stages[stage_id]
    layer = stage.layers[layer_index]
    try:
        signature = WorkSignature.from_node(layer.node)
    except ValueError as exc:
        append_violation(
            violations,
            "layer_device_assignment_invalid",
            f"stage {stage_id} layer {layer_index} cannot validate its Device "
            f"Assignment: {exc}",
        )
        return
    if layer.device_name is None:
        append_violation(
            violations,
            "layer_device_assignment_invalid",
            f"stage {stage_id} layer {layer_index} node {layer.node.name} with "
            f"{signature} has no retained Device name",
        )
        return
    for tile in stage.submesh.tiles:
        configured_name = tile.device_assignment.assignments.get(signature)
        try:
            device = tile.device_by_name(layer.device_name)
        except ValueError as exc:
            append_violation(
                violations,
                "layer_device_assignment_invalid",
                f"stage {stage_id} layer {layer_index} node {layer.node.name}: {exc}",
            )
            return
        if configured_name != layer.device_name or not device.supports(signature):
            append_violation(
                violations,
                "layer_device_assignment_invalid",
                f"stage {stage_id} layer {layer_index} node {layer.node.name} with "
                f"{signature} retains {layer.device_name} on tile {tile.tile_id}, "
                f"but configured assignment is {configured_name!r} and the Device "
                f"capability match is {device.supports(signature)}",
            )
            return


def _validate_initializer_input(
    violations: list[ConstraintViolation],
    stage_id: int,
    layer_index: int,
    input_index: int,
    source: InitializerInput,
    execution_plan: ExecutionPlan,
) -> None:
    stage = execution_plan.stages[stage_id]
    layer_input = stage.layers[layer_index].inputs[input_index]
    if layer_input.tensor_id >= len(execution_plan.tensors):
        return
    tensor = execution_plan.tensors[layer_input.tensor_id]
    if not tensor.is_initializer:
        append_violation(
            violations,
            "initializer_input_tensor_mismatch",
            f"stage {stage_id} layer {layer_index} input {input_index} "
            "does not reference an initializer tensor",
        )
    destination_tile_ids = [
        destination.tile_id
        for destination in source.destinations
    ]
    if (
        set(destination_tile_ids) != set(stage.submesh.tile_ids)
        or len(destination_tile_ids) != len(stage.submesh.tile_ids)
    ):
        append_violation(
            violations,
            "initializer_residency_tiles_mismatch",
            f"stage {stage_id} layer {layer_index} input {input_index} "
            "residency does not cover exactly the stage tiles",
        )
    for destination in source.destinations:
        if not execution_plan.mesh.contains_tile_id(destination.tile_id):
            append_violation(
                violations,
                "initializer_destination_tile_out_of_mesh",
                f"initializer destination tile {destination.tile_id} is outside mesh",
            )
        elif destination.tile_id not in stage.submesh.tile_ids:
            append_violation(
                violations,
                "initializer_destination_tile_outside_stage",
                f"initializer destination tile {destination.tile_id} is outside "
                f"stage {stage_id}",
            )
        _validate_slice(
            violations,
            destination.tensor_slice,
            tensor,
            "initializer_slice_invalid",
        )


def _validate_transition_source(
    violations: list[ConstraintViolation],
    stage_id: int,
    layer_index: int,
    input_index: int,
    source: TransitionSource,
    execution_plan: ExecutionPlan,
) -> None:
    layer_input = execution_plan.stages[stage_id].layers[layer_index].inputs[
        input_index
    ]
    if layer_index != 0:
        append_violation(
            violations,
            "transition_source_not_on_first_layer",
            f"stage {stage_id} layer {layer_index} input {input_index} "
            "references a Transition outside the first layer",
        )
    if source.transition_id >= len(execution_plan.transitions):
        append_violation(
            violations,
            "transition_reference_out_of_range",
            f"stage {stage_id} input {input_index} references missing "
            f"transition {source.transition_id}",
        )
        return
    transition = execution_plan.transitions[source.transition_id]
    if isinstance(transition, OutputTransition):
        append_violation(
            violations,
            "transition_source_references_output",
            f"stage {stage_id} input {input_index} references an Output Transition",
        )
        return
    if (
        transition.destination_stage_id != stage_id
        or transition.destination_input_index != input_index
    ):
        append_violation(
            violations,
            "transition_destination_mismatch",
            f"transition {source.transition_id} does not target stage {stage_id} "
            f"input {input_index}",
        )
    if transition.tensor_id != layer_input.tensor_id:
        append_violation(
            violations,
            "transition_tensor_mismatch",
            f"transition {source.transition_id} tensor does not match its binding",
        )


def _validate_local_input(
    violations: list[ConstraintViolation],
    stage_id: int,
    layer_index: int,
    input_index: int,
    source: LocalInput,
    execution_plan: ExecutionPlan,
) -> None:
    stage = execution_plan.stages[stage_id]
    layer_input = stage.layers[layer_index].inputs[input_index]
    if source.tensor_id != layer_input.tensor_id:
        append_violation(
            violations,
            "local_input_tensor_mismatch",
            f"stage {stage_id} layer {layer_index} input {input_index} "
            "does not match its local tensor",
        )
    if source.layer_idx >= layer_index:
        append_violation(
            violations,
            "local_output_layer_not_previous",
            f"stage {stage_id} layer {layer_index} input {input_index} "
            "does not reference a previous layer",
        )
    elif not any(
        output.tensor_id == source.tensor_id
        for output in stage.layers[source.layer_idx].outputs
    ):
        append_violation(
            violations,
            "local_output_tensor_missing",
            f"stage {stage_id} layer {layer_index} input {input_index} "
            "references a tensor absent from the local source layer",
        )


def _validate_transition(
    violations: list[ConstraintViolation],
    transition_id: int,
    transition: Transition,
    execution_plan: ExecutionPlan,
) -> None:
    if not 0 <= transition.tensor_id < len(execution_plan.tensors):
        append_violation(
            violations,
            "transition_tensor_out_of_range",
            f"transition {transition_id} references missing tensor "
            f"{transition.tensor_id}",
        )
        return
    tensor = execution_plan.tensors[transition.tensor_id]
    if tensor.is_initializer:
        append_violation(
            violations,
            "initializer_transition",
            f"transition {transition_id} references initializer tensor "
            f"{transition.tensor_id}",
        )
    if isinstance(transition, InputTransition):
        _validate_destination(
            violations,
            transition_id,
            transition,
            execution_plan,
        )
        for destination in transition.destinations:
            _validate_physical_tile(
                violations,
                destination.tile_id,
                execution_plan,
                transition.destination_stage_id,
                "transition_destination_tile",
            )
            _validate_slice(
                violations,
                destination.tensor_slice,
                tensor,
                "transition_slice_invalid",
            )
    elif isinstance(transition, IntermediateTransition):
        if transition.source_stage_id == transition.destination_stage_id:
            append_violation(
                violations,
                "intermediate_transition_within_stage",
                f"transition {transition_id} connects stage "
                f"{transition.source_stage_id} to itself",
            )
        _validate_source(
            violations,
            transition_id,
            transition,
            execution_plan,
        )
        _validate_destination(
            violations,
            transition_id,
            transition,
            execution_plan,
        )
        for transfer in transition.transfers:
            _validate_physical_tile(
                violations,
                transfer.source_tile_id,
                execution_plan,
                transition.source_stage_id,
                "transfer_source_tile",
            )
            _validate_physical_tile(
                violations,
                transfer.destination_tile_id,
                execution_plan,
                transition.destination_stage_id,
                "transfer_destination_tile",
            )
            _validate_subslice(
                violations,
                transfer.source_subslice,
                tensor,
                "transfer_source_subslice_invalid",
            )
            _validate_subslice(
                violations,
                transfer.destination_subslice,
                tensor,
                "transfer_destination_subslice_invalid",
            )
            if (
                isinstance(transfer.source_subslice, TensorSubSlice)
                and isinstance(transfer.destination_subslice, TensorSubSlice)
                and _global_subslice_region(transfer.source_subslice)
                != _global_subslice_region(transfer.destination_subslice)
            ):
                append_violation(
                    violations,
                    "transfer_subslice_region_mismatch",
                    f"transition {transition_id} transfer source and destination "
                    "refer to different Tensor regions",
                )
    else:
        _validate_source(
            violations,
            transition_id,
            transition,
            execution_plan,
        )
        for source in transition.sources:
            _validate_physical_tile(
                violations,
                source.tile_id,
                execution_plan,
                transition.source_stage_id,
                "transition_source_tile",
            )
            _validate_slice(
                violations,
                source.tensor_slice,
                tensor,
                "transition_slice_invalid",
            )


def _validate_source(
    violations: list[ConstraintViolation],
    transition_id: int,
    transition: IntermediateTransition | OutputTransition,
    execution_plan: ExecutionPlan,
) -> None:
    if not 0 <= transition.source_stage_id < len(execution_plan.stages):
        append_violation(
            violations,
            "transition_source_stage_out_of_range",
            f"transition {transition_id} references missing source stage",
        )
        return
    outputs = execution_plan.stages[transition.source_stage_id].layers[-1].outputs
    if not 0 <= transition.source_output_index < len(outputs):
        append_violation(
            violations,
            "transition_source_output_out_of_range",
            f"transition {transition_id} references missing source output",
        )
    elif outputs[transition.source_output_index].tensor_id != transition.tensor_id:
        append_violation(
            violations,
            "transition_source_tensor_mismatch",
            f"transition {transition_id} does not match its source output tensor",
        )


def _validate_destination(
    violations: list[ConstraintViolation],
    transition_id: int,
    transition: InputTransition | IntermediateTransition,
    execution_plan: ExecutionPlan,
) -> None:
    if not 0 <= transition.destination_stage_id < len(execution_plan.stages):
        append_violation(
            violations,
            "transition_destination_stage_out_of_range",
            f"transition {transition_id} references missing destination stage",
        )
        return
    inputs = execution_plan.stages[
        transition.destination_stage_id
    ].layers[0].inputs
    if not 0 <= transition.destination_input_index < len(inputs):
        append_violation(
            violations,
            "transition_destination_input_out_of_range",
            f"transition {transition_id} references missing destination input",
        )
        return
    destination_input = inputs[transition.destination_input_index]
    if destination_input.tensor_id != transition.tensor_id:
        append_violation(
            violations,
            "transition_destination_tensor_mismatch",
            f"transition {transition_id} does not match destination input tensor",
        )
    if (
        not isinstance(destination_input.source, TransitionSource)
        or destination_input.source.transition_id != transition_id
    ):
        append_violation(
            violations,
            "transition_destination_binding_mismatch",
            f"transition {transition_id} is not referenced by its destination input",
        )


def _validate_physical_tile(
    violations: list[ConstraintViolation],
    tile_id: int,
    execution_plan: ExecutionPlan,
    stage_id: int,
    prefix: str,
) -> None:
    if not execution_plan.mesh.contains_tile_id(tile_id):
        append_violation(
            violations,
            f"{prefix}_out_of_mesh",
            f"tile {tile_id} is outside the physical mesh",
        )
    elif (
        stage_id < len(execution_plan.stages)
        and tile_id not in execution_plan.stages[stage_id].submesh.tile_ids
    ):
        append_violation(
            violations,
            f"{prefix}_outside_stage",
            f"tile {tile_id} is outside stage {stage_id}",
        )


def _validate_slice(
    violations: list[ConstraintViolation],
    tensor_slice: object,
    tensor,
    kind: str,
) -> None:
    if not isinstance(tensor_slice, TensorSlice):
        append_violation(violations, kind, "value is not a TensorSlice")
        return
    if tensor_slice.rank != tensor.rank or any(
        dimension.start + dimension.length > tensor_length
        for dimension, tensor_length in zip(tensor_slice.dims, tensor.dims)
    ):
        append_violation(violations, kind, "TensorSlice is outside its Tensor")


def _validate_subslice(
    violations: list[ConstraintViolation],
    subslice: object,
    tensor,
    kind: str,
) -> None:
    if not isinstance(subslice, TensorSubSlice):
        append_violation(violations, kind, "value is not a TensorSubSlice")
        return
    _validate_slice(violations, subslice.parent, tensor, kind)


def _global_subslice_region(
    subslice: TensorSubSlice,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (parent.start + relative.start, relative.length)
        for parent, relative in zip(subslice.parent.dims, subslice.dims)
    )


__all__ = ["require_valid_execution_plan", "validate_execution_plan"]
