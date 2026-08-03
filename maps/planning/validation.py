"""Constraints and validation for physical Execution Plans."""

from __future__ import annotations

from dataclasses import dataclass, field

from maps.hardware import WorkKind, WorkSignature
from maps.operations.collective import AllReducePayload
from maps.planning.execution_plan import (
    CollectiveGroup,
    ExecutionPlan,
    InitializerInput,
    LocalInput,
    TransitionSource,
)
from maps.planning.mapping import (
    LayoutAxisMode,
    TensorLayout,
    TensorSlice,
    TensorSubSlice,
)
from maps.planning.stages import derive_virtual_collective_groups
from maps.planning.execution_plan import (
    estimate_stage_l1_memory_for_tile,
    estimate_stage_l2_memory,
)
from maps.planning.transitions.contracts import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
    Transition,
)


@dataclass(frozen=True)
class PlanningConstraints:
    """Hard legality constraints checked against a completed Execution Plan."""

    max_stage_operations: int = 0
    enforce_l1_capacity: bool = True
    enforce_l2_capacity: bool = True

    def __post_init__(self) -> None:
        if self.max_stage_operations < 0:
            raise ValueError("max_stage_operations must be >= 0")


@dataclass(frozen=True)
class ConstraintViolation:
    """One categorized Planning legality violation."""

    kind: str
    message: str


@dataclass(frozen=True)
class ConstraintReport:
    """Complete non-throwing result of Execution Plan validation."""

    violations: tuple[ConstraintViolation, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        """Return whether validation found no violations."""

        return not self.violations


def append_violation(
    violations: list[ConstraintViolation],
    kind: str,
    message: str,
) -> None:
    """Append one consistently constructed violation to an accumulator."""

    violations.append(ConstraintViolation(kind=kind, message=message))


def validate_execution_plan(
    execution_plan: ExecutionPlan,
    constraints: PlanningConstraints,
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
    constraints: PlanningConstraints,
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
    constraints: PlanningConstraints,
) -> bool:
    stage = execution_plan.stages[stage_id]
    if stage.submesh.mesh != execution_plan.mesh:
        append_violation(
            violations,
            "stage_mesh_mismatch",
            f"stage {stage_id} submesh belongs to a different mesh",
        )
    if (
        constraints.max_stage_operations > 0
        and len({layer.source_operation for layer in stage.layers})
        > constraints.max_stage_operations
    ):
        append_violation(
            violations,
            "stage_operation_limit_exceeded",
            f"stage {stage_id} exceeds "
            f"max_stage_operations={constraints.max_stage_operations}",
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
        _validate_collective_groups(
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
                _validate_partial_local_input(
                    violations,
                    stage_id,
                    layer_index,
                    input_index,
                    source,
                    execution_plan,
                )
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


def _layout_is_partial(layout: TensorLayout) -> bool:
    return any(
        axis.mode is LayoutAxisMode.PARTIAL
        for axis in (layout.mesh_x, layout.mesh_y)
    )


def _validate_partial_local_input(
    violations: list[ConstraintViolation],
    stage_id: int,
    layer_index: int,
    input_index: int,
    source: LocalInput,
    execution_plan: ExecutionPlan,
) -> None:
    stage = execution_plan.stages[stage_id]
    if source.layer_idx < 0 or source.layer_idx >= len(stage.layers):
        return
    producer = stage.layers[source.layer_idx]
    output = next(
        (
            candidate
            for candidate in producer.outputs
            if candidate.tensor_id == source.tensor_id
        ),
        None,
    )
    if output is None or not _layout_is_partial(output.layout):
        return
    consumer = stage.layers[layer_index]
    if not isinstance(consumer.node.payload, AllReducePayload):
        append_violation(
            violations,
            "partial_value_consumed_by_ordinary_layer",
            f"stage {stage_id} layer {layer_index} input {input_index} consumes "
            f"unresolved Partial Value tensor {source.tensor_id}",
        )


def _validate_collective_groups(
    violations: list[ConstraintViolation],
    stage_id: int,
    layer_index: int,
    execution_plan: ExecutionPlan,
) -> None:
    stage = execution_plan.stages[stage_id]
    layer = stage.layers[layer_index]
    is_collective = isinstance(layer.node.payload, AllReducePayload)
    if not is_collective:
        if layer.collective_groups:
            append_violation(
                violations,
                "collective_groups_on_ordinary_layer",
                f"stage {stage_id} layer {layer_index} is not a collective",
            )
        return
    expected_work_kind = {
        "sum": WorkKind.ALL_REDUCE_SUM,
        "max": WorkKind.ALL_REDUCE_MAX,
    }[layer.node.payload.reduction]
    if layer.node.payload.work_kind is not expected_work_kind:
        append_violation(
            violations,
            "collective_work_kind_collision",
            f"stage {stage_id} layer {layer_index} reduction "
            f"{layer.node.payload.reduction} uses {layer.node.payload.work_kind.name}",
        )
    if not layer.collective_groups:
        append_violation(
            violations,
            "collective_groups_missing",
            f"stage {stage_id} layer {layer_index} has no Collective Groups",
        )
        return

    participant_ids = tuple(
        tile_id
        for group in layer.collective_groups
        for tile_id in group.tile_ids
    )
    if len(participant_ids) != len(set(participant_ids)):
        append_violation(
            violations,
            "collective_groups_overlap",
            f"stage {stage_id} layer {layer_index} assigns a tile more than once",
        )
    stage_tile_ids = set(stage.submesh.tile_ids)
    if not set(participant_ids) <= stage_tile_ids:
        append_violation(
            violations,
            "collective_group_binding_invalid",
            f"stage {stage_id} layer {layer_index} includes tiles outside its Submesh",
        )
    expected_groups = _expected_physical_collective_groups(
        stage_id,
        layer_index,
        execution_plan,
    )
    if expected_groups is not None and layer.collective_groups != expected_groups:
        append_violation(
            violations,
            "collective_group_binding_invalid",
            f"stage {stage_id} layer {layer_index} groups do not match its "
            "ownership-derived physical binding",
        )
    if not layer.outputs:
        return
    output = layer.outputs[0]
    if _layout_is_partial(output.layout):
        append_violation(
            violations,
            "collective_output_remains_partial",
            f"stage {stage_id} layer {layer_index} does not resolve its Partial Value",
        )


def _expected_physical_collective_groups(
    stage_id: int,
    layer_index: int,
    execution_plan: ExecutionPlan,
) -> tuple[CollectiveGroup, ...] | None:
    stage = execution_plan.stages[stage_id]
    layer = stage.layers[layer_index]
    if not layer.inputs or not isinstance(layer.inputs[0].source, LocalInput):
        return None
    source = layer.inputs[0].source
    if source.layer_idx < 0 or source.layer_idx >= len(stage.layers):
        return None
    producer_output = next(
        (
            output
            for output in stage.layers[source.layer_idx].outputs
            if output.tensor_id == source.tensor_id
        ),
        None,
    )
    if producer_output is None or source.tensor_id >= len(execution_plan.tensors):
        return None
    virtual_groups = derive_virtual_collective_groups(
        execution_plan.tensors[source.tensor_id],
        producer_output.layout,
    )
    return tuple(
        CollectiveGroup(
            tuple(
                sorted(
                    stage.virtual_to_physical.get(tile_id, tile_id)
                    for tile_id in virtual_group.virtual_tile_ids
                )
            )
        )
        for virtual_group in virtual_groups
    )


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
    ):
        append_violation(
            violations,
            "transition_destination_mismatch",
            f"transition {source.transition_id} does not target stage {stage_id} "
            f"for tensor {layer_input.tensor_id}",
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
    outputs = tuple(
        output
        for layer in execution_plan.stages[transition.source_stage_id].layers
        for output in layer.outputs
        if output.tensor_id == transition.tensor_id
    )
    if not outputs:
        append_violation(
            violations,
            "transition_source_tensor_mismatch",
            f"transition {transition_id} tensor is not resident in its source stage",
        )
    elif any(_layout_is_partial(output.layout) for output in outputs):
        append_violation(
            violations,
            "partial_value_transition_escape",
            f"transition {transition_id} exposes unresolved Partial Value tensor "
            f"{transition.tensor_id}",
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
    destination_inputs = tuple(
        layer_input
        for layer in execution_plan.stages[transition.destination_stage_id].layers
        for layer_input in layer.inputs
        if layer_input.tensor_id == transition.tensor_id
    )
    if not destination_inputs:
        append_violation(
            violations,
            "transition_destination_tensor_missing",
            f"transition {transition_id} tensor is not consumed in its destination stage",
        )
        return
    if any(
        not isinstance(destination_input.source, TransitionSource)
        or destination_input.source.transition_id != transition_id
        for destination_input in destination_inputs
    ):
        append_violation(
            violations,
            "transition_destination_binding_mismatch",
            f"transition {transition_id} is not referenced by every destination consumer",
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


__all__ = [
    "ConstraintReport",
    "ConstraintViolation",
    "PlanningConstraints",
    "require_valid_execution_plan",
    "validate_execution_plan",
]
