"""L1 and L2 residency estimates used by Planning validation."""

from __future__ import annotations

from typing import cast

from maps.hardware import Tile
from maps.planning.layouts import (
    TensorRange,
    TensorSlice,
    tensor_slice_num_bytes,
    tile_tensor_slice,
)
from maps.graph import Tensor
from maps.operations.contracts import OpPayload
from maps.planning.execution_plan import (
    ExecutionPlan,
    InitializerInput,
    Layer,
    LocalInput,
    TransitionSource,
    Stage,
)
from maps.planning.allocation.memory import permanent_l1_allocation_bytes
from maps.planning.transitions.contracts import InputTransition


def estimate_stage_l1_memory_for_tile(
    stage: Stage,
    execution_plan: ExecutionPlan,
    tile: Tile,
) -> int:
    """Estimate the backend's permanent allocation for one stage tile."""

    virtual_tile = virtual_tile_for_stage_tile(stage, execution_plan, tile)
    allocation_sizes = []
    for layer in stage.layers:
        for binding_idx, binding in enumerate(layer.inputs):
            if isinstance(binding.source, LocalInput):
                continue
            tensor = execution_plan.tensors[binding.tensor_id]
            if isinstance(binding.source, InitializerInput):
                destination = next(
                    (
                        destination
                        for destination in binding.source.destinations
                        if destination.tile_id == tile.tile_id
                    ),
                    None,
                )
                if destination is None:
                    continue
                tensor_slice = destination.tensor_slice
                slot_count = 1
            else:
                tensor_slice = infer_input_slice_for_tile(
                    layer,
                    binding_idx,
                    execution_plan,
                    virtual_tile,
                )
                slot_count = execution_plan.execution.num_token_slots
            allocation_sizes.append(
                tensor_slice_num_bytes(tensor, tensor_slice) * slot_count
            )
        for binding in layer.outputs:
            tensor = execution_plan.tensors[binding.tensor_id]
            tensor_slice = tile_tensor_slice(tensor, binding.layout, virtual_tile)
            allocation_sizes.append(
                tensor_slice_num_bytes(tensor, tensor_slice)
                * execution_plan.execution.num_token_slots
            )
    return permanent_l1_allocation_bytes(allocation_sizes)


def estimate_stage_l2_memory(
    stage: Stage,
    execution_plan: ExecutionPlan,
) -> int:
    """Estimate L2 storage needed for a stage's external input bindings."""

    l2_memory = 0
    for layer in stage.layers:
        for binding_idx, binding in enumerate(layer.inputs):
            is_runtime_input = (
                isinstance(binding.source, TransitionSource)
                and binding.source.transition_id < len(execution_plan.transitions)
                and isinstance(
                    execution_plan.transitions[binding.source.transition_id],
                    InputTransition,
                )
            )
            if not is_runtime_input:
                continue
            tensor = execution_plan.tensors[binding.tensor_id]
            max_binding_bytes = 0
            for tile in stage.submesh.tiles:
                virtual_tile = virtual_tile_for_stage_tile(
                    stage,
                    execution_plan,
                    tile,
                )
                tensor_slice = infer_input_slice_for_tile(
                    layer,
                    binding_idx,
                    execution_plan,
                    virtual_tile,
                )
                max_binding_bytes = max(
                    max_binding_bytes,
                    tensor_slice_num_bytes(tensor, tensor_slice),
                )
            l2_memory += max_binding_bytes
    return l2_memory


def infer_input_slice_for_tile(
    layer: Layer,
    binding_idx: int,
    execution_plan: ExecutionPlan,
    tile: Tile,
) -> TensorSlice:
    """Infer an input slice from tile work, falling back to the full tensor."""

    tensor = execution_plan.tensors[layer.inputs[binding_idx].tensor_id]
    node = layer.node
    if node.payload is not None and layer.outputs:
        output_layouts = tuple(output.layout for output in layer.outputs)
        tile_work = cast(OpPayload, node.payload).build_tile_work(
            output_layouts=output_layouts,
            tile=tile,
        )
        for reference in tile_work.input_slices:
            if tensor == reference.tensor:
                return reference.tensor_slice
    return _default_tensor_slice(tensor)


def virtual_tile_for_stage_tile(
    stage: Stage,
    execution_plan: ExecutionPlan,
    tile: Tile,
) -> Tile:
    """Translate one physical stage tile to the virtual layout tile."""

    if not stage.physical_to_virtual:
        return tile
    return execution_plan.mesh.tile_by_id(stage.physical_to_virtual[tile.tile_id])


def _default_tensor_slice(tensor: Tensor) -> TensorSlice:
    """Return a slice covering an entire tensor."""

    return TensorSlice(
        rank=tensor.rank,
        dims=tuple(
            TensorRange(start=0, length=dimension)
            for dimension in tensor.dims
        ),
    )
