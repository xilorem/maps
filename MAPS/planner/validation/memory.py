"""L1 and L2 residency estimates used by planner validation."""

from __future__ import annotations

from MAPS.core.layout import TensorRange, TensorSlice, tile_tensor_slice
from MAPS.core.tensor import Tensor
from MAPS.pipeline.layer import ExternalInput, Layer, LocalInput
from MAPS.pipeline.pipeline import Pipeline
from MAPS.pipeline.stage import Stage
from MAPS.planner.workload.memory import permanent_l1_allocation_bytes


def estimate_stage_l1_memory_for_tile(
    stage: Stage,
    pipeline: Pipeline,
    tile,
) -> int:
    """Estimate the backend's permanent allocation for one stage tile."""

    virtual_tile = virtual_tile_for_stage_tile(stage, pipeline, tile)
    allocation_sizes = []
    for layer in stage.layers:
        for binding_idx, binding in enumerate(layer.inputs):
            if isinstance(binding.source, LocalInput):
                continue
            tensor = pipeline.tensors[binding.tensor_id]
            tensor_slice = infer_input_slice_for_tile(
                layer,
                binding_idx,
                pipeline,
                virtual_tile,
            )
            slot_count = (
                1
                if tensor.is_initializer
                else pipeline.execution.num_token_slots
            )
            allocation_sizes.append(tensor.slice_num_bytes(tensor_slice) * slot_count)
        for binding in layer.outputs:
            tensor = pipeline.tensors[binding.tensor_id]
            tensor_slice = tile_tensor_slice(tensor, binding.layout, virtual_tile)
            allocation_sizes.append(
                tensor.slice_num_bytes(tensor_slice)
                * pipeline.execution.num_token_slots
            )
    return permanent_l1_allocation_bytes(allocation_sizes)


def estimate_stage_l2_memory(stage: Stage, pipeline: Pipeline) -> int:
    """Estimate L2 storage needed for a stage's external input bindings."""

    l2_memory = 0
    for layer in stage.layers:
        for binding_idx, binding in enumerate(layer.inputs):
            if not isinstance(binding.source, ExternalInput):
                continue
            tensor = pipeline.tensors[binding.tensor_id]
            max_binding_bytes = 0
            for tile in stage.submesh.tiles:
                virtual_tile = virtual_tile_for_stage_tile(stage, pipeline, tile)
                tensor_slice = infer_input_slice_for_tile(
                    layer,
                    binding_idx,
                    pipeline,
                    virtual_tile,
                )
                max_binding_bytes = max(
                    max_binding_bytes,
                    tensor.slice_num_bytes(tensor_slice),
                )
            l2_memory += max_binding_bytes
    return l2_memory


def infer_input_slice_for_tile(
    layer: Layer,
    binding_idx: int,
    pipeline: Pipeline,
    tile,
) -> TensorSlice:
    """Infer an input slice from tile work, falling back to the full tensor."""

    tensor = pipeline.tensors[layer.inputs[binding_idx].tensor_id]
    node = layer.node
    if node.payload is not None and layer.outputs:
        output_layouts = tuple(output.layout for output in layer.outputs)
        tile_work = node.payload.build_tile_work(output_layouts=output_layouts, tile=tile)
        for reference in tile_work.input_slices:
            if tensor == reference.tensor:
                return reference.tensor_slice
    return _default_tensor_slice(tensor)


def virtual_tile_for_stage_tile(stage: Stage, pipeline: Pipeline, tile):
    """Translate one physical stage tile to the virtual layout tile."""

    if not stage.physical_to_virtual:
        return tile
    return pipeline.mesh.tile_by_id(stage.physical_to_virtual[tile.tile_id])


def _default_tensor_slice(tensor: Tensor) -> TensorSlice:
    """Return a slice covering an entire tensor."""

    return TensorSlice(
        rank=tensor.rank,
        dims=tuple(
            TensorRange(start=0, length=dimension)
            for dimension in tensor.dims
        ),
    )
