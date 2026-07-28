"""Per-tile, stage-local permanent L1 allocation accounting."""

from __future__ import annotations

L1_ALLOCATION_ALIGNMENT_BYTES = 16


def permanent_l1_allocation_for_stage(
    stage_nodes: tuple,
    node_output_layouts: tuple[tuple, ...],
    submesh,
    initializer_tensors: frozenset,
    num_token_slots: int = 2,
) -> int:
    """Return the greatest permanent L1 allocation on any virtual tile."""

    return max(
        (
            permanent_l1_allocation_for_tile(
                stage_nodes,
                node_output_layouts,
                tile,
                initializer_tensors,
                num_token_slots,
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def permanent_l1_allocation_for_tile(
    stage_nodes: tuple,
    node_output_layouts: tuple[tuple, ...],
    tile,
    initializer_tensors: frozenset,
    num_token_slots: int = 2,
) -> int:
    """Mirror the backend's monotonic, non-reusing tile-L1 allocator."""

    works = tuple(
        node.payload.build_tile_work(output_layouts=layouts, tile=tile)
        for node, layouts in zip(stage_nodes, node_output_layouts)
    )
    produced_tensors = set()
    allocation_sizes = []
    for work in works:
        for reference in work.input_slices:
            if reference.tensor in produced_tensors:
                continue
            slot_count = (
                1
                if _is_initializer(reference.tensor, initializer_tensors)
                else num_token_slots
            )
            allocation_sizes.append(reference.num_bytes * slot_count)

        for reference in work.output_slices:
            allocation_sizes.append(reference.num_bytes * num_token_slots)
            produced_tensors.add(reference.tensor)

    return permanent_l1_allocation_bytes(allocation_sizes)


def permanent_l1_allocation_bytes(allocation_sizes) -> int:
    """Return the final offset of the backend's monotonic L1 allocator."""

    next_offset = 0
    for allocation_size in allocation_sizes:
        next_offset = _align_to(next_offset, L1_ALLOCATION_ALIGNMENT_BYTES)
        next_offset += allocation_size
    return next_offset


def _is_initializer(tensor: object, initializer_tensors: frozenset) -> bool:
    return tensor in initializer_tensors or getattr(tensor, "is_initializer", False)


def _align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
