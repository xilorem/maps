"""Legacy bridge to Planning-owned memory checks."""

from maps.planning.memory import (
    estimate_stage_l1_memory_for_tile,
    estimate_stage_l2_memory,
    infer_input_slice_for_tile,
    virtual_tile_for_stage_tile,
)

__all__ = [
    "estimate_stage_l1_memory_for_tile",
    "estimate_stage_l2_memory",
    "infer_input_slice_for_tile",
    "virtual_tile_for_stage_tile",
]
