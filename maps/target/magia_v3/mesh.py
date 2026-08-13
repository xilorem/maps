"""MAGIA-v3 Mesh construction and deployment memory contracts."""

from maps.hardware import L1Memory, L2Memory, Mesh, Tile
from maps.target.magia.mesh import (
    L1_BANDWIDTH_BYTES,
    MESH_HEIGHT,
    MESH_WIDTH,
    NOC_CHANNEL_WIDTH_BYTES,
    NOC_HOP_LATENCY_CYCLES,
    NOC_WIDE_CHANNEL_WIDTH_BYTES,
    _noc,
)

from .devices import DEVICE_ASSIGNMENT, TILE_DEVICES


L1_SIZE_BYTES = 1024 * 1024
L1_DATA_BYTES = 0xC0000
L1_TASK_SCRATCH_BYTES = 0x10000
L1_USABLE_BYTES = L1_DATA_BYTES
L1_STACK_BYTES = 0x10000
L1_FIFO_OFFSET = 0xD0000
L1_READY_OFFSET = 0xE0000
L1_CONTROL_OFFSET = 0xF0000

L2_BULK_BASE = 0xC0000000
L2_BULK_END = 0xCC000000
L2_ARENA_BASE = 0xCC020000
L2_ARENA_END = 0xCCF20000
L2_SIZE_BYTES = L2_BULK_END - L2_BULK_BASE
L2_BANDWIDTH_BYTES = 32


def build_mesh(width: int = MESH_WIDTH, height: int = MESH_HEIGHT) -> Mesh:
    """Build a planner-ready MAGIA-v3 Mesh with configurable compute shape."""

    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(L2_SIZE_BYTES, L2_BANDWIDTH_BYTES),
        noc=_noc(width, height),
        tiles=tuple(
            Tile(
                tile_id=y * width + x,
                x=x,
                y=y,
                memory=L1Memory(L1_USABLE_BYTES, L1_BANDWIDTH_BYTES),
                devices=TILE_DEVICES,
                device_assignment=DEVICE_ASSIGNMENT,
            )
            for y in range(height)
            for x in range(width)
        ),
    )


__all__ = [
    "L1_BANDWIDTH_BYTES",
    "L1_CONTROL_OFFSET",
    "L1_DATA_BYTES",
    "L1_FIFO_OFFSET",
    "L1_READY_OFFSET",
    "L1_SIZE_BYTES",
    "L1_STACK_BYTES",
    "L1_TASK_SCRATCH_BYTES",
    "L1_USABLE_BYTES",
    "L2_ARENA_BASE",
    "L2_ARENA_END",
    "L2_BANDWIDTH_BYTES",
    "L2_BULK_BASE",
    "L2_BULK_END",
    "L2_SIZE_BYTES",
    "MESH_HEIGHT",
    "MESH_WIDTH",
    "NOC_CHANNEL_WIDTH_BYTES",
    "NOC_HOP_LATENCY_CYCLES",
    "NOC_WIDE_CHANNEL_WIDTH_BYTES",
    "build_mesh",
]
