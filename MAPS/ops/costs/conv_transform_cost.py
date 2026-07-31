"""Provisional memory-movement cost for Conv-to-GEMM transforms."""

from __future__ import annotations

from MAPS.arch import Device, Tile
from MAPS.ops.common.cost import OpCostModel
from MAPS.ops.defs.conv_transforms import TransformTileWork


class ConvTransformCostModel(OpCostModel):
    """Estimate scalar-core transform cycles from visible L1 traffic."""

    diagnostic_label = (
        "provisional Conv-to-GEMM transform estimate pending measured "
        "bytes-read/bytes-written/core-L1 models"
    )

    def cost(
        self,
        tile_work: TransformTileWork,
        tile: Tile,
        assigned_device: Device | None = None,
    ) -> int:
        del assigned_device
        bytes_read = sum(
            ref.tensor.slice_num_bytes(ref.tensor_slice)
            for ref in tile_work.input_slices
        )
        bytes_written = sum(
            ref.tensor.slice_num_bytes(ref.tensor_slice)
            for ref in tile_work.output_slices
        )
        total_bytes = bytes_read + bytes_written
        return (total_bytes + tile.memory.bandwidth - 1) // tile.memory.bandwidth
