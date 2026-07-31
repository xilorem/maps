"""Tensor layout IR matching the runtime-side layout structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from MAPS.arch import Tile

from .submesh import Submesh
from .tensor import TENSOR_MAX_DIMS, Tensor

TENSOR_AXIS_NONE: int | None = None


class LayoutAxisMode(IntEnum):
    NONE = 0
    SHARD = 1
    PARTIAL = 2
    REPLICATE = 3


@dataclass(frozen=True)
class LayoutAxis:
    """One mesh-axis policy applied to one tensor axis."""

    mode: LayoutAxisMode
    tensor_axis: int | None = TENSOR_AXIS_NONE

    def validate_for(self, tensor: Tensor) -> None:
        if self.mode in (LayoutAxisMode.NONE, LayoutAxisMode.REPLICATE):
            return
        if self.tensor_axis is None:
            raise ValueError("tensor_axis must be set for shard/partial modes")
        if self.tensor_axis < 0 or self.tensor_axis >= tensor.rank:
            raise ValueError("tensor_axis out of range for tensor rank")


@dataclass(frozen=True)
class TensorLayout:
    """Distribution policy for one tensor on one submesh."""

    submesh: Submesh
    mesh_x: LayoutAxis
    mesh_y: LayoutAxis
    logical_width: int | None = None
    logical_height: int | None = None

    @property
    def effective_logical_width(self) -> int:
        return self.logical_width if self.logical_width is not None else self.submesh.width

    @property
    def effective_logical_height(self) -> int:
        return self.logical_height if self.logical_height is not None else self.submesh.height

    def validate_for(self, tensor: Tensor) -> None:
        self.mesh_x.validate_for(tensor)
        self.mesh_y.validate_for(tensor)
        logical_width = self.effective_logical_width
        logical_height = self.effective_logical_height
        if logical_width <= 0:
            raise ValueError("logical_width must be > 0")
        if logical_height <= 0:
            raise ValueError("logical_height must be > 0")
        if logical_width * logical_height != self.submesh.num_tiles:
            raise ValueError("logical shape area must match submesh tile count")


@dataclass(frozen=True)
class TensorRange:
    """One 1D interval on a tensor axis."""

    start: int
    length: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.length < 0:
            raise ValueError("length must be >= 0")


@dataclass(frozen=True)
class TensorSlice:
    """One concrete multi-dimensional slice of a tensor."""

    rank: int
    dims: tuple[TensorRange, ...]

    def __post_init__(self) -> None:
        if self.rank < 0 or self.rank > TENSOR_MAX_DIMS:
            raise ValueError(f"rank must be in [0, {TENSOR_MAX_DIMS}]")
        if len(self.dims) != self.rank:
            raise ValueError("dims length must match rank")

    @property
    def num_elements(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim.length
        return total


def tensor_slice_num_bytes(tensor: Tensor, tensor_slice: TensorSlice) -> int:
    """Return placement storage for one slice of a logical Tensor."""

    return tensor_slice.num_elements * tensor.elem_bytes


@dataclass(frozen=True)
class TensorSubSlice:
    """One concrete multi-dimensional subslice relative to a parent slice."""

    parent: TensorSlice
    dims: tuple[TensorRange, ...]

    def __post_init__(self) -> None:
        if len(self.dims) != self.parent.rank:
            raise ValueError("dims length must match parent slice rank")
        for parent_dim, dim in zip(self.parent.dims, self.dims):
            if dim.start + dim.length > parent_dim.length:
                raise ValueError("subslice range must fit inside parent slice")

    @property
    def rank(self) -> int:
        return self.parent.rank

    @property
    def num_elements(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim.length
        return total


@dataclass(frozen=True)
class TensorSliceRef:
    """A concrete slice tied to its logical tensor."""

    tensor: Tensor
    tensor_slice: TensorSlice

    @property
    def num_bytes(self) -> int:
        return tensor_slice_num_bytes(self.tensor, self.tensor_slice)


def partition_range(total_length: int,
                    num_parts: int,
                    part_idx: int) -> TensorRange:
    """Return the balanced partition owned by one partition index.

    The axis is split as evenly as possible across ``num_parts``. When the
    division is not exact, the first ``total_length % num_parts`` partitions get
    one extra element.
    """
    if total_length < 0:
        raise ValueError("total_length must be non-negative")
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if part_idx < 0 or part_idx >= num_parts:
        raise ValueError("part_idx must be in [0, num_parts)")

    base = total_length // num_parts
    remainder = total_length % num_parts

    start = part_idx * base + min(part_idx, remainder)
    length = base + (1 if part_idx < remainder else 0)
    return TensorRange(start=start, length=length)


def _apply_layout_axis(current_range: TensorRange,
                       axis: LayoutAxis,
                       num_parts: int,
                       part_idx: int) -> TensorRange:
    """Apply one mesh-axis policy to one tensor-axis range."""

    if axis.mode in (LayoutAxisMode.NONE, LayoutAxisMode.REPLICATE):
        return current_range
    if axis.mode is LayoutAxisMode.PARTIAL:
        raise NotImplementedError("PARTIAL ownership is not implemented yet")
    if axis.mode is not LayoutAxisMode.SHARD:
        raise ValueError(f"unsupported layout axis mode: {axis.mode}")

    local_range = partition_range(current_range.length, num_parts, part_idx)
    return TensorRange(
        start=current_range.start + local_range.start,
        length=local_range.length,
    )


def tile_tensor_slice(tensor: Tensor, layout: TensorLayout, tile: Tile) -> TensorSlice:
    """Return the concrete tensor slice owned by one tile."""

    layout.validate_for(tensor)
    if tile.tile_id not in layout.submesh.tile_ids:
        raise ValueError(
            f"tile {tile.tile_id} is not inside submesh {layout.submesh.submesh_id}"
        )

    logical_width = layout.effective_logical_width
    logical_height = layout.effective_logical_height
    tile_ids = tuple(candidate.tile_id for candidate in layout.submesh.tiles)
    tile_ordinal = tile_ids.index(tile.tile_id)
    logical_x = tile_ordinal % logical_width
    logical_y = tile_ordinal // logical_width
    dims = [TensorRange(start=0, length=dim) for dim in tensor.dims]

    if layout.mesh_x.tensor_axis is not None:
        axis = layout.mesh_x.tensor_axis
        dims[axis] = _apply_layout_axis(
            dims[axis], layout.mesh_x, logical_width, logical_x
        )

    if layout.mesh_y.tensor_axis is not None:
        axis = layout.mesh_y.tensor_axis
        dims[axis] = _apply_layout_axis(
            dims[axis], layout.mesh_y, logical_height, logical_y
        )

    return TensorSlice(rank=tensor.rank, dims=tuple(dims))
