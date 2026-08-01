"""Planning-owned Tensor placement layouts and concrete slices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from maps.graph import TENSOR_MAX_DIMS, Tensor
from maps.hardware import Mesh, Tile

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


def _adjacent_tile_ids(mesh: Mesh, tile_id: int) -> set[int]:
    tile = mesh.tile_by_id(tile_id)
    neighbors: set[int] = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x = tile.x + dx
        y = tile.y + dy
        if 0 <= x < mesh.width and 0 <= y < mesh.height:
            neighbors.add(mesh.tile_id(x, y))
    return neighbors


def _is_connected_tile_set(mesh: Mesh, tile_ids: frozenset[int]) -> bool:
    if not tile_ids:
        return False
    start = next(iter(tile_ids))
    seen = {start}
    stack = [start]
    while stack:
        tile_id = stack.pop()
        for neighbor in _adjacent_tile_ids(mesh, tile_id):
            if neighbor in tile_ids and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(tile_ids)


@dataclass(frozen=True, init=False)
class Submesh:
    """One connected placed submesh inside a mesh.

    The shape does not need to be rectangular, but it must be 4-neighbor connected.
    """

    mesh: Mesh
    submesh_id: int
    tile_ids: frozenset[int] | set[int]

    def __init__(
        self,
        mesh: Mesh,
        submesh_id: int,
        tile_ids: frozenset[int] | set[int] | None = None,
        x0: int | None = None,
        y0: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if tile_ids is None:
            if x0 is None or y0 is None or width is None or height is None:
                raise TypeError("Submesh requires tile_ids or x0, y0, width, and height")
            tile_ids = frozenset(
                mesh.tile_id(x, y)
                for y in range(y0, y0 + height)
                for x in range(x0, x0 + width)
            )

        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "submesh_id", submesh_id)
        object.__setattr__(self, "tile_ids", tile_ids)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.submesh_id < 0:
            raise ValueError("submesh_id must be >= 0")

        tile_ids = frozenset(self.tile_ids)
        object.__setattr__(self, "tile_ids", tile_ids)

        if not tile_ids:
            raise ValueError("tile_ids must not be empty")

        if len(tile_ids) > 1:
            isolated = [
                tile_id
                for tile_id in tile_ids
                if not (_adjacent_tile_ids(self.mesh, tile_id) & tile_ids)
            ]
            if isolated:
                raise ValueError(
                    f"submesh contains isolated tiles: {isolated}"
                )
            if not _is_connected_tile_set(self.mesh, tile_ids):
                raise ValueError(
                    f"submesh tiles must form one connected component: {set(tile_ids)}"
                )

    @property
    def num_tiles(self) -> int:
        """Return the number of tiles covered by this submesh."""
        return len(self.tile_ids)

    @property
    def tiles(self) -> tuple[Tile, ...]:
        """Return mesh tiles covered by this submesh in row-major order."""
        return tuple(
            self.mesh.tile_by_id(tile_id)
            for tile_id in sorted(
                self.tile_ids,
                key=lambda tile_id: (
                    self.mesh.tile_by_id(tile_id).y,
                    self.mesh.tile_by_id(tile_id).x,
                ),
            )
        )

    @property
    def tile_mask(self) -> int:
        """Return a bit mask for fast overlap checks."""
        mask = 0
        for tile_id in self.tile_ids:
            mask |= 1 << tile_id
        return mask

    @property
    def x0(self) -> int:
        """Left edge of the bounding box. Compatibility with old rectangular code."""
        return min(tile.x for tile in self.tiles)

    @property
    def y0(self) -> int:
        """Top edge of the bounding box. Compatibility with old rectangular code."""
        return min(tile.y for tile in self.tiles)

    @property
    def width(self) -> int:
        """Bounding-box width. Compatibility with old rectangular code."""
        xs = [tile.x for tile in self.tiles]
        return max(xs) - min(xs) + 1

    @property
    def height(self) -> int:
        """Bounding-box height. Compatibility with old rectangular code."""
        ys = [tile.y for tile in self.tiles]
        return max(ys) - min(ys) + 1

    def intersects_tile_ids(self, tile_ids: set[int]) -> bool:
        """Return whether this submesh overlaps any global mesh tile id in the set."""
        return bool(self.tile_ids & tile_ids)

    def global_to_local(self, tile_id: int) -> tuple[int, int]:
        """Return the local bounding-box coordinate for a global tile id."""
        if tile_id not in self.tile_ids:
            raise ValueError(f"tile_id {tile_id} is not inside submesh {self.submesh_id}")
        tile = self.mesh.tile_by_id(tile_id)
        return tile.x - self.x0, tile.y - self.y0

    def local_to_global(self, local_x: int, local_y: int) -> int:
        """Return the global tile id for a local bounding-box coordinate."""
        tile_id = self.mesh.tile_id(self.x0 + local_x, self.y0 + local_y)
        if tile_id not in self.tile_ids:
            raise ValueError(f"local coordinate {(local_x, local_y)} is not inside submesh {self.submesh_id}")
        return tile_id
