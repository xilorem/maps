"""Primitive Split and Static Slice semantics, Tile Work, and costing."""

from __future__ import annotations

from dataclasses import dataclass

from maps.hardware import Tile, WorkKind
from maps.graph import Tensor
from maps.planning.mapping import (
    TensorLayout,
    TensorRange,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from maps.planning.mapping import Submesh

from .contracts import OpCostModel, OpPayload, TileWork, sharded_layout


@dataclass(frozen=True)
class StaticSliceTileWork(TileWork):
    """One tile's materialized copy from an offset rectangular input region."""

    x: Tensor
    output: Tensor
    input_slice: TensorSlice
    output_slice: TensorSlice
    work_kind: WorkKind = WorkKind.SLICE

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.x, self.input_slice),)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.output, self.output_slice),)

    def operation_count(self) -> int:
        return self.output_slice.num_elements


@dataclass(frozen=True)
class StaticSlicePayload(OpPayload):
    """Materialize a unit-stride rectangular slice at fixed input offsets."""

    x: Tensor
    output: Tensor
    offsets: tuple[int, ...]
    work_kind: WorkKind = WorkKind.SLICE

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.SLICE:
            raise ValueError("StaticSlice must use SLICE work")
        if self.x.rank != self.output.rank:
            raise ValueError("StaticSlice input and output ranks must match")
        if len(self.offsets) != self.x.rank:
            raise ValueError("StaticSlice offsets must match input rank")
        if any(offset < 0 for offset in self.offsets):
            raise ValueError("StaticSlice offsets must be nonnegative")
        for offset, input_dim, output_dim in zip(
            self.offsets,
            self.x.dims,
            self.output.dims,
        ):
            if offset + output_dim > input_dim:
                raise ValueError("StaticSlice output region must fit inside input")
        if self.x.elem_bytes != self.output.elem_bytes or self.x.dtype != self.output.dtype:
            raise ValueError(
                "StaticSlice input and output element representations must match"
            )

    @property
    def cost_model(self) -> OpCostModel:
        from .elementwise import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (sharded_layout(self.output, submesh, logical_shape),)

    def required_input_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.output.rank:
            raise ValueError("StaticSlice output slice rank must match output tensor rank")
        return TensorSlice(
            rank=self.x.rank,
            dims=tuple(
                TensorRange(
                    start=offset + output_range.start,
                    length=output_range.length,
                )
                for offset, output_range in zip(self.offsets, output_slice.dims)
            ),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> StaticSliceTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        return StaticSliceTileWork(
            x=self.x,
            output=self.output,
            input_slice=self.required_input_slice(output_slice),
            output_slice=output_slice,
        )


@dataclass(frozen=True)
class SplitTileWork(TileWork):
    """One tile's ordered Split output slices and offset input regions."""

    x: Tensor
    outputs: tuple[Tensor, ...]
    input_regions: tuple[TensorSlice, ...]
    output_regions: tuple[TensorSlice, ...]
    work_kind: WorkKind = WorkKind.SPLIT

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return tuple(
            TensorSliceRef(self.x, input_region)
            for input_region in self.input_regions
        )

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return tuple(
            TensorSliceRef(output, output_region)
            for output, output_region in zip(self.outputs, self.output_regions)
        )

    def operation_count(self) -> int:
        return sum(output.num_elements for output in self.output_regions)


@dataclass(frozen=True)
class SplitPayload(OpPayload):
    """Static multi-output split with normalized axis and sizes."""

    x: Tensor
    outputs: tuple[Tensor, ...]
    axis: int
    sizes: tuple[int, ...]
    work_kind: WorkKind = WorkKind.SPLIT

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.SPLIT:
            raise ValueError("Split must use SPLIT work")
        if self.axis < 0 or self.axis >= self.x.rank:
            raise ValueError("Split axis must be within input tensor rank")
        if not self.outputs:
            raise ValueError("Split must have at least one output")
        if len(self.sizes) != len(self.outputs):
            raise ValueError("Split sizes must match output count")
        if any(size <= 0 for size in self.sizes):
            raise ValueError("Split sizes must be positive")
        if sum(self.sizes) != self.x.dims[self.axis]:
            raise ValueError("Split sizes must sum to the split input dimension")

        for output, size in zip(self.outputs, self.sizes):
            if output.rank != self.x.rank:
                raise ValueError("Split input and output ranks must match")
            expected_dims = list(self.x.dims)
            expected_dims[self.axis] = size
            if output.dims != tuple(expected_dims):
                raise ValueError("Split output shape does not match its size")
            if output.elem_bytes != self.x.elem_bytes or output.dtype != self.x.dtype:
                raise ValueError(
                    "Split input and output element representations must match"
                )

    @property
    def cost_model(self) -> OpCostModel:
        from .elementwise import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return tuple(
            sharded_layout(output, submesh, logical_shape)
            for output in self.outputs
        )

    def required_input_slice(
        self,
        output_index: int,
        output_slice: TensorSlice,
    ) -> TensorSlice:
        split_offset = sum(self.sizes[:output_index])
        return TensorSlice(
            rank=self.x.rank,
            dims=tuple(
                TensorRange(
                    start=(
                        output_range.start + split_offset
                        if axis == self.axis
                        else output_range.start
                    ),
                    length=output_range.length,
                )
                for axis, output_range in enumerate(output_slice.dims)
            ),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> SplitTileWork:
        if len(output_layouts) != len(self.outputs):
            raise ValueError(
                f"Split expects {len(self.outputs)} output layouts, "
                f"got {len(output_layouts)}"
            )
        output_regions = tuple(
            tile_tensor_slice(output, layout, tile)
            for output, layout in zip(self.outputs, output_layouts)
        )
        return SplitTileWork(
            x=self.x,
            outputs=self.outputs,
            input_regions=tuple(
                self.required_input_slice(output_index, output_region)
                for output_index, output_region in enumerate(output_regions)
            ),
            output_regions=output_regions,
        )
