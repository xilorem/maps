"""GEMM semantics, layout, Tile Work, and Device costing."""

from __future__ import annotations

from dataclasses import dataclass

from maps.planning.mapping import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from maps.planning.mapping import Submesh
from maps.graph import Tensor
from maps.hardware import Device, Tile, WorkKind

from .broadcasting import broadcast_input_slice, validate_broadcastable_to
from .contracts import OpCostModel, OpPayload, TileWork, require_tile_device


@dataclass(frozen=True)
class GemmTileWork(TileWork):
    """Concrete GEMM slices associated with one tile."""

    output_slice: TensorSlice
    x_slice: TensorSlice
    w_slice: TensorSlice
    y_slice: TensorSlice | None
    x: Tensor
    w: Tensor
    output: Tensor
    y: Tensor | None = None

    @property
    def work_kind(self) -> WorkKind:
        return WorkKind.GEMM

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        refs = [
            TensorSliceRef(tensor=self.x, tensor_slice=self.x_slice),
            TensorSliceRef(tensor=self.w, tensor_slice=self.w_slice),
        ]
        if self.y is not None and self.y_slice is not None:
            refs.append(TensorSliceRef(tensor=self.y, tensor_slice=self.y_slice))
        return tuple(refs)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)

    def operation_count(self) -> int:
        return self.output_slice.num_elements * self.x_slice.dims[-1].length

    def dimensions(self) -> tuple[int, int, int, int]:
        batch_volume = 1
        for dimension in self.output_slice.dims[:-2]:
            batch_volume *= dimension.length
        return (
            batch_volume,
            self.output_slice.dims[-2].length,
            self.output_slice.dims[-1].length,
            self.x_slice.dims[-1].length,
        )


@dataclass(frozen=True)
class GemmCostModel(OpCostModel):
    """GEMM cycles evaluated by the Device assigned to one tile."""

    def cost(
        self,
        tile_work: TileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        return require_tile_device(tile, assigned_device).cycles(tile_work)


def _full_range(dimension: int) -> TensorRange:
    return TensorRange(start=0, length=dimension)


@dataclass(frozen=True)
class GemmPayload(OpPayload):
    """Source-independent GEMM Operation."""

    x: Tensor
    w: Tensor
    y: Tensor | None
    output: Tensor
    transpose_w: bool = False

    @property
    def work_kind(self) -> WorkKind:
        return WorkKind.GEMM

    def __post_init__(self) -> None:
        for tensor_name, tensor in (("X", self.x), ("W", self.w), ("output", self.output)):
            if tensor.rank < 2:
                raise ValueError(f"{tensor_name} tensor rank must be >= 2 for GEMM")
        if self.x.elem_bytes != self.w.elem_bytes or self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("GEMM tensors must agree on element size")
        if self.x.rank != self.w.rank or self.x.rank != self.output.rank:
            raise ValueError(
                "GEMM currently requires X, W, and output to have equal rank; "
                "broadcasted batch dimensions must be normalized before planning"
            )
        if self.x.dims[:-2] != self.w.dims[:-2] or self.x.dims[:-2] != self.output.dims[:-2]:
            raise ValueError(
                "GEMM currently requires identical batch dimensions for X, W, and output"
            )
        w_k = self.w.dims[-1] if self.transpose_w else self.w.dims[-2]
        w_n = self.w.dims[-2] if self.transpose_w else self.w.dims[-1]
        if self.x.dims[-1] != w_k:
            raise ValueError("GEMM tensors must agree on K dimension")
        if self.x.dims[-2] != self.output.dims[-2]:
            raise ValueError("GEMM X and output must agree on M dimension")
        if w_n != self.output.dims[-1]:
            raise ValueError("GEMM W and output must agree on N dimension")
        if self.y is not None:
            validate_broadcastable_to(self.y, self.output, "GEMM Y")
            if self.y.elem_bytes != self.output.elem_bytes:
                raise ValueError("Y input element size must match output tensor")

    @property
    def cost_model(self) -> OpCostModel:
        return GemmCostModel()

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        logical_width, logical_height = (None, None)
        if logical_shape is not None:
            logical_width, logical_height = logical_shape
        return (
            TensorLayout(
                submesh=submesh,
                mesh_x=LayoutAxis(
                    mode=LayoutAxisMode.SHARD,
                    tensor_axis=self.output.rank - 1,
                ),
                mesh_y=LayoutAxis(
                    mode=LayoutAxisMode.SHARD,
                    tensor_axis=self.output.rank - 2,
                ),
                logical_width=logical_width,
                logical_height=logical_height,
            ),
        )

    def required_x_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.x.rank:
            raise ValueError("output slice rank must match X tensor rank")
        return TensorSlice(
            rank=self.x.rank,
            dims=output_slice.dims[:-2]
            + (output_slice.dims[-2], _full_range(self.x.dims[-1])),
        )

    def required_w_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.w.rank:
            raise ValueError("output slice rank must match W tensor rank")
        matrix_dims = (
            (output_slice.dims[-1], _full_range(self.w.dims[-1]))
            if self.transpose_w
            else (_full_range(self.w.dims[-2]), output_slice.dims[-1])
        )
        return TensorSlice(
            rank=self.w.rank,
            dims=output_slice.dims[:-2] + matrix_dims,
        )

    def required_y_slice(self, output_slice: TensorSlice) -> TensorSlice | None:
        if self.y is None:
            return None
        return broadcast_input_slice(self.y, self.output, output_slice, "GEMM Y")

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> GemmTileWork:
        output_slice = tile_tensor_slice(
            tensor=self.output,
            layout=self.single_output_layout(output_layouts),
            tile=tile,
        )
        return GemmTileWork(
            output_slice=output_slice,
            x_slice=self.required_x_slice(output_slice),
            w_slice=self.required_w_slice(output_slice),
            y_slice=self.required_y_slice(output_slice),
            x=self.x,
            w=self.w,
            y=self.y,
            output=self.output,
        )


__all__ = ["GemmCostModel", "GemmPayload", "GemmTileWork"]
