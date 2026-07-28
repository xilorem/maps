"""Direct tile-local NCHW Conv2D operation."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Tile, WorkKind
from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.cost import OpCostModel
from MAPS.ops.common.payload import OpPayload
from MAPS.ops.common.tile_work import TileWork


def _full_range(length: int) -> TensorRange:
    return TensorRange(start=0, length=length)


def _clamped_interval(start: int, end: int, length: int) -> TensorRange:
    clamped_start = min(max(start, 0), length)
    clamped_end = min(max(end, 0), length)
    return TensorRange(
        start=clamped_start,
        length=max(0, clamped_end - clamped_start),
    )


@dataclass(frozen=True)
class Conv2DTileWork(TileWork):
    """Exact operands and boundary padding used by one direct Conv2D tile."""

    x: Tensor
    w: Tensor
    b: Tensor | None
    output: Tensor
    input_slice: TensorSlice
    weight_slice: TensorSlice
    bias_slice: TensorSlice | None
    output_slice: TensorSlice
    kernel_shape: tuple[int, int]
    strides: tuple[int, int]
    dilations: tuple[int, int]
    local_padding: tuple[int, int, int, int]
    work_kind: WorkKind = WorkKind.CONV2D

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        refs = [
            TensorSliceRef(self.x, self.input_slice),
            TensorSliceRef(self.w, self.weight_slice),
        ]
        if self.b is not None and self.bias_slice is not None:
            refs.append(TensorSliceRef(self.b, self.bias_slice))
        return tuple(refs)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.output, self.output_slice),)

    def operation_count(self) -> int:
        input_channels = self.w.dims[1]
        kernel_h, kernel_w = self.kernel_shape
        return (
            self.output_slice.num_elements
            * input_channels
            * kernel_h
            * kernel_w
        )


@dataclass(frozen=True)
class Conv2DPayload(OpPayload):
    """Dense NCHW convolution with an OIHW initializer and optional bias."""

    x: Tensor
    w: Tensor
    b: Tensor | None
    output: Tensor
    strides: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilations: tuple[int, int] = (1, 1)
    work_kind: WorkKind = WorkKind.CONV2D

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.CONV2D:
            raise ValueError("Conv2D must use CONV2D work")
        if self.x.rank != 4 or self.w.rank != 4 or self.output.rank != 4:
            raise ValueError("Conv2D input, weight, and output must be rank 4")
        if any(value <= 0 for value in self.strides + self.dilations):
            raise ValueError("Conv2D strides and dilations must be > 0")
        if any(value < 0 for value in self.pads):
            raise ValueError("Conv2D pads must be >= 0")
        if (
            self.x.elem_bytes != self.w.elem_bytes
            or self.x.elem_bytes != self.output.elem_bytes
        ):
            raise ValueError("Conv2D tensors must agree on element size")
        if self.w.dims[1] != self.x.dims[1]:
            raise ValueError("Conv2D weights must contain all input channels")
        if self.w.dims[0] != self.output.dims[1]:
            raise ValueError("Conv2D output channels must match weights")
        if len(self.strides) != 2 or len(self.dilations) != 2 or len(self.pads) != 4:
            raise ValueError("Conv2D convolution attributes have invalid rank")
        if self.b is not None:
            if self.b.dims != (self.output.dims[1],):
                raise ValueError("Conv2D bias shape must match output channels")
            if self.b.elem_bytes != self.output.elem_bytes:
                raise ValueError("Conv2D bias element size must match output")
        if self.x.dims[0] != self.output.dims[0]:
            raise ValueError("Conv2D input and output batch dimensions must match")
        input_h, input_w = self.x.dims[2:]
        kernel_h, kernel_w = self.w.dims[2:]
        stride_h, stride_w = self.strides
        pad_top, pad_left, pad_bottom, pad_right = self.pads
        dilation_h, dilation_w = self.dilations
        expected_h = (
            input_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        expected_w = (
            input_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        if self.output.dims[2:] != (expected_h, expected_w):
            raise ValueError("Conv2D output spatial dimensions do not match parameters")

    @property
    def cost_model(self) -> OpCostModel:
        from MAPS.ops.costs.conv2d_cost import Conv2DCostModel

        return Conv2DCostModel()

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        logical_width = logical_shape[0] if logical_shape is not None else None
        logical_height = logical_shape[1] if logical_shape is not None else None
        return (
            TensorLayout(
                submesh=submesh,
                mesh_x=LayoutAxis(LayoutAxisMode.SHARD, tensor_axis=1),
                mesh_y=LayoutAxis(LayoutAxisMode.SHARD, tensor_axis=2),
                logical_width=logical_width,
                logical_height=logical_height,
            ),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> Conv2DTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        output_h = output_slice.dims[2]
        stride_h, stride_w = self.strides
        dilation_h, dilation_w = self.dilations
        pad_top, pad_left, _, _ = self.pads
        kernel_h, kernel_w = self.w.dims[2:]

        if output_h.length == 0:
            theoretical_h_start = theoretical_h_end = 0
            theoretical_w_start = theoretical_w_end = 0
        else:
            theoretical_h_start = output_h.start * stride_h - pad_top
            theoretical_h_end = (
                (output_h.start + output_h.length - 1) * stride_h
                - pad_top
                + dilation_h * (kernel_h - 1)
                + 1
            )
            theoretical_w_start = -pad_left
            theoretical_w_end = (
                (self.output.dims[3] - 1) * stride_w
                - pad_left
                + dilation_w * (kernel_w - 1)
                + 1
            )
        input_h = _clamped_interval(
            theoretical_h_start, theoretical_h_end, self.x.dims[2]
        )
        input_w = _clamped_interval(
            theoretical_w_start, theoretical_w_end, self.x.dims[3]
        )
        local_padding = (
            max(0, -theoretical_h_start),
            max(0, -theoretical_w_start),
            max(0, theoretical_h_end - self.x.dims[2]),
            max(0, theoretical_w_end - self.x.dims[3]),
        )
        output_channels = output_slice.dims[1]
        input_slice = TensorSlice(
            rank=4,
            dims=(
                _full_range(self.x.dims[0]),
                _full_range(self.x.dims[1]),
                input_h,
                input_w,
            ),
        )
        weight_slice = TensorSlice(
            rank=4,
            dims=(
                output_channels,
                _full_range(self.w.dims[1]),
                _full_range(kernel_h),
                _full_range(kernel_w),
            ),
        )
        bias_slice = (
            TensorSlice(rank=1, dims=(output_channels,))
            if self.b is not None
            else None
        )
        return Conv2DTileWork(
            x=self.x,
            w=self.w,
            b=self.b,
            output=self.output,
            input_slice=input_slice,
            weight_slice=weight_slice,
            bias_slice=bias_slice,
            output_slice=output_slice,
            kernel_shape=(kernel_h, kernel_w),
            strides=self.strides,
            dilations=self.dilations,
            local_padding=local_padding,
        )
