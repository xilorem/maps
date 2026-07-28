"""Tile-local depthwise convolution operation."""

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


@dataclass(frozen=True)
class DepthwiseConvTileWork(TileWork):
    """Exact tensor slices used by one tile's depthwise convolution."""

    x: Tensor
    w: Tensor
    b: Tensor | None
    output: Tensor
    input_slice: TensorSlice
    weight_slice: TensorSlice
    bias_slice: TensorSlice | None
    output_slice: TensorSlice
    kernel_shape: tuple[int, int]
    work_kind: WorkKind = WorkKind.DEPTHWISE_CONV

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        refs = [
            TensorSliceRef(tensor=self.x, tensor_slice=self.input_slice),
            TensorSliceRef(tensor=self.w, tensor_slice=self.weight_slice),
        ]
        if self.b is not None and self.bias_slice is not None:
            refs.append(TensorSliceRef(tensor=self.b, tensor_slice=self.bias_slice))
        return tuple(refs)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)

    def operation_count(self) -> int:
        kernel_h, kernel_w = self.kernel_shape
        return self.output_slice.num_elements * kernel_h * kernel_w


@dataclass(frozen=True)
class DepthwiseConvPayload(OpPayload):
    """NCHW depthwise convolution with optional channel multiplier."""

    x: Tensor
    w: Tensor
    b: Tensor | None
    output: Tensor
    strides: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilations: tuple[int, int] = (1, 1)
    work_kind: WorkKind = WorkKind.DEPTHWISE_CONV

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.DEPTHWISE_CONV:
            raise ValueError("DepthwiseConv must use DEPTHWISE_CONV work")
        self.validate_shapes()

    @property
    def channel_multiplier(self) -> int:
        return self.output.dims[1] // self.x.dims[1]

    @property
    def cost_model(self) -> OpCostModel:
        from MAPS.ops.costs.elementwise_cost import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)

    def validate_shapes(self) -> None:
        if self.x.rank != 4 or self.w.rank != 4 or self.output.rank != 4:
            raise ValueError("DepthwiseConv input, weight, and output must be rank 4")
        if len(self.strides) != 2 or len(self.pads) != 4 or len(self.dilations) != 2:
            raise ValueError("DepthwiseConv convolution attributes have invalid rank")
        if any(value <= 0 for value in self.strides + self.dilations):
            raise ValueError("DepthwiseConv strides and dilations must be > 0")
        if any(value < 0 for value in self.pads):
            raise ValueError("DepthwiseConv pads must be >= 0")
        if (
            self.x.elem_bytes != self.w.elem_bytes
            or self.x.elem_bytes != self.output.elem_bytes
        ):
            raise ValueError("DepthwiseConv tensors must agree on element size")

        batch, in_channels, input_h, input_w = self.x.dims
        out_channels, weight_channels, kernel_h, kernel_w = self.w.dims
        out_batch, actual_out_channels, output_h, output_w = self.output.dims
        if weight_channels != 1:
            raise ValueError("DepthwiseConv weights must have one input channel per filter")
        if out_channels % in_channels:
            raise ValueError("DepthwiseConv output channels must be a multiple of input channels")
        if batch != out_batch or out_channels != actual_out_channels:
            raise ValueError("DepthwiseConv input, weight, and output channels do not match")
        if self.b is not None:
            if self.b.dims != (out_channels,):
                raise ValueError("DepthwiseConv bias shape must match output channels")
            if self.b.elem_bytes != self.output.elem_bytes:
                raise ValueError("DepthwiseConv bias element size must match output")

        stride_h, stride_w = self.strides
        pad_top, pad_left, pad_bottom, pad_right = self.pads
        dilation_h, dilation_w = self.dilations
        expected_h = (
            input_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        expected_w = (
            input_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        if (output_h, output_w) != (expected_h, expected_w):
            raise ValueError("DepthwiseConv output spatial dimensions do not match parameters")

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        logical_width = None
        logical_height = None
        if logical_shape is not None:
            logical_width, logical_height = logical_shape
        return (
            TensorLayout(
                submesh=submesh,
                mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
                mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
                logical_width=logical_width,
                logical_height=logical_height,
            ),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> DepthwiseConvTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        output_channels = output_slice.dims[1]
        multiplier = self.channel_multiplier
        input_channel_start = output_channels.start // multiplier
        input_channel_end = (
            output_channels.start + output_channels.length + multiplier - 1
        ) // multiplier
        input_channels = TensorRange(
            start=input_channel_start,
            length=input_channel_end - input_channel_start,
        )
        input_slice = TensorSlice(
            rank=4,
            dims=(
                _full_range(self.x.dims[0]),
                input_channels,
                _full_range(self.x.dims[2]),
                _full_range(self.x.dims[3]),
            ),
        )
        weight_slice = TensorSlice(
            rank=4,
            dims=(
                output_channels,
                _full_range(1),
                _full_range(self.w.dims[2]),
                _full_range(self.w.dims[3]),
            ),
        )
        bias_slice = None
        if self.b is not None:
            bias_slice = TensorSlice(rank=1, dims=(output_channels,))
        return DepthwiseConvTileWork(
            x=self.x,
            w=self.w,
            b=self.b,
            output=self.output,
            input_slice=input_slice,
            weight_slice=weight_slice,
            bias_slice=bias_slice,
            output_slice=output_slice,
            kernel_shape=self.w.dims[2:],
        )
