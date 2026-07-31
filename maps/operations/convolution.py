"""Convolution semantics, decomposition, Tile Work, and costing."""

from __future__ import annotations

from dataclasses import dataclass

from maps.hardware import Device, Tile, WorkKind
from maps.planning.layouts import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from maps.planning.submesh import Submesh
from maps.graph import Node, OpKind, Tensor
from .contracts import CompositeOpPayload, OpCostModel, OpPayload, TileWork, require_tile_device
from .depthwise_convolution import DepthwiseConvPayload


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


@dataclass(frozen=True)
class Conv2DCostModel(OpCostModel):
    """Compute-only Conv2D model backed by explicitly advertised devices.

    This provisional model accounts for MAC throughput only. Patch-address
    generation, packing, and boundary overhead are intentionally not modeled.
    """

    def cost(
        self,
        tile_work: TileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        return require_tile_device(tile, assigned_device).cycles(tile_work)


@dataclass(frozen=True)
class ConvPayload(CompositeOpPayload):
    """NCHW convolution lowered to a canonical primitive.

    The planner-side convention is:
    - ``x`` has shape ``[N, C, H, W]``
    - ``w`` has shape ``[OC, C / group, KH, KW]``
    - optional ``b`` has shape ``[OC]``
    - ``output`` has shape ``[N, OC, OH, OW]``
    """

    x: Tensor
    w: Tensor
    b: Tensor | None
    output: Tensor
    strides: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilations: tuple[int, int] = (1, 1)
    group: int = 1

    def __post_init__(self) -> None:
        if len(self.strides) != 2:
            raise ValueError("Conv strides must have length 2")
        if len(self.pads) != 4:
            raise ValueError("Conv pads must have length 4")
        if len(self.dilations) != 2:
            raise ValueError("Conv dilations must have length 2")
        if any(value <= 0 for value in self.strides):
            raise ValueError("Conv strides must be > 0")
        if any(value < 0 for value in self.pads):
            raise ValueError("Conv pads must be >= 0")
        if any(value <= 0 for value in self.dilations):
            raise ValueError("Conv dilations must be > 0")
        if self.group <= 0:
            raise ValueError("Conv group must be > 0")
        self.validate_shapes()

    def validate_shapes(self) -> None:
        if self.x.rank != 4:
            raise ValueError("Conv X tensor must be NCHW rank 4")
        if self.w.rank != 4:
            raise ValueError("Conv W tensor must be OIHW rank 4")
        if self.output.rank != 4:
            raise ValueError("Conv output tensor must be NCHW rank 4")
        if self.x.elem_bytes != self.w.elem_bytes or self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("Conv tensors must agree on element size")
        if self.b is not None:
            if self.b.rank != 1:
                raise ValueError("Conv bias tensor must be rank 1")
            if self.b.elem_bytes != self.output.elem_bytes:
                raise ValueError("Conv bias element size must match output tensor")

        batch, in_channels, input_h, input_w = self.x.dims
        out_channels, weight_channels, kernel_h, kernel_w = self.w.dims
        out_batch, out_channels_actual, output_h, output_w = self.output.dims
        if batch != out_batch:
            raise ValueError("Conv input and output batch dimensions must match")
        if in_channels % self.group:
            raise ValueError("Conv input channels must be divisible by group")
        if out_channels % self.group:
            raise ValueError("Conv output channels must be divisible by group")
        if weight_channels != in_channels // self.group:
            raise ValueError("Conv weight channels must equal input channels per group")
        if out_channels != out_channels_actual:
            raise ValueError("Conv weight output channels must match output channels")
        if self.b is not None and self.b.dims != (out_channels,):
            raise ValueError("Conv bias shape must match output channels")

        stride_h, stride_w = self.strides
        pad_top, pad_left, pad_bottom, pad_right = self.pads
        dilation_h, dilation_w = self.dilations
        expected_h = (
            input_h
            + pad_top
            + pad_bottom
            - dilation_h * (kernel_h - 1)
            - 1
        ) // stride_h + 1
        expected_w = (
            input_w
            + pad_left
            + pad_right
            - dilation_w * (kernel_w - 1)
            - 1
        ) // stride_w + 1
        if (output_h, output_w) != (expected_h, expected_w):
            raise ValueError("Conv output spatial dimensions do not match parameters")

    def decompose(self, node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
        return decompose_conv_node(node)


def decompose_conv_node(node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
    """Lower one NCHW Conv to direct dense or specialized depthwise work."""

    if not isinstance(node.payload, ConvPayload):
        raise TypeError("decompose_conv_node expects a Node with ConvPayload payload")

    op = node.payload
    if op.group != 1 and op.group == op.x.dims[1]:
        attributes = dict(node.attributes)
        attributes["stage_group_id"] = f"{node.name}::depthwise_conv"
        attributes["conv_step"] = "depthwise_conv"
        return (
            (),
            (
                Node(
                    name=f"{node.name}__depthwise",
                    kind=OpKind.CONV,
                    inputs=tuple(
                        tensor
                        for tensor in (op.x, op.w, op.b)
                        if tensor is not None
                    ),
                    outputs=(op.output,),
                    payload=DepthwiseConvPayload(
                        x=op.x,
                        w=op.w,
                        b=op.b,
                        output=op.output,
                        strides=op.strides,
                        pads=op.pads,
                        dilations=op.dilations,
                    ),
                    attributes=attributes,
                ),
            ),
        )
    if op.group != 1:
        raise NotImplementedError(
            f"Conv group={op.group} is not depthwise; general grouped Conv "
            "is not implemented"
        )

    attributes = dict(node.attributes)
    attributes.pop("stage_group_id", None)
    return (), (
        Node(
            name=node.name,
            kind=OpKind.CONV,
            inputs=tuple(tensor for tensor in (op.x, op.w, op.b) if tensor is not None),
            outputs=(op.output,),
            payload=Conv2DPayload(
                x=op.x,
                w=op.w,
                b=op.b,
                output=op.output,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
            ),
            attributes=attributes,
        ),
    )
