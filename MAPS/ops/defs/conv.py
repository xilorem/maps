"""ONNX Conv semantics and canonical primitive lowering."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import WorkKind
from MAPS.core.graph import Node, OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.common.payload import CompositeOpPayload
from MAPS.ops.registry import register_op
from MAPS.ops.spec import OpSpec
from MAPS.ops.defs.depthwise_conv import DepthwiseConvPayload
from MAPS.ops.defs.direct_conv import Conv2DPayload

@dataclass(frozen=True)
class ConvPayload(CompositeOpPayload):
    """Frontend NCHW Conv payload lowered to a canonical primitive.

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


def lower_conv_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, ConvPayload]:
    """Lower one ONNX Conv node into scheduler-side Conv semantics."""

    if len(inputs) not in (2, 3):
        raise ValueError(f"Conv node '{node_name}' must have 2 or 3 inputs")
    if len(outputs) != 1:
        raise ValueError(f"Conv node '{node_name}' must have exactly 1 output")
    if "auto_pad" in attributes and attributes["auto_pad"] != "NOTSET":
        raise NotImplementedError("Conv auto_pad is not implemented")

    return (
        OpKind.CONV,
        ConvPayload(
            x=inputs[0],
            w=inputs[1],
            b=inputs[2] if len(inputs) == 3 else None,
            output=outputs[0],
            strides=tuple(attributes.get("strides", (1, 1))),
            pads=tuple(attributes.get("pads", (0, 0, 0, 0))),
            dilations=tuple(attributes.get("dilations", (1, 1))),
            group=int(attributes.get("group", 1)),
        ),
    )


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


register_op(
    OpSpec(
        name="conv",
        onnx_names=("Conv",),
        lower_onnx=lower_conv_node,
        work_kinds=(
            WorkKind.CONV2D,
            WorkKind.DEPTHWISE_CONV,
        ),
    )
)
