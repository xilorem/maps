"""MAGIA dense-convolution lowering into explicit data transforms and GEMM."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from MAPS.arch import WorkSignature
from MAPS.core.constants import Constant, ConstantStore
from MAPS.core.dtype import TensorDType
from MAPS.core.graph import Graph, Node, OpKind
from MAPS.core.tensor import Tensor
from MAPS.importers.model import ImportedModel
from MAPS.ops.defs.conv_transforms import (
    ChannelShardedGemmPayload,
    Im2ColPayload,
    OutputReformatPayload,
)
from MAPS.ops.defs.direct_conv import Conv2DPayload

from .effects import RewriteEffect, RewriteTransformResult
from .graph_utils import build_graph_edges_from_nodes


def lower_fp16_convolutions(model: ImportedModel) -> RewriteTransformResult:
    """Pack immutable OIHW filters and expose FP16 Conv execution explicitly."""

    tensors = {tensor.name: tensor for tensor in model.graph.tensors}
    initializers = {tensor.name: tensor for tensor in model.graph.initializers}
    constants = model.constants
    node_names = {node.name for node in model.graph.nodes}
    lowered_nodes: list[Node] = []
    effects: list[RewriteEffect] = []
    packed_weights: dict[str, Tensor] = {}
    consumers = {
        tensor.name: tuple(
            node
            for node in model.graph.nodes
            if tensor in node.inputs
        )
        for tensor in model.graph.initializers
    }

    for node in model.graph.nodes:
        if not _is_fp16_dense_conv(node):
            lowered_nodes.append(node)
            continue

        op = node.payload
        if op.w.name not in initializers:
            raise ValueError(
                f"node {node.name} Conv-to-GEMM requires immutable initializer "
                f"weights, but {op.w.name} is a runtime value"
            )

        source_signature = WorkSignature.from_node(node)
        packed_weight = packed_weights.get(op.w.name)
        if packed_weight is None:
            replace_initializer = all(
                _is_fp16_dense_conv(consumer)
                and consumer.payload.w.name == op.w.name
                for consumer in consumers[op.w.name]
            )
            packed_name = (
                op.w.name
                if replace_initializer
                else f"{op.w.name}__conv_to_gemm_packed"
            )
            packed_weight, packed_constant = _pack_weight(
                op.w,
                model.constants.get(op.w.name),
                packed_name,
            )
            if replace_initializer:
                tensors[packed_weight.name] = packed_weight
                initializers[packed_weight.name] = packed_weight
                constants = constants.replace(packed_constant)
            else:
                _add_tensor(packed_weight, tensors)
                initializers[packed_weight.name] = packed_weight
                constants = ConstantStore(constants.constants + (packed_constant,))
            packed_weights[op.w.name] = packed_weight

        n, _, _, _ = op.x.dims
        _, _, output_h, output_w = op.output.dims
        output_channels, input_channels, kernel_h, kernel_w = op.w.dims
        matrix_rows = n * output_h * output_w
        matrix_depth = input_channels * kernel_h * kernel_w
        im2col_output = Tensor(
            name=f"{node.name}__im2col_output",
            rank=2,
            dims=(matrix_rows, matrix_depth),
            elem_bytes=op.x.elem_bytes,
            dtype=op.x.dtype,
        )
        gemm_output = Tensor(
            name=f"{node.name}__gemm_output",
            rank=2,
            dims=(matrix_rows, output_channels),
            elem_bytes=op.output.elem_bytes,
            dtype=op.output.dtype,
        )
        _add_tensor(im2col_output, tensors)
        _add_tensor(gemm_output, tensors)

        stage_group = f"{node.name}::conv_to_gemm"
        im2col = Node(
            name=f"{node.name}__im2col",
            kind=OpKind.TRANSFORM,
            inputs=(op.x,),
            outputs=(im2col_output,),
            payload=Im2ColPayload(
                x=op.x,
                output=im2col_output,
                kernel_shape=(kernel_h, kernel_w),
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
            ),
            attributes={"stage_group_id": stage_group, "conv_step": "im2col"},
        )
        gemm_inputs = (im2col_output, packed_weight) + (
            (op.b,) if op.b is not None else ()
        )
        gemm = Node(
            name=f"{node.name}__gemm",
            kind=OpKind.GEMM,
            inputs=gemm_inputs,
            outputs=(gemm_output,),
            payload=ChannelShardedGemmPayload(
                x=im2col_output,
                w=packed_weight,
                y=op.b,
                output=gemm_output,
            ),
            attributes={"stage_group_id": stage_group, "conv_step": "gemm"},
        )
        output_reformat = Node(
            name=f"{node.name}__output_reformat",
            kind=OpKind.TRANSFORM,
            inputs=(gemm_output,),
            outputs=(op.output,),
            payload=OutputReformatPayload(x=gemm_output, output=op.output),
            attributes={
                "stage_group_id": stage_group,
                "conv_step": "output_reformat",
            },
        )
        replacements = (im2col, gemm, output_reformat)
        for replacement in replacements:
            _reserve_node_name(replacement.name, node.name, node_names)
        lowered_nodes.extend(replacements)
        effects.append(
            RewriteEffect(
                source_node=node.name,
                original_signature=source_signature,
                resulting_signatures=tuple(
                    WorkSignature.from_node(replacement)
                    for replacement in replacements
                ),
                converted_initializers=(packed_weight.name,),
            )
        )

    nodes = tuple(lowered_nodes)
    graph = Graph(
        name=model.graph.name,
        tensors=tuple(tensors.values()),
        nodes=nodes,
        edges=build_graph_edges_from_nodes(
            nodes,
            tensors,
            tuple(tensor.name for tensor in model.graph.outputs),
        ),
        inputs=model.graph.inputs,
        outputs=model.graph.outputs,
        initializers=tuple(initializers.values()),
    )
    return RewriteTransformResult(
        model=ImportedModel(graph=graph, constants=constants),
        effects=tuple(effects),
    )


def _is_fp16_dense_conv(node: Node) -> bool:
    return isinstance(node.payload, Conv2DPayload) and all(
        tensor.dtype is TensorDType.FLOAT16
        for tensor in node.inputs + node.outputs
    )


def _pack_weight(
    weight: Tensor,
    constant: Constant,
    packed_name: str,
) -> tuple[Tensor, Constant]:
    output_channels, input_channels, kernel_h, kernel_w = weight.dims
    packed_shape = (input_channels * kernel_h * kernel_w, output_channels)
    values = np.frombuffer(constant.data, dtype="<f2").reshape(weight.dims)
    packed = values.transpose(1, 2, 3, 0).reshape(packed_shape)
    return (
        replace(weight, name=packed_name, rank=2, dims=packed_shape),
        replace(
            constant,
            name=packed_name,
            shape=packed_shape,
            data=packed.tobytes(),
        ),
    )


def _add_tensor(tensor: Tensor, tensors: dict[str, Tensor]) -> None:
    if tensor.name in tensors:
        raise ValueError(f"generated tensor name collision: '{tensor.name}'")
    tensors[tensor.name] = tensor


def _reserve_node_name(name: str, source_name: str, node_names: set[str]) -> None:
    if name != source_name and name in node_names:
        raise ValueError(f"generated node name collision: '{name}'")
    node_names.add(name)
