"""MAGIA-owned Target Specialization policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from maps.graph import (
    Constant,
    ConstantStore,
    Graph,
    ImportedModel,
    Node,
    OpKind,
    Tensor,
    TensorDType,
    dtype_elem_bytes,
)
from maps.graph.model import build_graph_edges_from_nodes
from maps.graph.rewrite import add_generated_tensor, reserve_generated_node_name
from maps.hardware import Mesh, WorkKind, WorkSignature
from maps.operations.cast import CastPayload
from maps.operations.convolution import Conv2DPayload
from maps.operations.convolution_transforms import (
    ChannelShardedGemmPayload,
    Im2ColPayload,
    OutputReformatPayload,
)
from maps.operations.gemm import GemmPayload
from maps.target.contracts import (
    PrecisionLoweringRecipe,
    RewriteEvent,
    RewriteReport,
    SpecializationOptions,
    SpecializationResult,
)

from .devices import REDMULE_DEVICE


PRECISION_LOWERING_RECIPES = tuple(
    PrecisionLoweringRecipe(
        source_signature=WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT32,) * input_count,
            (TensorDType.FLOAT32,),
        ),
        target_signature=WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT16,) * input_count,
            (TensorDType.FLOAT16,),
        ),
        device_name=REDMULE_DEVICE.name,
    )
    for input_count in (2, 3)
)


def _events(
    rewrite_name: str,
    effects: tuple[RewriteEffect, ...],
) -> tuple[RewriteEvent, ...]:
    return tuple(
        RewriteEvent(
            rewrite_name=rewrite_name,
            source_node=effect.source_node,
            original_signature=effect.original_signature,
            resulting_signatures=effect.resulting_signatures,
            converted_initializers=effect.converted_initializers,
        )
        for effect in effects
    )


def specialize(
    model: ImportedModel,
    mesh: Mesh,
    options: SpecializationOptions | None = None,
) -> SpecializationResult:
    """Specialize one hardware-independent Imported Model for MAGIA."""

    options = options or SpecializationOptions(enable_precision_lowering=True)
    model.validate()
    convolution = lower_convolutions(model)
    events = list(_events("conv_to_gemm", convolution.effects))
    rewritten = convolution.model
    if options.enable_precision_lowering:
        precision = precision_lower_model(
            rewritten,
            mesh,
            PRECISION_LOWERING_RECIPES,
        )
        rewritten = precision.model
        events.extend(_events("precision_lowering", precision.effects))
    rewritten.validate()
    return SpecializationResult(rewritten, RewriteReport(tuple(events)))


@dataclass(frozen=True)
class RewriteEffect:
    """One source Node's provenance before a rewrite name is applied."""

    source_node: str
    original_signature: WorkSignature | None
    resulting_signatures: tuple[WorkSignature, ...]
    converted_initializers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteTransformResult:
    """One rewritten Imported Model and its unstamped provenance effects."""

    model: ImportedModel
    effects: tuple[RewriteEffect, ...] = ()


def lower_convolutions(model: ImportedModel) -> RewriteTransformResult:
    """Pack immutable OIHW filters and expose dense Conv execution explicitly."""

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
        if not _is_supported_dense_conv(node):
            lowered_nodes.append(node)
            continue

        op = cast(Conv2DPayload, node.payload)
        if op.x.dims[0] != 1:
            raise ValueError(
                f"node {node.name} Conv-to-GEMM supports only batch size 1, "
                f"got {op.x.dims[0]}"
            )
        x_dtype = cast(TensorDType, op.x.dtype)
        weight_dtype = cast(TensorDType, op.w.dtype)
        output_dtype = cast(TensorDType, op.output.dtype)
        if op.w.name not in initializers:
            raise ValueError(
                f"node {node.name} Conv-to-GEMM requires immutable initializer "
                f"weights, but {op.w.name} is a runtime value"
            )

        source_signature = WorkSignature.from_node(node)
        packed_weight = packed_weights.get(op.w.name)
        if packed_weight is None:
            replace_initializer = all(
                _is_supported_dense_conv(consumer)
                and cast(Conv2DPayload, consumer.payload).w.name == op.w.name
                for consumer in consumers[op.w.name]
            )
            packed_name = (
                op.w.name
                if replace_initializer
                else _generated_name(
                    node.name,
                    "input_1",
                    "weight_packed",
                    weight_dtype,
                )
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
                add_generated_tensor(packed_weight, tensors)
                initializers[packed_weight.name] = packed_weight
                constants = ConstantStore(constants.constants + (packed_constant,))
            packed_weights[op.w.name] = packed_weight

        n, _, _, _ = op.x.dims
        _, _, output_h, output_w = op.output.dims
        output_channels, input_channels, kernel_h, kernel_w = op.w.dims
        matrix_rows = n * output_h * output_w
        matrix_depth = input_channels * kernel_h * kernel_w
        im2col_output = Tensor(
            name=_generated_name(
                node.name,
                "input_0",
                "im2col_output",
                x_dtype,
            ),
            rank=2,
            dims=(matrix_rows, matrix_depth),
            elem_bytes=op.x.elem_bytes,
            dtype=x_dtype,
        )
        gemm_output = Tensor(
            name=_generated_name(
                node.name,
                "output_0",
                "gemm_result",
                output_dtype,
            ),
            rank=2,
            dims=(matrix_rows, output_channels),
            elem_bytes=op.output.elem_bytes,
            dtype=output_dtype,
        )
        add_generated_tensor(im2col_output, tensors)
        add_generated_tensor(gemm_output, tensors)

        im2col = Node(
            name=_generated_name(
                node.name,
                "input_0",
                "im2col",
                x_dtype,
            ),
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
            attributes={"conv_step": "im2col"},
            source_operation=node.source_operation,
        )
        gemm_inputs = (im2col_output, packed_weight) + (
            (op.b,) if op.b is not None else ()
        )
        gemm = Node(
            name=_generated_name(
                node.name,
                "output_0",
                "gemm",
                output_dtype,
            ),
            kind=OpKind.GEMM,
            inputs=gemm_inputs,
            outputs=(gemm_output,),
            payload=ChannelShardedGemmPayload(
                x=im2col_output,
                w=packed_weight,
                y=op.b,
                output=gemm_output,
                row_granularity=output_w,
            ),
            attributes={"conv_step": "gemm"},
            source_operation=node.source_operation,
        )
        output_reformat = Node(
            name=_generated_name(
                node.name,
                "output_0",
                "reformat",
                output_dtype,
            ),
            kind=OpKind.TRANSFORM,
            inputs=(gemm_output,),
            outputs=(op.output,),
            payload=OutputReformatPayload(x=gemm_output, output=op.output),
            attributes={"conv_step": "output_reformat"},
            source_operation=node.source_operation,
        )
        replacements = (im2col, gemm, output_reformat)
        for replacement in replacements:
            reserve_generated_node_name(replacement.name, node_names)
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


def _is_supported_dense_conv(node: Node) -> bool:
    if not isinstance(node.payload, Conv2DPayload):
        return False
    dtypes = {tensor.dtype for tensor in node.inputs + node.outputs}
    return len(dtypes) == 1 and dtypes.pop() in {
        TensorDType.FLOAT16,
        TensorDType.FLOAT32,
    }


def _generated_name(
    source_node: str,
    operand_position: str,
    role: str,
    target_dtype: TensorDType,
) -> str:
    return f"{source_node}__{operand_position}_{role}_{target_dtype.value}"


def _pack_weight(
    weight: Tensor,
    constant: Constant,
    packed_name: str,
) -> tuple[Tensor, Constant]:
    output_channels, input_channels, kernel_h, kernel_w = weight.dims
    packed_shape = (input_channels * kernel_h * kernel_w, output_channels)
    numpy_dtype = {
        TensorDType.FLOAT16: np.dtype("<f2"),
        TensorDType.FLOAT32: np.dtype("<f4"),
    }[cast(TensorDType, weight.dtype)]
    values = np.frombuffer(constant.data, dtype=numpy_dtype).reshape(weight.dims)
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


def precision_lower_model(
    model: ImportedModel,
    mesh: Mesh,
    recipes: tuple[PrecisionLoweringRecipe, ...] = (),
) -> RewriteTransformResult:
    """Apply every matching target recipe without cost-based selection."""

    recipe_by_signature = {
        recipe.source_signature: recipe
        for recipe in recipes
    }
    tensors = {tensor.name: tensor for tensor in model.graph.tensors}
    constants = model.constants
    original_constants = model.constants
    initializers = {tensor.name: tensor for tensor in model.graph.initializers}
    node_names = {node.name for node in model.graph.nodes}
    lowered_nodes: list[Node] = []
    effects: list[RewriteEffect] = []
    node_rewrites = {}
    initializer_target_dtypes: dict[str, list[TensorDType]] = {
        name: [] for name in initializers
    }
    converted_initializer_cache: dict[tuple[str, TensorDType], Tensor] = {}

    for node in model.graph.nodes:
        source_signature = WorkSignature.from_node(node)
        recipe = recipe_by_signature.get(source_signature)
        if recipe is not None and (
            len(recipe.target_signature.input_dtypes) != len(node.inputs)
            or len(recipe.target_signature.output_dtypes) != len(node.outputs)
        ):
            raise ValueError(
                f"node {node.name} Precision Lowering Recipe target arity does "
                "not match the source operation"
            )
        node_rewrites[id(node)] = (source_signature, recipe)
        for input_index, tensor in enumerate(node.inputs):
            if tensor.name not in initializers:
                continue
            target_dtype = (
                recipe.target_signature.input_dtypes[input_index]
                if recipe is not None
                else cast(TensorDType, tensor.dtype)
            )
            initializer_target_dtypes[tensor.name].append(target_dtype)

    for node in model.graph.nodes:
        source_signature, recipe = node_rewrites[id(node)]
        if recipe is None:
            lowered_nodes.append(node)
            continue
        if not isinstance(node.payload, GemmPayload):
            raise ValueError(
                f"node {node.name} with {source_signature} matches a Precision "
                "Lowering Recipe but has no supported operation rewrite"
            )
        _require_assignment(mesh, node.name, recipe.target_signature, recipe.device_name)

        replacement_nodes: list[Node] = []
        target_inputs: list[Tensor] = []
        converted_initializers: list[str] = []
        for input_index, (tensor, target_dtype) in enumerate(
            zip(node.inputs, recipe.target_signature.input_dtypes)
        ):
            if tensor.dtype is target_dtype:
                target_inputs.append(tensor)
                continue
            cast_signature = WorkSignature(
                WorkKind.CAST,
                (cast(TensorDType, tensor.dtype),),
                (target_dtype,),
            )
            _require_assignment(mesh, node.name, cast_signature)
            if tensor.name in initializers:
                cache_key = (tensor.name, target_dtype)
                converted = converted_initializer_cache.get(cache_key)
                if converted is None:
                    convert_in_place = all(
                        required_dtype is target_dtype
                        for required_dtype in initializer_target_dtypes[tensor.name]
                    )
                    converted_name = (
                        tensor.name
                        if convert_in_place
                        else f"{node.name}__input_{input_index}_{target_dtype.value}"
                    )
                    converted = replace(
                        tensor,
                        name=converted_name,
                        elem_bytes=dtype_elem_bytes(target_dtype),
                        dtype=target_dtype,
                    )
                    converted_constant = replace(
                        _convert_constant(
                            original_constants.get(tensor.name),
                            target_dtype,
                        ),
                        name=converted_name,
                    )
                    if convert_in_place:
                        tensors[tensor.name] = converted
                        initializers[tensor.name] = converted
                        constants = constants.replace(converted_constant)
                    else:
                        add_generated_tensor(converted, tensors)
                        initializers[converted.name] = converted
                        constants = ConstantStore(
                            constants.constants + (converted_constant,)
                        )
                    converted_initializer_cache[cache_key] = converted
                    converted_initializers.append(converted.name)
                target_inputs.append(converted)
                continue

            cast_tensor = Tensor(
                name=f"{node.name}__input_{input_index}_{target_dtype.value}",
                rank=tensor.rank,
                dims=tensor.dims,
                elem_bytes=dtype_elem_bytes(target_dtype),
                dtype=target_dtype,
            )
            cast_node = Node(
                name=f"{node.name}__input_{input_index}_cast_{target_dtype.value}",
                kind=OpKind.TRANSFORM,
                inputs=(tensor,),
                outputs=(cast_tensor,),
                payload=CastPayload(x=tensor, output=cast_tensor),
                source_operation=_decomposed_source_operation(node),
            )
            reserve_generated_node_name(cast_node.name, node_names)
            add_generated_tensor(cast_tensor, tensors)
            replacement_nodes.append(cast_node)
            target_inputs.append(cast_tensor)

        target_outputs = tuple(
            Tensor(
                name=f"{node.name}__output_{output_index}_{target_dtype.value}",
                rank=output.rank,
                dims=output.dims,
                elem_bytes=dtype_elem_bytes(target_dtype),
                dtype=target_dtype,
            )
            for output_index, (output, target_dtype) in enumerate(
                zip(node.outputs, recipe.target_signature.output_dtypes)
            )
        )
        for target_output in target_outputs:
            add_generated_tensor(target_output, tensors)

        lowered_gemm_payload = replace(
            node.payload,
            x=target_inputs[0],
            w=target_inputs[1],
            y=target_inputs[2] if len(target_inputs) == 3 else None,
            output=target_outputs[0],
        )
        lowered_gemm = Node(
            name=node.name,
            kind=node.kind,
            inputs=tuple(target_inputs),
            outputs=target_outputs,
            payload=lowered_gemm_payload,
            attributes=node.attributes,
            source_operation=node.source_operation,
        )
        if WorkSignature.from_node(lowered_gemm) != recipe.target_signature:
            raise ValueError(
                f"node {node.name} Precision Lowering produced a Work Signature "
                f"inconsistent with recipe target {recipe.target_signature}"
            )
        replacement_nodes.append(lowered_gemm)

        for output_index, (target_output, output) in enumerate(
            zip(target_outputs, node.outputs)
        ):
            restore_signature = WorkSignature(
                WorkKind.CAST,
                (cast(TensorDType, target_output.dtype),),
                (cast(TensorDType, output.dtype),),
            )
            _require_assignment(mesh, node.name, restore_signature)
            output_dtype = cast(TensorDType, output.dtype)
            restore_node = Node(
                name=f"{node.name}__output_{output_index}_cast_{output_dtype.value}",
                kind=OpKind.TRANSFORM,
                inputs=(target_output,),
                outputs=(output,),
                payload=CastPayload(x=target_output, output=output),
                source_operation=_decomposed_source_operation(node),
            )
            reserve_generated_node_name(restore_node.name, node_names)
            replacement_nodes.append(restore_node)

        lowered_nodes.extend(replacement_nodes)
        effects.append(
            RewriteEffect(
                source_node=node.name,
                original_signature=source_signature,
                resulting_signatures=tuple(
                    WorkSignature.from_node(result) for result in replacement_nodes
                ),
                converted_initializers=tuple(converted_initializers),
            )
        )

    graph = Graph(
        name=model.graph.name,
        tensors=tuple(tensors.values()),
        nodes=tuple(lowered_nodes),
        edges=build_graph_edges_from_nodes(
            tuple(lowered_nodes),
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


def _decomposed_source_operation(node: Node) -> str | None:
    if node.source_operation == node.name:
        return None
    return node.source_operation


def _require_assignment(
    mesh: Mesh,
    node_name: str,
    signature: WorkSignature,
    expected_device_name: str | None = None,
) -> None:
    for tile in mesh.tiles:
        try:
            device = tile.assigned_device(signature)
        except ValueError as exc:
            raise ValueError(
                f"node {node_name} cannot apply Precision Lowering for "
                f"{signature}: {exc}"
            ) from exc
        if expected_device_name is not None and device.name != expected_device_name:
            raise ValueError(
                f"node {node_name} Precision Lowering target {signature} requires "
                f"device {expected_device_name}, but tile {tile.tile_id} assigns "
                f"{device.name}"
            )


def _convert_constant(constant: Constant, target_dtype: TensorDType) -> Constant:
    numpy_dtypes = {
        TensorDType.FLOAT16: np.dtype("<f2"),
        TensorDType.FLOAT32: np.dtype("<f4"),
    }
    source_dtype = numpy_dtypes.get(constant.dtype)
    converted_dtype = numpy_dtypes.get(target_dtype)
    if source_dtype is None or converted_dtype is None:
        raise ValueError(
            f"initializer '{constant.name}' cannot be converted from "
            f"{constant.dtype.value} to {target_dtype.value}"
        )
    values = np.frombuffer(constant.data, dtype=source_dtype)
    return Constant(
        name=constant.name,
        dtype=target_dtype,
        shape=constant.shape,
        data=values.astype(converted_dtype).tobytes(),
    )


__all__ = ["PRECISION_LOWERING_RECIPES", "specialize"]
