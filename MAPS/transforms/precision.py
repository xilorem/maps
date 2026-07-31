"""Recipe-driven, operation-local Precision Lowering."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from MAPS.arch import Mesh, WorkKind, WorkSignature
from MAPS.core.constants import Constant, ConstantStore
from MAPS.core.dtype import TensorDType, dtype_elem_bytes
from MAPS.core.graph import Graph, Node, OpKind
from MAPS.core.tensor import Tensor
from MAPS.importers.model import ImportedModel
from MAPS.ops.defs.cast import CastPayload
from MAPS.ops.defs.gemm import GemmPayload

from .effects import RewriteEffect, RewriteTransformResult
from .graph_utils import (
    add_generated_tensor,
    build_graph_edges_from_nodes,
    reserve_generated_node_name,
)


def precision_lower_model(
    model: ImportedModel,
    mesh: Mesh,
) -> RewriteTransformResult:
    """Apply every matching Mesh recipe without cost-based selection."""

    recipes = {
        recipe.source_signature: recipe
        for recipe in mesh.precision_lowering_recipes
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
        recipe = recipes.get(source_signature)
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
                else tensor.dtype
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
                (tensor.dtype,),
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

        lowered_gemm = Node(
            name=node.name,
            kind=node.kind,
            inputs=tuple(target_inputs),
            outputs=target_outputs,
            payload=GemmPayload(
                x=target_inputs[0],
                w=target_inputs[1],
                y=target_inputs[2] if len(target_inputs) == 3 else None,
                output=target_outputs[0],
                transpose_w=node.payload.transpose_w,
            ),
            attributes=node.attributes,
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
                (target_output.dtype,),
                (output.dtype,),
            )
            _require_assignment(mesh, node.name, restore_signature)
            restore_node = Node(
                name=f"{node.name}__output_{output_index}_cast_{output.dtype.value}",
                kind=OpKind.TRANSFORM,
                inputs=(target_output,),
                outputs=(output,),
                payload=CastPayload(x=target_output, output=output),
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


__all__ = ["precision_lower_model"]
