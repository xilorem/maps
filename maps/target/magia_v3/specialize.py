"""Whole-Graph Precision Specialization owned by the MAGIA-v3 Target."""

from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np

from maps.graph import (
    Constant,
    ConstantStore,
    Graph,
    ImportedModel,
    Node,
    Tensor,
    TensorDType,
)
from maps.graph.model import build_graph_edges_from_nodes
from maps.hardware import Mesh, WorkSignature
from maps.target.contracts import (
    RewriteEvent,
    RewriteReport,
    SpecializationOptions,
    SpecializationResult,
)
from maps.target.magia.specialize import _events, lower_convolutions


def _specialized_tensor(tensor: Tensor) -> Tensor:
    if tensor.dtype is not TensorDType.FLOAT32:
        return tensor
    return replace(tensor, dtype=TensorDType.FLOAT16, elem_bytes=2)


def _replace_tensors(value: Any, tensors: dict[str, Tensor]) -> Any:
    if isinstance(value, Tensor):
        return tensors[value.name]
    if isinstance(value, tuple):
        return tuple(_replace_tensors(item, tensors) for item in value)
    if isinstance(value, list):
        return [_replace_tensors(item, tensors) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_tensors(item, tensors) for key, item in value.items()
        }
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _replace_tensors(getattr(value, field.name), tensors)
                for field in fields(value)
                if field.init
            },
        )
    return value


def _specialize_floats(model: ImportedModel) -> ImportedModel:
    tensors = {
        tensor.name: _specialized_tensor(tensor) for tensor in model.graph.tensors
    }
    nodes = tuple(
        replace(
            node,
            inputs=tuple(tensors[tensor.name] for tensor in node.inputs),
            outputs=tuple(tensors[tensor.name] for tensor in node.outputs),
            payload=_replace_tensors(node.payload, tensors),
        )
        for node in model.graph.nodes
    )
    graph = Graph(
        name=model.graph.name,
        tensors=tuple(tensors[tensor.name] for tensor in model.graph.tensors),
        nodes=nodes,
        edges=build_graph_edges_from_nodes(
            nodes,
            tensors,
            tuple(tensor.name for tensor in model.graph.outputs),
        ),
        inputs=tuple(tensors[tensor.name] for tensor in model.graph.inputs),
        outputs=tuple(tensors[tensor.name] for tensor in model.graph.outputs),
        initializers=tuple(
            tensors[tensor.name] for tensor in model.graph.initializers
        ),
    )
    constants = ConstantStore(
        tuple(_specialized_constant(constant) for constant in model.constants.constants)
    )
    return ImportedModel(graph=graph, constants=constants)


def _specialized_constant(constant: Constant) -> Constant:
    if constant.dtype is not TensorDType.FLOAT32:
        return constant
    values = np.frombuffer(constant.data, dtype=np.dtype("<f4"))
    return replace(
        constant,
        dtype=TensorDType.FLOAT16,
        data=values.astype(np.dtype("<f2")).tobytes(),
    )


def specialize(
    model: ImportedModel,
    mesh: Mesh,
    options: SpecializationOptions | None = None,
) -> SpecializationResult:
    """Specialize every runtime floating value to MAGIA-v3-native FP16."""

    del mesh, options
    model.validate()
    specialized = _specialize_floats(model)
    converted_initializers = tuple(
        constant.name
        for constant in model.constants.constants
        if constant.dtype is TensorDType.FLOAT32
    )
    event = RewriteEvent(
        rewrite_name="whole_graph_precision_specialization",
        source_node=model.graph.name,
        original_signature=None,
        resulting_signatures=tuple(
            WorkSignature.from_node(node) for node in specialized.graph.nodes
        ),
        converted_initializers=converted_initializers,
    )
    convolution = lower_convolutions(specialized, spatz_conv_gemm=True)
    convolution.model.validate()
    return SpecializationResult(
        convolution.model,
        RewriteReport((event,) + _events("conv_to_gemm", convolution.effects)),
    )


__all__ = ["specialize"]
