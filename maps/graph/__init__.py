"""Hardware-independent logical computation and model import."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .model import (
    Constant,
    ConstantStore,
    Edge,
    Graph,
    ImportedModel,
    Node,
    OpKind,
    TENSOR_MAX_DIMS,
    Tensor,
    TensorDType,
    dtype_elem_bytes,
    validate_constants,
    validate_imported_model,
)

if TYPE_CHECKING:
    from onnx import ModelProto

    from .onnx.importer import InputShapes
    from .rewrite import GraphRewrite, GraphRewriteEffect


def import_onnx_graph(
    path: str | Path,
    *,
    input_shapes: InputShapes | None = None,
) -> Graph:
    from .onnx import import_onnx_graph as import_graph

    return import_graph(path, input_shapes=input_shapes)


def import_onnx_model(
    path: str | Path,
    *,
    input_shapes: InputShapes | None = None,
) -> ImportedModel:
    from .onnx import import_onnx_model as import_model

    return import_model(path, input_shapes=input_shapes)


def load_onnx_model(path: str | Path) -> ModelProto:
    from .onnx import load_onnx_model as load_model

    return load_model(path)


def prepare_onnx_model(
    model: ModelProto,
    input_shapes: InputShapes | None = None,
) -> ModelProto:
    from .onnx import prepare_onnx_model as prepare_model

    return prepare_model(model, input_shapes)


def decompose_graph(graph: Graph) -> Graph:
    from .rewrite import decompose_graph as decompose

    return decompose(graph)


def run_graph_rewrites(model: ImportedModel) -> ImportedModel:
    from .rewrite import run_graph_rewrites as run_rewrites

    return run_rewrites(model)


def run_graph_rewrites_with_effects(
    model: ImportedModel,
) -> tuple[ImportedModel, tuple[GraphRewriteEffect, ...]]:
    from .rewrite import run_graph_rewrites_with_effects as run_rewrites

    return run_rewrites(model)


def __getattr__(name: str):
    """Load type-like rewrite contracts without coupling the Graph model."""

    if name in {"GraphRewrite", "GraphRewriteEffect"}:
        from . import rewrite

        return getattr(rewrite, name)
    raise AttributeError(name)


__all__ = [
    "Constant",
    "ConstantStore",
    "Edge",
    "Graph",
    "GraphRewrite",
    "GraphRewriteEffect",
    "ImportedModel",
    "InputShapes",
    "Node",
    "OpKind",
    "TENSOR_MAX_DIMS",
    "Tensor",
    "TensorDType",
    "decompose_graph",
    "dtype_elem_bytes",
    "import_onnx_graph",
    "import_onnx_model",
    "load_onnx_model",
    "prepare_onnx_model",
    "run_graph_rewrites",
    "run_graph_rewrites_with_effects",
    "validate_constants",
    "validate_imported_model",
]
