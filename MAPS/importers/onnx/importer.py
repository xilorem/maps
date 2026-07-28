"""ONNX importer entry points."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from MAPS.core.graph import Graph
from MAPS.importers.model import ImportedModel
from MAPS.transforms import decompose_graph

from .graph_parser import parse_graph
from .preprocess import InputShapes, prepare_onnx_model
from .tensor_parser import parse_constants

if TYPE_CHECKING:
    from onnx import ModelProto


def load_onnx_model(path: str | Path) -> "ModelProto":
    """Load and validate one ONNX model from disk."""

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The optional 'onnx' package is required to load ONNX models"
        ) from exc

    model_path = Path(path)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return model


def import_onnx_graph(
    path: str | Path,
    *,
    input_shapes: InputShapes | None = None,
) -> Graph:
    """Import one ONNX model directly into the shared scheduler graph IR."""

    return import_onnx_model(path, input_shapes=input_shapes).graph


def import_onnx_model(
    path: str | Path,
    *,
    input_shapes: InputShapes | None = None,
) -> ImportedModel:
    """Import one specialized, statically shaped ONNX model."""

    onnx_model = prepare_onnx_model(
        load_onnx_model(path),
        input_shapes=input_shapes,
    )
    graph = parse_graph(
        onnx_model.graph,
        graph_name=onnx_model.graph.name or Path(path).stem,
    )
    graph = decompose_graph(graph)
    constants = parse_constants(
        onnx_model.graph,
        names={tensor.name for tensor in graph.initializers},
    )

    return ImportedModel(graph=graph, constants=constants)
