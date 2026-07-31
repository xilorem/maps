"""Hardware-independent logical computation and model import."""

from __future__ import annotations

from .constants import Constant, ConstantStore, validate_constants
from .dtype import TensorDType, dtype_elem_bytes
from .graph import Edge, Graph, Node, OpKind
from .model import ImportedModel, validate_imported_model
from .tensor import TENSOR_MAX_DIMS, Tensor

_ONNX_EXPORTS = {
    "InputShapes",
    "import_onnx_graph",
    "import_onnx_model",
    "load_onnx_model",
    "prepare_onnx_model",
}
_REWRITE_EXPORTS = {"GraphRewrite", "run_graph_rewrites"}


def __getattr__(name: str):
    """Load optional adapters without coupling the logical model to them."""

    if name == "decompose_graph":
        from .decompose import decompose_graph

        return decompose_graph
    if name in _ONNX_EXPORTS:
        from . import onnx

        return getattr(onnx, name)
    if name in _REWRITE_EXPORTS:
        from . import rewrites

        return getattr(rewrites, name)
    raise AttributeError(name)


__all__ = [
    "Constant",
    "ConstantStore",
    "Edge",
    "Graph",
    "GraphRewrite",
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
    "validate_constants",
    "validate_imported_model",
]
