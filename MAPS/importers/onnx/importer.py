"""Migration bridge to the lowercase Graph-owned ONNX adapter."""

from maps.graph.onnx.importer import (
    import_onnx_graph,
    import_onnx_model,
    load_onnx_model,
)

__all__ = ["import_onnx_graph", "import_onnx_model", "load_onnx_model"]
