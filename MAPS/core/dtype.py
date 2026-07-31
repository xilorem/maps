"""Migration bridge to the lowercase Graph-owned dtype model."""

from maps.graph.dtype import TensorDType, dtype_elem_bytes

__all__ = ["TensorDType", "dtype_elem_bytes"]
