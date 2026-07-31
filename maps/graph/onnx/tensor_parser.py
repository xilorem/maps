"""ONNX tensor parsing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from maps.graph.constants import Constant, ConstantStore
from maps.graph.dtype import TensorDType
from maps.graph.tensor import TENSOR_MAX_DIMS, Tensor

if TYPE_CHECKING:
    from onnx import GraphProto, TensorProto, ValueInfoProto


_ONNX_DTYPE_ELEM_BYTES: dict[int, int] = {
    1: 4,   # FLOAT
    2: 1,   # UINT8
    3: 1,   # INT8
    4: 2,   # UINT16
    5: 2,   # INT16
    6: 4,   # INT32
    7: 8,   # INT64
    9: 1,   # BOOL
    10: 2,  # FLOAT16
    11: 8,  # DOUBLE
    12: 4,  # UINT32
    13: 8,  # UINT64
    14: 8,  # COMPLEX64
    15: 16, # COMPLEX128
    16: 2,  # BFLOAT16
}

_ONNX_DTYPES: dict[int, TensorDType] = {
    1: TensorDType.FLOAT32,
    2: TensorDType.UINT8,
    6: TensorDType.INT32,
    7: TensorDType.INT64,
    9: TensorDType.BOOL,
    10: TensorDType.FLOAT16,
}


def onnx_dtype_elem_bytes(dtype: int) -> int | None:
    """Return the element size in bytes for one ONNX tensor dtype."""

    return _ONNX_DTYPE_ELEM_BYTES.get(dtype)


def onnx_tensor_dtype(dtype: int) -> TensorDType | None:
    return _ONNX_DTYPES.get(dtype)


def parse_value_shape(value: "ValueInfoProto") -> tuple[int, ...]:
    """Extract a concrete shape from ONNX value info when available.

    If any dimension is symbolic or unknown, return an empty shape for now.
    """

    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return ()

    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(dim.dim_value)
            continue
        return ()
    return tuple(dims)


def parse_value_tensor(
    value: "ValueInfoProto",
) -> tuple[str, tuple[int, ...], int | None, TensorDType | None]:
    """Extract tensor metadata from one ONNX value-info entry."""

    tensor_type = value.type.tensor_type
    elem_type = tensor_type.elem_type if tensor_type.HasField("elem_type") else 0
    return (
        value.name,
        parse_value_shape(value),
        onnx_dtype_elem_bytes(elem_type),
        onnx_tensor_dtype(elem_type),
    )


def parse_initializer_tensor(
    initializer: "TensorProto",
) -> tuple[str, tuple[int, ...], int | None, TensorDType | None]:
    """Extract tensor metadata from one ONNX initializer."""

    return (
        initializer.name,
        tuple(int(dim) for dim in initializer.dims),
        onnx_dtype_elem_bytes(initializer.data_type),
        onnx_tensor_dtype(initializer.data_type),
    )


def _merge_tensor_metadata(
    metadata: dict[str, dict[str, object]],
    name: str,
    shape: tuple[int, ...],
    elem_bytes: int | None,
    dtype: TensorDType | None,
) -> None:
    """Merge one shape / dtype observation into the graph tensor metadata table."""

    record = metadata.setdefault(name, {"shape": (), "elem_bytes": None, "dtype": None})
    if not record["shape"] and shape:
        record["shape"] = shape
    if record["elem_bytes"] is None and elem_bytes is not None:
        record["elem_bytes"] = elem_bytes
    if record["dtype"] is None and dtype is not None:
        record["dtype"] = dtype


def collect_scheduler_tensors(graph: "GraphProto") -> dict[str, Tensor]:
    """Collect scheduler-side logical tensors from one ONNX graph."""

    metadata: dict[str, dict[str, object]] = {}
    initializer_names = {initializer.name for initializer in graph.initializer}

    for value in graph.input:
        name, shape, elem_bytes, dtype = parse_value_tensor(value)
        _merge_tensor_metadata(metadata, name, shape, elem_bytes, dtype)

    for value in graph.output:
        name, shape, elem_bytes, dtype = parse_value_tensor(value)
        _merge_tensor_metadata(metadata, name, shape, elem_bytes, dtype)

    for value in graph.value_info:
        name, shape, elem_bytes, dtype = parse_value_tensor(value)
        _merge_tensor_metadata(metadata, name, shape, elem_bytes, dtype)

    for initializer in graph.initializer:
        name, shape, elem_bytes, dtype = parse_initializer_tensor(initializer)
        metadata[name] = {
            "shape": shape,
            "elem_bytes": elem_bytes,
            "dtype": dtype,
        }

    tensors: dict[str, Tensor] = {}
    for name, record in metadata.items():
        shape = record["shape"]
        elem_bytes = record["elem_bytes"]
        if not shape or elem_bytes is None:
            continue
        if len(shape) > TENSOR_MAX_DIMS:
            raise ValueError(
                f"tensor '{name}' has rank {len(shape)}; "
                f"the runtime ABI supports at most {TENSOR_MAX_DIMS}"
            )
        tensors[name] = Tensor(
            name=name,
            rank=len(shape),
            dims=shape,
            elem_bytes=elem_bytes,
            is_initializer=name in initializer_names,
            dtype=record["dtype"],
        )

    return tensors


def parse_constants(
    graph: "GraphProto",
    names: set[str] | None = None,
) -> ConstantStore:
    """Decode all supported ONNX initializers into owned, C-order bytes."""

    import numpy as np
    from onnx import numpy_helper

    constants: list[Constant] = []
    for initializer in graph.initializer:
        if names is not None and initializer.name not in names:
            continue
        dtype = onnx_tensor_dtype(initializer.data_type)
        if dtype is None:
            raise ValueError(
                f"initializer '{initializer.name}' uses unsupported ONNX dtype "
                f"{initializer.data_type}"
            )
        array = numpy_helper.to_array(initializer)
        declared_shape = tuple(int(dimension) for dimension in initializer.dims)
        if tuple(array.shape) != declared_shape:
            raise ValueError(f"initializer '{initializer.name}' decoded shape mismatch")
        contiguous = np.ascontiguousarray(array)
        if contiguous.dtype.itemsize > 1:
            contiguous = contiguous.astype(contiguous.dtype.newbyteorder("<"), copy=False)
        constants.append(Constant(
            name=initializer.name,
            dtype=dtype,
            shape=declared_shape,
            data=contiguous.tobytes(order="C"),
        ))
    return ConstantStore(tuple(constants))
