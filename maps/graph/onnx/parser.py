"""Graph-level ONNX parsing orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from maps.graph.model import (
    TENSOR_MAX_DIMS,
    Constant,
    ConstantStore,
    Graph,
    Node,
    Tensor,
    TensorDType,
    build_graph_edges_from_nodes,
)

from .operations import (
    LoweredOperation,
    STATIC_INPUT_VALUES,
    get_operation_converter,
    onnx_tensor_dtype,
)

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto, TensorProto, ValueInfoProto


def _static_integer_inputs(graph: "GraphProto") -> dict[str, tuple[int, ...]]:
    """Decode small integer initializers used as compile-time op configuration."""

    from onnx import TensorProto, numpy_helper

    values = {}
    for initializer in graph.initializer:
        if initializer.data_type not in (TensorProto.INT32, TensorProto.INT64):
            continue
        array = numpy_helper.to_array(initializer)
        values[initializer.name] = tuple(int(value) for value in array.flat)
    return values


def parse_graph(graph: "GraphProto", *, graph_name: str | None = None) -> Graph:
    """Parse one ONNX graph into the shared scheduler graph IR."""

    tensors = collect_scheduler_tensors(graph)
    static_input_values = _static_integer_inputs(graph)
    nodes = []
    for node_idx, node in enumerate(graph.node):
        nodes.append(
            parse_node(
                node,
                node_idx,
                tensors,
                static_input_values=static_input_values,
            )
        )

    initializer_names = {initializer.name for initializer in graph.initializer}
    graph_input_names = {value.name for value in graph.input if value.name not in initializer_names}
    graph_output_names = tuple(value.name for value in graph.output)
    lowered_nodes = tuple(nodes)
    live_tensor_names = {
        tensor.name
        for node in lowered_nodes
        for tensor in node.inputs + node.outputs
    } | {
        value.name for value in graph.input
        if value.name not in initializer_names
    } | set(graph_output_names)

    return Graph(
        name=graph_name or graph.name,
        tensors=tuple(
            tensor for tensor in tensors.values()
            if tensor.name in live_tensor_names
        ),
        nodes=lowered_nodes,
        edges=build_graph_edges_from_nodes(lowered_nodes, tensors, graph_output_names),
        inputs=tuple(tensors[value.name] for value in graph.input if value.name in graph_input_names),
        outputs=tuple(tensors[value.name] for value in graph.output),
        initializers=tuple(
            tensors[initializer.name]
            for initializer in graph.initializer
            if (
                initializer.name in tensors
                and initializer.name in live_tensor_names
            )
        ),
    )


def node_name(node: "NodeProto", node_idx: int) -> str:
    """Return a stable node name for one ONNX node."""

    return node.name or f"{node.op_type}_{node_idx}"


def node_inputs(node: "NodeProto") -> tuple[str, ...]:
    """Return non-empty ONNX node inputs."""

    return tuple(value for value in node.input if value)


def node_outputs(node: "NodeProto") -> tuple[str, ...]:
    """Return non-empty ONNX node outputs."""

    return tuple(value for value in node.output if value)


def parse_node_attributes(node: "NodeProto") -> dict[str, object]:
    """Extract ONNX node attributes as graph-node metadata."""

    attributes: dict[str, object] = {}
    for attr in node.attribute:
        if attr.type == attr.INT:
            attributes[attr.name] = attr.i
        elif attr.type == attr.FLOAT:
            attributes[attr.name] = attr.f
        elif attr.type == attr.STRING:
            attributes[attr.name] = attr.s.decode("utf-8")
        elif attr.type == attr.INTS:
            attributes[attr.name] = tuple(attr.ints)
        elif attr.type == attr.FLOATS:
            attributes[attr.name] = tuple(attr.floats)
        elif attr.type == attr.STRINGS:
            attributes[attr.name] = tuple(value.decode("utf-8") for value in attr.strings)
    return attributes


def resolve_node_tensors(
    node_name_value: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    tensors: dict[str, Tensor],
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Resolve ONNX input/output names to scheduler tensors."""

    missing_inputs = tuple(name for name in input_names if name not in tensors)
    if missing_inputs:
        raise ValueError(
            f"unknown input tensor for node '{node_name_value}': {missing_inputs[0]}"
        )

    missing_outputs = tuple(name for name in output_names if name not in tensors)
    if missing_outputs:
        raise ValueError(
            f"unknown output tensor for node '{node_name_value}': {missing_outputs[0]}"
        )

    return (
        tuple(tensors[name] for name in input_names),
        tuple(tensors[name] for name in output_names),
    )


def parse_node(
    node: "NodeProto",
    node_idx: int,
    tensors: dict[str, Tensor],
    static_input_values: dict[str, tuple[int, ...]] | None = None,
) -> Node:
    """Lower one raw ONNX node into one graph node."""

    node_name_value = node_name(node, node_idx)
    input_names = node_inputs(node)
    output_names = node_outputs(node)
    input_tensors, output_tensors = resolve_node_tensors(
        node_name_value,
        input_names,
        output_names,
        tensors,
    )

    attributes = parse_node_attributes(node)
    lowering_attributes = dict(attributes)
    node_static_inputs = {
        name: static_input_values[name]
        for name in input_names
        if static_input_values is not None and name in static_input_values
    }
    if node_static_inputs:
        lowering_attributes[STATIC_INPUT_VALUES] = node_static_inputs
    lowerer = get_operation_converter(node.op_type)
    if lowerer is None:
        raise NotImplementedError(f"unsupported ONNX op_type: {node.op_type}")

    lowered = lowerer(
        node_name_value,
        input_tensors,
        output_tensors,
        lowering_attributes,
    )
    if isinstance(lowered, LoweredOperation):
        kind = lowered.kind
        payload = lowered.payload
        input_tensors = lowered.inputs
        output_tensors = lowered.outputs
    else:
        kind, payload = lowered

    return Node(
        name=node_name_value,
        kind=kind,
        inputs=input_tensors,
        outputs=output_tensors,
        payload=payload,
        attributes=attributes,
    )
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


def onnx_dtype_elem_bytes(dtype: int) -> int | None:
    """Return the element size in bytes for one ONNX tensor dtype."""

    return _ONNX_DTYPE_ELEM_BYTES.get(dtype)


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
