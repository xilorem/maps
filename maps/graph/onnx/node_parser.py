"""ONNX node parsing and lowering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from maps.graph.graph import Node
from maps.graph.onnx.operations import get_operation_converter
from maps.graph.tensor import Tensor
from MAPS.ops.registry import get_onnx_lowerer
from MAPS.ops.spec import LoweredOperation, STATIC_INPUT_VALUES

if TYPE_CHECKING:
    from onnx import NodeProto


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
    lowerer = get_operation_converter(node.op_type) or get_onnx_lowerer(node.op_type)
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
