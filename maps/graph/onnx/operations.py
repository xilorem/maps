"""Explicit ONNX conversion into source-independent maps Operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from maps.graph import OpKind, Tensor
from maps.operations import OperationPayload
from maps.operations.cast import CastPayload
from maps.operations.elementwise import (
    BINARY_ELEMENTWISE_OPS,
    UNARY_ELEMENTWISE_OPS,
    BinaryElementwisePayload,
    UnaryElementwisePayload,
)
from maps.operations.gemm import GemmPayload

from .tensor_parser import onnx_tensor_dtype

OperationConversion = Callable[
    [str, tuple[Tensor, ...], tuple[Tensor, ...], dict[str, object]],
    tuple[OpKind, OperationPayload],
]


def _require_arity(
    external_name: str,
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    *,
    input_count: int,
) -> None:
    if len(inputs) != input_count:
        raise ValueError(
            f"{external_name} node '{node_name}' must have exactly {input_count} inputs"
        )
    if len(outputs) != 1:
        raise ValueError(f"{external_name} node '{node_name}' must have exactly 1 output")


def convert_gemm(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, OperationPayload]:
    if len(inputs) not in (2, 3):
        raise ValueError(f"Gemm node '{node_name}' must have 2 or 3 inputs")
    if len(outputs) != 1:
        raise ValueError(f"Gemm node '{node_name}' must have exactly 1 output")
    for attribute, default in {"alpha": 1.0, "beta": 1.0, "transA": 0}.items():
        value = attributes.get(attribute, default)
        if value != default:
            raise NotImplementedError(
                f"Gemm node '{node_name}' uses unsupported {attribute}={value}; "
                f"only {attribute}={default} is currently supported"
            )
    transpose_w = int(cast(int, attributes.get("transB", 0)))
    if transpose_w not in (0, 1):
        raise ValueError(f"Gemm node '{node_name}' transB must be 0 or 1")
    if any(tensor.rank != 2 for tensor in inputs[:2] + outputs):
        raise NotImplementedError(
            f"Gemm node '{node_name}' uses non-matrix operands; maps currently "
            "supports ONNX Gemm only for rank-2 A, B, and output tensors"
        )
    return OpKind.GEMM, GemmPayload(
        x=inputs[0],
        w=inputs[1],
        y=inputs[2] if len(inputs) == 3 else None,
        output=outputs[0],
        transpose_w=bool(transpose_w),
    )


def convert_matmul(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, OperationPayload]:
    del attributes
    _require_arity("MatMul", node_name, inputs, outputs, input_count=2)
    x, w = inputs
    output = outputs[0]
    if min(x.rank, w.rank, output.rank) < 2:
        raise NotImplementedError(
            f"MatMul node '{node_name}' uses vector operands; maps currently "
            "supports matrix operands only"
        )
    if x.rank != w.rank or x.rank != output.rank:
        raise NotImplementedError(
            f"MatMul node '{node_name}' uses broadcasted operand ranks; maps "
            "currently requires equal operand and output ranks"
        )
    if x.dims[:-2] != w.dims[:-2] or x.dims[:-2] != output.dims[:-2]:
        raise NotImplementedError(
            f"MatMul node '{node_name}' uses broadcasted batch dimensions; maps "
            "currently requires identical batch dimensions"
        )
    return OpKind.GEMM, GemmPayload(x=x, w=w, y=None, output=output)


def convert_cast(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, OperationPayload]:
    _require_arity("Cast", node_name, inputs, outputs, input_count=1)
    if "to" not in attributes:
        raise ValueError(f"Cast node '{node_name}' must declare its destination dtype")
    destination_dtype = onnx_tensor_dtype(int(cast(int, attributes["to"])))
    if destination_dtype is None:
        raise NotImplementedError(
            f"Cast node '{node_name}' uses unsupported destination dtype "
            f"{attributes['to']}"
        )
    if outputs[0].dtype is not destination_dtype:
        raise ValueError(
            f"Cast node '{node_name}' destination dtype does not match its output Tensor"
        )
    return OpKind.TRANSFORM, CastPayload(x=inputs[0], output=outputs[0])


def _unary_converter(operation_name: str, external_name: str) -> OperationConversion:
    def convert(
        node_name: str,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        attributes: dict[str, object],
    ) -> tuple[OpKind, OperationPayload]:
        del attributes
        _require_arity(external_name, node_name, inputs, outputs, input_count=1)
        return OpKind.ELEMENTWISE, UnaryElementwisePayload(
            op_name=operation_name,
            x=inputs[0],
            output=outputs[0],
        )

    return convert


def _binary_converter(operation_name: str, external_name: str) -> OperationConversion:
    def convert(
        node_name: str,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        attributes: dict[str, object],
    ) -> tuple[OpKind, OperationPayload]:
        del attributes
        _require_arity(external_name, node_name, inputs, outputs, input_count=2)
        return OpKind.ELEMENTWISE, BinaryElementwisePayload(
            op_name=operation_name,
            lhs=inputs[0],
            rhs=inputs[1],
            output=outputs[0],
        )

    return convert


ONNX_OPERATION_CONVERTERS: dict[str, OperationConversion] = {
    "Gemm": convert_gemm,
    "MatMul": convert_matmul,
    "Cast": convert_cast,
    **{
        external_name: _unary_converter(operation_name, external_name)
        for external_name, operation_name in {
            "Abs": "abs",
            "Exp": "exp",
            "Log": "log",
            "Neg": "neg",
            "Relu": "relu",
            "Sigmoid": "sigmoid",
            "Sqrt": "sqrt",
        }.items()
    },
    **{
        external_name: _binary_converter(operation_name, external_name)
        for external_name, operation_name in {
            "Add": "add",
            "Div": "div",
            "Mul": "mul",
            "Pow": "pow",
            "Sub": "sub",
        }.items()
    },
}

assert set(UNARY_ELEMENTWISE_OPS) == {
    "abs", "exp", "log", "neg", "relu", "sigmoid", "sqrt"
}
assert set(BINARY_ELEMENTWISE_OPS) == {"add", "div", "mul", "pow", "sub"}


def get_operation_converter(external_name: str) -> OperationConversion | None:
    return ONNX_OPERATION_CONVERTERS.get(external_name)


__all__ = ["ONNX_OPERATION_CONVERTERS", "OperationConversion", "get_operation_converter"]
