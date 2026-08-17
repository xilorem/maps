"""Explicit ONNX conversion into source-independent maps Operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import cast

from maps.graph import OpKind, Tensor, TensorDType
from maps.operations import OperationPayload
from maps.operations.cast import CastPayload
from maps.operations.convolution import ConvPayload
from maps.operations.elementwise import (
    BINARY_ELEMENTWISE_OPS,
    UNARY_ELEMENTWISE_OPS,
    BinaryElementwisePayload,
    UnaryElementwisePayload,
)
from maps.operations.gemm import GemmPayload
from maps.operations.normalization import GroupNormalizationPayload
from maps.operations.rearrangement import ReshapePayload, TransposePayload
from maps.operations.reduction import GlobalAveragePoolPayload, ReduceSumPayload
from maps.operations.softmax import SoftmaxPayload
from maps.operations.split import SplitPayload

STATIC_INPUT_VALUES = "_static_input_values"

_ONNX_DTYPES: dict[int, TensorDType] = {
    1: TensorDType.FLOAT32,
    2: TensorDType.UINT8,
    6: TensorDType.INT32,
    7: TensorDType.INT64,
    9: TensorDType.BOOL,
    10: TensorDType.FLOAT16,
}


def onnx_tensor_dtype(dtype: int) -> TensorDType | None:
    return _ONNX_DTYPES.get(dtype)


@dataclass(frozen=True)
class LoweredOperation:
    """Canonical operands and payload produced by one ONNX converter."""

    kind: OpKind
    payload: OperationPayload
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]


OperationConversion = Callable[
    [str, tuple[Tensor, ...], tuple[Tensor, ...], dict[str, object]],
    tuple[OpKind, OperationPayload] | LoweredOperation,
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


def convert_conv(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, ConvPayload]:
    """Lower one ONNX Conv node into scheduler-side Conv semantics."""

    if len(inputs) not in (2, 3):
        raise ValueError(f"Conv node '{node_name}' must have 2 or 3 inputs")
    if len(outputs) != 1:
        raise ValueError(f"Conv node '{node_name}' must have exactly 1 output")
    if "auto_pad" in attributes and attributes["auto_pad"] != "NOTSET":
        raise NotImplementedError("Conv auto_pad is not implemented")

    return (
        OpKind.CONV,
        ConvPayload(
            x=inputs[0],
            w=inputs[1],
            b=inputs[2] if len(inputs) == 3 else None,
            output=outputs[0],
            strides=cast(tuple[int, int], attributes.get("strides", (1, 1))),
            pads=cast(tuple[int, int, int, int], attributes.get("pads", (0, 0, 0, 0))),
            dilations=cast(tuple[int, int], attributes.get("dilations", (1, 1))),
            group=int(cast(int, attributes.get("group", 1))),
        ),
    )


def convert_softmax(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, SoftmaxPayload]:
    """Lower one ONNX Softmax node into one high-level maps softmax operation."""

    if len(inputs) != 1:
        raise ValueError(f"Softmax node '{node_name}' must have exactly 1 input")
    if len(outputs) != 1:
        raise ValueError(f"Softmax node '{node_name}' must have exactly 1 output")

    x = inputs[0]
    output = outputs[0]
    axis = int(cast(int, attributes.get("axis", -1)))
    if axis < 0:
        axis += x.rank
    return OpKind.CUSTOM, SoftmaxPayload(x=x, output=output, axis=axis)


def convert_group_normalization(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, GroupNormalizationPayload]:
    if len(inputs) != 3 or len(outputs) != 1:
        raise ValueError(
            f"GroupNormalization node '{node_name}' must have 3 inputs and 1 output"
        )
    if "num_groups" not in attributes:
        raise ValueError("GroupNormalization num_groups attribute is required")
    return (
        OpKind.CUSTOM,
        GroupNormalizationPayload(
            x=inputs[0],
            scale=inputs[1],
            bias=inputs[2],
            output=outputs[0],
            num_groups=int(cast(int, attributes["num_groups"])),
            epsilon=float(cast(float, attributes.get("epsilon", 1e-5))),
            stash_type=int(cast(int, attributes.get("stash_type", 1))),
        ),
    )


def convert_reshape(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> LoweredOperation:
    if len(inputs) != 2 or len(outputs) != 1:
        raise ValueError(f"Reshape node '{node_name}' must have 2 inputs and 1 output")
    if not inputs[1].is_initializer:
        raise NotImplementedError(
            f"Reshape node '{node_name}' requires a static shape initializer"
        )
    if int(cast(int, attributes.get("allowzero", 0))) != 0:
        raise NotImplementedError("Reshape allowzero is not implemented")
    payload = ReshapePayload(x=inputs[0], output=outputs[0])
    return LoweredOperation(
        kind=OpKind.TRANSFORM,
        payload=payload,
        inputs=(inputs[0],),
        outputs=outputs,
    )


def convert_transpose(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, TransposePayload]:
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Transpose node '{node_name}' must have 1 input and 1 output")
    permutation = cast(
        tuple[int, ...],
        attributes.get("perm", tuple(reversed(range(inputs[0].rank)))),
    )
    return (
        OpKind.TRANSFORM,
        TransposePayload(inputs[0], outputs[0], permutation),
    )


def convert_flatten(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, ReshapePayload]:
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Flatten node '{node_name}' must have 1 input and 1 output")
    unknown_attributes = set(attributes) - {"axis"}
    if unknown_attributes:
        attribute = sorted(unknown_attributes)[0]
        raise NotImplementedError(f"Flatten attribute '{attribute}' is not implemented")

    x = inputs[0]
    axis = int(cast(int, attributes.get("axis", 1)))
    if axis < 0:
        axis += x.rank
    if axis < 0 or axis > x.rank:
        raise ValueError(f"Flatten node '{node_name}' axis must be in [0, input rank]")

    expected_dims = (prod(x.dims[:axis]), prod(x.dims[axis:]))
    if outputs[0].dims != expected_dims:
        raise ValueError(
            f"Flatten node '{node_name}' output shape must be {expected_dims}"
        )
    return OpKind.TRANSFORM, ReshapePayload(x=x, output=outputs[0])


def _infer_single_reduced_axis(node_name: str, x: Tensor, output: Tensor) -> int:
    if x.rank != output.rank:
        raise NotImplementedError(
            f"ReduceSum node '{node_name}' requires keepdims=1"
        )
    candidates = tuple(
        axis
        for axis, (input_dim, output_dim) in enumerate(zip(x.dims, output.dims))
        if input_dim != output_dim and output_dim == 1
    )
    if len(candidates) != 1:
        raise NotImplementedError(
            f"ReduceSum node '{node_name}' must reduce exactly one statically "
            "identifiable axis"
        )
    return candidates[0]


def convert_reduce_sum(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> LoweredOperation:
    if len(inputs) not in (1, 2) or len(outputs) != 1:
        raise ValueError(f"ReduceSum node '{node_name}' must have 1 or 2 inputs and 1 output")
    unknown_attributes = set(attributes) - {
        "axes",
        "keepdims",
        "noop_with_empty_axes",
        STATIC_INPUT_VALUES,
    }
    if unknown_attributes:
        attribute = sorted(unknown_attributes)[0]
        raise NotImplementedError(f"ReduceSum attribute '{attribute}' is not implemented")
    if int(cast(int, attributes.get("keepdims", 1))) != 1:
        raise NotImplementedError(f"ReduceSum node '{node_name}' requires keepdims=1")
    if int(cast(int, attributes.get("noop_with_empty_axes", 0))) != 0:
        raise NotImplementedError("ReduceSum noop_with_empty_axes is not implemented")
    if len(inputs) == 2:
        axes = inputs[1]
        if (
            not axes.is_initializer
            or axes.dtype is not TensorDType.INT64
            or axes.rank != 1
            or axes.dims != (1,)
        ):
            raise NotImplementedError(
                f"ReduceSum node '{node_name}' requires one static INT64 axis"
            )
        static_inputs = cast(
            dict[str, tuple[int, ...]],
            attributes.get(STATIC_INPUT_VALUES, {}),
        )
        declared_axes = tuple(static_inputs.get(axes.name, ()))
        if len(declared_axes) != 1:
            raise NotImplementedError(
                f"ReduceSum node '{node_name}' requires one static INT64 axis"
            )
    elif "axes" not in attributes or len(cast(tuple[int, ...], attributes["axes"])) != 1:
        raise NotImplementedError(
            f"ReduceSum node '{node_name}' requires one static axis"
        )
    else:
        declared_axes = cast(tuple[int, ...], attributes["axes"])

    axis = _infer_single_reduced_axis(node_name, inputs[0], outputs[0])
    declared_axis = int(declared_axes[0])
    if declared_axis < 0:
        declared_axis += inputs[0].rank
    if declared_axis != axis:
        raise ValueError(f"ReduceSum node '{node_name}' axes do not match output shape")
    return LoweredOperation(
        kind=OpKind.CUSTOM,
        payload=ReduceSumPayload(inputs[0], outputs[0], axis),
        inputs=(inputs[0],),
        outputs=outputs,
    )


def convert_global_average_pool(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, GlobalAveragePoolPayload]:
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"GlobalAveragePool node '{node_name}' must have 1 input and 1 output"
        )
    if attributes:
        attribute = sorted(attributes)[0]
        raise NotImplementedError(
            f"GlobalAveragePool attribute '{attribute}' is not implemented"
        )
    return OpKind.CUSTOM, GlobalAveragePoolPayload(inputs[0], outputs[0])


def _normalized_axis(node_name: str, x: Tensor, attributes: dict[str, object]) -> int:
    axis = int(cast(int, attributes.get("axis", 0)))
    if axis < 0:
        axis += x.rank
    if axis < 0 or axis >= x.rank:
        raise ValueError(f"Split node '{node_name}' axis must be within input rank")
    return axis


def _num_output_sizes(node_name: str, dimension: int, num_outputs: int) -> tuple[int, ...]:
    if num_outputs <= 0:
        raise ValueError(f"Split node '{node_name}' num_outputs must be positive")
    chunk_size = (dimension + num_outputs - 1) // num_outputs
    sizes = (chunk_size,) * (num_outputs - 1) + (
        dimension - chunk_size * (num_outputs - 1),
    )
    if any(size <= 0 for size in sizes):
        raise ValueError(f"Split node '{node_name}' produces a zero-sized output")
    return sizes


def convert_split(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> LoweredOperation:
    """Normalize the supported static ONNX Split forms."""

    if len(inputs) not in (1, 2):
        raise ValueError(f"Split node '{node_name}' must have 1 or 2 inputs")
    if not outputs:
        raise ValueError(f"Split node '{node_name}' must have at least one output")

    unknown_attributes = set(attributes) - {
        "axis",
        "num_outputs",
        STATIC_INPUT_VALUES,
    }
    if unknown_attributes:
        attribute = sorted(unknown_attributes)[0]
        raise NotImplementedError(f"Split attribute '{attribute}' is not implemented")

    x = inputs[0]
    axis = _normalized_axis(node_name, x, attributes)
    has_split_input = len(inputs) == 2
    has_num_outputs = "num_outputs" in attributes
    if has_split_input == has_num_outputs:
        raise ValueError(
            f"Split node '{node_name}' must provide exactly one of "
            "a split initializer or num_outputs"
        )

    if has_split_input:
        split = inputs[1]
        if not split.is_initializer:
            raise NotImplementedError(
                f"Split node '{node_name}' requires a static split initializer"
            )
        if (
            split.dtype is not TensorDType.INT64
            or split.rank != 1
            or split.dims != (len(outputs),)
        ):
            raise ValueError(
                f"Split node '{node_name}' split initializer must be a rank-one "
                "INT64 tensor with one value per output"
            )
        sizes = tuple(output.dims[axis] for output in outputs)
    else:
        num_outputs = int(cast(int, attributes["num_outputs"]))
        if num_outputs != len(outputs):
            raise ValueError(
                f"Split node '{node_name}' num_outputs must match output count"
            )
        sizes = _num_output_sizes(node_name, x.dims[axis], num_outputs)

    payload = SplitPayload(x=x, outputs=outputs, axis=axis, sizes=sizes)
    return LoweredOperation(
        kind=OpKind.TRANSFORM,
        payload=payload,
        inputs=(x,),
        outputs=outputs,
    )


ONNX_OPERATION_CONVERTERS: dict[str, OperationConversion] = {
    "Gemm": convert_gemm,
    "MatMul": convert_matmul,
    "Cast": convert_cast,
    "Conv": convert_conv,
    "Flatten": convert_flatten,
    "GlobalAveragePool": convert_global_average_pool,
    "GroupNormalization": convert_group_normalization,
    "ReduceSum": convert_reduce_sum,
    "Reshape": convert_reshape,
    "Softmax": convert_softmax,
    "Split": convert_split,
    "Transpose": convert_transpose,
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

assert set(UNARY_ELEMENTWISE_OPS) - {"softmaxexp"} == {
    "abs", "exp", "log", "neg", "relu", "sigmoid", "sqrt"
}
assert set(BINARY_ELEMENTWISE_OPS) == {"add", "div", "mul", "pow", "sub"}


def get_operation_converter(external_name: str) -> OperationConversion | None:
    return ONNX_OPERATION_CONVERTERS.get(external_name)


__all__ = [
    "LoweredOperation",
    "ONNX_OPERATION_CONVERTERS",
    "OperationConversion",
    "STATIC_INPUT_VALUES",
    "get_operation_converter",
]
