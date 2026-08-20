"""Deterministic host oracle for the complete MAGIA-v3 MobileViT plan."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


INPUT_SEED = 20260729
INPUT_SHAPE = (1, 3, 256, 256)
INPUT_SHA256 = "21e57ef38318ef852e2f7292c9c2ad939247cd449bd3859ecca839169c34315c"
SDK_LOGITS_SHA256 = "a8cd37d9d74e744d99751cd3bcfd1a086f9946e18db64502009183ba26603bb3"
MAPS_LOGITS_SHA256 = "c8950756e3868bb514165a4b32a5747227c38776df24b9bbef9c52c075c86e91"
TOP1_CLASS = 574

# The input and SDK logits reproduce the SDK MobileViT-v2 test-data generator.
# Frozen before full-model GVSoC execution. The absolute bound is the next
# power-of-two FP16 step above the measured 0.02734375 maximum difference. The
# relative bound rounds up the 0.0229008 maximum for SDK logits with |x| >= 1;
# near-zero logits are governed by the absolute bound.
ABSOLUTE_LOGIT_TOLERANCE = 0.03125
RELATIVE_LOGIT_TOLERANCE = 0.025
SOFTMAX_COEFFICIENT = np.float16(1486.0)
SIGMOID_COEFFICIENT = np.float16(1477.0)
FAST_EXP_BIAS = np.float16(15360.0)
SIGMOID_MINIMUM = np.float16(-11.0)
SDK_VLMAX = 256

# These are the ordinary 32x32 plan's GroupNormalization collective widths.
_GROUP_NORM_WIDTH_SHARDS = {256: 5, 64: 2, 16: 1}


def mobilevit_input() -> np.ndarray:
    """Return the deterministic FP16 input used by the SDK reference."""

    return np.random.default_rng(INPUT_SEED).standard_normal(INPUT_SHAPE).astype(
        np.float16
    )


def _fp16(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float16)


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _lane_fold(values: np.ndarray) -> np.float16:
    lanes = np.zeros(min(SDK_VLMAX, values.size), dtype=np.float16)
    for offset in range(0, values.size, SDK_VLMAX):
        chunk = values[offset : offset + SDK_VLMAX]
        lanes[: chunk.size] = _fp16(
            lanes[: chunk.size].astype(np.float64) + chunk.astype(np.float64)
        )
    accumulator = np.float16(0.0)
    for value in lanes:
        accumulator = np.float16(accumulator + value)
    return accumulator


def _core_sum(values: np.ndarray) -> np.float16:
    flat = values.reshape(-1)
    accumulator = np.float32(flat[0])
    for value in flat[1:]:
        accumulator += np.float32(value)
    return np.float16(accumulator)


def _im2col(
    values: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    output_h: int,
    output_w: int,
    channel_offset: int,
    channels: int,
) -> np.ndarray:
    input_h, input_w = values.shape[1:]
    padded = np.zeros(
        (channels, input_h + 2 * pad_h, input_w + 2 * pad_w),
        dtype=np.float16,
    )
    padded[:, pad_h : pad_h + input_h, pad_w : pad_w + input_w] = values[
        channel_offset : channel_offset + channels
    ]
    output = np.empty(
        (channels * kernel_h * kernel_w, output_h * output_w),
        dtype=np.float16,
    )
    output_rows = np.arange(output_h) * stride_h
    output_columns = np.arange(output_w) * stride_w
    channel_indices = np.arange(channels)
    for kernel_row in range(kernel_h):
        for kernel_column in range(kernel_w):
            patch = padded[
                :,
                kernel_row + output_rows[:, None],
                kernel_column + output_columns[None, :],
            ]
            output[
                (channel_indices * kernel_h + kernel_row) * kernel_w
                + kernel_column
            ] = patch.reshape(channels, -1)
    return output


def _redmule_gemm(
    lhs: np.ndarray,
    rhs: np.ndarray,
    bias: np.ndarray | None,
) -> np.ndarray:
    accumulator = np.zeros((lhs.shape[0], rhs.shape[1]), dtype=np.float64)
    lhs64 = lhs.astype(np.float64)
    rhs64 = rhs.astype(np.float64)
    for reduction_index in range(lhs.shape[1]):
        accumulator += (
            lhs64[:, reduction_index : reduction_index + 1]
            * rhs64[reduction_index : reduction_index + 1]
        )
        accumulator = accumulator.astype(np.float16).astype(np.float64)
    if bias is not None:
        accumulator = _fp16(accumulator + bias.astype(np.float64)[:, None])
    return accumulator.astype(np.float16)


def _convolution(
    values: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None,
    attributes: dict[str, Any],
) -> np.ndarray:
    kernel_h, kernel_w = attributes["kernel_shape"]
    stride_h, stride_w = attributes["strides"]
    pad_h, pad_w = attributes["pads"][:2]
    groups = int(attributes.get("group", 1))
    output_h = (values.shape[2] + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (values.shape[3] + 2 * pad_w - kernel_w) // stride_w + 1
    input_channels_per_group = values.shape[1] // groups
    output_channels_per_group = weights.shape[0] // groups
    output = np.empty(
        (values.shape[0], weights.shape[0], output_h, output_w),
        dtype=np.float16,
    )
    for batch in range(values.shape[0]):
        for group in range(groups):
            columns = _im2col(
                values[batch],
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                output_h,
                output_w,
                group * input_channels_per_group,
                input_channels_per_group,
            )
            first_channel = group * output_channels_per_group
            last_channel = first_channel + output_channels_per_group
            matrix = weights[first_channel:last_channel].reshape(
                output_channels_per_group, -1
            )
            group_bias = bias[first_channel:last_channel] if bias is not None else None
            output[batch, first_channel:last_channel] = _redmule_gemm(
                matrix,
                columns,
                group_bias,
            ).reshape(output_channels_per_group, output_h, output_w)
    return output


def _fast_exp(values: np.ndarray, coefficient: np.float16) -> np.ndarray:
    with np.errstate(over="ignore"):
        scaled = _fp16(_fp16(values) * coefficient)
    biased = _fp16(scaled + FAST_EXP_BIAS)
    bits = np.clip(np.trunc(biased.astype(np.float64)), 0, 65535).astype(
        np.uint16
    )
    return bits.view(np.float16)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clamped = np.maximum(values.astype(np.float16), SIGMOID_MINIMUM)
    exponential = _fast_exp(_fp16(-clamped), SIGMOID_COEFFICIENT)
    return _fp16(np.float16(1.0) / _fp16(np.float16(1.0) + exponential))


def _softmax(values: np.ndarray, *, maps_contract: bool) -> np.ndarray:
    rows = values.reshape(-1, values.shape[-1])
    output = np.empty_like(rows)
    for row_index, row in enumerate(rows):
        exponentials = _fast_exp(_fp16(row - row.max()), SOFTMAX_COEFFICIENT)
        denominator = (
            _core_sum(exponentials) if maps_contract else _lane_fold(exponentials)
        )
        output[row_index] = _fp16(exponentials / denominator)
    return output.reshape(values.shape)


def _sdk_group_normalization(
    values: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    groups: int,
    epsilon: float,
) -> np.ndarray:
    _, channels, height, width = values.shape
    channels_per_group = channels // groups
    elements_per_group = channels_per_group * height * width
    statistic_scale = 1
    while statistic_scale * statistic_scale < elements_per_group:
        statistic_scale <<= 1
    inverse_scale = np.float16(1.0 / statistic_scale)
    output = np.empty_like(values)
    for group in range(groups):
        channel_slice = slice(
            group * channels_per_group, (group + 1) * channels_per_group
        )
        block = values[0, channel_slice].reshape(-1)
        mean = np.float16(
            np.float32(_lane_fold(_fp16(block * inverse_scale)))
            * (np.float32(statistic_scale) / np.float32(elements_per_group))
        )
        centered = _fp16(block - mean)
        variance = np.float32(
            _lane_fold(_fp16(_fp16(centered * inverse_scale) ** 2))
        ) * (
            np.float32(statistic_scale * statistic_scale)
            / np.float32(elements_per_group)
        )
        inverse_stddev = np.float16(
            np.float32(1.0)
            / np.sqrt(variance + np.float32(np.float16(epsilon)))
        )
        normalized = _fp16(centered * inverse_stddev).reshape(
            channels_per_group, height, width
        )
        output[0, channel_slice] = _fp16(
            _fp16(normalized * scale[channel_slice, None, None])
            + bias[channel_slice, None, None]
        )
    return output


def _maps_group_normalization(
    values: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    groups: int,
    epsilon: float,
) -> np.ndarray:
    _, channels, height, width = values.shape
    channels_per_group = channels // groups
    elements_per_group = channels_per_group * height * width
    statistic_scale = 1
    while statistic_scale * statistic_scale < elements_per_group:
        statistic_scale <<= 1
    inverse_scale = np.float16(1.0 / statistic_scale)
    mean_rescale = np.float32(statistic_scale) / np.float32(elements_per_group)
    variance_rescale = np.float32(
        statistic_scale * statistic_scale
    ) / np.float32(elements_per_group)
    width_shards = _GROUP_NORM_WIDTH_SHARDS[width]
    output = np.empty_like(values)
    for group in range(groups):
        channel_slice = slice(
            group * channels_per_group, (group + 1) * channels_per_group
        )
        shards = np.array_split(values[0, channel_slice], width_shards, axis=2)
        partial_means: list[np.float16] = []
        for shard in shards:
            partial_means.append(
                np.float16(
                    np.float32(
                        _lane_fold(_fp16(shard.reshape(-1) * inverse_scale))
                    )
                    * mean_rescale
                )
            )
        mean = np.float16(sum(np.float32(value) for value in partial_means))
        partial_variances: list[np.float16] = []
        for shard in shards:
            centered = _fp16(shard.reshape(-1) - mean)
            partial_variances.append(
                np.float16(
                    np.float32(
                        _lane_fold(_fp16(_fp16(centered * inverse_scale) ** 2))
                    )
                    * variance_rescale
                )
            )
        variance = np.float16(
            sum(np.float32(value) for value in partial_variances)
        )
        normalized = (
            values[0, channel_slice].astype(np.float32) - np.float32(mean)
        ) / np.sqrt(np.float32(variance) + np.float32(epsilon))
        output[0, channel_slice] = _fp16(
            normalized * scale[channel_slice, None, None].astype(np.float32)
            + bias[channel_slice, None, None].astype(np.float32)
        )
    return output


def _reduce_last(values: np.ndarray, *, maps_contract: bool) -> np.ndarray:
    rows = values.reshape(-1, values.shape[-1])
    output = np.empty(rows.shape[0], dtype=np.float16)
    for row_index, row in enumerate(rows):
        output[row_index] = _core_sum(row) if maps_contract else _lane_fold(row)
    return output.reshape((*values.shape[:-1], 1))


def _global_average(values: np.ndarray, *, maps_contract: bool) -> np.ndarray:
    if not maps_contract:
        rows = values.reshape(values.shape[0] * values.shape[1], -1)
        output = np.empty(rows.shape[0], dtype=np.float16)
        for row_index, row in enumerate(rows):
            output[row_index] = np.float16(
                _lane_fold(row) / np.float16(row.size)
            )
        return output.reshape(values.shape[0], values.shape[1], 1, 1)

    # MAPS lowers GlobalAveragePool to two ReduceSum kernels, each of which
    # stores an FP16 result, followed by an FP16 scalar multiplication.
    width_sums = np.empty(values.shape[:3], dtype=np.float16)
    for batch in range(values.shape[0]):
        for channel in range(values.shape[1]):
            for row in range(values.shape[2]):
                width_sums[batch, channel, row] = _core_sum(
                    values[batch, channel, row]
                )
    output = np.empty(values.shape[:2], dtype=np.float16)
    factor = np.float16(1.0 / (values.shape[2] * values.shape[3]))
    for batch in range(values.shape[0]):
        for channel in range(values.shape[1]):
            output[batch, channel] = np.float16(
                _core_sum(width_sums[batch, channel]) * factor
            )
    return output[:, :, None, None]


def mobilevit_logits(
    model_path: Path,
    *,
    maps_contract: bool,
) -> np.ndarray:
    """Evaluate logits using either MAPS or SDK FP16 execution arithmetic."""

    model = onnx.load(model_path)
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    tensors: dict[str, np.ndarray] = {
        model.graph.input[0].name: mobilevit_input()
    }
    logits: np.ndarray | None = None
    for node in model.graph.node:
        attributes = _attributes(node)
        values = tensors[node.input[0]]
        if node.op_type == "Conv":
            weights = _fp16(initializers[node.input[1]])
            bias = _fp16(initializers[node.input[2]]) if len(node.input) > 2 else None
            result = _convolution(values, weights, bias, attributes)
        elif node.op_type == "Relu":
            result = _fp16(np.maximum(values, np.float16(0.0)))
        elif node.op_type == "Sigmoid":
            result = _sigmoid(values)
        elif node.op_type == "Add":
            result = _fp16(
                values.astype(np.float64)
                + tensors[node.input[1]].astype(np.float64)
            )
        elif node.op_type == "Mul":
            result = _fp16(
                values.astype(np.float64)
                * tensors[node.input[1]].astype(np.float64)
            )
        elif node.op_type == "GroupNormalization":
            normalize = (
                _maps_group_normalization
                if maps_contract
                else _sdk_group_normalization
            )
            result = normalize(
                values,
                _fp16(initializers[node.input[1]]),
                _fp16(initializers[node.input[2]]),
                int(attributes["num_groups"]),
                float(attributes["epsilon"]),
            )
        elif node.op_type == "Softmax":
            result = _softmax(values, maps_contract=maps_contract)
        elif node.op_type == "ReduceSum":
            result = _reduce_last(values, maps_contract=maps_contract)
        elif node.op_type == "Transpose":
            result = np.ascontiguousarray(values.transpose(attributes["perm"]))
        elif node.op_type == "Reshape":
            result = values.reshape(
                tuple(int(value) for value in initializers[node.input[1]])
            )
        elif node.op_type == "Split":
            sizes = tuple(int(value) for value in initializers[node.input[1]])
            boundaries = np.cumsum(sizes[:-1])
            for name, output in zip(
                node.output,
                np.split(values, boundaries, axis=int(attributes["axis"])),
            ):
                tensors[name] = output
            continue
        elif node.op_type == "GlobalAveragePool":
            result = _global_average(values, maps_contract=maps_contract)
        elif node.op_type == "Flatten":
            result = values.reshape(values.shape[0], -1)
        elif node.op_type == "Gemm":
            weights = _fp16(initializers[node.input[1]])
            bias = _fp16(initializers[node.input[2]])
            result = _redmule_gemm(weights, values.reshape(-1, 1), bias).reshape(
                1, -1
            )
            logits = result
        else:
            raise ValueError(f"unsupported MobileViT oracle operation {node.op_type}")
        tensors[node.output[0]] = result
    if logits is None:
        raise ValueError("MobileViT model has no Gemm logits")
    return logits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("mobilenet.onnx"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    sdk_logits = mobilevit_logits(arguments.model, maps_contract=False)
    maps_logits = mobilevit_logits(arguments.model, maps_contract=True)
    differences = np.abs(
        maps_logits.astype(np.float64) - sdk_logits.astype(np.float64)
    )
    relative = differences / np.maximum(
        np.abs(sdk_logits.astype(np.float64)), np.finfo(np.float16).tiny
    )
    if arguments.output is not None:
        np.save(arguments.output, maps_logits)
    print(f"SDK logits sha256: {sha256(sdk_logits.tobytes()).hexdigest()}")
    print(f"MAPS logits sha256: {sha256(maps_logits.tobytes()).hexdigest()}")
    print(f"top-1 class: {int(np.argmax(maps_logits))}")
    print(f"maximum absolute error: {float(differences.max()):.9g}")
    print(f"maximum relative error: {float(relative.max()):.9g}")
    print(
        "frozen tolerances: "
        f"atol={ABSOLUTE_LOGIT_TOLERANCE:.9g} "
        f"rtol={RELATIVE_LOGIT_TOLERANCE:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
