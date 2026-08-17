"""Reproducibly extract the MobileViT nodes 170..179 acceptance slice."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference, utils

from maps.deployment import build_application, validate_application


FIRST_NODE_INDEX = 170
LAST_NODE_INDEX = 179
INPUT_SHAPE = (1, 128, 4, 16)
INPUT_SEED = 170179
REFERENCE_ATOL = 0.5
REFERENCE_RTOL = 0.05
_GROUP_NORM_SHARDS = 16
_SOFTMAX_AXIS_SHARDS = 3


def _fp16(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float16)


def _sequential_last_axis_sum(values: np.ndarray) -> np.ndarray:
    rows = values.reshape(-1, values.shape[-1])
    result = np.empty(rows.shape[0], dtype=np.float16)
    for row_index, row in enumerate(rows):
        accumulator = np.float32(row[0])
        for value in row[1:]:
            accumulator += np.float32(value)
        result[row_index] = np.float16(accumulator)
    return result.reshape((*values.shape[:-1], 1))


def softmax_exp_fp16(values: np.ndarray) -> np.ndarray:
    """Match the SDK Softmax task's FP16 bit-level exponential."""

    scaled = _fp16(_fp16(values) * np.float16(1486.0))
    biased = _fp16(scaled + np.float16(15360.0))
    bits = np.clip(biased.astype(np.float32), 0, 65535).astype(np.uint16)
    return bits.view(np.float16)


def _conv_1x1_fp16(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    rows = values.transpose(0, 2, 3, 1).reshape(-1, values.shape[1])
    matrix = weight[:, :, 0, 0]
    output = _fp16(rows.astype(np.float32) @ matrix.astype(np.float32).T)
    output = _fp16(output.astype(np.float32) + bias.astype(np.float32))
    return output.reshape(
        values.shape[0], values.shape[2], values.shape[3], weight.shape[0]
    ).transpose(0, 3, 1, 2)


def _group_normalize_fp16(
    values: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    group_elements = values.size
    reduction_scale = np.float16(1.0 / group_elements)
    channel_shards = np.array_split(values, _GROUP_NORM_SHARDS, axis=1)

    partial_means: list[np.float16] = []
    for shard in channel_shards:
        accumulator = np.float16(0.0)
        for value in shard.reshape(-1):
            accumulator = np.float16(
                accumulator + np.float16(value * reduction_scale)
            )
        partial_means.append(accumulator)
    mean = np.float16(sum(np.float32(value) for value in partial_means))

    partial_variances: list[np.float16] = []
    for shard in channel_shards:
        accumulator = np.float16(0.0)
        for value in shard.reshape(-1):
            centered = np.float16(value - mean)
            term = np.float16(np.float16(centered * reduction_scale) * centered)
            accumulator = np.float16(accumulator + term)
        partial_variances.append(accumulator)
    variance = np.float16(sum(np.float32(value) for value in partial_variances))

    normalized = (
        (values.astype(np.float32) - np.float32(mean))
        / np.sqrt(np.float32(variance) + np.float32(epsilon))
    )
    affine_scale = scale.reshape(1, -1, 1, 1).astype(np.float32)
    affine_bias = bias.reshape(1, -1, 1, 1).astype(np.float32)
    return _fp16(normalized * affine_scale + affine_bias)


def mobilevit_slice_reference(model_path: Path, input_path: Path) -> np.ndarray:
    """Evaluate the accepted slice with its MAGIA-v3 FP16 arithmetic contract."""

    model = onnx.load(model_path)
    initializers = {
        item.name: _fp16(numpy_helper.to_array(item))
        for item in model.graph.initializer
        if numpy_helper.to_array(item).dtype.kind == "f"
    }
    values = np.fromfile(input_path, dtype=np.dtype("<f2")).reshape(INPUT_SHAPE)

    group_norm = model.graph.node[0]
    epsilon = float(onnx.helper.get_attribute_value(group_norm.attribute[0]))
    normalized = _group_normalize_fp16(
        values,
        initializers[group_norm.input[1]],
        initializers[group_norm.input[2]],
        epsilon,
    )

    qkv = _conv_1x1_fp16(
        normalized,
        initializers[model.graph.node[1].input[1]],
        initializers[model.graph.node[1].input[2]],
    )
    query, key, value = np.split(qkv, (1, 129), axis=1)

    query_shards = np.array_split(query, _SOFTMAX_AXIS_SHARDS, axis=-1)
    local_maxima = [shard.max(axis=-1, keepdims=True) for shard in query_shards]
    maximum = _fp16(np.maximum.reduce(local_maxima))
    shifted = _fp16(query.astype(np.float32) - maximum.astype(np.float32))
    exponentials = softmax_exp_fp16(shifted)
    local_sums = [
        _sequential_last_axis_sum(shard)
        for shard in np.array_split(exponentials, _SOFTMAX_AXIS_SHARDS, axis=-1)
    ]
    denominator = _fp16(sum(part.astype(np.float32) for part in local_sums))
    attention = _fp16(exponentials.astype(np.float32) / denominator.astype(np.float32))

    weighted_key = _fp16(key.astype(np.float32) * attention.astype(np.float32))
    context_scale = _sequential_last_axis_sum(weighted_key)
    activated_value = _fp16(np.maximum(value, np.float16(0.0)))
    context = _fp16(
        activated_value.astype(np.float32) * context_scale.astype(np.float32)
    )

    projected = _conv_1x1_fp16(
        context,
        initializers[model.graph.node[8].input[1]],
        initializers[model.graph.node[8].input[2]],
    )
    return _fp16(values.astype(np.float32) + projected.astype(np.float32))


def _validation_application_source(
    reference: np.ndarray,
    output_name: str,
) -> str:
    reference_bits = reference.view(np.uint16).reshape(-1)
    rows = [
        "  " + ", ".join(f"0x{int(value):04x}u" for value in reference_bits[index:index + 8])
        for index in range(0, reference_bits.size, 8)
    ]
    reference_values = ",\n".join(rows)
    return f'''/* MobileViT slice numerical acceptance customization. */
#include "mobilevit_slice.h"

#include <stddef.h>
#include "utils/maps_operations.h"
#include "utils/printf.h"

extern const uint8_t mobilevit_slice_input_base_module_stages_stages_4_stages_4_1_reshape_1_output_0_start[];

static const uint16_t mobilevit_slice_reference[]
    __attribute__((aligned(16), section(".l2_bulk.maps_reference"))) = {{
{reference_values}
}};

void mobilevit_slice_handle_input(uint32_t token) {{
  uint8_t *input = mobilevit_slice_input_base_module_stages_stages_4_stages_4_1_reshape_1_output_0(token);
  const uint8_t *data = mobilevit_slice_input_base_module_stages_stages_4_stages_4_1_reshape_1_output_0_start;
  for (size_t index = 0; index < 16384u; ++index)
    input[index] = data[index + token * 16384u];
}}

void mobilevit_slice_handle_output(uint32_t token) {{
  const uint16_t *actual = (const uint16_t *)mobilevit_slice_output_{output_name}(token);
  uint32_t mismatches = 0u;
  uint32_t nonfinite = 0u;
  for (size_t index = 0; index < {reference_bits.size}u; ++index) {{
    const uint16_t actual_bits = actual[index];
    const float actual_value = maps_operation_f16_to_f32(actual_bits);
    const float expected = maps_operation_f16_to_f32(mobilevit_slice_reference[index]);
    const float difference = actual_value > expected
        ? actual_value - expected : expected - actual_value;
    const float expected_absolute = expected < 0.0f ? -expected : expected;
    if ((actual_bits & 0x7c00u) == 0x7c00u)
      ++nonfinite;
    if (difference > {REFERENCE_ATOL}f + {REFERENCE_RTOL}f * expected_absolute)
      ++mismatches;
  }}
  printf("mobilevit_slice reference mismatches: %u nonfinite: %u\\n",
         mismatches, nonfinite);
}}
'''


def configure_mobilevit_slice_validation(
    application: Path,
    reference: np.ndarray,
) -> None:
    """Install the numerical acceptance handler in the user-owned source."""

    manifest = validate_application(application)
    output_name = manifest["tensors"]["outputs"][0]["normalized_name"]
    (application / "src/application.c").write_text(
        _validation_application_source(reference, output_name),
        encoding="utf-8",
    )


def build_mobilevit_slice_application(
    model_path: Path,
    input_path: Path,
    application: Path,
) -> Path:
    """Plan the accepted slice on 8x8 and add its numerical comparison."""

    model = onnx.load(model_path)
    output = build_application(
        model_path,
        application,
        name="mobilevit_slice",
        target="magia-v3",
        mesh_width=8,
        mesh_height=8,
        num_token_slots=1,
        inputs={model.graph.input[0].name: input_path},
    )
    configure_mobilevit_slice_validation(
        output,
        mobilevit_slice_reference(model_path, input_path),
    )
    return output


def extract_mobilevit_slice(
    source: Path,
    model_output: Path,
    input_output: Path,
) -> tuple[Path, Path]:
    """Write the agreed ONNX slice and its deterministic FP16 Runtime Input."""

    model = shape_inference.infer_shapes(onnx.load(source))
    if len(model.graph.node) <= LAST_NODE_INDEX:
        raise ValueError("source model does not contain MobileViT nodes 170..179")
    first = model.graph.node[FIRST_NODE_INDEX]
    last = model.graph.node[LAST_NODE_INDEX]
    utils.extract_model(
        source,
        model_output,
        [first.input[0]],
        [last.output[0]],
        check_model=True,
        infer_shapes=True,
    )

    extracted = onnx.load(model_output)
    extracted.graph.name = "mobilevit_nodes_170_179"
    onnx.save(extracted, model_output)

    values = np.random.default_rng(INPUT_SEED).uniform(
        -1.0,
        1.0,
        size=INPUT_SHAPE,
    ).astype(np.dtype("<f2"))
    input_output.write_bytes(values.tobytes())
    return model_output, input_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("model_output", type=Path)
    parser.add_argument("input_output", type=Path)
    parser.add_argument("--application", type=Path)
    args = parser.parse_args(argv)
    extract_mobilevit_slice(args.source, args.model_output, args.input_output)
    if args.application is not None:
        build_mobilevit_slice_application(
            args.model_output,
            args.input_output,
            args.application,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
