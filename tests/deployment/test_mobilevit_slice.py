from pathlib import Path

import numpy as np
import onnx

from examples.mobilevit_slice import (
    INPUT_SEED,
    REFERENCE_ATOL,
    REFERENCE_RTOL,
    _group_normalize_fp16,
    _validation_application_source,
    build_mobilevit_slice_application,
    extract_mobilevit_slice,
    mobilevit_slice_reference,
    softmax_exp_fp16,
)
from maps.deployment import validate_application


PROJECT_ROOT = Path(__file__).parents[2]
SOURCE_MODEL = PROJECT_ROOT / "examples/mobilenet.onnx"


def test_mobilevit_slice_is_reproducibly_extracted_from_nodes_170_through_179(
    tmp_path: Path,
) -> None:
    first_model, first_input = extract_mobilevit_slice(
        SOURCE_MODEL,
        tmp_path / "first.onnx",
        tmp_path / "first.bin",
    )
    second_model, second_input = extract_mobilevit_slice(
        SOURCE_MODEL,
        tmp_path / "second.onnx",
        tmp_path / "second.bin",
    )

    source = onnx.load(SOURCE_MODEL)
    extracted = onnx.load(first_model)
    assert [node.SerializeToString() for node in extracted.graph.node] == [
        node.SerializeToString() for node in source.graph.node[170:180]
    ]
    assert [dimension.dim_value for dimension in extracted.graph.input[0].type.tensor_type.shape.dim] == [1, 128, 4, 16]
    assert [dimension.dim_value for dimension in extracted.graph.output[0].type.tensor_type.shape.dim] == [1, 128, 4, 16]
    source_initializers = {item.name: item for item in source.graph.initializer}
    assert len(extracted.graph.initializer) == 8
    assert all(
        initializer.SerializeToString()
        == source_initializers[initializer.name].SerializeToString()
        for initializer in extracted.graph.initializer
    )
    assert first_model.read_bytes() == second_model.read_bytes()
    assert first_input.read_bytes() == second_input.read_bytes()
    expected = np.random.default_rng(INPUT_SEED).uniform(
        -1.0, 1.0, size=(1, 128, 4, 16)
    ).astype(np.dtype("<f2"))
    assert first_input.read_bytes() == expected.tobytes()


def test_mobilevit_slice_reference_is_deterministic_and_finite(tmp_path: Path) -> None:
    model_path, input_path = extract_mobilevit_slice(
        SOURCE_MODEL,
        tmp_path / "slice.onnx",
        tmp_path / "input.bin",
    )

    first = mobilevit_slice_reference(model_path, input_path)
    second = mobilevit_slice_reference(model_path, input_path)

    assert first.shape == (1, 128, 4, 16)
    assert first.dtype == np.float16
    assert np.isfinite(first).all()
    assert first.tobytes() == second.tobytes()
    assert REFERENCE_ATOL == 0.5
    assert REFERENCE_RTOL == 0.05


def test_softmax_exp_reference_matches_sdk_fp16_bit_approximation() -> None:
    values = np.asarray([-16.0, -1.0, 0.0], dtype=np.float16)

    result = softmax_exp_fp16(values)

    scaled = np.float16(values * np.float16(1486.0))
    biased = np.float16(scaled + np.float16(15360.0))
    expected = np.clip(biased.astype(np.float32), 0, 65535).astype(
        np.uint16
    ).view(np.float16)
    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))


def test_scaled_group_normalization_remains_finite_for_large_fp16_group() -> None:
    values = np.resize(
        np.asarray([-200.0, 200.0], dtype=np.float16),
        (1, 128, 4, 16),
    )

    result = _group_normalize_fp16(
        values,
        np.ones(128, dtype=np.float16),
        np.zeros(128, dtype=np.float16),
        1e-5,
    )

    assert np.isfinite(result).all()
    np.testing.assert_allclose(
        result.astype(np.float32),
        np.sign(values).astype(np.float32),
        atol=0.02,
        rtol=0.0,
    )


def test_validation_handler_embeds_reference_and_declared_tolerances() -> None:
    source = _validation_application_source(
        np.asarray([1.0, -2.0], dtype=np.float16),
        "slice_output",
    )

    assert "0x3c00u, 0xc000u" in source
    assert "difference > 0.5f + 0.05f * expected_absolute" in source
    assert "mobilevit_slice_output_slice_output(token)" in source
    assert "reference mismatches: %u nonfinite: %u" in source


def test_mobilevit_slice_uses_the_ordinary_8x8_application_workflow(
    tmp_path: Path,
) -> None:
    model_path, input_path = extract_mobilevit_slice(
        SOURCE_MODEL,
        tmp_path / "slice.onnx",
        tmp_path / "input.bin",
    )

    application = build_mobilevit_slice_application(
        model_path,
        input_path,
        tmp_path / "application",
    )

    manifest = validate_application(application)
    assert manifest["planned_mesh"] == {"width": 8, "height": 8}
    assert manifest["execution"] == {"tokens": 1, "token_slots": 1}
    assert len(manifest["active_physical_tiles"]) == 64
    assert manifest["abi"] == {
        "descriptor": 1,
        "kernel": 2,
        "operation": 2,
        "task_bundle": 2,
    }
    assert {
        "softmax_exp_fp16_spatz_task",
        "group_reduce_fp16_spatz_task",
        "group_centered_reduce_fp16_spatz_task",
        "group_normalize_fp16_spatz_task",
    }.issubset(manifest["tasks"])
    source = (application / "src/application.c").read_text()
    assert "reference mismatches: %u nonfinite: %u" in source
    tile_sources = "\n".join(
        path.read_text() for path in (application / "src/tiles").glob("*.c")
    )
    assert ".kind = OP_SPLIT" in tile_sources
    assert ".num_outputs = 3u" in tile_sources
    split_tile = (application / "src/tiles/tile_33.c").read_text()
    split_op = split_tile.split("static const op_desc_t", 1)[1].split(
        "static const fifo_send_desc_t", 1
    )[0]
    assert split_op.count(".slice_id =") == 4
    assert split_op.count(".shape = {1, 128, 2, 3, 0, 0}") == 2
    runner = (application / "src/mobilevit_slice_runner.c").read_text()
    assert "MAPS_OP_ABI_VERSION == 2u" in runner
    assert "MAPS_KERNEL_ABI_VERSION == 2u" in runner
    assert "MAPS_TASK_BUNDLE_ABI_VERSION == 2u" in runner
