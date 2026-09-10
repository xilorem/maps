from pathlib import Path

import onnx
from onnx import TensorProto, helper

from maps.deployment import build_application, validate_application


def _write_model(path: Path) -> Path:
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT16, [2, 3, 5]
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 3, 5])
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["input"], ["output"], axis=-1)],
        "tile_local_reducemax",
        [input_tensor],
        [output],
    )
    onnx.save(helper.make_model(graph), path)
    return path


def test_generated_magia_v3_reducemax_bundles_its_tile_local_spatz_task(
    tmp_path: Path,
) -> None:
    application = build_application(
        _write_model(tmp_path / "reducemax.onnx"),
        tmp_path / "reducemax",
        target="magia-v3",
        mesh_width=1,
        mesh_height=1,
        num_token_slots=1,
    )

    assert "reducemax_fp16_spatz_task" in validate_application(application)["tasks"]
    cmake = (application / "CMakeLists.txt").read_text()
    assert "reducemax/spatz_task/reducemax_fp16_spatz_task.c" in cmake
    assert "MAPS_HAS_REDUCE_MAX_SPATZ_TASK=1" in cmake
    runner = (application / "src/reducemax_runner.c").read_text()
    assert "runtime.reducemax_fp16_task = REDUCEMAX_FP16_SPATZ_TASK" in runner
    tile = (application / "src/tiles/tile_00.c").read_text()
    assert ".kind = OP_REDUCE_MAX" in tile
