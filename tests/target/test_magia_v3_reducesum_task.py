from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from maps.deployment import build_application, validate_application
from maps.graph import TensorDType
from maps.hardware import WorkKind, WorkSignature
from maps.target import magia_v3


def _write_model(path: Path) -> Path:
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT16, [2, 3, 5]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [2, 3, 1]
    )
    axes = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceSum", ["input", "axes"], ["output"], keepdims=1)],
        "tile_local_reducesum",
        [input_tensor],
        [output],
        [axes],
    )
    onnx.save(helper.make_model(graph), path)
    return path


def test_generated_magia_v3_reducesum_bundles_its_tile_local_spatz_task(
    tmp_path: Path,
) -> None:
    application = build_application(
        _write_model(tmp_path / "reducesum.onnx"), tmp_path / "reducesum",
        target="magia-v3", mesh_width=1, mesh_height=1, num_token_slots=1,
    )
    signature = WorkSignature(
        WorkKind.REDUCE_SUM, (TensorDType.FLOAT16,), (TensorDType.FLOAT16,)
    )
    assert magia_v3.SPATZ_DEVICE.supports(signature)
    assert magia_v3.build_mesh(width=1, height=1).tiles[0].assigned_device(
        signature
    ) is magia_v3.SPATZ_DEVICE
    assert validate_application(application)["tasks"] == ["reducesum_fp16_spatz_task"]
    cmake = (application / "CMakeLists.txt").read_text()
    assert "reducesum/spatz_task/reducesum_fp16_spatz_task.c" in cmake
    assert "MAPS_HAS_REDUCE_SUM_SPATZ_TASK=1" in cmake
    runner = (application / "src/reducesum_runner.c").read_text()
    assert "runtime.reducesum_fp16_task = REDUCESUM_FP16_SPATZ_TASK" in runner
    assert ".kind = OP_REDUCE_SUM" in (application / "src/tiles/tile_00.c").read_text()
