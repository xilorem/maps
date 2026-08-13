import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from maps.cli import main
from maps.deployment import validate_application
from maps.graph import TensorDType, import_onnx_model
from maps.hardware import WorkKind, WorkSignature
from maps.operations.cast import CastPayload
from maps.target import magia, magia_v3


def _write_float_model_with_integer_control(path: Path) -> Path:
    runtime_input = helper.make_tensor_value_info(
        "runtime_input", TensorProto.FLOAT, [2, 2]
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])
    weight = numpy_helper.from_array(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        name="weight",
    )
    control_shape = numpy_helper.from_array(
        np.array([2, 2], dtype=np.int64), name="control_shape"
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Reshape", ["runtime_input", "control_shape"], ["reshaped"]
            ),
            helper.make_node("MatMul", ["reshaped", "weight"], ["output"]),
        ],
        "whole_graph_precision",
        [runtime_input],
        [output],
        [weight, control_shape],
    )
    onnx.save(helper.make_model(graph), path)
    return path


def test_magia_v3_is_distinct_and_specializes_runtime_floats_to_fp16(
    tmp_path: Path,
) -> None:
    imported = import_onnx_model(
        _write_float_model_with_integer_control(tmp_path / "precision.onnx")
    )

    result = magia_v3.specialize(imported, magia_v3.build_mesh(width=1, height=1))

    assert magia.SPATZ_DEVICE.vlen_bits == 512
    assert magia_v3.SPATZ_DEVICE.vlen_bits == 256
    assert magia.L2_SIZE_BYTES != magia_v3.L2_SIZE_BYTES
    assert [tensor.dtype for tensor in result.model.graph.inputs] == [
        TensorDType.FLOAT16
    ]
    assert [tensor.dtype for tensor in result.model.graph.outputs] == [
        TensorDType.FLOAT16
    ]
    assert result.model.constants.get("weight").dtype is TensorDType.FLOAT16
    reshape = result.model.graph.nodes[0]
    assert reshape.outputs[0].dims == (2, 2)
    assert not any(
        isinstance(node.payload, CastPayload) for node in result.model.graph.nodes
    )
    assert [event.rewrite_name for event in result.report.events] == [
        "whole_graph_precision_specialization"
    ]
    signature = WorkSignature(
        WorkKind.GEMM,
        (TensorDType.FLOAT16, TensorDType.FLOAT16),
        (TensorDType.FLOAT16,),
    )
    capable = [
        device.name
        for device in magia_v3.TILE_DEVICES
        if signature in device.capabilities
    ]
    assert capable == ["redmule"]
    assert magia_v3.build_mesh(width=1, height=1).tiles[0].assigned_device(
        signature
    ) is magia_v3.REDMULE_DEVICE


def test_magia_v3_identity_travels_through_the_ordinary_workflow(
    tmp_path: Path,
    capsys,
) -> None:
    model = _write_float_model_with_integer_control(tmp_path / "baseline.onnx")
    plan_path = tmp_path / "baseline.plan.json"
    application = tmp_path / "baseline"

    assert main(
        [
            "plan",
            str(model),
            "--target",
            "magia-v3",
            "--mesh",
            "2x1",
            "--token-slots",
            "1",
            "--output",
            str(plan_path),
        ]
    ) == 0
    assert '"target": "magia-v3"' in plan_path.read_text()
    capsys.readouterr()

    assert main(
        [
            "build",
            str(model),
            "--target",
            "magia-v3",
            "--mesh",
            "2x1",
            "--token-slots",
            "1",
            "--output",
            str(application),
        ]
    ) == 0
    capsys.readouterr()
    manifest = validate_application(application)
    assert manifest["application"]["target"] == "magia-v3"
    assert manifest["execution"] == {"token_slots": 1, "tokens": 1}
    assert manifest["tensors"]["inputs"][0]["dtype"] == "float16"
    assert manifest["tensors"]["outputs"][0]["dtype"] == "float16"
    assert manifest["abi"] == {
        "descriptor": 1,
        "kernel": 1,
        "operation": 1,
        "task_bundle": 1,
    }
    assert manifest["memory"]["initializers_region"] == "l2_bulk"
    assert manifest["memory"]["runtime_region"] == "l2_arena"
    assert manifest["provenance"]["rewrite_report"][0]["rewrite_name"] == (
        "whole_graph_precision_specialization"
    )
    generated_data = (application / "src/baseline.c").read_text()
    assert 'section(".l2_arena.maps_application")' in generated_data
    initializers = (application / "src/baseline_initializers.S.in").read_text()
    assert ".l2_bulk.maps_initializers" in initializers
    tile_sources = "\n".join(
        path.read_text() for path in (application / "src/tiles").glob("*.c")
    )
    assert "FIFO and planned L1 data overlap task scratch" in tile_sources
    assert "task scratch overlaps ready state" in tile_sources

    assert main(["inspect", str(application)]) == 0
    inspection = capsys.readouterr().out
    assert "Target: magia-v3" in inspection
    assert "Kernel ABI: 1" in inspection
    assert "Task bundle: 1" in inspection
    assert "Packed Initializers: 8 bytes" in inspection
    assert "Required L2:" in inspection
    assert "Maximum tile L1:" in inspection
    assert "Initializer region: l2_bulk" in inspection
    assert "Runtime region: l2_arena" in inspection
    assert main(["verify", str(application)]) == 0


def test_generated_magia_v3_application_runs_in_configured_sdk(
    tmp_path: Path,
) -> None:
    sdk_setting = os.environ.get("MAGIA_V3_SDK_ROOT")
    if sdk_setting is None:
        pytest.skip(
            "set MAGIA_V3_SDK_ROOT to enable the MAGIA-v3 GVSoC acceptance check"
        )
    sdk = Path(sdk_setting).resolve()
    gvrun = sdk / "gvsoc/install/bin/gvrun"
    bootrom = sdk / "build/bin/bootrom/spatz_init.bin"
    if not gvrun.is_file():
        pytest.skip("the configured MAGIA-v3 SDK has no built GVSoC target")
    if shutil.which("riscv32-unknown-elf-gcc") is None:
        pytest.skip("the MAGIA SDK GCC toolchain is unavailable")
    if not bootrom.is_file():
        bootrom = tmp_path / "unused_spatz_bootrom.bin"
        bootrom.write_bytes(bytes(256))

    model = _write_float_model_with_integer_control(tmp_path / "e2e.onnx")
    application = tmp_path / "maps_v3_e2e"
    assert main(
        [
            "build",
            str(model),
            "--target",
            "magia-v3",
            "--mesh",
            "2x2",
            "--token-slots",
            "1",
            "--name",
            "maps_v3_e2e",
            "--output",
            str(application),
        ]
    ) == 0

    build = tmp_path / "sdk-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(sdk),
            "-B",
            str(build),
            "-DTARGET_PLATFORM=magia_v3",
            "-DTILES=2",
            "-DPULP_CORE_COUNT=8",
            "-DCOMPILER=GCC_PULP",
            "-DUSE_CCACHE=OFF",
            f"-DMAPS_APPLICATION_DIR={application}",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "maps_v3_e2e", "-j", "8"],
        check=True,
    )
    executable = build / "bin/maps_v3_e2e"
    sections = subprocess.run(
        ["riscv32-unknown-elf-readelf", "-S", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert re.search(r"\.l2_bulk\s+PROGBITS\s+c0000000", sections)
    assert re.search(r"\.l2_arena\s+NOBITS\s+cc020000", sections)
    assert re.search(r"\.vectors\s+PROGBITS\s+cc000000", sections)

    simulation = subprocess.run(
        [
            str(gvrun),
            "--target",
            "magia_v3",
            "--param",
            f"binary={executable}",
            "--work-dir",
            str(tmp_path / "gvsoc-work"),
            "--attr",
            "magia_v3/n_tiles_x=2",
            "--attr",
            "magia_v3/n_tiles_y=2",
            "--attr",
            "magia_v3/nb_pulp_cores=8",
            "--attr",
            f"magia_v3/spatz_romfile={bootrom}",
            "run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "maps_v3_e2e token 0 output output checksum: af8d639d" in (
        simulation.stdout + simulation.stderr
    )
