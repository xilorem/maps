from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import onnx
from onnx import TensorProto, helper
import pytest

from maps.cli import main
from maps.deployment import validate_application
from maps.deployment.bundle import write_deployment_bundle
from maps.deployment.workflow import build_magia_deployment_bundle


def _write_representative_model(path: Path) -> Path:
    lhs = helper.make_tensor_value_info("lhs/value", TensorProto.FLOAT, [2, 2])
    rhs = helper.make_tensor_value_info("rhs value", TensorProto.FLOAT, [2, 2])
    total = helper.make_tensor_value_info("sum/value", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("output-value", TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["lhs/value", "rhs value"], ["sum/value"]),
            helper.make_node("Relu", ["sum/value"], ["output-value"]),
        ],
        "application_handoff",
        [lhs, rhs],
        [output],
    )
    onnx.save(helper.make_model(graph), path)
    return path


def _generated_files(application: Path) -> dict[str, bytes]:
    manifest = validate_application(application)
    return {
        record["path"]: (application / record["path"]).read_bytes()
        for record in manifest["files"]["generated"]
    }


def _compiler_tool(repository: Path, name: str) -> Path:
    return repository / "maps-ir" / "build" / "tools" / "maps-translate" / name


def _build_ordinary_application(
    model: Path,
    output: Path,
    lhs: Path,
    rhs: Path,
) -> int:
    return main(
        [
            "build",
            str(model),
            "--name",
            "Complete Handoff",
            "--mesh",
            "4x4",
            "--token-slots",
            "3",
            "--input",
            f"lhs/value={lhs}",
            "--input",
            f"rhs value={rhs}",
            "--output",
            str(output),
        ]
    )


def test_complete_application_handoff_through_ordinary_and_expert_workflows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).parents[2]
    model = _write_representative_model(tmp_path / "Handoff Model.onnx")
    lhs = tmp_path / "lhs.raw"
    rhs = tmp_path / "rhs.raw"
    lhs.write_bytes(bytes(range(48)))
    rhs.write_bytes(bytes(reversed(range(48))))
    ordinary = tmp_path / "ordinary"

    assert _build_ordinary_application(model, ordinary, lhs, rhs) == 0
    build_output = capsys.readouterr().out
    assert "MAGIA Application:" in build_output
    assert "Execution Tokens: 3" in build_output
    assert "add_subdirectory(complete_handoff)" in build_output

    assert main(["inspect", str(ordinary)]) == 0
    inspection = capsys.readouterr().out
    assert "MAGIA Application: complete_handoff" in inspection
    assert "Execution Tokens: 3" in inspection
    assert "Token Slots: 3" in inspection
    assert "Runtime Input: lhs/value (lhs_value)" in inspection
    assert "Graph output: output-value (output_value)" in inspection
    assert main(["verify", str(ordinary)]) == 0
    assert "Valid MAGIA Application:" in capsys.readouterr().out

    manifest = validate_application(ordinary)
    active_tiles = manifest["active_physical_tiles"]
    assert active_tiles
    assert sorted(path.name for path in (ordinary / "src/tiles").glob("*.c")) == [
        f"tile_{tile_id:02d}.c" for tile_id in active_tiles
    ]
    assert manifest["memory"]["required_l2_bytes"] > 0
    assert (ordinary / "data/complete_handoff.initializers.bin").is_file()
    assert (
        ordinary / "data/complete_handoff.input_lhs_value.bin"
    ).read_bytes() == lhs.read_bytes()
    assert (
        ordinary / "data/complete_handoff.input_rhs_value.bin"
    ).read_bytes() == rhs.read_bytes()
    assert manifest["files"]["user_owned"] == [
        {"path": "src/application.c", "role": "application_source"}
    ]
    public_header = (ordinary / "include/complete_handoff.h").read_text()
    assert "complete_handoff_run" in public_header
    assert "slice_desc_t" not in public_header

    readme = (ordinary / "README.md").read_text()
    cmake = (ordinary / "CMakeLists.txt").read_text()
    assert "add_subdirectory(complete_handoff)" in readme
    assert "cmake --build build --target complete_handoff" in readme
    assert re.search(r"add_executable\(complete_handoff(?:\s|$)", cmake)
    assert "src/application.c" in readme

    bundle = build_magia_deployment_bundle(
        model,
        mesh_width=4,
        mesh_height=4,
        num_token_slots=3,
        progress=None,
    )
    serialized_bundle = tmp_path / "expert.bundle.json"
    initializers = tmp_path / "expert.initializers.bin"
    maps_mlir = tmp_path / "expert.mlir"
    expert = tmp_path / "expert"
    write_deployment_bundle(bundle, serialized_bundle, initializers)
    subprocess.run(
        [
            str(_compiler_tool(repository, "maps-plan-import")),
            str(serialized_bundle),
            "-o",
            str(maps_mlir),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(_compiler_tool(repository, "maps-codegen")),
            str(maps_mlir),
            "--target=magia-v2",
            "--maps-magia-output-stem=Complete Handoff",
            f"--maps-magia-weights-file={initializers}",
            "--maps-magia-num-tokens=3",
            f"--maps-magia-runtime-input=lhs/value={lhs}",
            f"--maps-magia-runtime-input=rhs value={rhs}",
            "-o",
            str(expert),
        ],
        check=True,
    )
    assert validate_application(expert) == manifest
    assert _generated_files(expert) == _generated_files(ordinary)

    customization = b"/* developer-owned handoff customization */\n"
    (ordinary / "src/application.c").write_bytes(customization)
    assert _build_ordinary_application(model, ordinary, lhs, rhs) == 0
    capsys.readouterr()
    assert (ordinary / "src/application.c").read_bytes() == customization
    assert _generated_files(ordinary) == _generated_files(expert)


def test_generated_application_compiles_in_configured_magia_sdk(
    tmp_path: Path,
) -> None:
    sdk_setting = os.environ.get("MAGIA_SDK_ROOT")
    if sdk_setting is None:
        pytest.skip("set MAGIA_SDK_ROOT to enable the external SDK compilation check")
    sdk_source = Path(sdk_setting).resolve()
    if shutil.which("riscv32-unknown-elf-gcc") is None:
        pytest.skip("the MAGIA SDK GCC toolchain is unavailable")

    repository = Path(__file__).parents[2]
    model = repository / "examples/simple_three_stage.onnx"
    application = tmp_path / "sdk-source/tests/magia/mesh/maps_handoff"
    sdk_copy = tmp_path / "sdk-source"
    shutil.copytree(
        sdk_source,
        sdk_copy,
        ignore=shutil.ignore_patterns(
            ".git", ".ccache", "build", "gvsoc", "gvsoc_venv", "gvsoc_work"
        ),
    )
    assert main(
        [
            "build",
            str(model),
            "--name",
            "Maps Handoff",
            "--mesh",
            "4x4",
            "--output",
            str(application),
        ]
    ) == 0

    registration = sdk_copy / "tests/magia/mesh/CMakeLists.txt"
    registration.write_text(
        registration.read_text() + "\nadd_subdirectory(maps_handoff)\n"
    )
    build = tmp_path / "sdk-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(sdk_copy),
            "-B",
            str(build),
            "-DTARGET_PLATFORM=magia_v2",
            "-DTILES=4",
            "-DCOMPILER=GCC_PULP",
            "-DSPATZ_TESTS=0",
            "-DUSE_CCACHE=OFF",
            "-DEVAL=0",
            "-DSTALLING=0",
            "-DFSYNC_MM=1",
            "-DIDMA_MM=1",
            "-DREDMULE_MM=1",
            "-DPROFILE_CMP=0",
            "-DPROFILE_CMI=0",
            "-DPROFILE_CMO=0",
            "-DPROFILE_SNC=0",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "maps_handoff", "-j2"],
        check=True,
    )
    assert (build / "bin/maps_handoff").is_file()
