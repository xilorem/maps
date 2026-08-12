from __future__ import annotations

import json
from pathlib import Path
import subprocess

import onnx
from onnx import TensorProto, helper
import pytest

from maps.cli import main
from maps.deployment import build_application
import maps.deployment.application as application_module


def _write_two_input_model(path: Path) -> Path:
    lhs = helper.make_tensor_value_info("lhs/value", TensorProto.FLOAT, [2, 2])
    rhs = helper.make_tensor_value_info("rhs value", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph(
        [helper.make_node("Add", ["lhs/value", "rhs value"], ["sum"])],
        "runtime_inputs",
        [lhs, rhs],
        [output],
    )
    onnx.save(helper.make_model(graph), path)
    return path


def test_build_application_publishes_a_named_magia_application(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    output = tmp_path / "chosen-location"

    application = build_application(
        repository / "examples" / "simple_three_stage.onnx",
        output,
        name="Three Stage Demo",
        mesh_width=4,
        mesh_height=4,
    )

    assert application == output
    manifest = json.loads((application / "manifest.json").read_text())
    assert manifest["application"] == {
        "name": "three_stage_demo",
        "target": "magia-v2",
    }
    assert manifest["planned_mesh"] == {"width": 4, "height": 4}
    assert (application / "include" / "three_stage_demo.h").is_file()
    assert "three_stage_demo_run" in (
        application / "include" / "three_stage_demo.h"
    ).read_text()
    assert list(application.rglob("*.json")) == [application / "manifest.json"]
    assert not list(application.rglob("*.mlir"))
    assert not list(tmp_path.glob(".chosen-location.staging-*"))
    tile_sources = sorted((application / "src" / "tiles").glob("tile_*.c"))
    assert [source.name for source in tile_sources] == [
        f"tile_{tile_id:02d}.c"
        for tile_id in manifest["active_physical_tiles"]
    ]
    tile_text = "\n".join(source.read_text() for source in tile_sources)
    assert "static const slice_desc_t" in tile_text
    assert "static const op_desc_t" in tile_text
    assert ".num_token_slots = THREE_STAGE_DEMO_NUM_TOKEN_SLOTS" in tile_text


def test_build_application_derives_tokens_and_embeds_supplied_runtime_input(
    tmp_path: Path,
) -> None:
    model = _write_two_input_model(tmp_path / "runtime-inputs.onnx")
    supplied = tmp_path / "tokens.raw"
    supplied.write_bytes(bytes(range(32)))

    application = build_application(
        model,
        tmp_path / "application",
        mesh_width=4,
        mesh_height=4,
        num_token_slots=3,
        inputs={"lhs/value": supplied},
    )

    manifest = json.loads((application / "manifest.json").read_text())
    assert manifest["execution"] == {"tokens": 2, "token_slots": 3}
    inputs = {
        entry["original_name"]: entry
        for entry in manifest["tensors"]["inputs"]
    }
    assert inputs["lhs/value"]["data"] == {
        "kind": "supplied",
        "path": "data/runtime_inputs.input_lhs_value.bin",
    }
    assert inputs["rhs value"]["data"] == {"kind": "synthetic"}
    embedded = application / inputs["lhs/value"]["data"]["path"]
    assert embedded.read_bytes() == supplied.read_bytes()
    assert (
        str(embedded.relative_to(application))
        in manifest["files"]["generated"]
    )

    assembly = (application / "src/runtime_inputs_initializers.S.in").read_text()
    assert (
        '.incbin "@CMAKE_CURRENT_SOURCE_DIR@/'
        'data/runtime_inputs.input_lhs_value.bin"'
        in assembly
    )
    assert ".global runtime_inputs_input_lhs_value_start" in assembly
    customization = (application / "src/application.c").read_text()
    assert "runtime_inputs_input_lhs_value_start" in customization
    assert "lhs_value[index] = lhs_value_data[index + token * 16u]" in customization
    assert "rhs_value[index] = (uint8_t)(index + token * 17u" in customization
    runner = (application / "src/runtime_inputs_runner.c").read_text()
    assert (
        "for (uint32_t token = 0u; "
        "token < RUNTIME_INPUTS_EXECUTION_TOKENS; ++token)"
        in runner
    )
    assert (
        "maps_fifo_run_tile_tokens(\n"
        "        &plan, RUNTIME_INPUTS_EXECUTION_TOKENS"
        in runner
    )
    assert runner.count("for (uint32_t token = 0u;") == 2


def test_build_command_accepts_repeated_runtime_inputs(
    tmp_path: Path,
) -> None:
    model = _write_two_input_model(tmp_path / "runtime-inputs.onnx")
    lhs = tmp_path / "lhs.raw"
    rhs = tmp_path / "rhs.raw"
    lhs.write_bytes(bytes(range(32)))
    rhs.write_bytes(bytes(reversed(range(32))))
    output = tmp_path / "application"

    assert main(
        [
            "build",
            str(model),
            "--mesh",
            "4x4",
            "--output",
            str(output),
            "--input",
            f"lhs/value={lhs}",
            "--input",
            f"rhs value={rhs}",
        ]
    ) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["execution"]["tokens"] == 2
    assert all(
        entry["data"]["kind"] == "supplied"
        for entry in manifest["tensors"]["inputs"]
    )


@pytest.mark.parametrize("data", (b"", bytes(15)))
def test_build_application_rejects_incomplete_runtime_input_values(
    tmp_path: Path,
    data: bytes,
) -> None:
    model = _write_two_input_model(tmp_path / "runtime-inputs.onnx")
    supplied = tmp_path / "invalid.raw"
    supplied.write_bytes(data)
    output = tmp_path / "application"

    with pytest.raises(ValueError, match="positive whole number"):
        build_application(
            model,
            output,
            mesh_width=4,
            mesh_height=4,
            inputs={"lhs/value": supplied},
        )

    assert not output.exists()


def test_build_application_rejects_mismatched_runtime_input_token_counts(
    tmp_path: Path,
) -> None:
    model = _write_two_input_model(tmp_path / "runtime-inputs.onnx")
    lhs = tmp_path / "lhs.raw"
    rhs = tmp_path / "rhs.raw"
    lhs.write_bytes(bytes(32))
    rhs.write_bytes(bytes(48))
    output = tmp_path / "application"

    with pytest.raises(ValueError, match="Execution Token counts do not match"):
        build_application(
            model,
            output,
            mesh_width=4,
            mesh_height=4,
            inputs={"lhs/value": lhs, "rhs value": rhs},
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("assignments", "message"),
    (
        ((("unknown", "input.raw"),), "unknown Runtime Input"),
        (
            (("lhs/value", "input.raw"), ("lhs/value", "input.raw")),
            "duplicate Runtime Input assignment",
        ),
    ),
)
def test_build_application_rejects_unknown_and_duplicate_runtime_inputs(
    tmp_path: Path,
    assignments: tuple[tuple[str, str], ...],
    message: str,
) -> None:
    model = _write_two_input_model(tmp_path / "runtime-inputs.onnx")
    supplied = tmp_path / "input.raw"
    supplied.write_bytes(bytes(16))
    resolved = tuple(
        (name, supplied if path == "input.raw" else path)
        for name, path in assignments
    )

    with pytest.raises(ValueError, match=message):
        build_application(
            model,
            tmp_path / "application",
            mesh_width=4,
            mesh_height=4,
            inputs=resolved,
        )


def test_build_application_without_input_files_emits_one_synthetic_token(
    tmp_path: Path,
) -> None:
    model = _write_two_input_model(tmp_path / "runtime-inputs.onnx")

    application = build_application(
        model,
        tmp_path / "application",
        mesh_width=4,
        mesh_height=4,
    )

    manifest = json.loads((application / "manifest.json").read_text())
    assert manifest["execution"]["tokens"] == 1
    assert all(
        entry["data"] == {"kind": "synthetic"}
        for entry in manifest["tensors"]["inputs"]
    )
    assert not list((application / "data").glob("*.input_*.bin"))


def test_build_command_uses_the_default_output_and_reports_sdk_handoff(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repository = Path(__file__).parents[2]
    monkeypatch.chdir(tmp_path)

    assert main(
        [
            "build",
            str(repository / "examples" / "simple_three_stage.onnx"),
            "--name",
            "Three Stage CLI",
            "--mesh",
            "4x4",
        ]
    ) == 0

    application = tmp_path / "build" / "three_stage_cli"
    manifest = json.loads((application / "manifest.json").read_text())
    assert manifest["application"]["name"] == "three_stage_cli"
    output = capsys.readouterr().out
    assert f"MAGIA Application: {Path('build/three_stage_cli')}" in output
    assert "Target: magia-v2" in output
    assert "Mesh: 4x4" in output
    assert "Execution Tokens: 1" in output
    assert "Active tiles:" in output
    assert "add_subdirectory(three_stage_cli)" in output
    assert "Deployment Bundle" not in output
    assert "maps MLIR" not in output
    assert "maps-plan-import" not in output
    assert "maps-codegen" not in output


def test_build_command_output_override_preserves_the_derived_identity(
    tmp_path: Path,
    capsys,
) -> None:
    repository = Path(__file__).parents[2]
    output = tmp_path / "unrelated-directory-name"

    assert main(
        [
            "build",
            str(repository / "examples" / "simple_three_stage.onnx"),
            "--output",
            str(output),
            "--mesh",
            "4x4",
        ]
    ) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["application"]["name"] == "simple_three_stage"
    assert (output / "include" / "simple_three_stage.h").is_file()
    assert f"MAGIA Application: {output}" in capsys.readouterr().out


def test_build_application_leaves_no_partial_output_after_backend_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = Path(__file__).parents[2]
    output = tmp_path / "failed-application"

    def fail_backend(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="backend failed")

    monkeypatch.setattr(application_module.subprocess, "run", fail_backend)

    with pytest.raises(RuntimeError, match="MAGIA Application generation failed"):
        build_application(
            repository / "examples" / "simple_three_stage.onnx",
            output,
            mesh_width=4,
            mesh_height=4,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".failed-application.staging-*"))


def test_build_application_leaves_no_partial_output_after_validation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = Path(__file__).parents[2]
    output = tmp_path / "invalid-application"

    def emit_invalid_application(arguments, **kwargs):
        destination = Path(arguments[arguments.index("-o") + 1])
        if Path(arguments[0]).name == "maps-plan-import":
            destination.write_text("module {}")
        else:
            destination.mkdir()
            (destination / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "application": {
                            "name": "simple_three_stage",
                            "target": "magia-v2",
                        },
                        "planned_mesh": {"width": 4, "height": 4},
                        "source_model": "simple_three_stage",
                        "abi": {"operation": 1, "descriptor": 1},
                        "execution": {"tokens": 1, "token_slots": 2},
                        "active_physical_tiles": [],
                        "tensors": {"inputs": [], "outputs": []},
                        "entry_points": {
                            "run": "simple_three_stage_run",
                            "handle_input": "simple_three_stage_handle_input",
                            "handle_output": "simple_three_stage_handle_output",
                        },
                        "memory": {
                            "initializers_bytes": 0,
                            "required_l2_bytes": 0,
                            "max_tile_l1_bytes": 0,
                        },
                        "files": {
                            "generated": [
                                "CMakeLists.txt",
                                "README.md",
                                "manifest.json",
                                "include/simple_three_stage.h",
                                "src/simple_three_stage.c",
                                "src/simple_three_stage_runner.c",
                                "src/simple_three_stage_initializers.S.in",
                                "data/simple_three_stage.initializers.bin",
                            ],
                            "user_owned": ["src/application.c"],
                        },
                    }
                )
            )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(
        application_module.subprocess,
        "run",
        emit_invalid_application,
    )

    with pytest.raises(ValueError, match="missing 'CMakeLists.txt'"):
        build_application(
            repository / "examples" / "simple_three_stage.onnx",
            output,
            mesh_width=4,
            mesh_height=4,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".invalid-application.staging-*"))


def test_build_application_hides_missing_compiler_tool_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = Path(__file__).parents[2]
    monkeypatch.setattr(
        application_module,
        "_default_compiler_tool",
        lambda name: tmp_path / name,
    )

    with pytest.raises(ValueError) as failure:
        build_application(
            repository / "examples" / "simple_three_stage.onnx",
            tmp_path / "application",
            mesh_width=4,
            mesh_height=4,
        )

    assert str(failure.value) == "MAGIA Application compiler is unavailable"
    assert "maps-plan-import" not in str(failure.value)
    assert "maps-codegen" not in str(failure.value)


def test_build_application_hides_compiler_launch_failure_details(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = Path(__file__).parents[2]

    def fail_launch(*args, **kwargs):
        raise PermissionError("maps-plan-import in a temporary directory")

    monkeypatch.setattr(application_module.subprocess, "run", fail_launch)

    with pytest.raises(RuntimeError) as failure:
        build_application(
            repository / "examples" / "simple_three_stage.onnx",
            tmp_path / "application",
            mesh_width=4,
            mesh_height=4,
        )

    assert str(failure.value) == "MAGIA Application generation failed"
    assert "maps-plan-import" not in str(failure.value)
    assert "temporary" not in str(failure.value)
