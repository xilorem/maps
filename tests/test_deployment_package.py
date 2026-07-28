from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

from MAPS.cli import main
import MAPS.cli as cli_module
from MAPS.deployment import (
    validate_deployment_package,
    write_deployment_package,
)
import MAPS.deployment.package as package_module


def _write_package(path: Path) -> Path:
    path.mkdir()
    artifacts = (
        ("model.h", "header", b"header\n"),
        ("model_data.c", "data_source", b"data\n"),
        ("model_weights.S", "weights_assembly", b"assembly\n"),
        ("model.weights.bin", "weights_image", b"MAPS"),
    )
    records = []
    for name, role, data in artifacts:
        (path / name).write_bytes(data)
        records.append({
            "path": name,
            "role": role,
            "byte_size": len(data),
            "sha256": sha256(data).hexdigest(),
        })
    manifest = {
        "schema_version": 1,
        "source_model": {
            "name": "fixture",
            "inputs": [{
                "id": 0,
                "name": "input",
                "dtype": "float32",
                "shape": [1, 2],
                "encoding": "raw",
                "endianness": "little",
                "tensor_bytes": 8,
            }],
            "outputs": [{
                "id": 1,
                "name": "output",
                "dtype": "float16",
                "shape": [1, 2],
                "encoding": "raw",
                "endianness": "little",
                "tensor_bytes": 4,
            }],
        },
        "target": {
            "architecture": "magia-v2",
            "mesh": {"width": 2, "height": 3},
        },
        "execution": {
            "num_token_slots": 2,
            "pipeline_token_capacity": 1,
        },
        "abi": {
            "operation_version": 1,
            "descriptor_version": 1,
            "descriptor_max_dimensions": 6,
        },
        "entry_points": {
            "init_l2_data": "maps_generated_init_l2_data",
            "init_tensors": "maps_generated_init_tensors",
            "fill_plan": "maps_generated_fill_plan",
            "check_output": "maps_generated_check_output",
        },
        "memory": {
            "ready_flags_offset": 0xD0000,
            "max_tile_data_bytes": 64,
            "weights_size": 4,
            "required_l2_bytes": 32,
        },
        "artifacts": records,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_validate_deployment_package_is_relocatable(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "first")
    relocated = tmp_path / "relocated"
    package.rename(relocated)

    manifest = validate_deployment_package(relocated)

    assert manifest["source_model"]["name"] == "fixture"
    assert manifest["target"]["mesh"] == {"width": 2, "height": 3}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda package: (package / "model.h").write_text("corrupt"), "checksum mismatch"),
        (lambda package: (package / "extra.txt").write_text("extra"), "undeclared files"),
        (
            lambda package: _rewrite_manifest(
                package,
                lambda manifest: manifest["abi"].update(operation_version=2),
            ),
            "incompatible deployment ABI",
        ),
        (
            lambda package: _rewrite_manifest(
                package,
                lambda manifest: manifest["artifacts"][0].update(path="../model.h"),
            ),
            "unsafe package artifact path",
        ),
    ),
)
def test_validate_deployment_package_rejects_invalid_packages(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    package = _write_package(tmp_path / "package")
    mutation(package)

    with pytest.raises(ValueError, match=message):
        validate_deployment_package(package)


def _rewrite_manifest(path: Path, mutate) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))


def test_write_deployment_package_verifies_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    translator = tmp_path / "maps-translate"
    translator.write_bytes(b"executable")
    output = tmp_path / "published.maps"
    monkeypatch.setattr(package_module, "build_pipeline_bundle", lambda *args, **kwargs: object())

    def fake_write_bundle(bundle, pipeline_json, packed_weights):
        Path(pipeline_json).write_text("{}")
        Path(packed_weights).write_bytes(b"MAPS")

    def fake_run(arguments, check):
        assert check is True
        package_argument = next(
            argument for argument in arguments
            if argument.startswith("--maps-magia-package-dir=")
        )
        _write_package(Path(package_argument.split("=", maxsplit=1)[1]))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(package_module, "write_pipeline_bundle", fake_write_bundle)
    monkeypatch.setattr(package_module.subprocess, "run", fake_run)

    result = write_deployment_package(
        model,
        output,
        mesh_width=2,
        mesh_height=3,
        maps_translate=translator,
    )

    assert result == output
    validate_deployment_package(output)
    assert not list(tmp_path.glob(".published.maps.staging-*"))


def test_write_deployment_package_leaves_no_output_after_backend_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    translator = tmp_path / "maps-translate"
    translator.write_bytes(b"executable")
    output = tmp_path / "failed.maps"
    monkeypatch.setattr(package_module, "build_pipeline_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        package_module,
        "write_pipeline_bundle",
        lambda bundle, pipeline_json, packed_weights: None,
    )

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "maps-translate")

    monkeypatch.setattr(package_module.subprocess, "run", fail)

    with pytest.raises(subprocess.CalledProcessError):
        write_deployment_package(model, output, maps_translate=translator)

    assert not output.exists()
    assert not list(tmp_path.glob(".failed.maps.staging-*"))


def test_package_inspect_and_verify_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = _write_package(tmp_path / "package")

    assert main(["package", "inspect", str(package)]) == 0
    assert "Model: fixture" in capsys.readouterr().out
    assert main(["package", "verify", str(package)]) == 0
    assert "Valid deployment package:" in capsys.readouterr().out


def test_package_build_command_parses_target_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_write(model, output, **options):
        captured.update(model=model, output=output, **options)
        return output

    monkeypatch.setattr(cli_module, "write_deployment_package", fake_write)
    model = tmp_path / "model.onnx"
    output = tmp_path / "model.maps"

    assert main([
        "package",
        str(model),
        "--mesh",
        "2x3",
        "--token-slots",
        "4",
        "--pipeline-token-capacity",
        "2",
        "--output",
        str(output),
    ]) == 0

    assert captured["model"] == model
    assert captured["output"] == output
    assert captured["mesh_width"] == 2
    assert captured["mesh_height"] == 3
    assert captured["num_token_slots"] == 4
    assert captured["pipeline_token_capacity"] == 2
    assert f"Deployment package: {output}" in capsys.readouterr().out
