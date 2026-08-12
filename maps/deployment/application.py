"""High-level construction of relocatable MAGIA Applications."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Callable

from maps.target import magia

from .bundle import write_deployment_bundle
from .workflow import build_magia_deployment_bundle


MAGIA_V2_TARGET = "magia-v2"


def normalize_application_name(value: str) -> str:
    """Return the backend's deterministic snake-case application identity."""

    normalized: list[str] = []
    separator = False
    previous_was_lower_or_digit = False
    for character in value:
        if not character.isascii() or not character.isalnum():
            separator = bool(normalized)
            previous_was_lower_or_digit = False
            continue
        uppercase = character.isupper()
        if (
            normalized
            and (separator or (uppercase and previous_was_lower_or_digit))
            and normalized[-1] != "_"
        ):
            normalized.append("_")
        normalized.append(character.lower())
        separator = False
        previous_was_lower_or_digit = character.islower() or character.isdigit()
    while normalized and normalized[-1] == "_":
        normalized.pop()
    name = "".join(normalized) or "application"
    if name[0].isdigit():
        name = f"application_{name}"
    return name


def _default_compiler_tool(name: str) -> Path:
    executable = shutil.which(name)
    if executable:
        return Path(executable)
    return (
        Path(__file__).resolve().parents[2]
        / "maps-ir"
        / "build"
        / "tools"
        / "maps-translate"
        / name
    )


def _run_backend(arguments: list[str]) -> None:
    try:
        subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("MAGIA Application generation failed") from exc


def _application_manifest(application: Path) -> dict[str, Any]:
    manifest_path = application / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("generated application has no manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("generated application manifest is unreadable or invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("generated application manifest must be an object")
    return manifest


def _validate_generated_application(
    application: Path,
    *,
    name: str,
    mesh_width: int,
    mesh_height: int,
    num_token_slots: int,
) -> dict[str, Any]:
    manifest = _application_manifest(application)
    if manifest.get("application") != {"name": name, "target": MAGIA_V2_TARGET}:
        raise ValueError("generated application identity does not match the build")
    if manifest.get("planned_mesh") != {
        "width": mesh_width,
        "height": mesh_height,
    }:
        raise ValueError("generated application Mesh does not match the build")
    execution = manifest.get("execution")
    active_tiles = manifest.get("active_physical_tiles")
    abi = manifest.get("abi")
    tensors = manifest.get("tensors")
    entry_points = manifest.get("entry_points")
    memory = manifest.get("memory")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("source_model"), str)
        or not isinstance(execution, dict)
        or execution.get("tokens") != 1
        or execution.get("token_slots") != num_token_slots
        or not isinstance(active_tiles, list)
        or not isinstance(abi, dict)
        or not all(
            isinstance(abi.get(key), int) for key in ("operation", "descriptor")
        )
        or not isinstance(tensors, dict)
        or not all(
            isinstance(tensors.get(key), list) for key in ("inputs", "outputs")
        )
        or not isinstance(entry_points, dict)
        or entry_points.get("run") != f"{name}_run"
        or entry_points.get("handle_input") != f"{name}_handle_input"
        or entry_points.get("handle_output") != f"{name}_handle_output"
        or not isinstance(memory, dict)
        or not all(
            isinstance(memory.get(key), int)
            for key in (
                "initializers_bytes",
                "required_l2_bytes",
                "max_tile_l1_bytes",
            )
        )
    ):
        raise ValueError("generated application manifest contract is incomplete")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("generated application file manifest is invalid")
    for ownership in ("generated", "user_owned"):
        paths = files.get(ownership)
        if not isinstance(paths, list) or not paths:
            raise ValueError("generated application file manifest is invalid")
        for value in paths:
            if not isinstance(value, str) or not value:
                raise ValueError("generated application file path is invalid")
            relative = PurePosixPath(value)
            if (
                relative.is_absolute()
                or "." in relative.parts
                or ".." in relative.parts
            ):
                raise ValueError(f"unsafe generated application path '{value}'")
            artifact = application / Path(*relative.parts)
            if not artifact.is_file() or artifact.is_symlink():
                raise ValueError(f"generated application is missing '{value}'")
    required_generated = {
        "CMakeLists.txt",
        "README.md",
        "manifest.json",
        f"include/{name}.h",
        f"src/{name}.c",
        f"src/{name}_runner.c",
        f"src/{name}_initializers.S.in",
        f"data/{name}.initializers.bin",
    }
    if not required_generated.issubset(set(files["generated"])):
        raise ValueError("generated application file contract is incomplete")
    if "src/application.c" not in files["user_owned"]:
        raise ValueError("generated application customization source is missing")
    return manifest


def build_application(
    model_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    name: str | None = None,
    target: str = MAGIA_V2_TARGET,
    mesh_width: int = magia.MESH_WIDTH,
    mesh_height: int = magia.MESH_HEIGHT,
    num_token_slots: int = 2,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Build, validate, and atomically publish one MAGIA Application."""

    if target != MAGIA_V2_TARGET:
        raise ValueError(f"unsupported application Target '{target}'")
    model = Path(model_path)
    if not model.is_file():
        raise ValueError(f"source model does not exist: {model}")
    application_name = normalize_application_name(name or model.stem)
    output = Path(output_dir) if output_dir is not None else Path("build") / application_name
    if output.exists():
        raise FileExistsError(f"application output already exists: {output}")

    plan_import = _default_compiler_tool("maps-plan-import")
    codegen = _default_compiler_tool("maps-codegen")
    for tool in (plan_import, codegen):
        if not tool.is_file():
            raise ValueError("MAGIA Application compiler is unavailable")

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    report(
        f"Planning {model.name} for {target} on a "
        f"{mesh_width}x{mesh_height} Mesh..."
    )
    bundle = build_magia_deployment_bundle(
        model,
        mesh_width=mesh_width,
        mesh_height=mesh_height,
        num_token_slots=num_token_slots,
        progress=None,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        intermediates = staging_parent / "intermediates"
        application = staging_parent / "application"
        deployment_bundle = intermediates / "deployment-bundle.json"
        initializers = intermediates / "initializers.bin"
        maps_mlir = intermediates / "execution-plan.mlir"
        write_deployment_bundle(bundle, deployment_bundle, initializers)
        report("Generating the MAGIA Application...")
        _run_backend([str(plan_import), str(deployment_bundle), "-o", str(maps_mlir)])
        _run_backend(
            [
                str(codegen),
                str(maps_mlir),
                "-o",
                str(application),
                f"--target={target}",
                f"--maps-magia-output-stem={application_name}",
                f"--maps-magia-weights-file={initializers}",
            ]
        )
        _validate_generated_application(
            application,
            name=application_name,
            mesh_width=mesh_width,
            mesh_height=mesh_height,
            num_token_slots=num_token_slots,
        )
        if output.exists():
            raise FileExistsError(f"application output already exists: {output}")
        os.replace(application, output)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
    return output


def application_build_summary(application: str | Path) -> str:
    """Return the concise successful-build handoff for an application."""

    path = Path(application)
    manifest = _application_manifest(path)
    identity = manifest["application"]
    mesh = manifest["planned_mesh"]
    execution = manifest["execution"]
    active_tiles = manifest["active_physical_tiles"]
    return "\n".join(
        (
            f"MAGIA Application: {path}",
            f"Target: {identity['target']}",
            f"Mesh: {mesh['width']}x{mesh['height']}",
            f"Execution Tokens: {execution['tokens']}",
            f"Active tiles: {len(active_tiles)}",
            "SDK handoff: copy this directory into the MAGIA SDK application tree "
            f"and register it with add_subdirectory({identity['name']}).",
        )
    )


__all__ = [
    "MAGIA_V2_TARGET",
    "application_build_summary",
    "build_application",
    "normalize_application_name",
]
