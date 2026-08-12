"""High-level construction of relocatable MAGIA Applications."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping

from maps.target import magia

from .bundle import DeploymentBundle, write_deployment_bundle
from .application_validation import (
    APPLICATION_SCHEMA_VERSION,
    DESCRIPTOR_ABI_VERSION,
    MAGIA_V2_TARGET,
    OPERATION_ABI_VERSION,
    read_application_manifest,
    validate_application as _validate_application,
)
from .workflow import build_magia_deployment_bundle


_AT_FDCWD = -100
_RENAME_EXCHANGE = 2


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


def validate_application(application: str | Path) -> dict[str, Any]:
    return _validate_application(
        application,
        normalize_name=normalize_application_name,
    )


def _validate_generated_application(
    application: Path,
    *,
    name: str,
    mesh_width: int,
    mesh_height: int,
    num_token_slots: int,
    execution_tokens: int,
) -> dict[str, Any]:
    manifest = validate_application(application)
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
        or execution.get("tokens") != execution_tokens
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
    return manifest


def _prepare_runtime_inputs(
    bundle: DeploymentBundle,
    assignments: Mapping[str, str | Path] | Iterable[tuple[str, str | Path]],
) -> tuple[int, tuple[tuple[str, Path], ...]]:
    items = assignments.items() if isinstance(assignments, Mapping) else assignments
    runtime_inputs = {tensor.name: tensor for tensor in bundle.graph.inputs}
    supplied: list[tuple[str, Path]] = []
    counts: dict[str, int] = {}
    assigned_names: set[str] = set()
    for name, value in items:
        if name not in runtime_inputs:
            raise ValueError(f"unknown Runtime Input '{name}'")
        if name in assigned_names:
            raise ValueError(f"duplicate Runtime Input assignment '{name}'")
        assigned_names.add(name)
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"Runtime Input file does not exist: {path}")
        tensor = runtime_inputs[name]
        tensor_bytes = tensor.num_elements * tensor.elem_bytes
        file_bytes = path.stat().st_size
        if file_bytes == 0 or file_bytes % tensor_bytes:
            raise ValueError(
                f"Runtime Input '{name}' must contain a positive whole number "
                f"of {tensor_bytes}-byte Tensor values"
            )
        counts[name] = file_bytes // tensor_bytes
        supplied.append((name, path))
    token_counts = set(counts.values())
    if len(token_counts) > 1:
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise ValueError(
            f"Runtime Input Execution Token counts do not match: {details}"
        )
    return (next(iter(token_counts), 1), tuple(supplied))


def _generated_file_paths(manifest: dict[str, Any]) -> tuple[Path, ...]:
    return tuple(
        Path(*record["path"].split("/"))
        for record in manifest["files"]["generated"]
    )


def _prepare_regenerated_application(
    previous: Path,
    previous_manifest: dict[str, Any],
    generated: Path,
    generated_manifest: dict[str, Any],
    destination: Path,
) -> None:
    shutil.copytree(previous, destination, symlinks=True)
    for relative in _generated_file_paths(previous_manifest):
        (destination / relative).unlink()

    for relative in _generated_file_paths(generated_manifest):
        target = destination / relative
        parent = destination
        for part in relative.parts[:-1]:
            parent /= part
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(
                    f"generated file conflicts with developer path '{relative}'"
                )
            parent.mkdir(exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError(
                f"generated file conflicts with developer path '{relative}'"
            )
        shutil.copy2(generated / relative, target)
    shutil.copy2(generated / "manifest.json", destination / "manifest.json")


def _publish_application(
    candidate: Path,
    output: Path,
    *,
    replace_existing: bool,
) -> None:
    if not replace_existing:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"application output already exists: {output}")
        os.replace(candidate, output)
        return

    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        _AT_FDCWD,
        os.fsencode(candidate),
        _AT_FDCWD,
        os.fsencode(output),
        _RENAME_EXCHANGE,
    ) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), output)


def build_application(
    model_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    name: str | None = None,
    target: str = MAGIA_V2_TARGET,
    mesh_width: int = magia.MESH_WIDTH,
    mesh_height: int = magia.MESH_HEIGHT,
    num_token_slots: int = 2,
    inputs: Mapping[str, str | Path] | Iterable[tuple[str, str | Path]] = (),
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Build, validate, and atomically publish one MAGIA Application."""

    if target != MAGIA_V2_TARGET:
        raise ValueError(f"unsupported application Target '{target}'")
    model = Path(model_path)
    if not model.is_file():
        raise ValueError(f"source model does not exist: {model}")
    application_name = normalize_application_name(name or model.stem)
    output = (
        Path(output_dir)
        if output_dir is not None
        else Path("build") / application_name
    )
    output_exists = output.exists() or output.is_symlink()
    previous_manifest = validate_application(output) if output_exists else None

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
    execution_tokens, runtime_inputs = _prepare_runtime_inputs(bundle, inputs)

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
        codegen_arguments = [
            str(codegen),
            str(maps_mlir),
            "-o",
            str(application),
            f"--target={target}",
            f"--maps-magia-output-stem={application_name}",
            f"--maps-magia-weights-file={initializers}",
            f"--maps-magia-num-tokens={execution_tokens}",
        ]
        codegen_arguments.extend(
            f"--maps-magia-runtime-input={input_name}={input_path}"
            for input_name, input_path in runtime_inputs
        )
        _run_backend(codegen_arguments)
        generated_manifest = _validate_generated_application(
            application,
            name=application_name,
            mesh_width=mesh_width,
            mesh_height=mesh_height,
            num_token_slots=num_token_slots,
            execution_tokens=execution_tokens,
        )
        candidate = application
        if previous_manifest is not None:
            candidate = staging_parent / "regenerated-application"
            _prepare_regenerated_application(
                output,
                previous_manifest,
                application,
                generated_manifest,
                candidate,
            )
            _validate_generated_application(
                candidate,
                name=application_name,
                mesh_width=mesh_width,
                mesh_height=mesh_height,
                num_token_slots=num_token_slots,
                execution_tokens=execution_tokens,
            )
        _publish_application(
            candidate,
            output,
            replace_existing=previous_manifest is not None,
        )
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
    return output


def application_build_summary(application: str | Path) -> str:
    """Return the concise successful-build handoff for an application."""

    path = Path(application)
    manifest = read_application_manifest(path)
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


def application_summary(application: str | Path) -> str:
    """Return a concise human inspection of a valid MAGIA Application."""

    manifest = validate_application(application)
    identity = manifest["application"]
    mesh = manifest["planned_mesh"]
    execution = manifest["execution"]
    abi = manifest["abi"]
    lines = [
        f"MAGIA Application: {identity['name']}",
        f"Target: {identity['target']}",
        f"Mesh: {mesh['width']}x{mesh['height']}",
        f"Execution Tokens: {execution['tokens']}",
        f"Token Slots: {execution['token_slots']}",
        f"Active tiles: {len(manifest['active_physical_tiles'])}",
    ]
    labels = (("inputs", "Runtime Input"), ("outputs", "Graph output"))
    for tensor_kind, label in labels:
        for tensor in manifest["tensors"][tensor_kind]:
            shape = "x".join(str(dimension) for dimension in tensor["shape"])
            lines.append(
                f"{label}: {tensor['original_name']} "
                f"({tensor['normalized_name']}), {tensor['dtype']} "
                f"[{shape}], {tensor['byte_size']} bytes"
            )
    lines.extend(
        (
            f"Operation ABI: {abi['operation']}",
            f"Descriptor ABI: {abi['descriptor']}",
            f"Generated files: {len(manifest['files']['generated'])} verified",
            f"User-owned files: {len(manifest['files']['user_owned'])} present "
            "(content not verified)",
        )
    )
    return "\n".join(lines)


__all__ = [
    "APPLICATION_SCHEMA_VERSION",
    "DESCRIPTOR_ABI_VERSION",
    "MAGIA_V2_TARGET",
    "OPERATION_ABI_VERSION",
    "application_build_summary",
    "application_summary",
    "build_application",
    "normalize_application_name",
    "validate_application",
]
