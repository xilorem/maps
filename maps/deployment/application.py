"""High-level construction of relocatable MAGIA Applications."""

from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping

from maps.target import magia

from .bundle import DeploymentBundle, write_deployment_bundle
from .workflow import build_magia_deployment_bundle


MAGIA_V2_TARGET = "magia-v2"
APPLICATION_SCHEMA_VERSION = 1
OPERATION_ABI_VERSION = 1
DESCRIPTOR_ABI_VERSION = 1
_TENSOR_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


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
        raise ValueError(
            "generated application manifest is unreadable or invalid"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("generated application manifest must be an object")
    return manifest


def _safe_application_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"unsafe application file path '{value}'")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value != relative.as_posix()
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise ValueError(f"unsafe application file path '{value}'")
    return relative


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_tensor_records(
    records: Any,
    *,
    kind: str,
) -> tuple[set[int], set[str], set[str], set[str]]:
    if not isinstance(records, list):
        raise ValueError(f"application Runtime {kind} records must be a list")
    ids: set[int] = set()
    original_names: set[str] = set()
    normalized_names: set[str] = set()
    supplied_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"application Runtime {kind} record must be an object")
        tensor_id = record.get("id")
        original_name = record.get("original_name")
        normalized_name = record.get("normalized_name")
        dtype = record.get("dtype")
        shape = record.get("shape")
        byte_size = record.get("byte_size")
        if type(tensor_id) is not int or tensor_id < 0 or tensor_id in ids:
            raise ValueError(
                f"application Runtime {kind} Tensor id is invalid or duplicate"
            )
        if (
            not isinstance(original_name, str)
            or not original_name
            or original_name in original_names
        ):
            raise ValueError(
                f"application Runtime {kind} Tensor name is invalid or duplicate"
            )
        if (
            not isinstance(normalized_name, str)
            or normalized_name != normalize_application_name(original_name)
            or normalized_name in normalized_names
        ):
            raise ValueError(
                f"application Runtime {kind} normalized Tensor name is invalid or duplicate"
            )
        if dtype not in _TENSOR_DTYPE_BYTES:
            raise ValueError(f"application Runtime {kind} TensorDType is unsupported")
        if not isinstance(shape, list) or not shape or not all(
            _positive_integer(dimension) for dimension in shape
        ):
            raise ValueError(f"application Runtime {kind} Tensor shape is invalid")
        expected_bytes = _TENSOR_DTYPE_BYTES[dtype]
        for dimension in shape:
            expected_bytes *= dimension
        if byte_size != expected_bytes:
            raise ValueError(
                f"application Runtime {kind} Tensor byte size is inconsistent"
            )
        if kind == "Input":
            data = record.get("data")
            if not isinstance(data, dict) or data.get("kind") not in {
                "synthetic",
                "supplied",
            }:
                raise ValueError("application Runtime Input data source is invalid")
            if data["kind"] == "synthetic" and set(data) != {"kind"}:
                raise ValueError(
                    "application synthetic Runtime Input record is inconsistent"
                )
            if data["kind"] == "supplied":
                if set(data) != {"kind", "path"}:
                    raise ValueError(
                        "application supplied Runtime Input record is inconsistent"
                    )
                supplied_paths.add(_safe_application_path(data["path"]).as_posix())
        elif "data" in record:
            raise ValueError(
                "application graph output cannot declare Runtime Input data"
            )
        ids.add(tensor_id)
        original_names.add(original_name)
        normalized_names.add(normalized_name)
    return ids, original_names, normalized_names, supplied_paths


def validate_application(application: str | Path) -> dict[str, Any]:
    """Reopen and independently validate a generated MAGIA Application."""

    path = Path(application)
    manifest = _application_manifest(path)
    identity = manifest.get("application")
    mesh = manifest.get("planned_mesh")
    abi = manifest.get("abi")
    execution = manifest.get("execution")
    active_tiles = manifest.get("active_physical_tiles")
    entry_points = manifest.get("entry_points")
    memory = manifest.get("memory")
    if manifest.get("schema_version") != APPLICATION_SCHEMA_VERSION:
        raise ValueError("incompatible application schema version")
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("name"), str)
        or identity["name"] != normalize_application_name(identity["name"])
        or identity.get("target") != MAGIA_V2_TARGET
    ):
        raise ValueError("application identity or Target is invalid")
    name = identity["name"]
    if (
        not isinstance(manifest.get("source_model"), str)
        or not manifest["source_model"]
    ):
        raise ValueError("application source model identity is invalid")
    if (
        not isinstance(mesh, dict)
        or not _positive_integer(mesh.get("width"))
        or not _positive_integer(mesh.get("height"))
    ):
        raise ValueError("application planned Mesh is invalid")
    if abi != {
        "operation": OPERATION_ABI_VERSION,
        "descriptor": DESCRIPTOR_ABI_VERSION,
    }:
        raise ValueError("incompatible application ABI")
    if (
        not isinstance(execution, dict)
        or not _positive_integer(execution.get("tokens"))
        or not _positive_integer(execution.get("token_slots"))
    ):
        raise ValueError("application execution settings are invalid")
    if (
        not isinstance(active_tiles, list)
        or not active_tiles
        or any(type(tile) is not int for tile in active_tiles)
        or len(set(active_tiles)) != len(active_tiles)
        or active_tiles != sorted(active_tiles)
        or any(
            tile < 0 or tile >= mesh["width"] * mesh["height"]
            for tile in active_tiles
        )
    ):
        raise ValueError("application active physical tiles are invalid or duplicate")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != {"inputs", "outputs"}:
        raise ValueError("application Tensor records are incomplete")
    input_facts = _validate_tensor_records(tensors["inputs"], kind="Input")
    output_facts = _validate_tensor_records(tensors["outputs"], kind="Output")
    if input_facts[0] & output_facts[0]:
        raise ValueError("application Tensor ids are duplicate")
    if entry_points != {
        "run": f"{name}_run",
        "handle_input": f"{name}_handle_input",
        "handle_output": f"{name}_handle_output",
    }:
        raise ValueError("application entry points are inconsistent")
    if (
        not isinstance(memory, dict)
        or set(memory)
        != {
            "initializers_bytes",
            "required_l2_bytes",
            "max_tile_l1_bytes",
        }
        or not all(_nonnegative_integer(value) for value in memory.values())
    ):
        raise ValueError("application memory requirements are invalid")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"generated", "user_owned"}:
        raise ValueError("application file ownership is invalid")
    generated = files["generated"]
    user_owned = files["user_owned"]
    if not isinstance(generated, list) or not isinstance(user_owned, list):
        raise ValueError("application file records must be lists")
    declared_paths: set[str] = set()
    generated_paths: set[str] = set()
    for record in generated:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "role",
            "byte_size",
            "sha256",
        }:
            raise ValueError("application generated file record is incomplete")
        relative = _safe_application_path(record["path"])
        relative_text = relative.as_posix()
        if relative_text in declared_paths:
            raise ValueError(f"duplicate application file record '{relative_text}'")
        role = record["role"]
        byte_size = record["byte_size"]
        checksum = record["sha256"]
        if not isinstance(role, str) or not role:
            raise ValueError(
                f"application generated file role is invalid for '{relative_text}'"
            )
        if not _nonnegative_integer(byte_size):
            raise ValueError(
                f"application generated file byte size is invalid for '{relative_text}'"
            )
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError(
                f"application generated file checksum is invalid for '{relative_text}'"
            )
        artifact = path.joinpath(*relative.parts)
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"application is missing generated file '{relative_text}'")
        contents = artifact.read_bytes()
        if len(contents) != byte_size:
            raise ValueError(f"generated file byte size mismatch for '{relative_text}'")
        if sha256(contents).hexdigest() != checksum:
            raise ValueError(f"generated file checksum mismatch for '{relative_text}'")
        declared_paths.add(relative_text)
        generated_paths.add(relative_text)
    for record in user_owned:
        if not isinstance(record, dict) or set(record) != {"path", "role"}:
            raise ValueError("application user-owned file record is invalid")
        relative = _safe_application_path(record["path"])
        relative_text = relative.as_posix()
        if relative_text in declared_paths:
            raise ValueError(f"duplicate application file record '{relative_text}'")
        if not isinstance(record["role"], str) or not record["role"]:
            raise ValueError(
                f"application user-owned file role is invalid for '{relative_text}'"
            )
        artifact = path.joinpath(*relative.parts)
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(
                f"application is missing user-owned file '{relative_text}'"
            )
        declared_paths.add(relative_text)
    if user_owned != [{"path": "src/application.c", "role": "application_source"}]:
        raise ValueError("application customization source declaration is invalid")
    required_generated = {
        "CMakeLists.txt",
        "README.md",
        f"include/{name}.h",
        f"src/{name}.c",
        f"src/{name}_runner.c",
        f"src/{name}_initializers.S.in",
        f"data/{name}.initializers.bin",
    }
    expected_tiles = {f"src/tiles/tile_{tile:02d}.c" for tile in active_tiles}
    if not required_generated.issubset(generated_paths):
        raise ValueError("application generated file contract is incomplete")
    actual_tiles = {
        value for value in generated_paths if value.startswith("src/tiles/tile_")
    }
    if actual_tiles != expected_tiles:
        raise ValueError("application active tile files are inconsistent")
    if not input_facts[3].issubset(generated_paths):
        raise ValueError("application Runtime Input assets are undeclared")
    return manifest


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
        _validate_generated_application(
            application,
            name=application_name,
            mesh_width=mesh_width,
            mesh_height=mesh_height,
            num_token_slots=num_token_slots,
            execution_tokens=execution_tokens,
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
