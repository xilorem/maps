"""Independent validation and inspection of MAGIA Applications."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, NamedTuple


APPLICATION_SCHEMA_VERSION = 1
OPERATION_ABI_VERSION = 1
DESCRIPTOR_ABI_VERSION = 1
MAGIA_V2_TARGET = "magia-v2"
MAGIA_V3_TARGET = "magia-v3"
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


class _TensorFacts(NamedTuple):
    ids: set[int]
    supplied_tensor_bytes: dict[str, int]


def read_application_manifest(application: Path) -> dict[str, Any]:
    manifest_path = application / "manifest.json"
    if (
        application.is_symlink()
        or not application.is_dir()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not manifest_path.resolve().is_relative_to(application.resolve())
    ):
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


def _has_symlinked_parent(artifact: Path, application: Path) -> bool:
    parent = artifact.parent
    while parent != application:
        if parent.is_symlink():
            return True
        parent = parent.parent
    return False


def _validate_tensor_records(
    records: Any,
    *,
    kind: str,
    normalize_name: Callable[[str], str],
) -> _TensorFacts:
    if not isinstance(records, list):
        raise ValueError(f"application Runtime {kind} records must be a list")
    ids: set[int] = set()
    original_names: set[str] = set()
    normalized_names: set[str] = set()
    supplied_tensor_bytes: dict[str, int] = {}
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
            or normalized_name != normalize_name(original_name)
            or normalized_name in normalized_names
        ):
            raise ValueError(
                f"application Runtime {kind} normalized Tensor name is invalid "
                "or duplicate"
            )
        if dtype not in _TENSOR_DTYPE_BYTES:
            raise ValueError(f"application Runtime {kind} TensorDType is unsupported")
        if not isinstance(shape, list) or not all(
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
                supplied_path = _safe_application_path(data["path"]).as_posix()
                if supplied_path in supplied_tensor_bytes:
                    raise ValueError("application Runtime Input asset is duplicate")
                supplied_tensor_bytes[supplied_path] = byte_size
        elif "data" in record:
            raise ValueError(
                "application graph output cannot declare Runtime Input data"
            )
        ids.add(tensor_id)
        original_names.add(original_name)
        normalized_names.add(normalized_name)
    return _TensorFacts(ids=ids, supplied_tensor_bytes=supplied_tensor_bytes)


def _validate_manifest_contract(
    manifest: dict[str, Any], normalize_name: Callable[[str], str]
) -> tuple[str, list[int], int, _TensorFacts]:
    identity = manifest.get("application")
    mesh = manifest.get("planned_mesh")
    abi = manifest.get("abi")
    execution = manifest.get("execution")
    active_tiles = manifest.get("active_physical_tiles")
    if manifest.get("schema_version") != APPLICATION_SCHEMA_VERSION:
        raise ValueError("incompatible application schema version")
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("name"), str)
        or identity["name"] != normalize_name(identity["name"])
        or identity.get("target") not in {MAGIA_V2_TARGET, MAGIA_V3_TARGET}
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
    expected_abi = {
        "operation": OPERATION_ABI_VERSION,
        "descriptor": DESCRIPTOR_ABI_VERSION,
    }
    if identity["target"] == MAGIA_V3_TARGET:
        expected_abi.update(kernel=1, task_bundle=1)
    if abi != expected_abi:
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
    input_facts = _validate_tensor_records(
        tensors["inputs"], kind="Input", normalize_name=normalize_name
    )
    output_facts = _validate_tensor_records(
        tensors["outputs"], kind="Output", normalize_name=normalize_name
    )
    if input_facts.ids & output_facts.ids:
        raise ValueError("application Tensor ids are duplicate")
    if manifest.get("entry_points") != {
        "run": f"{name}_run",
        "handle_input": f"{name}_handle_input",
        "handle_output": f"{name}_handle_output",
    }:
        raise ValueError("application entry points are inconsistent")
    memory = manifest.get("memory")
    expected_memory_keys = {
        "initializers_bytes",
        "required_l2_bytes",
        "max_tile_l1_bytes",
    }
    if identity["target"] == MAGIA_V3_TARGET:
        expected_memory_keys.update({"initializers_region", "runtime_region"})
    if (
        not isinstance(memory, dict)
        or set(memory) != expected_memory_keys
        or not all(
            _nonnegative_integer(memory.get(key))
            for key in (
                "initializers_bytes",
                "required_l2_bytes",
                "max_tile_l1_bytes",
            )
        )
    ):
        raise ValueError("application memory requirements are invalid")
    if identity["target"] == MAGIA_V3_TARGET and (
        memory["initializers_region"] != "l2_bulk"
        or memory["runtime_region"] != "l2_arena"
    ):
        raise ValueError("application memory regions are invalid")
    return name, active_tiles, execution["tokens"], input_facts


def _validate_files(
    application: Path,
    manifest: dict[str, Any],
    *,
    name: str,
    active_tiles: list[int],
    execution_tokens: int,
    supplied_tensor_bytes: dict[str, int],
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        "generated",
        "trust_root",
        "user_owned",
    }:
        raise ValueError("application file ownership is invalid")
    generated = files["generated"]
    trust_root = files["trust_root"]
    user_owned = files["user_owned"]
    if (
        not isinstance(generated, list)
        or not isinstance(trust_root, list)
        or not isinstance(user_owned, list)
    ):
        raise ValueError("application file records must be lists")
    if application.is_symlink() or not application.is_dir():
        raise ValueError("unsafe MAGIA Application directory")
    application_root = application.resolve()
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
        artifact = application.joinpath(*relative.parts)
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or _has_symlinked_parent(artifact, application)
            or not artifact.resolve().is_relative_to(application_root)
        ):
            raise ValueError(f"application is missing generated file '{relative_text}'")
        contents = artifact.read_bytes()
        if len(contents) != byte_size:
            raise ValueError(f"generated file byte size mismatch for '{relative_text}'")
        if sha256(contents).hexdigest() != checksum:
            raise ValueError(f"generated file checksum mismatch for '{relative_text}'")
        declared_paths.add(relative_text)
        generated_paths.add(relative_text)
    if trust_root != [{"path": "manifest.json", "role": "application_manifest"}]:
        raise ValueError("application manifest trust-root declaration is invalid")
    declared_paths.add("manifest.json")
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
        artifact = application.joinpath(*relative.parts)
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or _has_symlinked_parent(artifact, application)
            or not artifact.resolve().is_relative_to(application_root)
        ):
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
    generated_by_path = {record["path"]: record for record in generated}
    expected_roles = {
        "CMakeLists.txt": "build_definition",
        "README.md": "integration_guide",
        f"include/{name}.h": "application_interface",
        f"src/{name}.c": "application_data",
        f"src/{name}_runner.c": "application_runner",
        f"src/{name}_initializers.S.in": "initializer_assembly",
        f"data/{name}.initializers.bin": "initializer_data",
        **{tile: "tile_plan" for tile in expected_tiles},
        **{supplied: "runtime_input_data" for supplied in supplied_tensor_bytes},
    }
    if any(
        generated_by_path[path]["role"] != role
        for path, role in expected_roles.items()
    ):
        raise ValueError("application generated file roles are inconsistent")
    for supplied_path, tensor_bytes in supplied_tensor_bytes.items():
        record = generated_by_path.get(supplied_path)
        if record is None:
            raise ValueError("application Runtime Input assets are undeclared")
        if record["byte_size"] != tensor_bytes * execution_tokens:
            raise ValueError("application Runtime Input asset size is inconsistent")


def validate_application(
    application: str | Path,
    *,
    normalize_name: Callable[[str], str],
) -> dict[str, Any]:
    """Reopen and independently validate a generated MAGIA Application."""

    path = Path(application)
    manifest = read_application_manifest(path)
    name, active_tiles, execution_tokens, input_facts = _validate_manifest_contract(
        manifest, normalize_name
    )
    _validate_files(
        path,
        manifest,
        name=name,
        active_tiles=active_tiles,
        execution_tokens=execution_tokens,
        supplied_tensor_bytes=input_facts.supplied_tensor_bytes,
    )
    return manifest
