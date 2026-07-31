"""Complete deployment-package construction and independent verification."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Callable

from MAPS.deployment.bundle import write_execution_plan_bundle
from MAPS.hw.chips import magia_mesh
from MAPS.pipeline import ExecutionContract
from MAPS.planner.contracts.options import (
    PlannerOptions,
    SpatialMappingOptions,
    WorkloadBalancingOptions,
)
from MAPS.planner.plan import build_execution_plan_bundle


PACKAGE_SCHEMA_VERSION = 1
OPERATION_ABI_VERSION = 1
DESCRIPTOR_ABI_VERSION = 1
DESCRIPTOR_MAX_DIMENSIONS = 6
SUPPORTED_TARGET = "magia-v2"

_DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "bfloat16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "bool": 1,
}
_ARTIFACT_ROLES = {
    "header",
    "data_source",
    "weights_assembly",
    "weights_image",
}
_ENTRY_POINTS = {
    "init_l2_data": "maps_generated_init_l2_data",
    "init_tensors": "maps_generated_init_tensors",
    "fill_plan": "maps_generated_fill_plan",
    "check_output": "maps_generated_check_output",
}


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"manifest field '{field}' must be an object")
    return value


def _positive_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"manifest field '{field}' must be a positive integer")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"manifest field '{field}' must be a nonnegative integer")
    return value


def _safe_artifact_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe package artifact path '{value}'")
    if len(path.parts) != 1:
        raise ValueError(f"runtime artifact must be at package root: '{value}'")
    return path


def _validate_tensor(tensor: object, field: str) -> None:
    record = _object(tensor, field)
    _nonnegative_integer(record.get("id"), f"{field}.id")
    if not isinstance(record.get("name"), str) or not record["name"]:
        raise ValueError(f"manifest field '{field}.name' must be nonempty")
    dtype = record.get("dtype")
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported tensor dtype in '{field}'")
    shape = record.get("shape")
    if not isinstance(shape, list) or any(
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 0
        for dimension in shape
    ):
        raise ValueError(f"manifest field '{field}.shape' is invalid")
    if record.get("encoding") != "raw" or record.get("endianness") != "little":
        raise ValueError(f"unsupported tensor encoding in '{field}'")
    elements = 1
    for dimension in shape:
        elements *= dimension
    tensor_bytes = _nonnegative_integer(
        record.get("tensor_bytes"), f"{field}.tensor_bytes"
    )
    if tensor_bytes != elements * _DTYPE_BYTES[dtype]:
        raise ValueError(f"tensor byte size mismatch in '{field}'")


def _validate_tensor_list(value: object, field: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"manifest field '{field}' must be an array")
    names: set[str] = set()
    ids: set[int] = set()
    for index, tensor in enumerate(value):
        _validate_tensor(tensor, f"{field}[{index}]")
        record = _object(tensor, f"{field}[{index}]")
        if record["name"] in names or record["id"] in ids:
            raise ValueError(f"duplicate tensor identity in '{field}'")
        names.add(record["name"])
        ids.add(record["id"])


def validate_deployment_package(package_dir: str | Path) -> dict[str, Any]:
    """Reopen and strictly validate a relocatable deployment package."""

    package = Path(package_dir)
    if not package.is_dir() or package.is_symlink():
        raise ValueError(f"deployment package is not a directory: {package}")
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("deployment package has no regular manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("deployment manifest is unreadable or invalid") from exc
    manifest = _object(manifest, "manifest")
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported deployment manifest schema")

    source = _object(manifest.get("source_model"), "source_model")
    if not isinstance(source.get("name"), str) or not source["name"]:
        raise ValueError("source model name must be nonempty")
    _validate_tensor_list(source.get("inputs"), "source_model.inputs")
    _validate_tensor_list(source.get("outputs"), "source_model.outputs")
    input_names = {tensor["name"] for tensor in source["inputs"]}
    output_names = {tensor["name"] for tensor in source["outputs"]}
    input_ids = {tensor["id"] for tensor in source["inputs"]}
    output_ids = {tensor["id"] for tensor in source["outputs"]}
    if input_names & output_names or input_ids & output_ids:
        raise ValueError("input and output tensor identities must be distinct")

    target = _object(manifest.get("target"), "target")
    if target.get("architecture") != SUPPORTED_TARGET:
        raise ValueError("unsupported deployment target")
    mesh = _object(target.get("mesh"), "target.mesh")
    _positive_integer(mesh.get("width"), "target.mesh.width")
    _positive_integer(mesh.get("height"), "target.mesh.height")

    execution = _object(manifest.get("execution"), "execution")
    _positive_integer(execution.get("num_token_slots"), "execution.num_token_slots")
    _positive_integer(
        execution.get("pipeline_token_capacity"),
        "execution.pipeline_token_capacity",
    )

    abi = _object(manifest.get("abi"), "abi")
    if (
        abi.get("operation_version") != OPERATION_ABI_VERSION
        or abi.get("descriptor_version") != DESCRIPTOR_ABI_VERSION
        or abi.get("descriptor_max_dimensions") != DESCRIPTOR_MAX_DIMENSIONS
    ):
        raise ValueError("incompatible deployment ABI")

    entry_points = _object(manifest.get("entry_points"), "entry_points")
    if entry_points != _ENTRY_POINTS:
        raise ValueError("invalid generated entry-point contract")

    memory = _object(manifest.get("memory"), "memory")
    ready_flags_offset = _positive_integer(
        memory.get("ready_flags_offset"), "memory.ready_flags_offset"
    )
    max_tile_data_bytes = _nonnegative_integer(
        memory.get("max_tile_data_bytes"), "memory.max_tile_data_bytes"
    )
    weights_size = _nonnegative_integer(
        memory.get("weights_size"), "memory.weights_size"
    )
    required_l2_bytes = _nonnegative_integer(
        memory.get("required_l2_bytes"), "memory.required_l2_bytes"
    )
    if max_tile_data_bytes > ready_flags_offset:
        raise ValueError("tile data overlaps the ready-flags region")
    if required_l2_bytes < weights_size:
        raise ValueError("required L2 capacity is smaller than packed weights")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(_ARTIFACT_ROLES):
        raise ValueError("deployment package must declare four runtime artifacts")
    declared_paths: set[PurePosixPath] = set()
    declared_roles: set[str] = set()
    weights_artifact_size: int | None = None
    for index, artifact in enumerate(artifacts):
        record = _object(artifact, f"artifacts[{index}]")
        relative = _safe_artifact_path(record.get("path"))
        role = record.get("role")
        if role not in _ARTIFACT_ROLES:
            raise ValueError(f"unknown artifact role '{role}'")
        if relative in declared_paths or role in declared_roles:
            raise ValueError("duplicate artifact path or role")
        expected_size = _nonnegative_integer(
            record.get("byte_size"), f"artifacts[{index}].byte_size"
        )
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"invalid artifact checksum for '{relative}'")
        artifact_path = package / Path(*relative.parts)
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise ValueError(f"missing regular package artifact '{relative}'")
        data = artifact_path.read_bytes()
        if len(data) != expected_size:
            raise ValueError(f"artifact size mismatch for '{relative}'")
        if sha256(data).hexdigest() != expected_sha256:
            raise ValueError(f"artifact checksum mismatch for '{relative}'")
        if role == "weights_image":
            weights_artifact_size = len(data)
        declared_paths.add(relative)
        declared_roles.add(role)
    if declared_roles != _ARTIFACT_ROLES or weights_artifact_size != weights_size:
        raise ValueError("runtime artifact contract is incomplete")

    actual_files: set[PurePosixPath] = set()
    for path in package.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"deployment package contains symlink '{path.name}'")
        if path.is_dir():
            raise ValueError(f"deployment package contains undeclared directory '{path.name}'")
        if path.is_file():
            actual_files.add(PurePosixPath(path.relative_to(package).as_posix()))
    if actual_files != declared_paths | {PurePosixPath("manifest.json")}:
        raise ValueError("deployment package contains missing or undeclared files")
    return manifest


def _default_maps_translate() -> Path:
    executable = shutil.which("maps-translate")
    if executable:
        return Path(executable)
    repository_executable = (
        Path(__file__).resolve().parents[2]
        / "maps-ir"
        / "build"
        / "tools"
        / "maps-translate"
        / "maps-translate"
    )
    return repository_executable


def write_deployment_package(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    target: str = SUPPORTED_TARGET,
    mesh_width: int = 32,
    mesh_height: int = 32,
    num_token_slots: int = 2,
    pipeline_token_capacity: int = 1,
    maps_translate: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Plan once, package in a staging directory, verify, then publish."""

    if target != SUPPORTED_TARGET:
        raise ValueError(f"unsupported deployment target '{target}'")
    if mesh_width <= 0 or mesh_height <= 0:
        raise ValueError("mesh dimensions must be positive")
    if num_token_slots <= 0 or pipeline_token_capacity <= 0:
        raise ValueError("token counts must be positive")
    model = Path(model_path)
    if not model.is_file():
        raise ValueError(f"source model does not exist: {model}")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"deployment output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    translator = Path(maps_translate) if maps_translate else _default_maps_translate()
    if not translator.is_file():
        raise ValueError(f"maps-translate executable does not exist: {translator}")

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    report(
        f"Planning {model.name} for {target} on a "
        f"{mesh_width}x{mesh_height} mesh..."
    )
    bundle = build_execution_plan_bundle(
        model,
        magia_mesh(width=mesh_width, height=mesh_height),
        PlannerOptions(
            execution=ExecutionContract(num_token_slots=num_token_slots),
            workload=WorkloadBalancingOptions(
                print_progress=progress is not None,
            ),
            spatial_mapping=SpatialMappingOptions(
                print_progress=progress is not None,
                print_mapping=False,
                print_costs=False,
            ),
            print_pipeline_cost=False,
        ),
    )

    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        intermediates = staging_parent / "intermediates"
        package = staging_parent / "package"
        intermediates.mkdir()
        execution_plan_json = intermediates / "execution-plan.json"
        packed_weights = intermediates / "execution-plan.weights.bin"
        report("Packing constants and serializing the deployment bundle...")
        write_execution_plan_bundle(
            bundle,
            execution_plan_json,
            packed_weights,
        )
        report("Lowering the bundle to MAGIA runtime artifacts...")
        subprocess.run(
            [
                str(translator),
                "--json-to-magia-package",
                f"--maps-magia-package-dir={package}",
                "--maps-magia-output-stem=model",
                f"--maps-magia-num-tokens={pipeline_token_capacity}",
                f"--maps-magia-weights-file={packed_weights}",
                str(execution_plan_json),
                "-o",
                str(intermediates / "discarded.mlir"),
            ],
            check=True,
        )
        report("Verifying the complete deployment package...")
        validate_deployment_package(package)
        if output.exists():
            raise FileExistsError(f"deployment output already exists: {output}")
        report(f"Publishing {output}...")
        os.replace(package, output)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
    return output


def package_summary(manifest: dict[str, Any]) -> str:
    """Return a compact human-readable package summary."""

    source = manifest["source_model"]
    target = manifest["target"]
    execution = manifest["execution"]
    mesh = target["mesh"]
    return "\n".join(
        (
            f"Model: {source['name']}",
            f"Target: {target['architecture']} ({mesh['width']}x{mesh['height']})",
            f"Inputs: {len(source['inputs'])}",
            f"Outputs: {len(source['outputs'])}",
            f"Token slots: {execution['num_token_slots']}",
            f"Pipeline token capacity: {execution['pipeline_token_capacity']}",
            f"Artifacts: {len(manifest['artifacts'])}",
        )
    )
