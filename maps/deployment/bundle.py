"""Deployment bundle model, JSON export, and independent artifact validation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from maps.graph import (
    ConstantStore,
    Graph,
    GraphRewriteEffect,
    Node,
    validate_constants,
)
from maps.hardware import WorkSignature
from maps.planning import (
    ExecutionPlan,
    PlanningConstraints,
    require_valid_execution_plan,
)
from maps.planning.transitions.contracts import InputTransition, OutputTransition
from maps.target import RewriteEvent, RewriteReport, SpecializationResult

from .serialization import execution_plan_payload
from .weights import PackedWeights, pack_weights


BUNDLE_SCHEMA_VERSION = 1

_DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "bool": 1,
}


@dataclass(frozen=True)
class DeploymentBundle:
    execution_plan: ExecutionPlan
    graph: Graph
    constants: ConstantStore
    rewrite_report: RewriteReport = RewriteReport()


def build_deployment_bundle(
    specialization: SpecializationResult,
    execution_plan: ExecutionPlan,
    *,
    graph_rewrite_effects: tuple[GraphRewriteEffect, ...] = (),
) -> DeploymentBundle:
    """Aggregate a specialized Imported Model and its Execution Plan."""

    graph_rewrite_events = tuple(
        RewriteEvent(
            rewrite_name=effect.rewrite_name,
            source_node=effect.source_node.name,
            original_signature=_node_signature(effect.source_node),
            resulting_signatures=tuple(
                WorkSignature.from_node(node)
                for node in effect.resulting_nodes
            ),
        )
        for effect in graph_rewrite_effects
    )
    return DeploymentBundle(
        execution_plan=execution_plan,
        graph=specialization.model.graph,
        constants=specialization.model.constants,
        rewrite_report=RewriteReport(
            graph_rewrite_events + specialization.report.events
        ),
    )


def _node_signature(node: Node) -> WorkSignature | None:
    try:
        return WorkSignature.from_node(node)
    except ValueError:
        return None


def _signature_payload(signature) -> dict[str, Any] | None:
    if signature is None:
        return None
    return {
        "work_kind": signature.work_kind.name,
        "input_dtypes": [dtype.value for dtype in signature.input_dtypes],
        "output_dtypes": [dtype.value for dtype in signature.output_dtypes],
    }


def _rewrite_report_payload(report: RewriteReport) -> list[dict[str, Any]]:
    return [
        {
            "rewrite_name": event.rewrite_name,
            "source_node": event.source_node,
            "original_signature": _signature_payload(event.original_signature),
            "resulting_signatures": [
                _signature_payload(signature)
                for signature in event.resulting_signatures
            ],
            "converted_initializers": list(event.converted_initializers),
        }
        for event in report.events
    ]


def _static_activation_bytes(bundle: DeploymentBundle) -> int:
    initializer_ids = {
        tensor_id
        for tensor_id, tensor in enumerate(bundle.execution_plan.tensors)
        if tensor.is_initializer
    }
    external_ids = {
        transition.tensor_id
        for transition in bundle.execution_plan.transitions
        if isinstance(transition, (InputTransition, OutputTransition))
    }
    return sum(
        bundle.execution_plan.tensors[tensor_id].num_elements
        * bundle.execution_plan.tensors[tensor_id].elem_bytes
        for tensor_id in external_ids - initializer_ids
    )


def _bundle_payload(
    bundle: DeploymentBundle,
    packed: PackedWeights,
    weights_file: str,
) -> dict[str, Any]:
    payload = execution_plan_payload(bundle.execution_plan)
    payload["provenance"] = {
        "rewrite_report": _rewrite_report_payload(bundle.rewrite_report),
    }
    payload["bundle"] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "weights_file": weights_file,
        "weights_size": len(packed.data),
        "weights_sha256": packed.sha256,
        "alignment": packed.alignment,
        "endianness": "little",
    }
    for initializer in packed.initializers:
        tensor = payload["tensors"][initializer.tensor_id]
        tensor["initializer"] = {
            "offset": initializer.offset,
            "byte_size": initializer.byte_size,
            "dtype": initializer.dtype.value,
            "shape": list(initializer.shape),
            "sha256": initializer.sha256,
        }
    return payload


def write_deployment_bundle(
    bundle: DeploymentBundle,
    output_execution_plan: str | Path,
    output_initializers: str | Path,
) -> tuple[Path, Path]:
    """Write a serialized Deployment Bundle and packed Initializer image."""

    validate_constants(bundle.graph, bundle.constants)
    require_valid_execution_plan(
        bundle.execution_plan,
        PlanningConstraints(),
        error_prefix="deployment bundle has an invalid Execution Plan",
    )
    packed = pack_weights(bundle.execution_plan, bundle.constants)
    required_l2 = len(packed.data) + _static_activation_bytes(bundle)
    capacity = bundle.execution_plan.mesh.l2_memory.size
    if required_l2 > capacity:
        raise ValueError(
            f"deployment bundle requires {required_l2} L2 bytes but mesh provides {capacity}"
        )

    bundle_path = Path(output_execution_plan)
    initializers_path = Path(output_initializers)
    if bundle_path.resolve() == initializers_path.resolve():
        raise ValueError(
            "Deployment Bundle document and Initializer image must use different paths"
        )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    initializers_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _bundle_payload(bundle, packed, initializers_path.name)
    initializers_path.write_bytes(packed.data)
    bundle_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_deployment_bundle(
        bundle_path,
        initializers_path,
        l2_capacity=capacity,
    )
    return bundle_path, initializers_path


def validate_deployment_bundle(
    deployment_bundle: str | Path,
    initializers: str | Path,
    *,
    l2_capacity: int | None = None,
) -> None:
    """Independently validate a serialized Deployment Bundle."""

    bundle_path = Path(deployment_bundle)
    initializers_path = Path(initializers)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    data = initializers_path.read_bytes()
    metadata = payload.get("bundle")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported or missing deployment bundle schema")
    if metadata.get("weights_file") != initializers_path.name:
        raise ValueError("bundle weights filename mismatch")
    if metadata.get("endianness") != "little":
        raise ValueError("bundle weights must be little-endian")
    if metadata.get("weights_size") != len(data):
        raise ValueError("bundle weights size mismatch")
    if metadata.get("weights_sha256") != sha256(data).hexdigest():
        raise ValueError("bundle weights checksum mismatch")
    alignment = metadata.get("alignment")
    if (
        not isinstance(alignment, int)
        or alignment <= 0
        or alignment & (alignment - 1)
    ):
        raise ValueError("bundle alignment is invalid")
    previous_end = 0
    seen_ranges: list[tuple[int, str]] = []
    for tensor in payload.get("tensors", []):
        initializer = tensor.get("initializer")
        if initializer is None:
            if tensor.get("is_initializer"):
                raise ValueError(f"initializer tensor '{tensor.get('name')}' has no metadata")
            continue
        if not tensor.get("is_initializer"):
            raise ValueError(f"non-initializer tensor '{tensor.get('name')}' has metadata")
        offset = initializer.get("offset")
        byte_size = initializer.get("byte_size")
        if not isinstance(offset, int) or not isinstance(byte_size, int) or byte_size < 0:
            raise ValueError("initializer range is invalid")
        if offset % alignment:
            raise ValueError(f"initializer tensor '{tensor.get('name')}' is not aligned")
        seen_ranges.append((offset, tensor.get("name", "")))
        end = offset + byte_size
        if end > len(data):
            raise ValueError(f"initializer tensor '{tensor.get('name')}' exceeds weights")
        tensor_data = data[offset:end]
        if initializer.get("sha256") != sha256(tensor_data).hexdigest():
            raise ValueError(f"initializer tensor '{tensor.get('name')}' checksum mismatch")
        if initializer.get("shape") != tensor.get("dims"):
            raise ValueError(f"initializer tensor '{tensor.get('name')}' shape mismatch")
        dtype = initializer.get("dtype")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"initializer tensor '{tensor.get('name')}' dtype is invalid")
        elements = 1
        for dimension in initializer["shape"]:
            elements *= dimension
        if byte_size != elements * _DTYPE_BYTES[dtype]:
            raise ValueError(f"initializer tensor '{tensor.get('name')}' byte size mismatch")

    for offset, name in sorted(seen_ranges):
        tensor = next(
            item for item in payload["tensors"]
            if item.get("name") == name and "initializer" in item
        )
        byte_size = tensor["initializer"]["byte_size"]
        if offset < previous_end:
            raise ValueError(f"initializer tensor '{name}' overlaps another initializer")
        if any(data[previous_end:offset]):
            raise ValueError("bundle padding must contain only zero bytes")
        previous_end = offset + byte_size

    if l2_capacity is not None:
        initializer_ids = {
            tensor_id for tensor_id, tensor in enumerate(payload.get("tensors", []))
            if tensor.get("is_initializer")
        }
        external_ids = {
            item["tensor_id"]
            for item in payload.get("transitions", [])
            if item.get("kind") in {"INPUT", "OUTPUT"}
        }
        activation_bytes = 0
        for tensor_id in external_ids - initializer_ids:
            tensor = payload["tensors"][tensor_id]
            elements = 1
            for dimension in tensor["dims"]:
                elements *= dimension
            activation_bytes += elements * tensor["elem_bytes"]
        if len(data) + activation_bytes > l2_capacity:
            raise ValueError("deployment bundle exceeds L2 capacity")
