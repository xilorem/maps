"""Stable JSON serialization for Execution Plans."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

from maps.graph import TensorDType
from maps.hardware import Mesh
from maps.planning.mapping import Submesh
from maps.planning import (
    ExecutionPlan,
    InitializerInput,
    LocalInput,
    TransitionSource,
)
from maps.operations.elementwise import (
    BinaryElementwisePayload,
    UnaryElementwisePayload,
)
from maps.planning.transitions.contracts import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
)

_RUNTIME_ELEMENTWISE_NAMES = {
    "abs": "Abs",
    "add": "Add",
    "div": "Div",
    "exp": "Exp",
    "log": "Log",
    "mul": "Mul",
    "neg": "Neg",
    "pow": "Pow",
    "relu": "Relu",
    "sigmoid": "Sigmoid",
    "sqrt": "Sqrt",
    "sub": "Sub",
}


def _elementwise_payload(
    value: BinaryElementwisePayload | UnaryElementwisePayload,
) -> dict[str, Any]:
    """Preserve the established runtime names for elementwise Operations."""

    return {
        field.name: (
            _RUNTIME_ELEMENTWISE_NAMES[value.op_name]
            if field.name == "op_name"
            else _to_jsonable(getattr(value, field.name))
        )
        for field in fields(value)
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, InputTransition):
        return _transition_payload("INPUT", value)
    if isinstance(value, IntermediateTransition):
        return _transition_payload("INTERMEDIATE", value)
    if isinstance(value, OutputTransition):
        return _transition_payload("OUTPUT", value)
    if isinstance(value, InitializerInput):
        return {
            "kind": "INITIALIZER",
            "destinations": _to_jsonable(value.destinations),
        }
    if isinstance(value, TransitionSource):
        return {
            "kind": "TRANSITION",
            "transition_id": value.transition_id,
        }
    if isinstance(value, LocalInput):
        return {
            "kind": "LOCAL",
            "layer_index": value.layer_idx,
            "tensor_id": value.tensor_id,
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, TensorDType):
        return value.value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mesh):
        return {
            "width": value.width,
            "height": value.height,
            "l2_memory": {"size": value.l2_memory.size},
            "tiles": [
                {"tile_id": tile.tile_id, "x": tile.x, "y": tile.y}
                for tile in value.tiles
            ],
        }
    if isinstance(value, Submesh):
        return {
            "submesh_id": value.submesh_id,
            "tile_ids": sorted(value.tile_ids),
        }
    if isinstance(value, (BinaryElementwisePayload, UnaryElementwisePayload)):
        return _elementwise_payload(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
            if field.name != "is_initializer"
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_to_jsonable(item) for item in sorted(value, key=repr)]
    return str(value)


def _transition_payload(
    kind: str,
    transition: InputTransition | IntermediateTransition | OutputTransition,
) -> dict[str, Any]:
    return {
        "kind": kind,
        **{
            field.name: _to_jsonable(getattr(transition, field.name))
            for field in fields(transition)
        },
    }


def execution_plan_payload(
    execution_plan: ExecutionPlan,
) -> dict[str, Any]:
    """Return the runtime-facing unified Execution Plan representation."""

    payload = _to_jsonable(execution_plan)
    if execution_plan.target is None:
        payload.pop("target")
    for tensor_payload, tensor in zip(payload["tensors"], execution_plan.tensors):
        tensor_payload["is_initializer"] = tensor.is_initializer
    return payload


def write_execution_plan(
    execution_plan: ExecutionPlan,
    output_path: str | Path,
) -> Path:
    """Write a plain serialized Execution Plan and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = execution_plan_payload(execution_plan)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "execution_plan_payload",
    "write_execution_plan",
]
