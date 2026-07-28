"""JSON export helpers for planned pipelines."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import TYPE_CHECKING

from MAPS.arch import Mesh
from MAPS.core.submesh import Submesh
from MAPS.core.dtype import TensorDType

if TYPE_CHECKING:
    from MAPS.pipeline.pipeline import Pipeline


def _to_jsonable(value: object) -> object:
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
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_to_jsonable(item) for item in sorted(value, key=lambda item: repr(item))]
    return str(value)


def _remove_initializer_markers(value: object) -> None:
    if isinstance(value, dict):
        value.pop("is_initializer", None)
        for item in value.values():
            _remove_initializer_markers(item)
    elif isinstance(value, list):
        for item in value:
            _remove_initializer_markers(item)


def pipeline_json_payload(pipeline: Pipeline) -> dict[str, object]:
    """Return the stable JSON representation shared by pipeline exporters."""

    payload = _to_jsonable(pipeline)
    initializer_flags = [
        tensor.is_initializer
        for tensor in pipeline.tensors
    ]
    _remove_initializer_markers(payload)
    for tensor, is_initializer in zip(payload["tensors"], initializer_flags):
        tensor["is_initializer"] = is_initializer
    return payload


def write_pipeline_json(pipeline: Pipeline, output_path: str | Path) -> Path:
    """Write one pipeline object to JSON and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pipeline_json_payload(pipeline)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
