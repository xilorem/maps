"""ONNX importer entry points."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from maps.graph.model import Graph
from maps.graph.model import ImportedModel

from .parser import parse_constants, parse_graph

if TYPE_CHECKING:
    from onnx import ModelProto


def load_onnx_model(path: str | Path) -> "ModelProto":
    """Load and validate one ONNX model from disk."""

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The optional 'onnx' package is required to load ONNX models"
        ) from exc

    model_path = Path(path)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return model


def import_onnx_graph(
    path: str | Path,
    *,
    input_shapes: InputShapes | None = None,
) -> Graph:
    """Import one ONNX model directly into the shared scheduler graph IR."""

    return import_onnx_model(path, input_shapes=input_shapes).graph


def import_onnx_model(
    path: str | Path,
    *,
    input_shapes: InputShapes | None = None,
) -> ImportedModel:
    """Import one specialized, statically shaped ONNX model."""

    onnx_model = prepare_onnx_model(
        load_onnx_model(path),
        input_shapes=input_shapes,
    )
    graph = parse_graph(
        onnx_model.graph,
        graph_name=onnx_model.graph.name or Path(path).stem,
    )
    constants = parse_constants(
        onnx_model.graph,
        names={tensor.name for tensor in graph.initializers},
    )

    model = ImportedModel(graph=graph, constants=constants)
    model.validate()
    return model
InputShapes = Mapping[str, tuple[int, ...]]


def _declared_shape(value: "ValueInfoProto") -> tuple[int | str | None, ...]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return ()

    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return tuple(shape)


def _specialize_inputs(model: "ModelProto", input_shapes: InputShapes) -> None:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = {
        value.name: value
        for value in model.graph.input
        if value.name not in initializer_names
    }
    unknown = sorted(set(input_shapes) - set(inputs))
    if unknown:
        raise ValueError(f"input shape override names unknown input '{unknown[0]}'")

    for name, shape in input_shapes.items():
        if (
            not shape
            or any(
                not isinstance(dimension, int) or dimension <= 0
                for dimension in shape
            )
        ):
            raise ValueError(
                f"input shape override for '{name}' must contain positive dimensions"
            )
        value = inputs[name]
        declared = _declared_shape(value)
        if declared and len(shape) != len(declared):
            raise ValueError(
                f"input shape override for '{name}' has rank {len(shape)}; "
                f"expected {len(declared)}"
            )
        for axis, (declared_dimension, override_dimension) in enumerate(
            zip(declared, shape)
        ):
            if (
                isinstance(declared_dimension, int)
                and declared_dimension != override_dimension
            ):
                raise ValueError(
                    f"input shape override for '{name}' changes concrete dimension "
                    f"{axis} from {declared_dimension} to {override_dimension}"
                )
        del value.type.tensor_type.shape.dim[:]
        for dimension in shape:
            value.type.tensor_type.shape.dim.add().dim_value = dimension


def _validate_static_shapes(model: "ModelProto") -> None:
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        shape = _declared_shape(value)
        if not value.type.tensor_type.HasField("shape"):
            raise ValueError(f"tensor '{value.name}' has no statically inferred shape")
        for dimension in shape:
            if isinstance(dimension, int) and dimension > 0:
                continue
            rendered = "unknown" if dimension is None else repr(dimension)
            raise ValueError(
                f"tensor '{value.name}' has dynamic dimension {rendered}; "
                "provide a concrete input shape override"
            )


def prepare_onnx_model(
    model: "ModelProto",
    input_shapes: InputShapes | None = None,
) -> "ModelProto":
    """Specialize inputs, infer shapes, and reject dynamic tensor dimensions."""

    import onnx

    prepared = deepcopy(model)
    _specialize_inputs(prepared, input_shapes or {})
    try:
        prepared = onnx.shape_inference.infer_shapes(
            prepared,
            strict_mode=True,
            data_prop=True,
        )
    except onnx.shape_inference.InferenceError as exc:
        raise ValueError(f"ONNX shape inference failed: {exc}") from exc
    _validate_static_shapes(prepared)
    onnx.checker.check_model(prepared)
    return prepared
