"""Central operation registry."""

from __future__ import annotations

from .spec import OnnxLoweringFn, OpSpec

_OPS_BY_NAME: dict[str, OpSpec] = {}
_OPS_BY_ONNX_NAME: dict[str, OpSpec] = {}
_BUILTINS_LOADED = False


def _ensure_builtins_registered() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from MAPS.ops.defs import (  # noqa: F401
        conv,
        elementwise,
        gemm,
        group_norm,
        rearrange,
        reduction,
        softmax,
        split,
    )

    _BUILTINS_LOADED = True


def register_op(spec: OpSpec) -> None:
    """Atomically register one operation frontend specification."""

    from maps.graph.onnx.operations import get_operation_converter

    existing = _OPS_BY_NAME.get(spec.name)
    if existing is not None:
        raise ValueError(f"duplicate op spec name: {spec.name}")

    for onnx_name in spec.onnx_names:
        if get_operation_converter(onnx_name) is not None:
            raise ValueError(f"duplicate ONNX op mapping for {onnx_name}")
        existing = _OPS_BY_ONNX_NAME.get(onnx_name)
        if existing is not None:
            raise ValueError(
                f"duplicate ONNX op mapping for {onnx_name}: {existing.name} vs {spec.name}"
            )

    _OPS_BY_NAME[spec.name] = spec
    for onnx_name in spec.onnx_names:
        _OPS_BY_ONNX_NAME[onnx_name] = spec


def get_op(name: str) -> OpSpec:
    """Return one registered op spec by canonical name."""

    _ensure_builtins_registered()
    try:
        return _OPS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown op spec: {name}") from exc


def get_onnx_lowerer(onnx_op_type: str) -> OnnxLoweringFn | None:
    """Return the registered ONNX lowerer for one external op type."""

    from maps.graph.onnx.operations import get_operation_converter

    explicit_converter = get_operation_converter(onnx_op_type)
    if explicit_converter is not None:
        return explicit_converter
    _ensure_builtins_registered()
    spec = _OPS_BY_ONNX_NAME.get(onnx_op_type)
    if spec is None:
        return None
    return spec.lower_onnx


def get_op_by_onnx_name(onnx_op_type: str) -> OpSpec | None:
    """Return the op spec that handles one ONNX op type."""

    _ensure_builtins_registered()
    return _OPS_BY_ONNX_NAME.get(onnx_op_type)


def registered_ops() -> tuple[OpSpec, ...]:
    """Return all registered operation specs."""

    _ensure_builtins_registered()
    return tuple(_OPS_BY_NAME.values())


def registered_onnx_lowerers() -> dict[str, OnnxLoweringFn]:
    """Return the registered ONNX lowerers keyed by ONNX op type."""

    from maps.graph.onnx.operations import ONNX_OPERATION_CONVERTERS

    _ensure_builtins_registered()
    registered = {
        onnx_name: spec.lower_onnx
        for onnx_name, spec in _OPS_BY_ONNX_NAME.items()
        if spec.lower_onnx is not None
    }
    return {**registered, **ONNX_OPERATION_CONVERTERS}
