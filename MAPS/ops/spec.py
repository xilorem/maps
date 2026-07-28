"""Operation metadata used by frontends and legalization passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from MAPS.arch import WorkKind
from MAPS.core.graph import OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.common import OperationPayload

STATIC_INPUT_VALUES = "_static_input_values"


@dataclass(frozen=True)
class LoweredOperation:
    """Canonical operands and payload produced by one frontend lowerer."""

    kind: OpKind
    payload: OperationPayload
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]


OnnxLoweringResult = tuple[OpKind, OperationPayload] | LoweredOperation

OnnxLoweringFn = Callable[
    [str, tuple[Tensor, ...], tuple[Tensor, ...], dict[str, object]],
    OnnxLoweringResult,
]


@dataclass(frozen=True)
class OpSpec:
    """Frontend metadata for one registered operation.

    Execution behavior belongs to the returned payload. The registry only
    connects external operation names to lowerers and exposes discoverable
    metadata; it is not involved in planning or composite decomposition.
    """

    name: str
    onnx_names: tuple[str, ...] = ()
    lower_onnx: OnnxLoweringFn | None = None
    work_kinds: tuple[WorkKind, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or self.name != self.name.lower():
            raise ValueError(
                "op spec name must be a lowercase Python-style identifier"
            )
        if len(set(self.onnx_names)) != len(self.onnx_names):
            raise ValueError(f"op spec {self.name} contains duplicate ONNX names")
        if any(not name for name in self.onnx_names):
            raise ValueError(f"op spec {self.name} contains an empty ONNX name")
        if self.onnx_names and self.lower_onnx is None:
            raise ValueError(
                f"op spec {self.name} declares ONNX names without an ONNX lowerer"
            )
        if len(set(self.work_kinds)) != len(self.work_kinds):
            raise ValueError(f"op spec {self.name} contains duplicate work kinds")
        if any(not isinstance(kind, WorkKind) for kind in self.work_kinds):
            raise TypeError(f"op spec {self.name} contains an invalid work kind")
