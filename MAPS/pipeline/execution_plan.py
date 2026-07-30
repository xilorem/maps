"""Complete physical execution-plan IR."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.arch import Mesh
from MAPS.core.tensor import Tensor
from MAPS.pipeline.execution import ExecutionContract
from MAPS.pipeline.stage import Stage
from MAPS.transitions.contracts import Transition


@dataclass(frozen=True)
class ExecutionPlan:
    """One complete physical execution decision."""

    name: str
    mesh: Mesh
    tensors: tuple[Tensor, ...] = field(default_factory=tuple)
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    transitions: tuple[Transition, ...] = field(default_factory=tuple)
    execution: ExecutionContract = field(default_factory=ExecutionContract)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("execution plan name must not be empty")
