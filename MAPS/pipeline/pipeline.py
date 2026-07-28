"""Pipeline IR for one scheduled execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.arch import Mesh
from MAPS.core.tensor import Tensor
from MAPS.pipeline.execution import ExecutionContract
from MAPS.pipeline.finalization import Finalization
from MAPS.pipeline.initialization import Initialization
from MAPS.pipeline.stage import Stage
from MAPS.transitions.model import Transition


@dataclass(frozen=True)
class Pipeline:
    """One generated pipeline plan."""

    name: str
    mesh: Mesh
    tensors: tuple[Tensor, ...] = field(default_factory=tuple)
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    initializations: tuple[Initialization, ...] = field(default_factory=tuple)
    finalizations: tuple[Finalization, ...] = field(default_factory=tuple)
    transitions: tuple[Transition, ...] = field(default_factory=tuple)
    execution: ExecutionContract = field(default_factory=ExecutionContract)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pipeline name must not be empty")
