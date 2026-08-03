"""Public configuration for Planning."""

from __future__ import annotations

from dataclasses import dataclass, field

from maps.planning.execution_plan import ExecutionContract


@dataclass(frozen=True)
class StageFormationOptions:
    """Control deterministic formation of graph Nodes into Stages."""

    max_stage_operations: int = 0

    def __post_init__(self) -> None:
        if self.max_stage_operations < 0:
            raise ValueError("max_stage_operations must be >= 0")


@dataclass(frozen=True)
class AllocationOptions:
    """Control virtual tile allocation and its diagnostics."""

    stage_latency_weight: float = 1.0
    communication_weight: float = 10.0
    print_progress: bool = False


@dataclass(frozen=True)
class PlacementOptions:
    """Control physical placement diagnostics."""

    print_progress: bool = False
    print_placement: bool = True
    print_costs: bool = False


@dataclass(frozen=True)
class PlanningOptions:
    """Complete configuration for planning a target-specialized Graph."""

    stage_formation: StageFormationOptions = field(
        default_factory=StageFormationOptions
    )
    allocation: AllocationOptions = field(default_factory=AllocationOptions)
    placement: PlacementOptions = field(default_factory=PlacementOptions)
    execution: ExecutionContract = field(default_factory=ExecutionContract)
    print_execution_plan_cost: bool = True


__all__ = [
    "AllocationOptions",
    "PlacementOptions",
    "PlanningOptions",
    "StageFormationOptions",
]
