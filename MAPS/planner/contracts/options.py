"""Configuration contracts for the planner and its individual passes."""

from __future__ import annotations

from dataclasses import dataclass, field

from maps.planning.options import (
    AllocationOptions as _AllocationOptions,
    StageFormationOptions as _StageFormationOptions,
)
from maps.planning.execution_plan import ExecutionContract


@dataclass(frozen=True)
class SpatialMappingOptions:
    """Control physical-placement diagnostics.

    Progress and result printing affect diagnostics only. The current heuristic
    mapper has no additional public search knobs.
    """

    print_progress: bool = False
    print_mapping: bool = True
    print_costs: bool = False


@dataclass(frozen=True)
class GraphRewriteOptions:
    """Control supported optional Graph Rewrites in canonical order."""

    enable_precision_lowering: bool = False


@dataclass(frozen=True)
class PlannerOptions:
    """Complete configuration for planning an already imported graph.

    Pass-specific options remain grouped by pass. ``execution`` is shared by
    Allocation feasibility and Execution Plan lowering.
    """

    stage_formation: _StageFormationOptions = field(
        default_factory=_StageFormationOptions
    )
    allocation: _AllocationOptions = field(default_factory=_AllocationOptions)
    spatial_mapping: SpatialMappingOptions = field(default_factory=SpatialMappingOptions)
    execution: ExecutionContract = field(default_factory=ExecutionContract)
    print_execution_plan_cost: bool = True
    graph_rewrites: GraphRewriteOptions = field(default_factory=GraphRewriteOptions)
