"""Configuration contracts for the planner and its individual passes."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.pipeline.execution import ExecutionContract


@dataclass(frozen=True)
class WorkloadBalancingOptions:
    """Weights and diagnostics used while allocating virtual stage tiles.

    ``compute_weight`` and ``communication_weight`` scale the two components of
    the bottleneck objective. They do not change legality: every returned plan
    must fit in tile L1 memory and the total allocation must fit on the mesh.
    """

    compute_weight: float = 1.0
    communication_weight: float = 10.0
    print_progress: bool = False


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
class StageSelectionOptions:
    """Deterministic graph-level stage coalescing configuration.

    Zero selects maximal eligible coalescing, one disables automatic
    coalescing, and larger values bound automatic groups by canonical nodes.
    """

    max_stage_nodes: int = 0

    def __post_init__(self) -> None:
        if self.max_stage_nodes < 0:
            raise ValueError("max_stage_nodes must be >= 0")


@dataclass(frozen=True)
class GraphRewriteOptions:
    """Control supported optional Graph Rewrites in canonical order."""

    enable_precision_lowering: bool = False


@dataclass(frozen=True)
class PlannerOptions:
    """Complete configuration for planning an already imported graph.

    Pass-specific options remain grouped by pass. ``execution`` is shared by
    workload feasibility and Execution Plan lowering.
    """

    stage_selection: StageSelectionOptions = field(default_factory=StageSelectionOptions)
    workload: WorkloadBalancingOptions = field(default_factory=WorkloadBalancingOptions)
    spatial_mapping: SpatialMappingOptions = field(default_factory=SpatialMappingOptions)
    execution: ExecutionContract = field(default_factory=ExecutionContract)
    print_execution_plan_cost: bool = True
    graph_rewrites: GraphRewriteOptions = field(default_factory=GraphRewriteOptions)
