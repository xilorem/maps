"""Validated inputs and graph classifications for workload balancing."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.graph import Graph
from MAPS.planner.contracts.stages import StageSelection
from MAPS.planner.validation.stages import validate_stage_selection


@dataclass(frozen=True)
class WorkloadContext:
    """Validated workload inputs shared by candidate allocation."""

    graph: Graph
    stage_selection: StageSelection
    initializer_tensors: frozenset


def build_workload_context(
    graph: Graph,
    stage_selection: StageSelection,
) -> WorkloadContext:
    """Validate Stage coverage and retain intrinsic workload inputs."""

    resolved_selection = validate_stage_selection(graph, stage_selection)
    initializer_tensors = frozenset(graph.initializers)
    return WorkloadContext(
        graph=graph,
        stage_selection=resolved_selection,
        initializer_tensors=initializer_tensors,
    )
