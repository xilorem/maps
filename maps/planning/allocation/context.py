"""Validated inputs and Graph classifications for Allocation."""

from __future__ import annotations

from dataclasses import dataclass

from maps.graph import Graph
from maps.planning.stages import StageFormation
from maps.planning.stage_validation import validate_stage_formation


@dataclass(frozen=True)
class AllocationContext:
    """Validated inputs shared by Stage Candidate Allocation."""

    graph: Graph
    stage_formation: StageFormation
    initializer_tensors: frozenset


def build_allocation_context(
    graph: Graph,
    stage_formation: StageFormation,
) -> AllocationContext:
    """Validate Stage coverage and retain intrinsic Allocation inputs."""

    resolved_selection = validate_stage_formation(graph, stage_formation)
    initializer_tensors = frozenset(graph.initializers)
    return AllocationContext(
        graph=graph,
        stage_formation=resolved_selection,
        initializer_tensors=initializer_tensors,
    )
