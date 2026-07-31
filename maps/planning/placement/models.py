"""Data contracts shared by Placement phases."""

from __future__ import annotations

from dataclasses import dataclass

from maps.planning.stages import StagePlacement


@dataclass(frozen=True)
class VirtualTraffic:
    """Virtual communication summary between already balanced stages.

    Matrices retain virtual tile ids so traffic can guide both physical-region
    selection and the later virtual-to-physical ownership assignment.
    """

    stage_comm: dict[tuple[int, int], int]
    edge_matrices: dict[tuple[int, int], dict[tuple[int, int], int]]
    input_weights: dict[int, dict[int, int]]
    output_weights: dict[int, dict[int, int]]
    l2_read_weights: dict[int, dict[int, int]]
    l2_write_weights: dict[int, dict[int, int]]
    communication_degree: dict[int, int]
    bottleneck_risk: dict[int, int]
    l2_pressure: dict[int, int]


@dataclass(frozen=True)
class TileIOScore:
    """Exact physical IO accounting for one tile."""

    tile_id: int
    stage_id: int | None
    tile_to_tile_writes: int
    l2_reads: int
    l2_writes: int
    consumer_stage_writes: dict[int, int]

    @property
    def score(self) -> int:
        """Return the additive physical IO score for one tile."""

        return self.tile_to_tile_writes + self.l2_reads + self.l2_writes


@dataclass(frozen=True)
class StageIOBreakdown:
    """Worst physical tile IO components for one placed stage."""

    physical_tile_id: int | None
    l2_read: int
    l2_write: int
    l1_write: int

    @property
    def total(self) -> int:
        """Return the additive physical IO score of the worst tile."""

        return self.l1_write + self.l2_read + self.l2_write


@dataclass(frozen=True)
class PlacementEvaluation:
    """Exact score for a complete ownership-aware Placement."""

    placements: dict[int, StagePlacement]
    tile_scores: dict[int, TileIOScore]
    stage_breakdowns: dict[int, StageIOBreakdown]
    objective: tuple[int, int, int, int]
    worst_tile_id: int | None


@dataclass(frozen=True)
class RepairCandidate:
    """A local collection of stages that may improve the current bottleneck."""

    stages: frozenset[int]
    priority: float
    reason: str
