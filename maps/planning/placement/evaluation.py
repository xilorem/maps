"""Exact ownership-aware scoring for complete Placements."""

from __future__ import annotations

from dataclasses import dataclass

from maps.graph import Node
from maps.hardware import Mesh
from maps.planning.mapping import tensor_slice_num_bytes
from maps.planning.stages import StagePlacement, StagePlan, virtual_submesh
from maps.planning.transitions import (
    VirtualInputTransition,
    VirtualIntermediateTransition,
    VirtualOutputTransition,
    VirtualTransfer,
    VirtualTransition,
)
from maps.planning.transitions.transport import TransportCostModel


@dataclass(frozen=True)
class _L1Transfer:
    source_stage_id: int
    destination_stage_id: int
    source_virtual_tile_id: int
    destination_virtual_tile_id: int
    bytes: int
    row_bytes: int | None
    rows: int


@dataclass(frozen=True)
class _L2Transfer:
    stage_id: int
    virtual_tile_id: int
    bytes: int
    row_bytes: int | None = None
    rows: int = 1


class PlacementEvaluator:
    """Compile logical transfers once and score complete or locally changed Placements."""

    def __init__(
        self,
        mesh: Mesh,
        stage_plans: dict[int, StagePlan],
        virtual_transitions: tuple[VirtualTransition, ...],
    ) -> None:
        self._mesh = mesh
        self._stage_plans = stage_plans
        self._model = TransportCostModel(mesh=mesh)
        (
            self._l1_transfers_by_source,
            self._l2_reads_by_stage,
            self._l2_writes_by_stage,
            self._source_dependencies_by_destination,
        ) = _compile_transfers(virtual_transitions)

    def evaluate(
        self,
        placements: dict[int, StagePlacement],
        previous: PlacementEvaluation | None = None,
        moved_stage_ids: frozenset[int] = frozenset(),
    ) -> PlacementEvaluation:
        """Return the exact score, reusing unaffected stage scores when possible."""

        if previous is None:
            score_stage_ids = frozenset(self._stage_plans)
        else:
            score_stage_ids = moved_stage_ids | frozenset(
                source_stage_id
                for destination_stage_id in moved_stage_ids
                for source_stage_id in self._source_dependencies_by_destination.get(
                    destination_stage_id,
                    (),
                )
            )

        tile_scores = self._score_stages(placements, score_stage_ids)
        if previous is not None:
            tile_scores.update(
                (tile_id, score)
                for tile_id, score in previous.tile_scores.items()
                if score.stage_id not in score_stage_ids
            )
        return _placement_evaluation(placements, tile_scores)

    def _score_stages(
        self,
        placements: dict[int, StagePlacement],
        stage_ids: frozenset[int],
    ) -> dict[int, TileIOScore]:
        tile_writes = {
            tile_id: 0
            for stage_id in stage_ids
            for tile_id in placements[stage_id].physical_submesh.tile_ids
        }
        tile_l2_reads = dict.fromkeys(tile_writes, 0)
        tile_l2_writes = dict.fromkeys(tile_writes, 0)
        consumer_stage_writes = {tile_id: {} for tile_id in tile_writes}

        for source_stage_id in stage_ids:
            source_placement = placements[source_stage_id]
            for transfer in self._l1_transfers_by_source.get(source_stage_id, ()):
                destination_placement = placements[transfer.destination_stage_id]
                source_tile_id = source_placement.physical_tile_id(
                    transfer.source_virtual_tile_id
                )
                destination_tile_id = destination_placement.physical_tile_id(
                    transfer.destination_virtual_tile_id
                )
                transfer_cost = self._model.l1_to_l1(
                    self._mesh.tile_by_id(source_tile_id),
                    self._mesh.tile_by_id(destination_tile_id),
                    transfer.bytes,
                    row_bytes=transfer.row_bytes,
                    rows=transfer.rows,
                )
                tile_writes[source_tile_id] += transfer_cost
                stage_writes = consumer_stage_writes[source_tile_id]
                destination_stage_id = transfer.destination_stage_id
                stage_writes[destination_stage_id] = (
                    stage_writes.get(destination_stage_id, 0) + transfer_cost
                )

            for transfer in self._l2_reads_by_stage.get(source_stage_id, ()):
                tile_id = source_placement.physical_tile_id(transfer.virtual_tile_id)
                tile_l2_reads[tile_id] += self._model.l2_to_l1(
                    self._mesh.tile_by_id(tile_id),
                    transfer.bytes,
                    row_bytes=transfer.row_bytes,
                    rows=transfer.rows,
                )
            for transfer in self._l2_writes_by_stage.get(source_stage_id, ()):
                tile_id = source_placement.physical_tile_id(transfer.virtual_tile_id)
                tile_l2_writes[tile_id] += self._model.l1_to_l2(
                    self._mesh.tile_by_id(tile_id),
                    transfer.bytes,
                    row_bytes=transfer.row_bytes,
                    rows=transfer.rows,
                )

        stage_of_tile = {
            tile_id: stage_id
            for stage_id in stage_ids
            for tile_id in placements[stage_id].physical_submesh.tile_ids
        }
        return {
            tile_id: TileIOScore(
                tile_id=tile_id,
                stage_id=stage_id,
                tile_to_tile_writes=tile_writes[tile_id],
                l2_reads=tile_l2_reads[tile_id],
                l2_writes=tile_l2_writes[tile_id],
                consumer_stage_writes=dict(
                    sorted(consumer_stage_writes[tile_id].items())
                ),
            )
            for tile_id, stage_id in stage_of_tile.items()
        }


def evaluate_placement(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    virtual_transitions: tuple[VirtualTransition, ...],
) -> PlacementEvaluation:
    """Compute the exact physical IO objective for a complete Placement.

    Contract:
        Every stage must have a disjoint physical placement and a complete
        virtual-to-physical ownership map. ``virtual_transitions`` must have
        been compiled from the supplied plans.

    Behavior:
        Graph inputs are charged as L2 reads on consumer tiles, graph outputs as
        L2 writes on producer tiles, and inter-stage Transfers as routed L1
        writes on producer tiles.  Strided transfers retain row information for
        the transport model.

    Returns:
        Per-tile and per-stage breakdowns plus a deterministic lexicographic
        objective used by local repair.
    """

    return PlacementEvaluator(
        mesh,
        stage_plans,
        virtual_transitions,
    ).evaluate(placements)


def _compile_transfers(
    virtual_transitions: tuple[VirtualTransition, ...],
) -> tuple[
    dict[int, tuple[_L1Transfer, ...]],
    dict[int, tuple[_L2Transfer, ...]],
    dict[int, tuple[_L2Transfer, ...]],
    dict[int, frozenset[int]],
]:
    l1_transfers: dict[int, list[_L1Transfer]] = {}
    l2_reads: dict[int, list[_L2Transfer]] = {}
    l2_writes: dict[int, list[_L2Transfer]] = {}
    source_dependencies: dict[int, set[int]] = {}

    for transition in virtual_transitions:
        if isinstance(transition, VirtualInputTransition):
            for destination in transition.destinations:
                l2_reads.setdefault(transition.destination_stage_id, []).append(
                    _L2Transfer(
                        stage_id=transition.destination_stage_id,
                        virtual_tile_id=destination.virtual_tile_id,
                        bytes=tensor_slice_num_bytes(
                            transition.tensor, destination.tensor_slice
                        ),
                    )
                )
        elif isinstance(transition, VirtualIntermediateTransition):
            if transition.transfers:
                source_dependencies.setdefault(
                    transition.destination_stage_id,
                    set(),
                ).add(transition.source_stage_id)
            for transfer in transition.transfers:
                row_bytes, rows = _transfer_row_shape(
                    transfer,
                    transition.tensor.elem_bytes,
                )
                l1_transfers.setdefault(transition.source_stage_id, []).append(
                    _L1Transfer(
                        source_stage_id=transition.source_stage_id,
                        destination_stage_id=transition.destination_stage_id,
                        source_virtual_tile_id=transfer.source_virtual_tile_id,
                        destination_virtual_tile_id=(
                            transfer.destination_virtual_tile_id
                        ),
                        bytes=(
                            transfer.source_subslice.num_elements
                            * transition.tensor.elem_bytes
                        ),
                        row_bytes=row_bytes,
                        rows=rows,
                    )
                )
        elif isinstance(transition, VirtualOutputTransition):
            for source in transition.sources:
                row_bytes, rows = _output_row_shape(
                    transition.tensor,
                    source.tensor_slice,
                )
                l2_writes.setdefault(transition.source_stage_id, []).append(
                    _L2Transfer(
                        stage_id=transition.source_stage_id,
                        virtual_tile_id=source.virtual_tile_id,
                        bytes=tensor_slice_num_bytes(
                            transition.tensor, source.tensor_slice
                        ),
                        row_bytes=row_bytes,
                        rows=rows,
                    )
                )

    return (
        {stage_id: tuple(transfers) for stage_id, transfers in l1_transfers.items()},
        {stage_id: tuple(transfers) for stage_id, transfers in l2_reads.items()},
        {stage_id: tuple(transfers) for stage_id, transfers in l2_writes.items()},
        {
            stage_id: frozenset(source_stage_ids)
            for stage_id, source_stage_ids in source_dependencies.items()
        },
    )


def _placement_evaluation(
    placements: dict[int, StagePlacement],
    tile_scores: dict[int, TileIOScore],
) -> PlacementEvaluation:
    objective = tile_score_objective(tile_scores)
    worst_tile_id = max(
        tile_scores,
        key=lambda tile_id: (tile_scores[tile_id].score, -tile_id),
        default=None,
    )
    stage_breakdowns = _stage_breakdowns(placements, tile_scores)
    return PlacementEvaluation(
        placements=placements,
        tile_scores=tile_scores,
        stage_breakdowns=stage_breakdowns,
        objective=objective,
        worst_tile_id=worst_tile_id,
    )


def tile_score_objective(
    tile_scores: dict[int, TileIOScore],
    k: int = 5,
) -> tuple[int, int, int, int]:
    """Return deterministic max-first aggregate Placement objectives."""

    scores = sorted(
        (score.score for score in tile_scores.values()),
        reverse=True,
    )
    return (
        scores[0] if scores else 0,
        scores[1] if len(scores) > 1 else 0,
        sum(scores[:k]),
        sum(scores),
    )


def _stage_breakdowns(
    placements: dict[int, StagePlacement],
    tile_scores: dict[int, TileIOScore],
) -> dict[int, StageIOBreakdown]:
    """Select the worst physical tile in every stage."""

    breakdowns = {}
    for stage_id, placement in placements.items():
        worst_tile = max(
            placement.physical_submesh.tile_ids,
            key=lambda tile_id: (tile_scores[tile_id].score, -tile_id),
            default=None,
        )
        if worst_tile is None:
            breakdowns[stage_id] = StageIOBreakdown(None, 0, 0, 0)
            continue
        score = tile_scores[worst_tile]
        breakdowns[stage_id] = StageIOBreakdown(
            physical_tile_id=worst_tile,
            l2_read=score.l2_reads,
            l2_write=score.l2_writes,
            l1_write=score.tile_to_tile_writes,
        )
    return breakdowns


def _transfer_row_shape(
    transfer: VirtualTransfer,
    element_bytes: int,
) -> tuple[int | None, int]:
    """Describe a canonical Virtual Transfer's strided rows."""

    if transfer.source_subslice.rank < 2:
        return None, 1
    source_inner = transfer.source_subslice.dims[-1]
    destination_inner = transfer.destination_subslice.dims[-1]
    if source_inner.length != destination_inner.length:
        return None, 1
    if (
        source_inner.length == transfer.source_subslice.parent.dims[-1].length
        and destination_inner.length
        == transfer.destination_subslice.parent.dims[-1].length
    ):
        return None, 1
    return (
        source_inner.length * element_bytes,
        transfer.source_subslice.num_elements // source_inner.length,
    )


def _output_row_shape(tensor, output_slice) -> tuple[int | None, int]:
    """Describe strided output rows for an L2 write."""

    if tensor.rank < 2 or output_slice.dims[-1].length >= tensor.dims[-1]:
        return None, 1
    return (
        output_slice.dims[-1].length * tensor.elem_bytes,
        output_slice.num_elements // output_slice.dims[-1].length,
    )


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


def build_virtual_traffic(
    virtual_transitions: tuple[VirtualTransition, ...],
    stage_plans: dict[int, StagePlan],
) -> VirtualTraffic:
    """Describe all stage communication before physical tiles are selected.

    Contract:
        Stage plans must contain final virtual layouts. ``virtual_transitions``
        must have been compiled from the same complete Stage Plan set. Physical
        placement is deliberately absent from this analysis.

    Returns:
        Per-edge virtual-tile byte matrices plus aggregate input, output, L2,
        communication-degree, and bottleneck-pressure weights.  These values are
        byte counts, not transport cycles.
    """

    stage_ids = tuple(stage_plans)

    stage_comm: dict[tuple[int, int], int] = {}
    edge_matrices: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    input_weights = _empty_stage_tile_weights(stage_plans)
    output_weights = _empty_stage_tile_weights(stage_plans)
    l2_read_weights = _empty_stage_tile_weights(stage_plans)
    l2_write_weights = _empty_stage_tile_weights(stage_plans)

    for transition in virtual_transitions:
        if isinstance(transition, VirtualInputTransition):
            for destination in transition.destinations:
                bytes_ = tensor_slice_num_bytes(
                    transition.tensor, destination.tensor_slice
                )
                tile_id = destination.virtual_tile_id
                input_weights[transition.destination_stage_id][tile_id] += bytes_
                l2_read_weights[transition.destination_stage_id][tile_id] += bytes_
        elif isinstance(transition, VirtualIntermediateTransition):
            edge = (
                transition.source_stage_id,
                transition.destination_stage_id,
            )
            matrix = edge_matrices.setdefault(edge, {})
            for transfer in transition.transfers:
                bytes_ = (
                    transfer.source_subslice.num_elements
                    * transition.tensor.elem_bytes
                )
                key = (
                    transfer.source_virtual_tile_id,
                    transfer.destination_virtual_tile_id,
                )
                matrix[key] = matrix.get(key, 0) + bytes_
                stage_comm[edge] = stage_comm.get(edge, 0) + bytes_
                output_weights[transition.source_stage_id][
                    transfer.source_virtual_tile_id
                ] += bytes_
                input_weights[transition.destination_stage_id][
                    transfer.destination_virtual_tile_id
                ] += bytes_
        elif isinstance(transition, VirtualOutputTransition):
            for source in transition.sources:
                bytes_ = tensor_slice_num_bytes(transition.tensor, source.tensor_slice)
                tile_id = source.virtual_tile_id
                output_weights[transition.source_stage_id][tile_id] += bytes_
                l2_write_weights[transition.source_stage_id][tile_id] += bytes_

    communication_degree = {
        stage_id: sum(
            weight
            for (source_stage_id, destination_stage_id), weight in stage_comm.items()
            if source_stage_id == stage_id or destination_stage_id == stage_id
        )
        for stage_id in stage_ids
    }
    bottleneck_risk = {
        stage_id: max(input_weights[stage_id].values(), default=0)
        for stage_id in stage_ids
    }
    l2_pressure = {
        stage_id: (
            sum(l2_read_weights[stage_id].values())
            + sum(l2_write_weights[stage_id].values())
        )
        for stage_id in stage_ids
    }
    return VirtualTraffic(
        stage_comm=stage_comm,
        edge_matrices=edge_matrices,
        input_weights=input_weights,
        output_weights=output_weights,
        l2_read_weights=l2_read_weights,
        l2_write_weights=l2_write_weights,
        communication_degree=communication_degree,
        bottleneck_risk=bottleneck_risk,
        l2_pressure=l2_pressure,
    )


def _empty_stage_tile_weights(
    stage_plans: dict[int, StagePlan],
) -> dict[int, dict[int, int]]:
    """Build zero-valued virtual-tile weights for every stage."""

    return {
        stage_id: {
            tile.tile_id: 0
            for tile in virtual_submesh(plan).tiles
        }
        for stage_id, plan in stage_plans.items()
    }


def print_placement_details(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    virtual_transitions: tuple[VirtualTransition, ...],
    label: str = "placement",
) -> None:
    """Print physical regions, ownership maps, and exact IO bottlenecks."""

    evaluation = evaluate_placement(
        mesh,
        stage_plans,
        placements,
        virtual_transitions,
    )
    print(f"\n[placement] chosen physical submeshes for {label}:")
    for stage_id in stage_plans:
        placement = placements[stage_id]
        submesh = placement.physical_submesh
        print(
            f"  stage={stage_id} name={_stage_name(stage_plans[stage_id].nodes)} "
            f"bbox=({submesh.x0},{submesh.y0},{submesh.width},{submesh.height}) "
            f"tiles={sorted(submesh.tile_ids)} "
            f"virtual_to_physical={dict(sorted(placement.virtual_to_physical.items()))}"
        )
    print_placement_grid(mesh, placements)
    print(f"[placement] stage worst physical-tile IO costs for {label}:")
    for stage_id in stage_plans:
        io_cost = evaluation.stage_breakdowns[stage_id]
        print(
            f"  stage={stage_id} name={_stage_name(stage_plans[stage_id].nodes)} "
            f"tile={io_cost.physical_tile_id} l2_read={io_cost.l2_read} "
            f"l2_write={io_cost.l2_write} l1_write={io_cost.l1_write} "
            f"total={io_cost.total}"
        )
    print(
        f"[placement] bottleneck for {label} "
        f"worst_stage_io={max((cost.total for cost in evaluation.stage_breakdowns.values()), default=0)} "
        f"objective={evaluation.objective}"
    )


def print_placement_grid(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
) -> None:
    """Print a compact mesh grid showing physical stage ownership."""

    owners = {
        tile_id: stage_id
        for stage_id, placement in placements.items()
        for tile_id in placement.physical_submesh.tile_ids
    }
    cell_width = max(1, *(len(str(stage_id)) for stage_id in placements))
    print("Placement mesh:")
    for y in range(mesh.height):
        cells = []
        for x in range(mesh.width):
            owner = owners.get(mesh.tile_id(x, y))
            cells.append(("." if owner is None else str(owner)).rjust(cell_width))
        print(" ".join(cells))


def _stage_name(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact selected-stage display name."""

    return "+".join(node.name for node in stage_nodes)
