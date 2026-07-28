"""Exact ownership-aware scoring for complete spatial mappings."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.core.layout import tile_tensor_slice
from MAPS.planner.contracts.queries import (
    node_output_index,
    node_output_layouts,
    required_input_slices,
)
from MAPS.planner.contracts.stages import StagePlacement, StagePlan
from MAPS.planner.spatial.models import MappingEvaluation, StageIOBreakdown, TileIOScore
from MAPS.transitions import build_direct_remap_fragments
from MAPS.transitions.transport import TransportCostModel


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


class MappingEvaluator:
    """Compile logical transfers once and score complete or locally changed mappings."""

    def __init__(
        self,
        graph: Graph,
        mesh: Mesh,
        stage_plans: dict[int, StagePlan],
        node_stage_ids: dict[int, int],
    ) -> None:
        self._mesh = mesh
        self._stage_plans = stage_plans
        self._model = TransportCostModel(mesh=mesh)
        (
            self._l1_transfers_by_source,
            self._l2_reads_by_stage,
            self._l2_writes_by_stage,
            self._source_dependencies_by_destination,
        ) = _compile_transfers(graph, stage_plans, node_stage_ids)

    def evaluate(
        self,
        placements: dict[int, StagePlacement],
        previous: MappingEvaluation | None = None,
        moved_stage_ids: frozenset[int] = frozenset(),
    ) -> MappingEvaluation:
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
        return _mapping_evaluation(placements, tile_scores)

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


def evaluate_mapping(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    node_stage_ids: dict[int, int],
) -> MappingEvaluation:
    """Compute the exact physical IO objective for a complete mapping.

    Contract:
        Every stage must have a disjoint physical placement and a complete
        virtual-to-physical ownership map.  ``node_stage_ids`` must cover all
        nodes contained in the supplied plans.

    Behavior:
        Graph inputs are charged as L2 reads on consumer tiles, graph outputs as
        L2 writes on producer tiles, and inter-stage fragments as routed L1
        writes on producer tiles.  Strided transfers retain row information for
        the transport model.

    Returns:
        Per-tile and per-stage breakdowns plus a deterministic lexicographic
        objective used by local repair.
    """

    return MappingEvaluator(
        graph,
        mesh,
        stage_plans,
        node_stage_ids,
    ).evaluate(placements)


def _compile_transfers(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
    node_stage_ids: dict[int, int],
) -> tuple[
    dict[int, tuple[_L1Transfer, ...]],
    dict[int, tuple[_L2Transfer, ...]],
    dict[int, tuple[_L2Transfer, ...]],
    dict[int, frozenset[int]],
]:
    producer_by_tensor = {
        tensor: node
        for node in graph.nodes
        for tensor in node.outputs
    }
    initializer_tensors = frozenset(graph.initializers)
    graph_inputs = frozenset(graph.inputs) - initializer_tensors
    graph_outputs = frozenset(graph.outputs)
    l1_transfers: dict[int, list[_L1Transfer]] = {}
    l2_reads: dict[int, list[_L2Transfer]] = {}
    l2_writes: dict[int, list[_L2Transfer]] = {}
    source_dependencies: dict[int, set[int]] = {}

    for destination_stage_id, destination_plan in stage_plans.items():
        for destination_node, output_layouts in zip(
            destination_plan.nodes,
            destination_plan.node_output_layouts,
        ):
            for tensor in destination_node.inputs:
                if tensor in initializer_tensors:
                    continue
                required_slices = required_input_slices(
                    tensor=tensor,
                    destination_node=destination_node,
                    destination_output_layouts=output_layouts,
                )
                source_node = producer_by_tensor.get(tensor)
                if tensor in graph_inputs or source_node is None:
                    for virtual_tile, tensor_slice in required_slices:
                        l2_reads.setdefault(destination_stage_id, []).append(
                            _L2Transfer(
                                stage_id=destination_stage_id,
                                virtual_tile_id=virtual_tile.tile_id,
                                bytes=tensor.slice_num_bytes(tensor_slice),
                            )
                        )
                    continue

                source_stage_id = node_stage_ids[id(source_node)]
                if source_stage_id == destination_stage_id:
                    continue
                source_layout = node_output_layouts(
                    stage_plans[source_stage_id],
                    source_node,
                )[node_output_index(source_node, tensor)]
                fragments = build_direct_remap_fragments(
                    tensor=tensor,
                    src_layout=source_layout,
                    dst_required_slices=required_slices,
                )
                source_dependencies.setdefault(destination_stage_id, set()).add(
                    source_stage_id
                )
                for fragment in fragments:
                    bytes_ = fragment.src_subslice.num_elements * tensor.elem_bytes
                    row_bytes, rows = _fragment_row_shape(fragment, tensor.elem_bytes)
                    l1_transfers.setdefault(source_stage_id, []).append(
                        _L1Transfer(
                            source_stage_id=source_stage_id,
                            destination_stage_id=destination_stage_id,
                            source_virtual_tile_id=fragment.src_hartid,
                            destination_virtual_tile_id=fragment.dst_hartid,
                            bytes=bytes_,
                            row_bytes=row_bytes,
                            rows=rows,
                        )
                    )

            for output_idx, tensor in enumerate(destination_node.outputs):
                if tensor not in graph_outputs:
                    continue
                output_layout = output_layouts[output_idx]
                for virtual_tile in output_layout.submesh.tiles:
                    output_slice = tile_tensor_slice(tensor, output_layout, virtual_tile)
                    row_bytes, rows = _output_row_shape(tensor, output_slice)
                    l2_writes.setdefault(destination_stage_id, []).append(
                        _L2Transfer(
                            stage_id=destination_stage_id,
                            virtual_tile_id=virtual_tile.tile_id,
                            bytes=tensor.slice_num_bytes(output_slice),
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


def _mapping_evaluation(
    placements: dict[int, StagePlacement],
    tile_scores: dict[int, TileIOScore],
) -> MappingEvaluation:
    objective = tile_score_objective(tile_scores)
    worst_tile_id = max(
        tile_scores,
        key=lambda tile_id: (tile_scores[tile_id].score, -tile_id),
        default=None,
    )
    stage_breakdowns = _stage_breakdowns(placements, tile_scores)
    return MappingEvaluation(
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
    """Return deterministic max-first aggregate mapping objectives."""

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


def _fragment_row_shape(fragment, element_bytes: int) -> tuple[int | None, int]:
    """Describe strided fragment rows for the transport model."""

    if fragment.src_subslice.rank < 2:
        return None, 1
    source_inner = fragment.src_subslice.dims[-1]
    destination_inner = fragment.dst_subslice.dims[-1]
    if source_inner.length != destination_inner.length:
        return None, 1
    if (
        source_inner.length == fragment.src_subslice.parent.dims[-1].length
        and destination_inner.length == fragment.dst_subslice.parent.dims[-1].length
    ):
        return None, 1
    return (
        source_inner.length * element_bytes,
        fragment.src_subslice.num_elements // source_inner.length,
    )


def _output_row_shape(tensor, output_slice) -> tuple[int | None, int]:
    """Describe strided output rows for an L2 write."""

    if tensor.rank < 2 or output_slice.dims[-1].length >= tensor.dims[-1]:
        return None, 1
    return (
        output_slice.dims[-1].length * tensor.elem_bytes,
        output_slice.num_elements // output_slice.dims[-1].length,
    )
