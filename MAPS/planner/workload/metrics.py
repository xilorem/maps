"""Compute and communication bottleneck metrics for workload allocation."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node
from MAPS.planner.contracts.stages import StagePlan, virtual_submesh
from MAPS.planner.workload.candidates import StageCandidate
from MAPS.transitions import build_virtual_transitions


@dataclass(frozen=True)
class StageMetricBreakdown:
    """Canonical intrinsic and communication costs for one selected Stage."""

    compute_cycles: int
    communication_cycles: int
    weighted_bottleneck: float


@dataclass(frozen=True)
class SelectionEvaluation:
    """Reusable global evaluation of one complete candidate selection."""

    stage_breakdowns: dict[int, StageMetricBreakdown]

    @property
    def metrics(self) -> dict[int, float]:
        """Return the weighted bottleneck used to order each Stage."""

        return {
            stage_id: breakdown.weighted_bottleneck
            for stage_id, breakdown in self.stage_breakdowns.items()
        }


def evaluate_candidate_selection(
    candidates: dict[int, StageCandidate],
    mesh: Mesh,
    compute_weight: float,
    communication_weight: float,
    graph: Graph,
) -> SelectionEvaluation:
    """Evaluate one complete Stage Candidate selection."""

    plans = {
        stage_id: candidate.plan
        for stage_id, candidate in candidates.items()
    }
    virtual_communication = _virtual_communication_cycles(graph, mesh, plans)
    return SelectionEvaluation(
        stage_breakdowns={
            stage_id: StageMetricBreakdown(
                compute_cycles=candidate.stage_compute,
                communication_cycles=max(
                    virtual_communication[stage_id].values(),
                    default=0,
                ),
                weighted_bottleneck=max(
                    compute_weight * candidate.stage_compute,
                    communication_weight
                    * max(virtual_communication[stage_id].values(), default=0),
                ),
            )
            for stage_id, candidate in candidates.items()
        }
    )


def _virtual_communication_cycles(
    graph: Graph,
    mesh: Mesh,
    plans: dict[int, StagePlan],
) -> dict[int, dict[int, int]]:
    """Estimate producer-side virtual-tile communication cycles."""

    # Virtual traffic is a pre-placement analysis shared by workload estimation
    # and spatial mapping; it does not depend on physical mapping decisions.
    from MAPS.planner.spatial.traffic import build_virtual_traffic

    virtual_transitions = build_virtual_transitions(graph, plans)
    traffic = build_virtual_traffic(virtual_transitions, plans)
    communication = {
        stage_id: {
            tile.tile_id: 0
            for tile in virtual_submesh(plan).tiles
        }
        for stage_id, plan in plans.items()
    }

    for stage_id, plan in plans.items():
        for virtual_tile in virtual_submesh(plan).tiles:
            tile_id = virtual_tile.tile_id
            l2_bytes = (
                traffic.l2_read_weights[stage_id][tile_id]
                + traffic.l2_write_weights[stage_id][tile_id]
            )
            if l2_bytes:
                communication[stage_id][tile_id] += _ceil_div(
                    l2_bytes,
                    min(virtual_tile.memory.bandwidth, mesh.l2_memory.bandwidth),
                )

    for (source_stage_id, _), matrix in traffic.edge_matrices.items():
        for (source_tile_id, destination_tile_id), bytes_ in matrix.items():
            source_tile = mesh.tile_by_id(source_tile_id)
            destination_tile = mesh.tile_by_id(destination_tile_id)
            communication[source_stage_id][source_tile_id] += _ceil_div(
                bytes_,
                min(source_tile.memory.bandwidth, destination_tile.memory.bandwidth),
            )
    return communication


def selection_objective(metrics: dict[int, float]) -> tuple[float, ...]:
    """Order stage metrics so candidates compare worst bottlenecks first."""

    return tuple(sorted(metrics.values(), reverse=True))


def worst_tile_compute_workload(
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple, ...],
    submesh,
    device_names: tuple[str | None, ...] = (),
) -> int:
    """Return the greatest accumulated compute cost on any stage tile."""

    return max(
        (
            sum(
                _node_compute_workload(
                    node,
                    output_layouts,
                    tile,
                    device_names[node_index] if device_names else None,
                )
                for node_index, (node, output_layouts) in enumerate(
                    zip(stage_nodes, node_output_layouts)
                )
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def _node_compute_workload(
    node: Node,
    output_layouts: tuple,
    tile,
    device_name: str | None,
) -> int:
    """Estimate compute cost for one node on one virtual tile."""

    tile_work = node.payload.build_tile_work(output_layouts=output_layouts, tile=tile)
    cost_model = node.payload.cost_model
    compute_cost = (
        cost_model.cost(tile_work, tile)
        if device_name is None
        else cost_model.cost(tile_work, tile, tile.device_by_name(device_name))
    )
    return int(compute_cost) + int(
        cost_model.placement_cost(node=node, output_layouts=output_layouts)
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return positive integer ceiling division."""

    if denominator <= 0:
        raise ValueError("denominator must be > 0")
    return (numerator + denominator - 1) // denominator
