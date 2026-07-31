"""Compute and communication bottleneck metrics for workload allocation."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node
from MAPS.core.layout import tile_tensor_slice
from MAPS.planner.contracts.stages import StagePlan, StageSelection, virtual_submesh
from MAPS.planner.workload.candidates import StageCandidate
from MAPS.planner.workload.submesh import representative_connected_submesh
from MAPS.transitions import build_virtual_transitions


def evaluate_candidate_selection(
    candidates: dict[int, StageCandidate],
    mesh: Mesh,
    compute_weight: float,
    communication_weight: float,
    graph: Graph,
) -> dict[int, float]:
    """Evaluate one complete Stage Candidate selection."""

    plans = {
        stage_id: candidate.plan
        for stage_id, candidate in candidates.items()
    }
    virtual_communication = virtual_communication_cycles(graph, mesh, plans)
    return {
        stage_id: max(
            (
                max(
                    compute_weight * fact.compute_cycles,
                    communication_weight
                    * virtual_communication[stage_id][fact.tile_id],
                )
                for fact in candidate.tile_facts
            ),
            default=0.0,
        )
        for stage_id, candidate in candidates.items()
    }


def estimate_selection_metrics(
    plans: dict[int, StagePlan],
    stage_selection: StageSelection,
    mesh: Mesh,
    compute_weight: float,
    communication_weight: float,
    graph: Graph,
) -> dict[int, float]:
    """Estimate the bottleneck metric of every stage in one allocation.

    Virtual communication is compiled for the complete candidate Stage Plan
    set, accounted per producer tile, and combined with compute using a max
    metric.
    """

    virtual_communication = virtual_communication_cycles(graph, mesh, plans)
    return {
        stage_id: _selection_metric_for_stage(
            stage_nodes=stage_nodes,
            plan=plans[stage_id],
            mesh=mesh,
            compute_weight=compute_weight,
            communication_weight=communication_weight,
            virtual_communication=virtual_communication,
        )
        for stage_id, stage_nodes in stage_selection.items()
    }


def _selection_metric_for_stage(
    stage_nodes: tuple[Node, ...],
    plan: StagePlan,
    mesh: Mesh,
    compute_weight: float,
    communication_weight: float,
    virtual_communication: dict[int, dict[int, int]],
) -> float:
    """Combine one stage's worst per-tile compute and communication costs."""

    submesh = representative_connected_submesh(mesh, plan.stage_id, plan.tile_count)
    compute_by_tile = {
        tile.tile_id: sum(
            _node_compute_workload(node, output_layouts, tile)
            for node, output_layouts in zip(stage_nodes, plan.node_output_layouts)
        )
        for tile in submesh.tiles
    }
    return max(
        (
            max(
                compute_weight * compute_by_tile[tile_id],
                communication_weight
                * virtual_communication[plan.stage_id][tile_id],
            )
            for tile_id in compute_by_tile
        ),
        default=0.0,
    )


def virtual_communication_cycles(
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
) -> int:
    """Return the greatest accumulated compute cost on any stage tile."""

    return max(
        (
            sum(
                _node_compute_workload(node, output_layouts, tile)
                for node, output_layouts in zip(stage_nodes, node_output_layouts)
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def _node_compute_workload(node: Node, output_layouts: tuple, tile) -> int:
    """Estimate compute cost for one node on one virtual tile."""

    tile_work = node.payload.build_tile_work(output_layouts=output_layouts, tile=tile)
    cost_model = node.payload.cost_model
    return int(cost_model.cost(tile_work, tile)) + int(
        cost_model.placement_cost(node=node, output_layouts=output_layouts)
    )


def worst_tile_l2_transfer_workload(
    stage_id: int,
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple, ...],
    submesh,
    mesh: Mesh,
    graph_inputs: frozenset,
    graph_outputs: frozenset,
    producer_stage_id_by_tensor: dict[object, int],
    initializer_tensors: frozenset,
) -> int:
    """Return the greatest boundary-transfer cost on any stage tile."""

    return max(
        (
            sum(
                _node_l2_transfer_workload(
                    stage_id=stage_id,
                    node=node,
                    output_layouts=output_layouts,
                    tile=tile,
                    mesh=mesh,
                    graph_inputs=graph_inputs,
                    graph_outputs=graph_outputs,
                    producer_stage_id_by_tensor=producer_stage_id_by_tensor,
                    initializer_tensors=initializer_tensors,
                )
                for node, output_layouts in zip(stage_nodes, node_output_layouts)
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def _node_l2_transfer_workload(
    stage_id: int,
    node: Node,
    output_layouts: tuple,
    tile,
    mesh: Mesh,
    graph_inputs: frozenset,
    graph_outputs: frozenset,
    producer_stage_id_by_tensor: dict[object, int],
    initializer_tensors: frozenset,
) -> int:
    """Estimate graph-boundary and peer traffic for one tile's node work."""

    tile_work = node.payload.build_tile_work(output_layouts=output_layouts, tile=tile)
    l2_read_cost = 0
    other_stage_read_cost = 0
    l2_write_cost = 0
    for reference in tile_work.input_slices:
        producer_stage_id = producer_stage_id_by_tensor.get(reference.tensor)
        if reference.tensor in initializer_tensors:
            continue
        if reference.tensor in graph_inputs or producer_stage_id is None:
            l2_read_cost += _one_hop_l2_transfer_cost(
                reference.num_bytes,
                tile.memory.bandwidth,
                mesh.l2_memory.bandwidth,
            )
        elif producer_stage_id != stage_id:
            other_stage_read_cost += _one_hop_peer_transfer_cost(
                reference.num_bytes,
                tile.memory.bandwidth,
            )

    for tensor, layout in zip(node.outputs, output_layouts):
        if tensor not in graph_outputs:
            continue
        l2_write_cost += _one_hop_l2_transfer_cost(
            tensor.slice_num_bytes(tile_tensor_slice(tensor, layout, tile)),
            tile.memory.bandwidth,
            mesh.l2_memory.bandwidth,
        )
    return l2_read_cost + other_stage_read_cost + l2_write_cost


def _one_hop_l2_transfer_cost(
    bytes_: int,
    l1_bandwidth: int,
    l2_bandwidth: int,
) -> int:
    """Convert an L1/L2 byte transfer to bottleneck-link cycles."""

    return _ceil_div(bytes_, min(l1_bandwidth, l2_bandwidth))


def _one_hop_peer_transfer_cost(bytes_: int, l1_bandwidth: int) -> int:
    """Convert a peer byte transfer to source-L1 cycles."""

    return _ceil_div(bytes_, l1_bandwidth)


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return positive integer ceiling division."""

    if denominator <= 0:
        raise ValueError("denominator must be > 0")
    return (numerator + denominator - 1) // denominator
