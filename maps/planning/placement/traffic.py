"""Virtual communication analysis performed before physical placement."""

from __future__ import annotations

from maps.planning.layouts import tensor_slice_num_bytes
from maps.planning.stages import StagePlan, virtual_submesh
from maps.planning.placement.models import VirtualTraffic
from maps.planning.transitions import (
    VirtualInputTransition,
    VirtualIntermediateTransition,
    VirtualOutputTransition,
    VirtualTransition,
)


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
