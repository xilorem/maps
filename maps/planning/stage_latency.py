"""Authoritative barrier-aware Stage Latency estimation."""

from __future__ import annotations

from typing import cast

from maps.graph import Node
from maps.hardware import Tile
from maps.operations.collective import AllReducePayload
from maps.operations.contracts import OpPayload, TileWork
from maps.planning.stages import VirtualCollectiveGroup


def estimate_stage_latency(
    *,
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple, ...],
    virtual_tiles: tuple[Tile, ...],
    device_names: tuple[str, ...],
    virtual_collective_groups: tuple[
        tuple[VirtualCollectiveGroup, ...], ...
    ],
    node_tile_work: tuple[tuple[TileWork, ...], ...] | None = None,
    physical_tiles_by_virtual_id: dict[int, Tile] | None = None,
) -> int:
    """Sum slowest-tile phases and synchronous collective Layer latencies."""

    physical_tiles_by_virtual_id = physical_tiles_by_virtual_id or {
        tile.tile_id: tile for tile in virtual_tiles
    }
    phase_cycles = {tile.tile_id: 0 for tile in virtual_tiles}
    latency = 0
    for node_index, (node, output_layouts, device_name) in enumerate(
        zip(stage_nodes, node_output_layouts, device_names)
    ):
        payload = cast(OpPayload, node.payload)
        cost_model = payload.cost_model
        work_by_tile_id = {
            tile.tile_id: (
                node_tile_work[node_index][tile_index]
                if node_tile_work is not None
                else payload.build_tile_work(
                    output_layouts=output_layouts,
                    tile=tile,
                )
            )
            for tile_index, tile in enumerate(virtual_tiles)
        }
        if isinstance(node.payload, AllReducePayload):
            latency += max(phase_cycles.values(), default=0)
            group_latencies = []
            for group in virtual_collective_groups[node_index]:
                representative_id = group.virtual_tile_ids[0]
                virtual_tile = next(
                    tile for tile in virtual_tiles if tile.tile_id == representative_id
                )
                physical_participants = tuple(
                    physical_tiles_by_virtual_id[tile_id]
                    for tile_id in group.virtual_tile_ids
                )
                group_latencies.append(
                    cost_model.collective_cost(
                        work_by_tile_id[representative_id],
                        virtual_tile,
                        virtual_tile.device_by_name(device_name),
                        physical_participants,
                    )
                )
            latency += max(group_latencies, default=0)
            phase_cycles = dict.fromkeys(phase_cycles, 0)
            continue

        placement_cycles = int(
            cost_model.placement_cost(node=node, output_layouts=output_layouts)
        )
        for tile in virtual_tiles:
            phase_cycles[tile.tile_id] += int(
                cost_model.cost(
                    work_by_tile_id[tile.tile_id],
                    tile,
                    tile.device_by_name(device_name),
                )
            ) + placement_cycles
    return latency + max(phase_cycles.values(), default=0)


__all__ = ["estimate_stage_latency"]
