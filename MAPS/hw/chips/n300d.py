
"""Tenstorrent Wormhole n300d single-ASIC chip description.

This module models one fixed Wormhole ASIC as:
- one logical 8x8 compute mesh used by the planner
- one larger physical 10x12 NoC used by transport costs

The DRAM endpoint coordinates follow the physical Wormhole NoC grid. The
logical 8x8 compute tiles are attached to a regular subset of NoC nodes so the
current MAPS planner can still operate on a dense compute mesh.

This is intentionally a per-ASIC model, not a full dual-ASIC n300d board model.
"""

from __future__ import annotations

from MAPS.arch import (
    EndpointKind,
    L1Memory,
    L2Memory,
    Mesh,
    NoC,
    NoCChannel,
    NoCEndpoint,
    NoCLink,
    NoCNode,
    RoutingPolicy,
    Tile,
    TrafficKind,
    TrafficPolicy,
)
from MAPS.hw.devices import (
    TENSIX_MATRIX_DEVICE,
    TENSIX_DEVICE_ASSIGNMENT,
    TENSIX_READ_CORE,
    TENSIX_SCALAR_DEVICE,
    TENSIX_VECTOR_DEVICE,
    TENSIX_WRITE_CORE,
)
from MAPS.utils.print_mesh import print_mesh


# Single-ASIC approximation of one Wormhole die.
N300D_MESH_WIDTH = 8
N300D_MESH_HEIGHT = 8

N300D_L1_SIZE_BYTES = 1464 * 1024
N300D_L1_USABLE_BYTES = 1464 * 1024
N300D_L1_STACK_BYTES = 0
N300D_L1_RESERVED_BYTES = 0
# MAPS uses tile memory bandwidth as the NoC-facing transfer limit. For
# Wormhole, use the 32 B/cycle NoC-facing port width rather than the aggregate
# internal SRAM bank bandwidth.
N300D_L1_BANDWIDTH_BYTES = 32
N300D_L2_SIZE_BYTES = 12 * 1024 * 1024 * 1024
N300D_L2_BANDWIDTH_BYTES = 288
N300D_NOC_CHANNEL_WIDTH_BYTES = 32
N300D_NOC_HOP_LATENCY_CYCLES = 9
N300D_NIU_LATENCY_CYCLES = 5
N300D_NOC_WIDTH_PADDING = 2
N300D_NOC_HEIGHT_PADDING = 4
N300D_NOC_WIDTH = N300D_MESH_WIDTH + N300D_NOC_WIDTH_PADDING
N300D_NOC_HEIGHT = N300D_MESH_HEIGHT + N300D_NOC_HEIGHT_PADDING
N300D_RESERVED_NOC_ROW_COUNT = 2

_N300D_TEMPLATE_NOC_HEIGHT = 12
_N300D_TEMPLATE_LEFT_L2_ROWS = (0, 1, 5, 6, 7, 11)

N300D_TILE_DEVICES = (
    TENSIX_READ_CORE,
    TENSIX_WRITE_CORE,
    TENSIX_SCALAR_DEVICE,
    TENSIX_VECTOR_DEVICE,
    TENSIX_MATRIX_DEVICE,
)


def _n300d_noc_node_id(x: int, y: int, width: int) -> int:
    return y * width + x


def _n300d_reserved_rows() -> tuple[int, ...]:
    start = N300D_NOC_HEIGHT // 2
    return tuple(range(start, start + N300D_RESERVED_NOC_ROW_COUNT))


def _n300d_middle_l2_x() -> int:
    left_compute_width = (N300D_MESH_WIDTH + 1) // 2
    return 1 + left_compute_width


def _n300d_tile_noc_coords() -> tuple[tuple[int, int], ...]:
    middle_l2_x = _n300d_middle_l2_x()
    left_positions = tuple(range(1, middle_l2_x))
    right_positions = tuple(range(middle_l2_x + 1, N300D_NOC_WIDTH))
    x_positions = left_positions + right_positions
    reserved_rows = set(_n300d_reserved_rows())
    y_positions = tuple(y for y in range(2, N300D_NOC_HEIGHT) if y not in reserved_rows)
    return tuple((x, y) for y in y_positions for x in x_positions)


def _n300d_l2_endpoint_coords() -> tuple[tuple[int, int], ...]:
    middle_l2_x = _n300d_middle_l2_x()
    left_rows = _N300D_TEMPLATE_LEFT_L2_ROWS
    middle_rows = tuple(range(N300D_NOC_HEIGHT))
    return tuple((0, y) for y in left_rows) + tuple((middle_l2_x, y) for y in middle_rows)


N300D_L2_ENDPOINT_COORDS = _n300d_l2_endpoint_coords()
N300D_TILE_NOC_COORDS = _n300d_tile_noc_coords()


def _n300d_noc_channels() -> tuple[NoCChannel, ...]:
    all_traffic = frozenset(
        {
            TrafficKind.READ_REQ,
            TrafficKind.WRITE_REQ,
            TrafficKind.READ_RSP,
            TrafficKind.WRITE_RSP,
            TrafficKind.WRITE_DATA,
        }
    )
    return (
        NoCChannel(
            channel_id=0,
            width_bytes=N300D_NOC_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=N300D_NOC_HOP_LATENCY_CYCLES,
            tag="noc0",
            supported_traffic=all_traffic,
        ),
        NoCChannel(
            channel_id=1,
            width_bytes=N300D_NOC_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=N300D_NOC_HOP_LATENCY_CYCLES,
            tag="noc1",
            supported_traffic=all_traffic,
        ),
    )


def _n300d_noc() -> NoC:
    attachment_channels = _n300d_noc_channels()
    tile_noc_coords = _n300d_tile_noc_coords()
    l2_endpoint_coords = _n300d_l2_endpoint_coords()
    nodes = tuple(
        NoCNode(
            node_id=_n300d_noc_node_id(x, y, N300D_NOC_WIDTH),
            x=x,
            y=y,
        )
        for y in range(N300D_NOC_HEIGHT)
        for x in range(N300D_NOC_WIDTH)
    )
    right_pairs = tuple(
        (
            _n300d_noc_node_id(x, y, N300D_NOC_WIDTH),
            _n300d_noc_node_id((x + 1) % N300D_NOC_WIDTH, y, N300D_NOC_WIDTH),
            0,
        )
        for y in range(N300D_NOC_HEIGHT)
        for x in range(N300D_NOC_WIDTH)
    )
    left_pairs = tuple(
        (
            _n300d_noc_node_id(x, y, N300D_NOC_WIDTH),
            _n300d_noc_node_id((x - 1) % N300D_NOC_WIDTH, y, N300D_NOC_WIDTH),
            1,
        )
        for y in range(N300D_NOC_HEIGHT)
        for x in range(N300D_NOC_WIDTH)
    )
    up_pairs = tuple(
        (
            _n300d_noc_node_id(x, y, N300D_NOC_WIDTH),
            _n300d_noc_node_id(x, (y + 1) % N300D_NOC_HEIGHT, N300D_NOC_WIDTH),
            0,
        )
        for y in range(N300D_NOC_HEIGHT)
        for x in range(N300D_NOC_WIDTH)
    )
    down_pairs = tuple(
        (
            _n300d_noc_node_id(x, y, N300D_NOC_WIDTH),
            _n300d_noc_node_id(x, (y - 1) % N300D_NOC_HEIGHT, N300D_NOC_WIDTH),
            1,
        )
        for y in range(N300D_NOC_HEIGHT)
        for x in range(N300D_NOC_WIDTH)
    )
    link_specs = right_pairs + left_pairs + up_pairs + down_pairs
    channels = _n300d_noc_channels()
    links = tuple(
        NoCLink(
            link_id=link_id,
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            channels=(channels[channel_id],),
        )
        for link_id, (src_node_id, dst_node_id, channel_id) in enumerate(link_specs)
    )
    l1_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=tile_id,
            kind=EndpointKind.L1,
            node_id=_n300d_noc_node_id(x, y, N300D_NOC_WIDTH),
            tile_id=tile_id,
            ingress_latency_cycles=N300D_NIU_LATENCY_CYCLES,
            egress_latency_cycles=N300D_NIU_LATENCY_CYCLES,
        )
        for tile_id, (x, y) in enumerate(tile_noc_coords)
    )
    l2_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=len(tile_noc_coords) + endpoint_index,
            kind=EndpointKind.L2,
            node_id=_n300d_noc_node_id(x, y, N300D_NOC_WIDTH),
            name=f"l2_{endpoint_index}",
            ingress_latency_cycles=N300D_NIU_LATENCY_CYCLES,
            egress_latency_cycles=N300D_NIU_LATENCY_CYCLES,
            ingress_channels=attachment_channels,
            egress_channels=attachment_channels,
        )
        for endpoint_index, (x, y) in enumerate(l2_endpoint_coords)
    )
    return NoC(
        nodes=nodes,
        links=links,
        endpoints=l1_endpoints + l2_endpoints,
        traffic_policy=TrafficPolicy(
            {
                TrafficKind.READ_REQ: (0, 1),
                TrafficKind.WRITE_REQ: (0, 1),
                TrafficKind.READ_RSP: (0, 1),
                TrafficKind.WRITE_RSP: (0, 1),
                TrafficKind.WRITE_DATA: (0, 1),
            }
        ),
        routing_policy=RoutingPolicy.TORUS_XY,
    )


def wormhole_n300d_mesh() -> Mesh:
    return Mesh(
        width=N300D_MESH_WIDTH,
        height=N300D_MESH_HEIGHT,
        l2_memory=L2Memory(size=N300D_L2_SIZE_BYTES, bandwidth=N300D_L2_BANDWIDTH_BYTES),
        noc=_n300d_noc(),
        tiles=tuple(
            Tile(
                tile_id=y * N300D_MESH_WIDTH + x,
                x=x,
                y=y,
                memory=L1Memory(size=N300D_L1_USABLE_BYTES, bandwidth=N300D_L1_BANDWIDTH_BYTES),
                devices=N300D_TILE_DEVICES,
                device_assignment=TENSIX_DEVICE_ASSIGNMENT,
            )
            for y in range(N300D_MESH_HEIGHT)
            for x in range(N300D_MESH_WIDTH)
        ),
    )


if __name__ == "__main__":
    print("Wormhole n300d")
    print_mesh(wormhole_n300d_mesh())
