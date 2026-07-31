"""Single-ASIC Wormhole N300D Mesh and physical torus NoC."""

from maps.hardware import (
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

from .devices import DEVICE_ASSIGNMENT, TILE_DEVICES

MESH_WIDTH = 8
MESH_HEIGHT = 8
L1_SIZE_BYTES = 1464 * 1024
L1_USABLE_BYTES = L1_SIZE_BYTES
L1_STACK_BYTES = 0
L1_RESERVED_BYTES = 0
L1_BANDWIDTH_BYTES = 32
L2_SIZE_BYTES = 12 * 1024 * 1024 * 1024
L2_BANDWIDTH_BYTES = 288
NOC_CHANNEL_WIDTH_BYTES = 32
NOC_HOP_LATENCY_CYCLES = 9
NIU_LATENCY_CYCLES = 5
NOC_WIDTH = MESH_WIDTH + 2
NOC_HEIGHT = MESH_HEIGHT + 4
RESERVED_NOC_ROW_COUNT = 2
_LEFT_L2_ROWS = (0, 1, 5, 6, 7, 11)


def _node_id(x: int, y: int) -> int:
    return y * NOC_WIDTH + x


def _reserved_rows() -> tuple[int, ...]:
    start = NOC_HEIGHT // 2
    return tuple(range(start, start + RESERVED_NOC_ROW_COUNT))


def _middle_l2_x() -> int:
    return 1 + (MESH_WIDTH + 1) // 2


def _tile_noc_coords() -> tuple[tuple[int, int], ...]:
    middle_l2_x = _middle_l2_x()
    x_positions = tuple(range(1, middle_l2_x)) + tuple(
        range(middle_l2_x + 1, NOC_WIDTH)
    )
    reserved_rows = set(_reserved_rows())
    y_positions = tuple(y for y in range(2, NOC_HEIGHT) if y not in reserved_rows)
    return tuple((x, y) for y in y_positions for x in x_positions)


def _l2_endpoint_coords() -> tuple[tuple[int, int], ...]:
    middle_l2_x = _middle_l2_x()
    return tuple((0, y) for y in _LEFT_L2_ROWS) + tuple(
        (middle_l2_x, y) for y in range(NOC_HEIGHT)
    )


TILE_NOC_COORDS = _tile_noc_coords()
L2_ENDPOINT_COORDS = _l2_endpoint_coords()


def _channels() -> tuple[NoCChannel, ...]:
    all_traffic = frozenset(TrafficKind)
    return (
        NoCChannel(0, NOC_CHANNEL_WIDTH_BYTES, NOC_HOP_LATENCY_CYCLES, "noc0", all_traffic),
        NoCChannel(1, NOC_CHANNEL_WIDTH_BYTES, NOC_HOP_LATENCY_CYCLES, "noc1", all_traffic),
    )


def _noc() -> NoC:
    nodes = tuple(
        NoCNode(_node_id(x, y), x, y)
        for y in range(NOC_HEIGHT)
        for x in range(NOC_WIDTH)
    )
    link_specs = (
        tuple((_node_id(x, y), _node_id((x + 1) % NOC_WIDTH, y), 0) for y in range(NOC_HEIGHT) for x in range(NOC_WIDTH))
        + tuple((_node_id(x, y), _node_id((x - 1) % NOC_WIDTH, y), 1) for y in range(NOC_HEIGHT) for x in range(NOC_WIDTH))
        + tuple((_node_id(x, y), _node_id(x, (y + 1) % NOC_HEIGHT), 0) for y in range(NOC_HEIGHT) for x in range(NOC_WIDTH))
        + tuple((_node_id(x, y), _node_id(x, (y - 1) % NOC_HEIGHT), 1) for y in range(NOC_HEIGHT) for x in range(NOC_WIDTH))
    )
    channels = _channels()
    links = tuple(
        NoCLink(index, source, destination, (channels[channel_id],))
        for index, (source, destination, channel_id) in enumerate(link_specs)
    )
    l1_endpoints = tuple(
        NoCEndpoint(
            tile_id,
            EndpointKind.L1,
            _node_id(x, y),
            tile_id=tile_id,
            ingress_latency_cycles=NIU_LATENCY_CYCLES,
            egress_latency_cycles=NIU_LATENCY_CYCLES,
        )
        for tile_id, (x, y) in enumerate(TILE_NOC_COORDS)
    )
    attachment_channels = _channels()
    l2_endpoints = tuple(
        NoCEndpoint(
            len(TILE_NOC_COORDS) + index,
            EndpointKind.L2,
            _node_id(x, y),
            name=f"l2_{index}",
            ingress_latency_cycles=NIU_LATENCY_CYCLES,
            egress_latency_cycles=NIU_LATENCY_CYCLES,
            ingress_channels=attachment_channels,
            egress_channels=attachment_channels,
        )
        for index, (x, y) in enumerate(L2_ENDPOINT_COORDS)
    )
    channels_by_traffic = {traffic_kind: (0, 1) for traffic_kind in TrafficKind}
    return NoC(
        nodes=nodes,
        links=links,
        endpoints=l1_endpoints + l2_endpoints,
        traffic_policy=TrafficPolicy(channels_by_traffic),
        routing_policy=RoutingPolicy.TORUS_XY,
    )


def build_mesh() -> Mesh:
    """Build the fixed single-ASIC N300D compute Mesh and Wormhole NoC."""

    return Mesh(
        width=MESH_WIDTH,
        height=MESH_HEIGHT,
        l2_memory=L2Memory(L2_SIZE_BYTES, L2_BANDWIDTH_BYTES),
        noc=_noc(),
        tiles=tuple(
            Tile(
                tile_id=y * MESH_WIDTH + x,
                x=x,
                y=y,
                memory=L1Memory(L1_USABLE_BYTES, L1_BANDWIDTH_BYTES),
                devices=TILE_DEVICES,
                device_assignment=DEVICE_ASSIGNMENT,
            )
            for y in range(MESH_HEIGHT)
            for x in range(MESH_WIDTH)
        ),
    )


__all__ = [
    "L1_BANDWIDTH_BYTES",
    "L1_USABLE_BYTES",
    "L2_BANDWIDTH_BYTES",
    "L2_ENDPOINT_COORDS",
    "MESH_HEIGHT",
    "MESH_WIDTH",
    "NIU_LATENCY_CYCLES",
    "NOC_CHANNEL_WIDTH_BYTES",
    "NOC_HEIGHT",
    "NOC_HOP_LATENCY_CYCLES",
    "NOC_WIDTH",
    "TILE_NOC_COORDS",
    "build_mesh",
]
