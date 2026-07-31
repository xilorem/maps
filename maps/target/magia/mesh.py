"""MAGIA Mesh construction and NoC topology."""

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
L1_SIZE_BYTES = 1024 * 10240
L1_DATA_BYTES = 0xD0000
L1_USABLE_BYTES = L1_DATA_BYTES
L1_STACK_BYTES = 64 * 1024
L1_RESERVED_BYTES = 64 * 1024
L1_BANDWIDTH_BYTES = 32
L2_SIZE_BYTES = 1024 * 1024 * 1024
L2_BANDWIDTH_BYTES = 32
NOC_CHANNEL_WIDTH_BYTES = 4
NOC_WIDE_CHANNEL_WIDTH_BYTES = 4
NOC_HOP_LATENCY_CYCLES = 2


def _node_id(x: int, y: int, width: int) -> int:
    return y * width + x


def _channels() -> tuple[NoCChannel, ...]:
    return (
        NoCChannel(0, NOC_CHANNEL_WIDTH_BYTES, NOC_HOP_LATENCY_CYCLES, "req", frozenset({TrafficKind.READ_REQ, TrafficKind.WRITE_REQ, TrafficKind.WRITE_DATA})),
        NoCChannel(1, NOC_CHANNEL_WIDTH_BYTES, NOC_HOP_LATENCY_CYCLES, "rsp", frozenset({TrafficKind.READ_RSP, TrafficKind.WRITE_RSP})),
        NoCChannel(2, NOC_WIDE_CHANNEL_WIDTH_BYTES, NOC_HOP_LATENCY_CYCLES, "wide", frozenset({TrafficKind.WRITE_REQ, TrafficKind.READ_RSP, TrafficKind.WRITE_DATA})),
    )


def _noc(width: int, height: int) -> NoC:
    nodes = tuple(NoCNode(_node_id(x, y, width), x, y) for y in range(height) for x in range(width))
    link_pairs = tuple(
        (_node_id(x, y, width), _node_id(x + 1, y, width))
        for y in range(height)
        for x in range(width - 1)
    ) + tuple(
        (_node_id(x, y, width), _node_id(x, y + 1, width))
        for y in range(height - 1)
        for x in range(width)
    )
    links = tuple(
        NoCLink(index, source, destination, _channels(), bidirectional=True)
        for index, (source, destination) in enumerate(link_pairs)
    )
    l1_endpoints = tuple(
        NoCEndpoint(tile_id, EndpointKind.L1, _node_id(tile_id % width, tile_id // width, width), tile_id=tile_id)
        for tile_id in range(width * height)
    )
    attachment_channels = _channels()
    l2_endpoints = tuple(
        NoCEndpoint(
            width * height + y,
            EndpointKind.L2,
            _node_id(0, y, width),
            name=f"l2_{y}",
            ingress_channels=attachment_channels,
            egress_channels=attachment_channels,
        )
        for y in range(height)
    )
    return NoC(
        nodes=nodes,
        links=links,
        endpoints=l1_endpoints + l2_endpoints,
        traffic_policy=TrafficPolicy({
            TrafficKind.READ_REQ: (0,),
            TrafficKind.WRITE_REQ: (2,),
            TrafficKind.READ_RSP: (2,),
            TrafficKind.WRITE_RSP: (1,),
            TrafficKind.WRITE_DATA: (2,),
        }),
        routing_policy=RoutingPolicy.XY,
    )


def build_mesh(width: int = MESH_WIDTH, height: int = MESH_HEIGHT) -> Mesh:
    """Build a planner-ready MAGIA Mesh with configurable compute shape."""

    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(L2_SIZE_BYTES, L2_BANDWIDTH_BYTES),
        noc=_noc(width, height),
        tiles=tuple(
            Tile(
                tile_id=y * width + x,
                x=x,
                y=y,
                memory=L1Memory(L1_USABLE_BYTES, L1_BANDWIDTH_BYTES),
                devices=TILE_DEVICES,
                device_assignment=DEVICE_ASSIGNMENT,
            )
            for y in range(height)
            for x in range(width)
        ),
    )


__all__ = [
    "L1_BANDWIDTH_BYTES",
    "L1_DATA_BYTES",
    "L1_SIZE_BYTES",
    "L1_USABLE_BYTES",
    "L2_BANDWIDTH_BYTES",
    "L2_SIZE_BYTES",
    "MESH_HEIGHT",
    "MESH_WIDTH",
    "NOC_CHANNEL_WIDTH_BYTES",
    "NOC_HOP_LATENCY_CYCLES",
    "NOC_WIDE_CHANNEL_WIDTH_BYTES",
    "build_mesh",
]
