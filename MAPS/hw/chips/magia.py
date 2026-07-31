"""MAGIA chip description."""

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
    MAGIA_CORE_DEVICE,
    MAGIA_DEVICE_ASSIGNMENT,
    MAGIA_IDMA_READ_DEVICE,
    MAGIA_IDMA_WRITE_DEVICE,
    MAGIA_PRECISION_LOWERING_RECIPES,
    MAGIA_REDMULE_DEVICE,
    MAGIA_SPATZ_DEVICE,
    MAGIA_TILE_DEVICES,
)
from MAPS.utils.print_mesh import print_mesh
from MAPS.planner.contracts.options import GraphRewriteOptions, PlannerOptions

MAGIA_MESH_WIDTH = 8
MAGIA_MESH_HEIGHT = 8

MAGIA_L1_SIZE_BYTES = 1024 * 10240
# Keep planner data allocation below the generated ready-flags region.
MAGIA_L1_DATA_BYTES = 0xD0000
MAGIA_L1_USABLE_BYTES = MAGIA_L1_DATA_BYTES
MAGIA_L1_STACK_BYTES = 64 * 1024
MAGIA_L1_RESERVED_BYTES = 64 * 1024
MAGIA_L1_BANDWIDTH_BYTES = 32
MAGIA_L2_SIZE_BYTES = 1024 * 1024 * 1024
MAGIA_L2_BANDWIDTH_BYTES = 32
MAGIA_NOC_CHANNEL_WIDTH_BYTES = 4
MAGIA_NOC_WIDE_CHANNEL_WIDTH_BYTES = 4
MAGIA_NOC_HOP_LATENCY_CYCLES = 2

def _magia_noc_node_id(x: int, y: int, width: int) -> int:
    return y * width + x


def _magia_noc_channels() -> tuple[NoCChannel, ...]:
    return (
        NoCChannel(
            channel_id=0,
            width_bytes=MAGIA_NOC_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=MAGIA_NOC_HOP_LATENCY_CYCLES,
            tag="req",
            supported_traffic=frozenset(
                {
                    TrafficKind.READ_REQ,
                    TrafficKind.WRITE_REQ,
                    TrafficKind.WRITE_DATA,
                }
            ),
        ),
        NoCChannel(
            channel_id=1,
            width_bytes=MAGIA_NOC_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=MAGIA_NOC_HOP_LATENCY_CYCLES,
            tag="rsp",
            supported_traffic=frozenset(
                {
                    TrafficKind.READ_RSP,
                    TrafficKind.WRITE_RSP,
                }
            ),
        ),
        NoCChannel(
            channel_id=2,
            width_bytes=MAGIA_NOC_WIDE_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=MAGIA_NOC_HOP_LATENCY_CYCLES,
            tag="wide",
            supported_traffic=frozenset(
                {
                    TrafficKind.WRITE_REQ,
                    TrafficKind.READ_RSP,
                    TrafficKind.WRITE_DATA,
                }
            ),
        ),
    )


def _magia_noc(width: int, height: int) -> NoC:
    attachment_channels = _magia_noc_channels()
    nodes = tuple(
        NoCNode(
            node_id=_magia_noc_node_id(x, y, width),
            x=x,
            y=y,
        )
        for y in range(height)
        for x in range(width)
    )
    link_pairs = tuple(
        (_magia_noc_node_id(x, y, width), _magia_noc_node_id(x + 1, y, width))
        for y in range(height)
        for x in range(width - 1)
    ) + tuple(
        (_magia_noc_node_id(x, y, width), _magia_noc_node_id(x, y + 1, width))
        for y in range(height - 1)
        for x in range(width)
    )
    links = tuple(
        NoCLink(
            link_id=link_id,
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            channels=_magia_noc_channels(),
            bidirectional=True,
        )
        for link_id, (src_node_id, dst_node_id) in enumerate(link_pairs)
    )
    l1_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=tile_id,
            kind=EndpointKind.L1,
            node_id=_magia_noc_node_id(tile_id % width, tile_id // width, width),
            tile_id=tile_id,
        )
        for tile_id in range(width * height)
    )
    l2_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=width * height + y,
            kind=EndpointKind.L2,
            node_id=_magia_noc_node_id(0, y, width),
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
        traffic_policy=TrafficPolicy(
            {
                TrafficKind.READ_REQ: (0,),
                TrafficKind.WRITE_REQ: (2,),
                TrafficKind.READ_RSP: (2,),
                TrafficKind.WRITE_RSP: (1,),
                TrafficKind.WRITE_DATA: (2,),
            }
        ),
        routing_policy=RoutingPolicy.XY,
    )


def magia_mesh(
    width: int = MAGIA_MESH_WIDTH,
    height: int = MAGIA_MESH_HEIGHT,
) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=MAGIA_L2_SIZE_BYTES, bandwidth=MAGIA_L2_BANDWIDTH_BYTES),
        noc=_magia_noc(width, height),
        tiles=tuple(
            Tile(
                tile_id=y * width + x,
                x=x,
                y=y,
                memory=L1Memory(size=MAGIA_L1_USABLE_BYTES, bandwidth=MAGIA_L1_BANDWIDTH_BYTES),
                devices=MAGIA_TILE_DEVICES,
                device_assignment=MAGIA_DEVICE_ASSIGNMENT,
            )
            for y in range(height)
            for x in range(width)
        ),
        precision_lowering_recipes=MAGIA_PRECISION_LOWERING_RECIPES,
        required_graph_rewrites=("conv_to_gemm",),
    )


def magia_planner_options(
    *,
    enable_precision_lowering: bool = True,
) -> PlannerOptions:
    """Return MAGIA defaults with optional precision lowering enabled."""

    return PlannerOptions(
        graph_rewrites=GraphRewriteOptions(
            enable_precision_lowering=enable_precision_lowering,
        )
    )


if __name__ == "__main__":
    print("MAGIA")
    print_mesh(magia_mesh())
