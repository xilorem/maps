from dataclasses import replace

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
    Tile,
    TrafficKind,
    TrafficPolicy,
)
from MAPS.hw.chips.magia import magia_mesh
from MAPS.hw.devices.generic import GENERIC_SCALAR_DEVICE
import pytest

from maps.planning.transitions import TransferKind, TransferLeg, TransportCostModel
from tests.noc_utils import rectangular_test_tiles


def _uniform_l1_only_mesh(
    width: int,
    height: int,
    *,
    link_width_bytes: int = 8,
    hop_latency_cycles: int = 1,
    memory_bandwidth: int = 64,
) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=memory_bandwidth),
        tiles=rectangular_test_tiles(
            width,
            height,
            memory=L1Memory(size=4096, bandwidth=memory_bandwidth),
            devices=(GENERIC_SCALAR_DEVICE,),
        ),
        noc=NoC(
            nodes=tuple(
                NoCNode(node_id=y * width + x, x=x, y=y)
                for y in range(height)
                for x in range(width)
            ),
            links=tuple(
                NoCLink(
                    link_id=link_id,
                    src_node_id=src_node_id,
                    dst_node_id=dst_node_id,
                    channels=(
                        NoCChannel(
                            channel_id=0,
                            width_bytes=link_width_bytes,
                            hop_latency_cycles=hop_latency_cycles,
                        ),
                    ),
                    bidirectional=True,
                )
                for link_id, (src_node_id, dst_node_id) in enumerate(
                    tuple(
                        (y * width + x, y * width + x + 1)
                        for y in range(height)
                        for x in range(width - 1)
                    )
                    + tuple(
                        (y * width + x, (y + 1) * width + x)
                        for y in range(height - 1)
                        for x in range(width)
                    )
                )
            ),
            endpoints=tuple(
                NoCEndpoint(
                    endpoint_id=tile_id,
                    kind=EndpointKind.L1,
                    node_id=tile_id,
                    tile_id=tile_id,
                )
                for tile_id in range(width * height)
            ),
        ),
    )


def test_transport_cost_requires_mesh_for_communication() -> None:
    tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=1),
        devices=(GENERIC_SCALAR_DEVICE,),
    )
    model = TransportCostModel(mesh=None)

    with pytest.raises(ValueError, match="requires a mesh"):
        model.l1_to_l1(tile, tile, 64)

    with pytest.raises(ValueError, match="requires a mesh"):
        model.l1_to_l2(tile, 64)


def test_magia_burst_rounding_applies_to_strided_l1_and_l2_writes() -> None:
    mesh = magia_mesh(width=2, height=1)
    model = TransportCostModel(mesh=mesh)
    src = mesh.tile(0, 0)
    dst = mesh.tile(1, 0)

    # idma1 emits two 6-byte rows as two bursts. At the 4-byte NoC width,
    # that costs four data cycles instead of three for one packed 12-byte run.
    assert model.l1_to_l1(src, dst, 12, row_bytes=6, rows=2) == model.l1_to_l1(src, dst, 12) + 1
    assert model.l1_to_l2(src, 12, row_bytes=6, rows=2) == model.l1_to_l2(src, 12) + 1


def test_l1_to_l1_transfer_cost_uses_noc_route_hops_when_available() -> None:
    mesh = Mesh(
        width=3,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        tiles=rectangular_test_tiles(3, 1, memory=L1Memory(size=4096, bandwidth=16)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
                NoCNode(node_id=2, x=2, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=2),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=1,
                    src_node_id=1,
                    dst_node_id=2,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=2),),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
                NoCEndpoint(endpoint_id=2, kind=EndpointKind.L1, node_id=2, tile_id=2),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l1(mesh.tile(0, 0), mesh.tile(2, 0), 64) > model.l1_to_l1(mesh.tile(0, 0), mesh.tile(1, 0), 64)


def test_l1_to_l1_transfer_cost_respects_write_traffic_policy_channel_selection() -> None:
    wide_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        tiles=rectangular_test_tiles(2, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(
                        NoCChannel(channel_id=0, width_bytes=4, hop_latency_cycles=1),
                        NoCChannel(channel_id=1, width_bytes=16, hop_latency_cycles=1),
                    ),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
            ),
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (1,),
                    TrafficKind.WRITE_DATA: (1,),
                    TrafficKind.WRITE_RSP: (1,),
                }
            ),
        ),
    )
    narrow_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        tiles=rectangular_test_tiles(2, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=wide_mesh.noc.nodes,
            links=wide_mesh.noc.links,
            endpoints=wide_mesh.noc.endpoints,
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (0,),
                    TrafficKind.WRITE_RSP: (0,),
                }
            ),
        ),
    )

    wide_cost = TransportCostModel(mesh=wide_mesh, read_request_bytes=64).l1_to_l1(
        wide_mesh.tile(0, 0),
        wide_mesh.tile(1, 0),
        64,
    )
    narrow_cost = TransportCostModel(mesh=narrow_mesh, read_request_bytes=64).l1_to_l1(
        narrow_mesh.tile(0, 0),
        narrow_mesh.tile(1, 0),
        64,
    )

    assert narrow_cost > wide_cost


def test_l2_transfer_cost_uses_noc_route_to_nearest_l2_endpoint_when_available() -> None:
    mesh = Mesh(
        width=3,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=16),
        tiles=rectangular_test_tiles(3, 1, memory=L1Memory(size=4096, bandwidth=16)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
                NoCNode(node_id=2, x=2, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=2),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=1,
                    src_node_id=1,
                    dst_node_id=2,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=2),),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
                NoCEndpoint(endpoint_id=2, kind=EndpointKind.L1, node_id=2, tile_id=2),
                NoCEndpoint(endpoint_id=3, kind=EndpointKind.L2, node_id=0, name="l2_west"),
                NoCEndpoint(endpoint_id=4, kind=EndpointKind.L2, node_id=2, name="l2_east"),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l2(mesh.tile(0, 0), 64) < model.l1_to_l2(mesh.tile(1, 0), 64)
    assert model.l2_to_l1(mesh.tile(2, 0), 64) < model.l2_to_l1(mesh.tile(1, 0), 64)


def test_l2_transfer_cost_respects_directional_traffic_policy_channel_selection() -> None:
    write_wide_read_narrow_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=rectangular_test_tiles(2, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(
                        NoCChannel(channel_id=0, width_bytes=4, hop_latency_cycles=1),
                        NoCChannel(channel_id=1, width_bytes=16, hop_latency_cycles=1),
                    ),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
                NoCEndpoint(endpoint_id=2, kind=EndpointKind.L2, node_id=1, name="l2"),
            ),
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (1,),
                    TrafficKind.READ_REQ: (0,),
                    TrafficKind.READ_RSP: (0,),
                    TrafficKind.WRITE_RSP: (0,),
                }
            ),
        ),
    )
    write_narrow_read_wide_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=rectangular_test_tiles(2, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=write_wide_read_narrow_mesh.noc.nodes,
            links=write_wide_read_narrow_mesh.noc.links,
            endpoints=write_wide_read_narrow_mesh.noc.endpoints,
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (0,),
                    TrafficKind.READ_REQ: (0,),
                    TrafficKind.READ_RSP: (1,),
                    TrafficKind.WRITE_RSP: (1,),
                }
            ),
        ),
    )

    write_wide_read_narrow_model = TransportCostModel(mesh=write_wide_read_narrow_mesh)
    write_narrow_read_wide_model = TransportCostModel(mesh=write_narrow_read_wide_mesh)

    assert (
        write_wide_read_narrow_model.l1_to_l2(write_wide_read_narrow_mesh.tile(0, 0), 64)
        < write_narrow_read_wide_model.l1_to_l2(write_narrow_read_wide_mesh.tile(0, 0), 64)
    )
    assert (
        write_wide_read_narrow_model.l2_to_l1(write_wide_read_narrow_mesh.tile(0, 0), 64)
        > write_narrow_read_wide_model.l2_to_l1(write_narrow_read_wide_mesh.tile(0, 0), 64)
    )


def test_l2_transfer_cost_includes_noc_endpoint_attachment_latency_without_hops() -> None:
    mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=16),
        tiles=rectangular_test_tiles(1, 1, memory=L1Memory(size=4096, bandwidth=16)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
            ),
            links=(),
            endpoints=(
                NoCEndpoint(
                    endpoint_id=0,
                    kind=EndpointKind.L1,
                    node_id=0,
                    tile_id=0,
                    ingress_latency_cycles=11,
                    egress_latency_cycles=3,
                ),
                NoCEndpoint(
                    endpoint_id=1,
                    kind=EndpointKind.L2,
                    node_id=0,
                    name="l2",
                    ingress_latency_cycles=5,
                    egress_latency_cycles=7,
                ),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l2(mesh.tile(0, 0), 64) == 40
    assert model.l2_to_l1(mesh.tile(0, 0), 64) == 31


def test_l2_transfer_cost_uses_noc_endpoint_attachment_bandwidth_without_hops() -> None:
    mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=rectangular_test_tiles(1, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
            ),
            links=(),
            endpoints=(
                NoCEndpoint(
                    endpoint_id=0,
                    kind=EndpointKind.L1,
                    node_id=0,
                    tile_id=0,
                    ingress_bandwidth_bytes=32,
                    egress_bandwidth_bytes=8,
                ),
                NoCEndpoint(
                    endpoint_id=1,
                    kind=EndpointKind.L2,
                    node_id=0,
                    name="l2",
                    ingress_bandwidth_bytes=16,
                    egress_bandwidth_bytes=4,
                ),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh, account_noc_contention=True)

    l1_to_l2_estimate = model.estimate(
        TransferLeg(
            kind=TransferKind.L1_TO_L2,
            bytes=64,
            src_tile=mesh.tile(0, 0),
        )
    )
    l2_to_l1_estimate = model.estimate(
        TransferLeg(
            kind=TransferKind.L2_TO_L1,
            bytes=64,
            dst_tile=mesh.tile(0, 0),
        )
    )

    assert model.l1_to_l2(mesh.tile(0, 0), 64) == 10
    assert model.l2_to_l1(mesh.tile(0, 0), 64) == 17
    assert l1_to_l2_estimate.resource_loads == {
        "tile:0:dma:idma_write": 10,
        "noc_endpoint:0:egress": 9,
        "noc_endpoint:1:ingress": 5,
        "noc_endpoint:1:egress": 1,
        "noc_endpoint:0:ingress": 1,
    }
    assert l2_to_l1_estimate.resource_loads == {
        "tile:0:dma:idma_read": 17,
        "noc_endpoint:0:egress": 1,
        "noc_endpoint:1:ingress": 1,
        "noc_endpoint:1:egress": 16,
        "noc_endpoint:0:ingress": 2,
    }


def test_l2_transfer_cost_uses_endpoint_attachment_channels_and_policy_without_internal_hops() -> None:
    mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=rectangular_test_tiles(1, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=(NoCNode(node_id=0, x=0, y=0),),
            links=(),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(
                    endpoint_id=1,
                    kind=EndpointKind.L2,
                    node_id=0,
                    name="l2",
                    ingress_channels=(
                        NoCChannel(
                            channel_id=0,
                            width_bytes=8,
                            hop_latency_cycles=2,
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
                            width_bytes=4,
                            hop_latency_cycles=3,
                            supported_traffic=frozenset(
                                {
                                    TrafficKind.READ_RSP,
                                    TrafficKind.WRITE_RSP,
                                }
                            ),
                        ),
                        NoCChannel(
                            channel_id=2,
                            width_bytes=32,
                            hop_latency_cycles=1,
                            supported_traffic=frozenset(
                                {
                                    TrafficKind.READ_RSP,
                                    TrafficKind.WRITE_DATA,
                                }
                            ),
                        ),
                    ),
                    egress_channels=(
                        NoCChannel(
                            channel_id=0,
                            width_bytes=8,
                            hop_latency_cycles=2,
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
                            width_bytes=4,
                            hop_latency_cycles=3,
                            supported_traffic=frozenset(
                                {
                                    TrafficKind.READ_RSP,
                                    TrafficKind.WRITE_RSP,
                                }
                            ),
                        ),
                        NoCChannel(
                            channel_id=2,
                            width_bytes=32,
                            hop_latency_cycles=1,
                            supported_traffic=frozenset(
                                {
                                    TrafficKind.READ_RSP,
                                    TrafficKind.WRITE_DATA,
                                }
                            ),
                        ),
                    ),
                ),
            ),
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.READ_REQ: (0,),
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.READ_RSP: (1,),
                    TrafficKind.WRITE_RSP: (1,),
                    TrafficKind.WRITE_DATA: (0,),
                }
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh, account_noc_contention=True)

    l1_to_l2_estimate = model.estimate(
        TransferLeg(
            kind=TransferKind.L1_TO_L2,
            bytes=64,
            src_tile=mesh.tile(0, 0),
        )
    )
    l2_to_l1_estimate = model.estimate(
        TransferLeg(
            kind=TransferKind.L2_TO_L1,
            bytes=64,
            dst_tile=mesh.tile(0, 0),
        )
    )

    assert model.l1_to_l2(mesh.tile(0, 0), 64) == 17
    assert model.l2_to_l1(mesh.tile(0, 0), 64) == 22
    assert l1_to_l2_estimate.resource_loads == {
        "tile:0:dma:idma_write": 17,
        "noc_endpoint_attachment:1:ingress:channel:0": 9,
        "noc_endpoint_attachment:1:egress:channel:1": 1,
    }
    assert l2_to_l1_estimate.resource_loads == {
        "tile:0:dma:idma_read": 22,
        "noc_endpoint_attachment:1:ingress:channel:0": 1,
        "noc_endpoint_attachment:1:egress:channel:1": 16,
    }


def test_l1_to_l2_transfer_cost_respects_write_rsp_traffic_policy_channel_selection() -> None:
    ack_wide_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=rectangular_test_tiles(2, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(
                        NoCChannel(channel_id=0, width_bytes=4, hop_latency_cycles=1),
                        NoCChannel(channel_id=1, width_bytes=16, hop_latency_cycles=1),
                    ),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=1, name="l2"),
            ),
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (0,),
                    TrafficKind.WRITE_RSP: (1,),
                }
            ),
        ),
    )
    ack_narrow_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=rectangular_test_tiles(2, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=ack_wide_mesh.noc.nodes,
            links=ack_wide_mesh.noc.links,
            endpoints=ack_wide_mesh.noc.endpoints,
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (0,),
                    TrafficKind.WRITE_RSP: (0,),
                }
            ),
        ),
    )

    ack_wide_model = TransportCostModel(mesh=ack_wide_mesh, write_response_bytes=64)
    ack_narrow_model = TransportCostModel(mesh=ack_narrow_mesh, write_response_bytes=64)

    assert ack_narrow_model.l1_to_l2(ack_narrow_mesh.tile(0, 0), 64) > (
        ack_wide_model.l1_to_l2(ack_wide_mesh.tile(0, 0), 64)
    )


def test_transport_cost_rounds_nonzero_flow_transfer_time_up_to_one_cycle() -> None:
    mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=rectangular_test_tiles(2, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=32),),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l1(mesh.tile(0, 0), mesh.tile(1, 0), 1) == 3


def test_l1_to_l1_delta_cache_reuses_uniform_no_contention_costs() -> None:
    mesh = _uniform_l1_only_mesh(3, 2)
    model = TransportCostModel(mesh=mesh)

    first_cost = model.l1_to_l1(mesh.tile(0, 0), mesh.tile(1, 1), 64)

    assert model._l1_to_l1_delta_cache_enabled is True
    assert len(model._flow_cost_cache) == 3
    assert len(model._l1_to_l1_delta_estimate_cache) == 1

    second_cost = model.l1_to_l1(mesh.tile(1, 0), mesh.tile(2, 1), 64)

    assert second_cost == first_cost
    assert len(model._estimate_cache) == 2
    assert len(model._flow_cost_cache) == 3
    assert len(model._l1_to_l1_delta_estimate_cache) == 1


def test_l1_to_l1_delta_cache_is_disabled_when_accounting_noc_contention() -> None:
    mesh = _uniform_l1_only_mesh(3, 2)
    model = TransportCostModel(
        mesh=mesh,
        account_noc_contention=True,
    )

    model.l1_to_l1(mesh.tile(0, 0), mesh.tile(1, 1), 64)
    model.l1_to_l1(mesh.tile(1, 0), mesh.tile(2, 1), 64)

    assert model._l1_to_l1_delta_cache_enabled is False
    assert len(model._l1_to_l1_delta_estimate_cache) == 0
    assert len(model._flow_cost_cache) == 6


def test_l1_to_l1_delta_cache_is_disabled_on_nonuniform_noc() -> None:
    mesh = Mesh(
        width=3,
        height=2,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=rectangular_test_tiles(
            3,
            2,
            memory=L1Memory(size=4096, bandwidth=64),
            devices=(GENERIC_SCALAR_DEVICE,),
        ),
        noc=NoC(
            nodes=tuple(
                NoCNode(node_id=y * 3 + x, x=x, y=y)
                for y in range(2)
                for x in range(3)
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=1),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=1,
                    src_node_id=1,
                    dst_node_id=2,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=1),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=2,
                    src_node_id=3,
                    dst_node_id=4,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=5),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=3,
                    src_node_id=4,
                    dst_node_id=5,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=5),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=4,
                    src_node_id=0,
                    dst_node_id=3,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=1),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=5,
                    src_node_id=1,
                    dst_node_id=4,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=1),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=6,
                    src_node_id=2,
                    dst_node_id=5,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=1),),
                    bidirectional=True,
                ),
            ),
            endpoints=tuple(
                NoCEndpoint(
                    endpoint_id=tile_id,
                    kind=EndpointKind.L1,
                    node_id=tile_id,
                    tile_id=tile_id,
                )
                for tile_id in range(6)
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    top_cost = model.l1_to_l1(mesh.tile(0, 0), mesh.tile(1, 0), 64)
    bottom_cost = model.l1_to_l1(mesh.tile(0, 1), mesh.tile(1, 1), 64)

    assert model._l1_to_l1_delta_cache_enabled is False
    assert len(model._l1_to_l1_delta_estimate_cache) == 0
    assert len(model._flow_cost_cache) == 6
    assert bottom_cost > top_cost
