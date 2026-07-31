import pytest

from maps.hardware import (
    EndpointKind,
    NoC,
    NoCChannel,
    NoCEndpoint,
    NoCLink,
    NoCNode,
    NoCRoute,
    RoutingPolicy,
    TrafficKind,
    TrafficPolicy,
)


def test_noc_preserves_lookup_helpers_and_link_directions() -> None:
    noc = NoC(
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
                    NoCChannel(channel_id=0, width_bytes=4, tag="req"),
                    NoCChannel(channel_id=1, width_bytes=32, tag="data"),
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
                TrafficKind.READ_REQ: (0,),
                TrafficKind.WRITE_DATA: (1,),
            }
        ),
    )

    assert noc.node_by_id(0).coords == (0, 0)
    assert noc.link_by_id(0).bidirectional is True
    assert noc.endpoint_by_id(1).name == "l2"
    assert noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_REQ) == (0,)
    assert tuple(link.link_id for link in noc.outgoing_links(0)) == (0,)
    assert tuple(link.link_id for link in noc.outgoing_links(1)) == (0,)
    assert tuple(link.link_id for link in noc.incoming_links(0)) == (0,)
    assert tuple(link.link_id for link in noc.incoming_links(1)) == (0,)


def test_noc_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValueError, match="node ids must be unique"):
        NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=0, x=1, y=0),
            ),
            links=(),
        )


def test_noc_rejects_link_referencing_unknown_node() -> None:
    with pytest.raises(ValueError, match="link references unknown node_id"):
        NoC(
            nodes=(NoCNode(node_id=0, x=0, y=0),),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=4),),
                ),
            ),
        )


def test_noc_link_rejects_empty_channels() -> None:
    with pytest.raises(ValueError, match="link must have at least one channel"):
        NoCLink(
            link_id=0,
            src_node_id=0,
            dst_node_id=1,
            channels=(),
        )


def test_noc_link_rejects_duplicate_channel_ids() -> None:
    with pytest.raises(ValueError, match="link channel ids must be unique"):
        NoCLink(
            link_id=0,
            src_node_id=0,
            dst_node_id=1,
            channels=(
                NoCChannel(channel_id=0, width_bytes=4),
                NoCChannel(channel_id=0, width_bytes=8),
            ),
        )


def test_noc_endpoint_rejects_negative_attachment_latency() -> None:
    with pytest.raises(ValueError, match="endpoint ingress_latency_cycles must be >= 0"):
        NoCEndpoint(
            endpoint_id=0,
            kind=EndpointKind.L1,
            node_id=0,
            ingress_latency_cycles=-1,
        )

    with pytest.raises(ValueError, match="endpoint egress_latency_cycles must be >= 0"):
        NoCEndpoint(
            endpoint_id=0,
            kind=EndpointKind.L1,
            node_id=0,
            egress_latency_cycles=-1,
        )


def test_noc_endpoint_rejects_non_positive_attachment_bandwidth() -> None:
    with pytest.raises(ValueError, match="endpoint ingress_bandwidth_bytes must be > 0"):
        NoCEndpoint(
            endpoint_id=0,
            kind=EndpointKind.L1,
            node_id=0,
            ingress_bandwidth_bytes=0,
        )

    with pytest.raises(ValueError, match="endpoint egress_bandwidth_bytes must be > 0"):
        NoCEndpoint(
            endpoint_id=0,
            kind=EndpointKind.L1,
            node_id=0,
            egress_bandwidth_bytes=-1,
        )


def test_noc_endpoint_rejects_duplicate_attachment_channel_ids() -> None:
    with pytest.raises(ValueError, match="endpoint ingress channel ids must be unique"):
        NoCEndpoint(
            endpoint_id=0,
            kind=EndpointKind.L1,
            node_id=0,
            ingress_channels=(
                NoCChannel(channel_id=0, width_bytes=4),
                NoCChannel(channel_id=0, width_bytes=8),
            ),
        )

    with pytest.raises(ValueError, match="endpoint egress channel ids must be unique"):
        NoCEndpoint(
            endpoint_id=0,
            kind=EndpointKind.L1,
            node_id=0,
            egress_channels=(
                NoCChannel(channel_id=1, width_bytes=4),
                NoCChannel(channel_id=1, width_bytes=8),
            ),
        )


def test_noc_rejects_policy_referencing_unknown_channel_ids() -> None:
    with pytest.raises(ValueError, match="references unknown channel ids"):
        NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=4),),
                ),
            ),
            traffic_policy=TrafficPolicy({TrafficKind.READ_REQ: (1,)}),
        )


def test_noc_accepts_policy_referencing_endpoint_attachment_channel_ids() -> None:
    noc = NoC(
        nodes=(NoCNode(node_id=0, x=0, y=0),),
        links=(),
        endpoints=(
            NoCEndpoint(
                endpoint_id=0,
                kind=EndpointKind.L2,
                node_id=0,
                ingress_channels=(NoCChannel(channel_id=7, width_bytes=4),),
            ),
        ),
        traffic_policy=TrafficPolicy({TrafficKind.READ_REQ: (7,)}),
    )

    assert noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_REQ) == (7,)


def test_noc_route_requires_one_more_node_than_links() -> None:
    with pytest.raises(ValueError, match="route node_ids length must be link_ids length \\+ 1"):
        NoCRoute(
            src_endpoint_id=0,
            dst_endpoint_id=1,
            node_ids=(0, 1),
            link_ids=(0, 1),
        )


def test_noc_routes_endpoints_with_xy_policy() -> None:
    noc = NoC(
        nodes=(
            NoCNode(node_id=0, x=0, y=0),
            NoCNode(node_id=1, x=1, y=0),
            NoCNode(node_id=2, x=0, y=1),
            NoCNode(node_id=3, x=1, y=1),
        ),
        links=(
            NoCLink(link_id=0, src_node_id=0, dst_node_id=1, channels=(NoCChannel(channel_id=0, width_bytes=4),), bidirectional=True),
            NoCLink(link_id=1, src_node_id=2, dst_node_id=3, channels=(NoCChannel(channel_id=0, width_bytes=4),), bidirectional=True),
            NoCLink(link_id=2, src_node_id=0, dst_node_id=2, channels=(NoCChannel(channel_id=0, width_bytes=4),), bidirectional=True),
            NoCLink(link_id=3, src_node_id=1, dst_node_id=3, channels=(NoCChannel(channel_id=0, width_bytes=4),), bidirectional=True),
        ),
        endpoints=(
            NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
            NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=3),
        ),
        routing_policy=RoutingPolicy.XY,
    )

    route = noc.route_endpoints(0, 1)

    assert route.src_endpoint_id == 0
    assert route.dst_endpoint_id == 1
    assert route.node_ids == (0, 1, 3)
    assert route.link_ids == (0, 3)


def test_noc_routes_same_node_without_links() -> None:
    noc = NoC(
        nodes=(NoCNode(node_id=0, x=0, y=0),),
        links=(),
        endpoints=(
            NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
            NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=0),
        ),
    )

    route = noc.route_endpoints(0, 1)

    assert route.node_ids == (0,)
    assert route.link_ids == ()


def test_noc_xy_routing_rejects_missing_required_link() -> None:
    noc = NoC(
        nodes=(
            NoCNode(node_id=0, x=0, y=0),
            NoCNode(node_id=1, x=1, y=0),
            NoCNode(node_id=2, x=0, y=1),
            NoCNode(node_id=3, x=1, y=1),
        ),
        links=(
            NoCLink(link_id=0, src_node_id=0, dst_node_id=2, channels=(NoCChannel(channel_id=0, width_bytes=4),), bidirectional=True),
            NoCLink(link_id=1, src_node_id=2, dst_node_id=3, channels=(NoCChannel(channel_id=0, width_bytes=4),), bidirectional=True),
        ),
        endpoints=(
            NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
            NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=3),
        ),
    )

    with pytest.raises(ValueError, match="no XY link from node 0 to node 1"):
        noc.route_endpoints(0, 1)


def test_noc_xy_routing_rejects_wrong_link_direction() -> None:
    noc = NoC(
        nodes=(
            NoCNode(node_id=0, x=0, y=0),
            NoCNode(node_id=1, x=1, y=0),
        ),
        links=(
            NoCLink(
                link_id=0,
                src_node_id=1,
                dst_node_id=0,
                channels=(NoCChannel(channel_id=0, width_bytes=4),),
            ),
        ),
        endpoints=(
            NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
            NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=1),
        ),
    )

    with pytest.raises(ValueError, match="no XY link from node 0 to node 1"):
        noc.route_endpoints(0, 1)


def test_noc_routes_endpoints_with_torus_xy_policy_using_wraparound() -> None:
    noc = NoC(
        nodes=tuple(
            NoCNode(node_id=y * 3 + x, x=x, y=y)
            for y in range(2)
            for x in range(3)
        ),
        links=tuple(
            NoCLink(
                link_id=link_id,
                src_node_id=src_node_id,
                dst_node_id=dst_node_id,
                channels=(NoCChannel(channel_id=0, width_bytes=4),),
                bidirectional=True,
            )
            for link_id, (src_node_id, dst_node_id) in enumerate(
                (
                    (0, 1), (1, 2), (2, 0),
                    (3, 4), (4, 5), (5, 3),
                    (0, 3), (1, 4), (2, 5),
                    (3, 0), (4, 1), (5, 2),
                )
            )
        ),
        endpoints=(
            NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=2, tile_id=0),
            NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=0),
        ),
        routing_policy=RoutingPolicy.TORUS_XY,
    )

    route = noc.route_endpoints(0, 1)

    assert route.node_ids == (2, 0)
    assert len(route.link_ids) == 1
