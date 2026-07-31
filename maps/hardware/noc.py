"""Network-on-chip topology, routing, traffic, and endpoint contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property


class TrafficKind(Enum):
    """Protocol-level traffic classes carried by the NoC."""

    READ_REQ = auto()
    WRITE_REQ = auto()
    READ_RSP = auto()
    WRITE_RSP = auto()
    WRITE_DATA = auto()


class EndpointKind(Enum):
    """Traffic endpoint kinds attached to NoC nodes."""

    L1 = auto()
    L2 = auto()
    EXTERNAL = auto()
    PERIPHERAL = auto()


class RoutingPolicy(Enum):
    """Deterministic node-to-node routing policies."""

    XY = auto()
    TORUS_XY = auto()


@dataclass(frozen=True)
class NoCChannel:
    """One physical channel available on one link."""

    channel_id: int
    width_bytes: int
    hop_latency_cycles: int = 0
    tag: str | None = None
    supported_traffic: frozenset[TrafficKind] = frozenset()

    def __post_init__(self) -> None:
        if self.channel_id < 0:
            raise ValueError("channel_id must be >= 0")
        if self.width_bytes <= 0:
            raise ValueError("channel width_bytes must be > 0")
        if self.hop_latency_cycles < 0:
            raise ValueError("channel hop_latency_cycles must be >= 0")
        if self.tag == "":
            raise ValueError("channel tag must not be empty")

        supported_traffic = frozenset(self.supported_traffic)
        if any(not isinstance(kind, TrafficKind) for kind in supported_traffic):
            raise ValueError("supported_traffic must contain TrafficKind values")
        object.__setattr__(self, "supported_traffic", supported_traffic)

    def supports(self, traffic_kind: TrafficKind) -> bool:
        return not self.supported_traffic or traffic_kind in self.supported_traffic


@dataclass(frozen=True)
class NoCNode:
    """One topology vertex in the NoC graph."""

    node_id: int
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("node_id must be >= 0")
        if self.x < 0 or self.y < 0:
            raise ValueError("node coordinates must be >= 0")

    @property
    def coords(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class NoCEndpoint:
    """One producer/consumer attachment to a NoC node."""

    endpoint_id: int
    kind: EndpointKind
    node_id: int
    name: str = ""
    tile_id: int | None = None
    ingress_latency_cycles: int = 0
    egress_latency_cycles: int = 0
    ingress_bandwidth_bytes: int | None = None
    egress_bandwidth_bytes: int | None = None
    ingress_channels: tuple[NoCChannel, ...] = ()
    egress_channels: tuple[NoCChannel, ...] = ()

    def __post_init__(self) -> None:
        if self.endpoint_id < 0:
            raise ValueError("endpoint_id must be >= 0")
        if self.node_id < 0:
            raise ValueError("endpoint node_id must be >= 0")
        if self.tile_id is not None and self.tile_id < 0:
            raise ValueError("endpoint tile_id must be >= 0")
        if self.ingress_latency_cycles < 0:
            raise ValueError("endpoint ingress_latency_cycles must be >= 0")
        if self.egress_latency_cycles < 0:
            raise ValueError("endpoint egress_latency_cycles must be >= 0")
        if self.ingress_bandwidth_bytes is not None and self.ingress_bandwidth_bytes <= 0:
            raise ValueError("endpoint ingress_bandwidth_bytes must be > 0")
        if self.egress_bandwidth_bytes is not None and self.egress_bandwidth_bytes <= 0:
            raise ValueError("endpoint egress_bandwidth_bytes must be > 0")

        ingress_channels = tuple(self.ingress_channels)
        egress_channels = tuple(self.egress_channels)
        if len({channel.channel_id for channel in ingress_channels}) != len(ingress_channels):
            raise ValueError("endpoint ingress channel ids must be unique")
        if len({channel.channel_id for channel in egress_channels}) != len(egress_channels):
            raise ValueError("endpoint egress channel ids must be unique")

        object.__setattr__(self, "ingress_channels", ingress_channels)
        object.__setattr__(self, "egress_channels", egress_channels)


@dataclass(frozen=True)
class NoCLink:
    """One graph edge plus the channels available on it."""

    link_id: int
    src_node_id: int
    dst_node_id: int
    channels: tuple[NoCChannel, ...]
    bidirectional: bool = False

    def __post_init__(self) -> None:
        if self.link_id < 0:
            raise ValueError("link_id must be >= 0")
        if self.src_node_id < 0 or self.dst_node_id < 0:
            raise ValueError("link node ids must be >= 0")
        if self.src_node_id == self.dst_node_id:
            raise ValueError("link must connect two distinct nodes")

        channels = tuple(self.channels)
        if not channels:
            raise ValueError("link must have at least one channel")

        channel_ids = set()
        for channel in channels:
            if channel.channel_id in channel_ids:
                raise ValueError("link channel ids must be unique")
            channel_ids.add(channel.channel_id)

        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class TrafficPolicy:
    """Traffic-to-channel selection rules."""

    channel_ids_by_traffic: dict[TrafficKind, tuple[int, ...]]

    def __post_init__(self) -> None:
        normalized: dict[TrafficKind, tuple[int, ...]] = {}
        for traffic_kind, channel_ids in self.channel_ids_by_traffic.items():
            if not isinstance(traffic_kind, TrafficKind):
                raise ValueError("traffic policy keys must be TrafficKind values")

            normalized_ids = tuple(channel_ids)
            if not normalized_ids:
                raise ValueError("traffic policy channel id lists must not be empty")
            if any(channel_id < 0 for channel_id in normalized_ids):
                raise ValueError("traffic policy channel ids must be >= 0")
            if len(set(normalized_ids)) != len(normalized_ids):
                raise ValueError("traffic policy channel ids must be unique per traffic kind")

            normalized[traffic_kind] = normalized_ids

        object.__setattr__(self, "channel_ids_by_traffic", normalized)

    def allowed_channel_ids(self, traffic_kind: TrafficKind) -> tuple[int, ...]:
        return self.channel_ids_by_traffic.get(traffic_kind, ())


@dataclass(frozen=True)
class NoCRoute:
    """One concrete route between two endpoints."""

    src_endpoint_id: int
    dst_endpoint_id: int
    node_ids: tuple[int, ...]
    link_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.src_endpoint_id < 0 or self.dst_endpoint_id < 0:
            raise ValueError("route endpoint ids must be >= 0")

        node_ids = tuple(self.node_ids)
        link_ids = tuple(self.link_ids)
        if not node_ids:
            raise ValueError("route must include at least one node")
        if len(node_ids) != len(link_ids) + 1:
            raise ValueError("route node_ids length must be link_ids length + 1")

        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "link_ids", link_ids)


@dataclass(frozen=True)
class NoC:
    """Static NoC topology description."""

    nodes: tuple[NoCNode, ...]
    links: tuple[NoCLink, ...]
    endpoints: tuple[NoCEndpoint, ...] = ()
    traffic_policy: TrafficPolicy | None = None
    routing_policy: RoutingPolicy = RoutingPolicy.XY

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        links = tuple(self.links)
        endpoints = tuple(self.endpoints)

        # check for wrong node append to the noc
        nodes_by_id: dict[int, NoCNode] = {}
        node_coords: set[tuple[int, int]] = set()

        for node in nodes:
            if node.node_id in nodes_by_id:
                raise ValueError("node ids must be unique")
            if node.coords in node_coords:
                raise ValueError("node coordinates must be unique")
            nodes_by_id[node.node_id] = node
            node_coords.add(node.coords)

        # check for wrong link append to the noc
        channel_ids: set[int] = set()
        seen_link_ids: set[int] = set()

        for link in links:
            if link.link_id in seen_link_ids:
                raise ValueError("link ids must be unique")
            if link.src_node_id not in nodes_by_id or link.dst_node_id not in nodes_by_id:
                raise ValueError("link references unknown node_id")
            seen_link_ids.add(link.link_id)
            channel_ids.update(channel.channel_id for channel in link.channels)

        # check for wrong endpoint append in the noc
        endpoints_by_tile_kind: dict[tuple[int, EndpointKind], NoCEndpoint] = {}
        seen_endpoint_ids: set[int] = set()

        for endpoint in endpoints:
            if endpoint.endpoint_id in seen_endpoint_ids:
                raise ValueError("endpoint ids must be unique")
            if endpoint.node_id not in nodes_by_id:
                raise ValueError("endpoint references unknown node_id")
            seen_endpoint_ids.add(endpoint.endpoint_id)
            if endpoint.tile_id is not None:
                key = (endpoint.tile_id, endpoint.kind)
                if key in endpoints_by_tile_kind:
                    raise ValueError(
                        f"multiple {endpoint.kind.name} endpoints for tile_id {endpoint.tile_id}"
                    )
                endpoints_by_tile_kind[key] = endpoint
            channel_ids.update(channel.channel_id for channel in endpoint.ingress_channels)
            channel_ids.update(channel.channel_id for channel in endpoint.egress_channels)

        # check for correct channel referencing in traffic policy
        if self.traffic_policy is not None:
            for traffic_kind, allowed_channel_ids in self.traffic_policy.channel_ids_by_traffic.items():
                unknown_channel_ids = set(allowed_channel_ids) - channel_ids
                if unknown_channel_ids:
                    raise ValueError(
                        f"traffic policy for {traffic_kind.name} references unknown channel ids"
                    )

        # append attributes in correct formatting after checks
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "endpoints", endpoints)

    @cached_property
    def _nodes_by_id(self) -> dict[int, NoCNode]:
        return {node.node_id: node for node in self.nodes}

    @cached_property
    def _nodes_by_coords(self) -> dict[tuple[int, int], NoCNode]:
        return {node.coords: node for node in self.nodes}

    @cached_property
    def _links_by_id(self) -> dict[int, NoCLink]:
        return {link.link_id: link for link in self.links}

    @cached_property
    def _endpoints_by_id(self) -> dict[int, NoCEndpoint]:
        return {endpoint.endpoint_id: endpoint for endpoint in self.endpoints}

    @cached_property
    def _endpoints_by_kind(self) -> dict[EndpointKind, tuple[NoCEndpoint, ...]]:
        endpoints_by_kind = {kind: [] for kind in EndpointKind}
        for endpoint in self.endpoints:
            endpoints_by_kind[endpoint.kind].append(endpoint)
        return {kind: tuple(kind_endpoints) for kind, kind_endpoints in endpoints_by_kind.items()}

    @cached_property
    def _endpoints_by_tile_kind(self) -> dict[tuple[int, EndpointKind], NoCEndpoint]:
        return {
            (endpoint.tile_id, endpoint.kind): endpoint
            for endpoint in self.endpoints
            if endpoint.tile_id is not None
        }

    @cached_property
    def _outgoing_links_by_node_id(self) -> dict[int, tuple[NoCLink, ...]]:
        outgoing_links_by_node_id = {node.node_id: [] for node in self.nodes}
        for link in self.links:
            outgoing_links_by_node_id[link.src_node_id].append(link)
            if link.bidirectional:
                outgoing_links_by_node_id[link.dst_node_id].append(link)
        return {
            node_id: tuple(node_links)
            for node_id, node_links in outgoing_links_by_node_id.items()
        }

    @cached_property
    def _incoming_links_by_node_id(self) -> dict[int, tuple[NoCLink, ...]]:
        incoming_links_by_node_id = {node.node_id: [] for node in self.nodes}
        for link in self.links:
            incoming_links_by_node_id[link.dst_node_id].append(link)
            if link.bidirectional:
                incoming_links_by_node_id[link.src_node_id].append(link)
        return {
            node_id: tuple(node_links)
            for node_id, node_links in incoming_links_by_node_id.items()
        }

    @cached_property
    def _links_between_nodes(self) -> dict[tuple[int, int], NoCLink]:
        links_between_nodes: dict[tuple[int, int], NoCLink] = {}
        for link in self.links:
            links_between_nodes.setdefault((link.src_node_id, link.dst_node_id), link)
            if link.bidirectional:
                links_between_nodes.setdefault((link.dst_node_id, link.src_node_id), link)
        return links_between_nodes

    @cached_property
    def _routes_by_endpoint_pair(self) -> dict[tuple[int, int], NoCRoute]:
        return {}

    def node_by_id(self, node_id: int) -> NoCNode:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as exc:
            raise ValueError(f"node_id out of bounds: {node_id}") from exc

    def link_by_id(self, link_id: int) -> NoCLink:
        try:
            return self._links_by_id[link_id]
        except KeyError as exc:
            raise ValueError(f"unknown link_id: {link_id}") from exc

    def node_at(self, x: int, y: int) -> NoCNode:
        try:
            return self._nodes_by_coords[(x, y)]
        except KeyError as exc:
            raise ValueError(f"no node at coordinates: ({x}, {y})") from exc

    def endpoint_by_id(self, endpoint_id: int) -> NoCEndpoint:
        try:
            return self._endpoints_by_id[endpoint_id]
        except KeyError as exc:
            raise ValueError(f"unknown endpoint_id: {endpoint_id}") from exc

    def endpoints_of_kind(self, kind: EndpointKind) -> tuple[NoCEndpoint, ...]:
        return self._endpoints_by_kind[kind]

    def endpoint_for_tile(self, tile_id: int, kind: EndpointKind = EndpointKind.L1) -> NoCEndpoint:
        try:
            return self._endpoints_by_tile_kind[(tile_id, kind)]
        except KeyError as exc:
            raise ValueError(f"no {kind.name} endpoint for tile_id {tile_id}") from exc

    def outgoing_links(self, node_id: int) -> tuple[NoCLink, ...]:
        self.node_by_id(node_id)
        return self._outgoing_links_by_node_id[node_id]

    def incoming_links(self, node_id: int) -> tuple[NoCLink, ...]:
        self.node_by_id(node_id)
        return self._incoming_links_by_node_id[node_id]

    def route_endpoints(self, src_endpoint_id: int, dst_endpoint_id: int) -> NoCRoute:
        cached = self._routes_by_endpoint_pair.get((src_endpoint_id, dst_endpoint_id))
        if cached is not None:
            return cached

        src_endpoint = self.endpoint_by_id(src_endpoint_id)
        dst_endpoint = self.endpoint_by_id(dst_endpoint_id)

        if self.routing_policy is RoutingPolicy.XY:
            node_ids, link_ids = self._route_nodes_xy(src_endpoint.node_id, dst_endpoint.node_id)
            route = NoCRoute(
                src_endpoint_id=src_endpoint_id,
                dst_endpoint_id=dst_endpoint_id,
                node_ids=node_ids,
                link_ids=link_ids,
            )
            self._routes_by_endpoint_pair[(src_endpoint_id, dst_endpoint_id)] = route
            return route
        if self.routing_policy is RoutingPolicy.TORUS_XY:
            node_ids, link_ids = self._route_nodes_torus_xy(src_endpoint.node_id, dst_endpoint.node_id)
            route = NoCRoute(
                src_endpoint_id=src_endpoint_id,
                dst_endpoint_id=dst_endpoint_id,
                node_ids=node_ids,
                link_ids=link_ids,
            )
            self._routes_by_endpoint_pair[(src_endpoint_id, dst_endpoint_id)] = route
            return route
        raise ValueError(f"unsupported routing policy: {self.routing_policy}")

    def _route_nodes_xy(self, src_node_id: int, dst_node_id: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        current = self.node_by_id(src_node_id)
        dst = self.node_by_id(dst_node_id)

        node_ids = [current.node_id]
        link_ids: list[int] = []

        while current.x != dst.x:
            next_x = current.x + (1 if dst.x > current.x else -1)
            current = self._append_xy_step(current, next_x, current.y, node_ids, link_ids)

        while current.y != dst.y:
            next_y = current.y + (1 if dst.y > current.y else -1)
            current = self._append_xy_step(current, current.x, next_y, node_ids, link_ids)

        return tuple(node_ids), tuple(link_ids)

    def _append_xy_step(
        self,
        current: NoCNode,
        next_x: int,
        next_y: int,
        node_ids: list[int],
        link_ids: list[int],
    ) -> NoCNode:
        next_node = self.node_at(next_x, next_y)
        link = self._link_between_nodes(current.node_id, next_node.node_id)
        if link is None:
            raise ValueError(
                f"no XY link from node {current.node_id} to node {next_node.node_id}"
            )

        node_ids.append(next_node.node_id)
        link_ids.append(link.link_id)
        return next_node

    def _route_nodes_torus_xy(self, src_node_id: int, dst_node_id: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        current = self.node_by_id(src_node_id)
        dst = self.node_by_id(dst_node_id)

        node_ids = [current.node_id]
        link_ids: list[int] = []
        width = self._torus_width
        height = self._torus_height

        while current.x != dst.x:
            next_x = self._torus_step(current.x, dst.x, width)
            current = self._append_xy_step(current, next_x, current.y, node_ids, link_ids)

        while current.y != dst.y:
            next_y = self._torus_step(current.y, dst.y, height)
            current = self._append_xy_step(current, current.x, next_y, node_ids, link_ids)

        return tuple(node_ids), tuple(link_ids)

    @cached_property
    def _torus_width(self) -> int:
        if not self.nodes:
            raise ValueError("torus routing requires at least one node")
        return max(node.x for node in self.nodes) + 1

    @cached_property
    def _torus_height(self) -> int:
        if not self.nodes:
            raise ValueError("torus routing requires at least one node")
        return max(node.y for node in self.nodes) + 1

    @staticmethod
    def _torus_step(current: int, dst: int, size: int) -> int:
        forward = (dst - current) % size
        backward = (current - dst) % size
        if forward <= backward:
            return (current + 1) % size
        return (current - 1) % size

    def _link_between_nodes(self, src_node_id: int, dst_node_id: int) -> NoCLink | None:
        return self._links_between_nodes.get((src_node_id, dst_node_id))
