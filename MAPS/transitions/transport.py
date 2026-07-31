"""Primitive transport-cost model for one transfer leg."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math

from MAPS.arch import DMADevice, DMAJob, EndpointKind, Mesh, NoCChannel, NoCRoute, RoutingPolicy, Tile, TrafficKind


class TransferKind(Enum):
    """Atomic transfer kinds supported by the current cost model."""

    L1_TO_L2 = auto()
    L2_TO_L1 = auto()
    L1_TO_L1 = auto()


@dataclass(frozen=True)
class TransferLeg:
    """
    One concrete transfer operation.

    A leg is the atomic costed movement used to build a full transition cost.
    For example:
    - one producer tile writing a SubSlice to L2
    - one L2 read into one consumer tile
    - one direct L1-to-L1 SubSlice transfer between two tiles
    """

    kind: TransferKind
    bytes: int
    src_tile: Tile | None = None
    dst_tile: Tile | None = None
    row_bytes: int | None = None
    rows: int = 1

    def __post_init__(self) -> None:
        if self.bytes <= 0:
            raise ValueError("transfer leg bytes must be > 0")
        if self.row_bytes is None:
            if self.rows != 1:
                raise ValueError("rows must be 1 when row_bytes is not specified")
        elif self.row_bytes <= 0 or self.rows <= 0:
            raise ValueError("row_bytes and rows must be > 0")
        elif self.bytes != self.row_bytes * self.rows:
            raise ValueError("bytes must equal row_bytes * rows for a row-strided transfer")

        if self.kind is TransferKind.L1_TO_L2:
            if self.src_tile is None:
                raise ValueError("L1_TO_L2 legs require src_tile")
        elif self.kind is TransferKind.L2_TO_L1:
            if self.dst_tile is None:
                raise ValueError("L2_TO_L1 legs require dst_tile")
        elif self.kind is TransferKind.L1_TO_L1:
            if self.src_tile is None or self.dst_tile is None:
                raise ValueError("L1_TO_L1 legs require src_tile and dst_tile")
        else:
            raise ValueError(f"unsupported transfer kind: {self.kind}")


@dataclass(frozen=True)
class TransferCostEstimate:
    """Cost plus shared-resource loads for one transfer leg."""

    total_cost: int
    resource_loads: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _NoCFlow:
    """One protocol flow routed over the NoC."""

    src_endpoint_id: int
    dst_endpoint_id: int
    bytes: int
    traffic_kind: TrafficKind
    bandwidth_limit: int | None = None
    row_bytes: int | None = None
    rows: int = 1
    burst_bytes: int | None = None


@dataclass(frozen=True)
class TransportCostModel:
    """Primitive latency model for one transfer leg."""

    mesh: Mesh | None = None
    account_noc_contention: bool = False
    read_request_bytes: int = 1
    write_request_bytes: int = 1
    write_response_bytes: int = 1
    _estimate_cache: dict[
        tuple[TransferKind, int, int | None, int | None, int | None, int], TransferCostEstimate
    ] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
        default_factory=dict,
    )
    _flow_cost_cache: dict[_NoCFlow, TransferCostEstimate] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
        default_factory=dict,
    )
    _route_channels_cache: dict[tuple[tuple[int, ...], TrafficKind], tuple[NoCChannel, ...]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
        default_factory=dict,
    )
    _attachment_channel_cache: dict[tuple[int, str, TrafficKind], NoCChannel | None] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
        default_factory=dict,
    )
    _allowed_channel_ids_cache: dict[TrafficKind, tuple[int, ...]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
        default_factory=dict,
    )
    _l1_to_l1_delta_estimate_cache: dict[tuple[int, int, int, int], TransferCostEstimate] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
        default_factory=dict,
    )
    _l1_to_l1_delta_cache_enabled: bool | None = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
        default=None,
    )

    def l1_to_l2(self, src: Tile, bytes_: int, *, row_bytes: int | None = None, rows: int = 1) -> int:
        return self.estimate(
            TransferLeg(
                kind=TransferKind.L1_TO_L2,
                bytes=bytes_,
                src_tile=src,
                row_bytes=row_bytes,
                rows=rows,
            )
        ).total_cost

    def l2_to_l1(self, dst: Tile, bytes_: int, *, row_bytes: int | None = None, rows: int = 1) -> int:
        return self.estimate(
            TransferLeg(
                kind=TransferKind.L2_TO_L1,
                bytes=bytes_,
                dst_tile=dst,
                row_bytes=row_bytes,
                rows=rows,
            )
        ).total_cost

    def l1_to_l1(self, src: Tile, dst: Tile, bytes_: int, *, row_bytes: int | None = None, rows: int = 1) -> int:
        return self.estimate(
            TransferLeg(
                kind=TransferKind.L1_TO_L1,
                bytes=bytes_,
                src_tile=src,
                dst_tile=dst,
                row_bytes=row_bytes,
                rows=rows,
            )
        ).total_cost

    def estimate(self, leg: TransferLeg) -> TransferCostEstimate:

        # try using exact transfer caching
        leg_key = self._estimate_cache_key(leg)
        cached = self._estimate_cache.get(leg_key)
        if cached is not None:
            return cached
        
        # try using cached delta value for l1 to l1 transfers
        if leg.kind is TransferKind.L1_TO_L1 and leg.row_bytes is None:
            delta_key = self._l1_to_l1_delta_cache_key(leg.src_tile, leg.dst_tile, leg.bytes)
            if delta_key is not None:
                cached = self._l1_to_l1_delta_estimate_cache.get(delta_key)
                if cached is not None:
                    self._estimate_cache[leg_key] = cached
                    return cached

        if leg.kind is TransferKind.L1_TO_L2:
            estimate = self._estimate_l1_to_l2(leg.src_tile, leg)
        elif leg.kind is TransferKind.L2_TO_L1:
            estimate = self._estimate_l2_to_l1(leg.dst_tile, leg)
        elif leg.kind is TransferKind.L1_TO_L1:
            estimate = self._estimate_l1_to_l1(leg.src_tile, leg.dst_tile, leg)
        else:
            raise ValueError(f"unsupported transfer kind: {leg.kind}")
        
        # cache the computation for later reuse
        self._estimate_cache[leg_key] = estimate

        # try to cache the delta of the computation
        if leg.kind is TransferKind.L1_TO_L1 and leg.row_bytes is None:
            delta_key = self._l1_to_l1_delta_cache_key(leg.src_tile, leg.dst_tile, leg.bytes)
            if delta_key is not None:
                self._l1_to_l1_delta_estimate_cache.setdefault(delta_key, estimate)

        return estimate


    def cost(self, leg: TransferLeg) -> int:
        return self.estimate(leg).total_cost

    def resource_loads(self, leg: TransferLeg) -> dict[str, int]:
        return self.estimate(leg).resource_loads

    def _estimate_l1_to_l2(self, src: Tile, leg: TransferLeg) -> TransferCostEstimate:
        self._require_noc_with_l2()
        src_endpoint = self.mesh.noc.endpoint_for_tile(src.tile_id, EndpointKind.L1)
        estimate = min(
            (
                self._route_protocol_cost(
                    flows=(
                        _NoCFlow(
                            src_endpoint_id=src_endpoint.endpoint_id,
                            dst_endpoint_id=l2_endpoint.endpoint_id,
                            bytes=self.write_request_bytes,
                            traffic_kind=TrafficKind.WRITE_REQ,
                        ),
                        _NoCFlow(
                            src_endpoint_id=src_endpoint.endpoint_id,
                            dst_endpoint_id=l2_endpoint.endpoint_id,
                            bytes=leg.bytes,
                            traffic_kind=TrafficKind.WRITE_DATA,
                            bandwidth_limit=min(src.memory.bandwidth, self.mesh.l2_memory.bandwidth),
                            row_bytes=leg.row_bytes,
                            rows=leg.rows,
                            burst_bytes=self._dma_burst_bytes(src, DMAJob.WRITEJOB),
                        ),
                        _NoCFlow(
                            src_endpoint_id=l2_endpoint.endpoint_id,
                            dst_endpoint_id=src_endpoint.endpoint_id,
                            bytes=self.write_response_bytes,
                            traffic_kind=TrafficKind.WRITE_RSP,
                        ),
                    ),
                )
                for l2_endpoint in self.mesh.noc.endpoints_of_kind(EndpointKind.L2)
            ),
            key=lambda estimate: estimate.total_cost,
        )
        return self._with_dma_resource_load(
            estimate=estimate,
            tile=src,
            job=DMAJob.WRITEJOB,
        )

    def _estimate_l2_to_l1(self, dst: Tile, leg: TransferLeg) -> TransferCostEstimate:
        self._require_noc_with_l2()
        dst_endpoint = self.mesh.noc.endpoint_for_tile(dst.tile_id, EndpointKind.L1)
        estimate = min(
            (
                self._route_protocol_cost(
                    flows=(
                        _NoCFlow(
                            src_endpoint_id=dst_endpoint.endpoint_id,
                            dst_endpoint_id=l2_endpoint.endpoint_id,
                            bytes=self.read_request_bytes,
                            traffic_kind=TrafficKind.READ_REQ,
                        ),
                        _NoCFlow(
                            src_endpoint_id=l2_endpoint.endpoint_id,
                            dst_endpoint_id=dst_endpoint.endpoint_id,
                            bytes=leg.bytes,
                            traffic_kind=TrafficKind.READ_RSP,
                            bandwidth_limit=min(self.mesh.l2_memory.bandwidth, dst.memory.bandwidth),
                            row_bytes=leg.row_bytes,
                            rows=leg.rows,
                            burst_bytes=self._dma_burst_bytes(dst, DMAJob.READJOB),
                        ),
                    ),
                )
                for l2_endpoint in self.mesh.noc.endpoints_of_kind(EndpointKind.L2)
            ),
            key=lambda estimate: estimate.total_cost,
        )
        return self._with_dma_resource_load(
            estimate=estimate,
            tile=dst,
            job=DMAJob.READJOB,
        )

    def _estimate_l1_to_l1(self, src: Tile, dst: Tile, leg: TransferLeg) -> TransferCostEstimate:
        """Estimate a producer-initiated write into a consumer's L1."""

        self._require_noc()
        src_endpoint = self.mesh.noc.endpoint_for_tile(src.tile_id, EndpointKind.L1)
        dst_endpoint = self.mesh.noc.endpoint_for_tile(dst.tile_id, EndpointKind.L1)
        estimate = self._route_protocol_cost(
            flows=(
                _NoCFlow(
                    src_endpoint_id=src_endpoint.endpoint_id,
                    dst_endpoint_id=dst_endpoint.endpoint_id,
                    bytes=self.write_request_bytes,
                    traffic_kind=TrafficKind.WRITE_REQ,
                ),
                _NoCFlow(
                    src_endpoint_id=src_endpoint.endpoint_id,
                    dst_endpoint_id=dst_endpoint.endpoint_id,
                    bytes=leg.bytes,
                    traffic_kind=TrafficKind.WRITE_DATA,
                    bandwidth_limit=min(src.memory.bandwidth, dst.memory.bandwidth),
                    row_bytes=leg.row_bytes,
                    rows=leg.rows,
                    burst_bytes=self._dma_burst_bytes(src, DMAJob.WRITEJOB),
                ),
                _NoCFlow(
                    src_endpoint_id=dst_endpoint.endpoint_id,
                    dst_endpoint_id=src_endpoint.endpoint_id,
                    bytes=self.write_response_bytes,
                    traffic_kind=TrafficKind.WRITE_RSP,
                ),
            ),
        )
        return self._with_dma_resource_load(
            estimate=estimate,
            tile=src,
            job=DMAJob.WRITEJOB,
        )

    def _route_protocol_cost(
        self,
        flows: tuple[_NoCFlow, ...],
    ) -> TransferCostEstimate:
        total_cost = 0
        resource_loads: dict[str, int] = {}

        for flow in flows:
            estimate = self._route_flow_cost(flow)
            total_cost += estimate.total_cost
            for resource_id, load in estimate.resource_loads.items():
                resource_loads[resource_id] = resource_loads.get(resource_id, 0) + load

        return TransferCostEstimate(total_cost=total_cost, resource_loads=resource_loads)

    def _route_flow_cost(
        self,
        flow: _NoCFlow,
    ) -> TransferCostEstimate:
        cached = self._flow_cost_cache.get(flow)
        if cached is not None:
            return cached

        src_endpoint = self.mesh.noc.endpoint_by_id(flow.src_endpoint_id)
        dst_endpoint = self.mesh.noc.endpoint_by_id(flow.dst_endpoint_id)
        route = self.mesh.noc.route_endpoints(flow.src_endpoint_id, flow.dst_endpoint_id)
        src_attachment_channel = self._endpoint_attachment_channel(
            endpoint_id=src_endpoint.endpoint_id,
            direction="egress",
            channels=src_endpoint.egress_channels,
            traffic_kind=flow.traffic_kind,
            resource_name=f"endpoint {src_endpoint.endpoint_id} egress attachment",
        )
        dst_attachment_channel = self._endpoint_attachment_channel(
            endpoint_id=dst_endpoint.endpoint_id,
            direction="ingress",
            channels=dst_endpoint.ingress_channels,
            traffic_kind=flow.traffic_kind,
            resource_name=f"endpoint {dst_endpoint.endpoint_id} ingress attachment",
        )
        route_channels = self._route_channels(route, flow.traffic_kind)
        src_endpoint_bandwidth = (
            src_endpoint.egress_bandwidth_bytes
            if src_endpoint.egress_bandwidth_bytes is not None
            else None
        )
        dst_endpoint_bandwidth = (
            dst_endpoint.ingress_bandwidth_bytes
            if dst_endpoint.ingress_bandwidth_bytes is not None
            else None
        )
        route_bandwidth = min(
            (channel.width_bytes for channel in route_channels),
            default=None,
        )
        bandwidth = self._min_bandwidth(
            flow.bandwidth_limit,
            src_endpoint_bandwidth,
            dst_endpoint_bandwidth,
            src_attachment_channel.width_bytes if src_attachment_channel is not None else None,
            route_bandwidth,
            dst_attachment_channel.width_bytes if dst_attachment_channel is not None else None,
        )
        total_cost = (
            src_endpoint.egress_latency_cycles
            + (src_attachment_channel.hop_latency_cycles if src_attachment_channel is not None else 0)
            + dst_endpoint.ingress_latency_cycles
            + self._flow_transfer_cycles(flow, bandwidth)
            + sum(channel.hop_latency_cycles for channel in route_channels)
            + (dst_attachment_channel.hop_latency_cycles if dst_attachment_channel is not None else 0)
        )
        resource_loads = {}
        if self.account_noc_contention:
            resource_loads = {
                self._route_resource_id(link_id, channel.channel_id): self._flow_transfer_cycles(flow, channel.width_bytes)
                for link_id, channel in zip(route.link_ids, route_channels)
            }
            if src_attachment_channel is not None:
                resource_loads[
                    self._endpoint_attachment_resource_id(
                        src_endpoint.endpoint_id,
                        "egress",
                        src_attachment_channel.channel_id,
                    )
                ] = self._flow_transfer_cycles(flow, src_attachment_channel.width_bytes)
            if dst_attachment_channel is not None:
                resource_loads[
                    self._endpoint_attachment_resource_id(
                        dst_endpoint.endpoint_id,
                        "ingress",
                        dst_attachment_channel.channel_id,
                    )
                ] = self._flow_transfer_cycles(flow, dst_attachment_channel.width_bytes)
            if src_endpoint.egress_bandwidth_bytes is not None:
                resource_loads[self._endpoint_resource_id(src_endpoint.endpoint_id, "egress")] = (
                    self._flow_transfer_cycles(flow, src_endpoint.egress_bandwidth_bytes)
                )
            if dst_endpoint.ingress_bandwidth_bytes is not None:
                resource_loads[self._endpoint_resource_id(dst_endpoint.endpoint_id, "ingress")] = (
                    self._flow_transfer_cycles(flow, dst_endpoint.ingress_bandwidth_bytes)
                )
        estimate = TransferCostEstimate(total_cost=total_cost, resource_loads=resource_loads)
        self._flow_cost_cache[flow] = estimate
        return estimate

    def _require_noc(self) -> None:
        if self.mesh is None:
            raise ValueError("transport cost model requires a mesh")

    def _require_noc_with_l2(self) -> None:
        self._require_noc()
        if not self.mesh.noc.endpoints_of_kind(EndpointKind.L2):
            raise ValueError("transport cost model requires at least one NoC L2 endpoint")

    def _route_channels(self, route: NoCRoute, traffic_kind: TrafficKind) -> tuple[NoCChannel, ...]:
        cached = self._route_channels_cache.get((route.link_ids, traffic_kind))
        if cached is not None:
            return cached

        selected_channels = []
        for link_id in route.link_ids:
            link = self.mesh.noc.link_by_id(link_id)
            selected_channels.append(
                self._select_channel(
                    link.channels,
                    traffic_kind,
                    f"link {link.link_id}",
                )
            )

        selected = tuple(selected_channels)
        self._route_channels_cache[(route.link_ids, traffic_kind)] = selected
        return selected

    def _endpoint_attachment_channel(
        self,
        endpoint_id: int,
        direction: str,
        channels: tuple[NoCChannel, ...],
        traffic_kind: TrafficKind,
        resource_name: str,
    ) -> NoCChannel | None:

        # try to hit cached channel selection
        cached = self._attachment_channel_cache.get((endpoint_id, direction, traffic_kind))
        if cached is not None or (endpoint_id, direction, traffic_kind) in self._attachment_channel_cache:
            return cached

        # fallback for no channel
        if not channels:
            self._attachment_channel_cache[(endpoint_id, direction, traffic_kind)] = None
            return None

        selected = self._select_channel(channels, traffic_kind, resource_name)
        self._attachment_channel_cache[(endpoint_id, direction, traffic_kind)] = selected
        return selected


    def _select_channel(
        self,
        channels: tuple[NoCChannel, ...],
        traffic_kind: TrafficKind,
        resource_name: str,
    ) -> NoCChannel:
        allowed_channel_ids = self._allowed_channel_ids_cache.get(traffic_kind)
        if allowed_channel_ids is None:
            allowed_channel_ids = ()
            if self.mesh.noc.traffic_policy is not None:
                allowed_channel_ids = self.mesh.noc.traffic_policy.allowed_channel_ids(traffic_kind)
            self._allowed_channel_ids_cache[traffic_kind] = allowed_channel_ids

        candidates = tuple(
            channel
            for channel in channels
            if channel.supports(traffic_kind)
            and (not allowed_channel_ids or channel.channel_id in allowed_channel_ids)
        )
        if not candidates:
            raise ValueError(f"no channel available for {traffic_kind.name} on {resource_name}")

        return max(candidates, key=lambda channel: (channel.width_bytes, -channel.hop_latency_cycles))

    @staticmethod
    def _route_resource_id(link_id: int, channel_id: int) -> str:
        return f"noc_link:{link_id}:channel:{channel_id}"
        dma_devices = []
        for device in self.devices:
            if isinstance(device, DMADevice):
                if device.job == job:
                    dma_devices.append(device)

    @staticmethod
    def _endpoint_resource_id(endpoint_id: int, direction: str) -> str:
        return f"noc_endpoint:{endpoint_id}:{direction}"

    @staticmethod
    def _endpoint_attachment_resource_id(endpoint_id: int, direction: str, channel_id: int) -> str:
        return f"noc_endpoint_attachment:{endpoint_id}:{direction}:channel:{channel_id}"

    def _with_dma_resource_load(
        self,
        estimate: TransferCostEstimate,
        tile: Tile,
        job: DMAJob,
    ) -> TransferCostEstimate:
        dma_devices = tile.dma_devices(job)
        if not dma_devices:
            return estimate

        resource_loads = dict(estimate.resource_loads)
        resource_id = self._dma_resource_id(tile.tile_id, dma_devices[0])
        resource_loads[resource_id] = resource_loads.get(resource_id, 0) + estimate.total_cost
        return TransferCostEstimate(
            total_cost=estimate.total_cost,
            resource_loads=resource_loads,
        )

    @staticmethod
    def _dma_burst_bytes(tile: Tile, job: DMAJob) -> int | None:
        dma_devices = tile.dma_devices(job)
        return dma_devices[0].burst_bytes if dma_devices else None

    @staticmethod
    def _dma_resource_id(tile_id: int, device: DMADevice) -> str:
        return f"tile:{tile_id}:dma:{device.name}"

    def _l1_to_l1_delta_cache_key(
        self,
        src: Tile,
        dst: Tile,
        bytes_: int,
    ) -> tuple[int, int, int, int] | None:
        if any(isinstance(device, DMADevice) for device in src.devices + dst.devices):
            return None
        if not self._can_use_l1_to_l1_delta_cache():
            return None

        src_endpoint = self.mesh.noc.endpoint_for_tile(src.tile_id, EndpointKind.L1)
        dst_endpoint = self.mesh.noc.endpoint_for_tile(dst.tile_id, EndpointKind.L1)
        src_node = self.mesh.noc.node_by_id(src_endpoint.node_id)
        dst_node = self.mesh.noc.node_by_id(dst_endpoint.node_id)
        return (
            dst_node.x - src_node.x,
            dst_node.y - src_node.y,
            bytes_,
            min(src.memory.bandwidth, dst.memory.bandwidth),
        )

    def _can_use_l1_to_l1_delta_cache(self) -> bool:
        cached = self._l1_to_l1_delta_cache_enabled
        if cached is not None:
            return cached

        enabled = self._compute_l1_to_l1_delta_cache_enabled()
        object.__setattr__(self, "_l1_to_l1_delta_cache_enabled", enabled)
        return enabled

    def _compute_l1_to_l1_delta_cache_enabled(self) -> bool:
        if self.account_noc_contention or self.mesh is None:
            return False
        if self.mesh.noc.routing_policy is not RoutingPolicy.XY:
            return False

        l1_endpoints = self.mesh.noc.endpoints_of_kind(EndpointKind.L1)
        if not l1_endpoints:
            return False
        if any(endpoint.ingress_channels or endpoint.egress_channels for endpoint in l1_endpoints):
            return False

        endpoint_signature = (
            l1_endpoints[0].ingress_latency_cycles,
            l1_endpoints[0].egress_latency_cycles,
            l1_endpoints[0].ingress_bandwidth_bytes,
            l1_endpoints[0].egress_bandwidth_bytes,
        )
        if any(
            (
                endpoint.ingress_latency_cycles,
                endpoint.egress_latency_cycles,
                endpoint.ingress_bandwidth_bytes,
                endpoint.egress_bandwidth_bytes,
            ) != endpoint_signature
            for endpoint in l1_endpoints[1:]
        ):
            return False

        max_x = max(node.x for node in self.mesh.noc.nodes)
        max_y = max(node.y for node in self.mesh.noc.nodes)
        expected_node_count = (max_x + 1) * (max_y + 1)
        if len(self.mesh.noc.nodes) != expected_node_count:
            return False

        channel_signature: tuple[tuple[int, int, int, str | None, frozenset[TrafficKind]], ...] | None = None
        for y in range(max_y + 1):
            for x in range(max_x + 1):
                current = self.mesh.noc.node_at(x, y)
                for next_x, next_y in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if next_x < 0 or next_x > max_x or next_y < 0 or next_y > max_y:
                        continue
                    next_node = self.mesh.noc.node_at(next_x, next_y)
                    link = self.mesh.noc._link_between_nodes(current.node_id, next_node.node_id)
                    if link is None:
                        return False
                    signature = tuple(
                        (
                            channel.channel_id,
                            channel.width_bytes,
                            channel.hop_latency_cycles,
                            channel.tag,
                            channel.supported_traffic,
                        )
                        for channel in link.channels
                    )
                    if channel_signature is None:
                        channel_signature = signature
                    elif signature != channel_signature:
                        return False

        return True

    @staticmethod
    def _estimate_cache_key(
        leg: TransferLeg,
    ) -> tuple[TransferKind, int, int | None, int | None, int | None, int]:
        return (
            leg.kind,
            leg.bytes,
            None if leg.src_tile is None else leg.src_tile.tile_id,
            None if leg.dst_tile is None else leg.dst_tile.tile_id,
            leg.row_bytes,
            leg.rows,
        )

    @staticmethod
    def _min_bandwidth(*bandwidths: int | None) -> int | None:
        finite = tuple(bandwidth for bandwidth in bandwidths if bandwidth is not None)
        return min(finite) if finite else None

    @staticmethod
    def _transfer_cycles(bytes_: int, bandwidth: int | None) -> int:
        if bytes_ <= 0:
            return 0
        if bandwidth is None:
            return 1
        return max(1, math.ceil(bytes_ / bandwidth))

    @classmethod
    def _flow_transfer_cycles(cls, flow: _NoCFlow, bandwidth: int | None) -> int:
        if flow.row_bytes is None or flow.burst_bytes is None:
            return cls._transfer_cycles(flow.bytes, bandwidth)

        row_cost = sum(
            cls._transfer_cycles(min(flow.burst_bytes, flow.row_bytes - offset), bandwidth)
            for offset in range(0, flow.row_bytes, flow.burst_bytes)
        )
        return flow.rows * row_cost
