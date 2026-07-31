"""Migration bridge to Hardware-owned NoC contracts."""

from maps.hardware.noc import (
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

__all__ = [
    "EndpointKind",
    "NoC",
    "NoCChannel",
    "NoCEndpoint",
    "NoCLink",
    "NoCNode",
    "NoCRoute",
    "RoutingPolicy",
    "TrafficKind",
    "TrafficPolicy",
]
