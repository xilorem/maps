"""Temporary migration boundary for hardware-owned Device Assignment."""

from MAPS.arch import WorkKind
from MAPS.core.graph import Node


def node_requires_fixed_device_assignment(node: Node) -> bool:
    """Return whether this migration stage plans the Node through Tile policy."""

    return getattr(node.payload, "work_kind", None) is WorkKind.GEMM


__all__ = ["node_requires_fixed_device_assignment"]
