"""Temporary migration boundary for hardware-owned Device Assignment."""

from MAPS.arch import WorkKind
from MAPS.core.graph import Node


def node_requires_fixed_device_assignment(node: Node) -> bool:
    """Return whether this migration stage plans the Node through Tile policy."""

    return getattr(node.payload, "work_kind", None) in {
        WorkKind.GEMM,
        WorkKind.CAST,
        WorkKind.IM2COL,
        WorkKind.OUTPUT_REFORMAT,
    }


__all__ = ["node_requires_fixed_device_assignment"]
