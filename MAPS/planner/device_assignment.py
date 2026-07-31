"""Resolve exact hardware-owned Device Assignments for graph Nodes."""

from MAPS.arch import WorkSignature
from MAPS.core.graph import Node


def assigned_device_name(node: Node, tiles: tuple) -> str:
    """Resolve one stable Device name for a Node across homogeneous Tiles."""

    signature = WorkSignature.from_node(node)
    try:
        assigned = tuple(tile.assigned_device(signature) for tile in tiles)
    except ValueError as exc:
        raise ValueError(f"node {node.name} with {signature}: {exc}") from exc
    device_names = {device.name for device in assigned}
    if len(device_names) != 1:
        raise ValueError(
            f"node {node.name} with {signature} has inconsistent fixed Device "
            f"assignments across tiles: {sorted(device_names)}"
        )
    return assigned[0].name


__all__ = ["assigned_device_name"]
