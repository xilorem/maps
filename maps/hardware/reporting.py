"""Mesh printing helpers."""

from __future__ import annotations

from collections import defaultdict

from .mesh import Mesh
from .noc import EndpointKind


def print_mesh(mesh: Mesh) -> None:
    """Print one mesh and its attached NoC topology."""

    labels_by_node_id: dict[int, list[str]] = defaultdict(list)
    for endpoint in mesh.noc.endpoints:
        if endpoint.kind is EndpointKind.L1:
            if endpoint.tile_id is None:
                labels_by_node_id[endpoint.node_id].append("L1")
            else:
                labels_by_node_id[endpoint.node_id].append(f"T{endpoint.tile_id}")
        elif endpoint.kind is EndpointKind.L2:
            labels_by_node_id[endpoint.node_id].append(endpoint.name or "L2")
        else:
            labels_by_node_id[endpoint.node_id].append(endpoint.kind.name)

    max_x = max(node.x for node in mesh.noc.nodes)
    max_y = max(node.y for node in mesh.noc.nodes)

    cell_strings: dict[tuple[int, int], str] = {}
    max_cell_width = 2
    for node in mesh.noc.nodes:
        labels = labels_by_node_id.get(node.node_id)
        if labels:
            cell = "/".join(labels)
        else:
            cell = "."
        cell_strings[(node.x, node.y)] = cell
        max_cell_width = max(max_cell_width, len(cell))

    for y in range(max_y + 1):
        row = " ".join(cell_strings[(x, y)].rjust(max_cell_width) for x in range(max_x + 1))
        print(row)
