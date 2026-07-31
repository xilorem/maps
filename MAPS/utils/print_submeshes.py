"""Submesh printing helpers."""

from __future__ import annotations

from collections import defaultdict
import re

from MAPS.arch import EndpointKind
from MAPS.pipeline.execution_plan import ExecutionPlan


def print_submeshes(execution_plan: ExecutionPlan) -> None:
    """Print one Execution Plan's submesh placement on the attached NoC."""

    mesh = execution_plan.mesh
    submesh_labels_by_tile_id: dict[int, list[str]] = defaultdict(list)
    for stage in execution_plan.stages:
        label = str(stage.submesh.submesh_id)
        for tile in stage.submesh.tiles:
            submesh_labels_by_tile_id[tile.tile_id].append(label)

    labels_by_node_id: dict[int, list[str]] = defaultdict(list)
    for endpoint in mesh.noc.endpoints:
        if endpoint.kind is EndpointKind.L1 and endpoint.tile_id is not None:
            labels = submesh_labels_by_tile_id.get(endpoint.tile_id)
            if labels:
                labels_by_node_id[endpoint.node_id].append("/".join(labels))
        elif endpoint.kind is EndpointKind.L2:
            labels_by_node_id[endpoint.node_id].append(_compact_l2_label(endpoint.name or "L2"))
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


def _compact_l2_label(label: str) -> str:
    match = re.fullmatch(r"l2_(\d+)", label)
    if match is None:
        return label
    return f"L{_base36(int(match.group(1)))}"


def _base36(value: int) -> str:
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value < 0:
        raise ValueError("base36 value must be >= 0")
    if value < 36:
        return digits[value]
    result = []
    while value:
        value, remainder = divmod(value, 36)
        result.append(digits[remainder])
    return "".join(reversed(result))
