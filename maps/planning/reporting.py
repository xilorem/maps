"""Planning-owned diagnostics for complete Execution Plans."""

from __future__ import annotations

from collections import defaultdict
import re

from maps.hardware import EndpointKind
from maps.planning.execution_plan import ExecutionPlan
from maps.planning.stages import StagePlacement, StagePlan, virtual_submesh
from maps.planning.placement.evaluation import evaluate_placement
from maps.planning.allocation.metrics import worst_tile_stage_compute
from maps.planning.transitions import VirtualTransition


def print_submeshes(execution_plan: ExecutionPlan) -> None:
    """Print one Execution Plan's Stage placement on the attached NoC."""

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
            labels_by_node_id[endpoint.node_id].append(
                _compact_l2_label(endpoint.name or "L2")
            )
        else:
            labels_by_node_id[endpoint.node_id].append(endpoint.kind.name)

    max_x = max(node.x for node in mesh.noc.nodes)
    max_y = max(node.y for node in mesh.noc.nodes)
    cell_strings: dict[tuple[int, int], str] = {}
    max_cell_width = 2

    for node in mesh.noc.nodes:
        labels = labels_by_node_id.get(node.node_id)
        cell = "/".join(labels) if labels else "."
        cell_strings[(node.x, node.y)] = cell
        max_cell_width = max(max_cell_width, len(cell))

    for y in range(max_y + 1):
        row = " ".join(
            cell_strings[(x, y)].rjust(max_cell_width)
            for x in range(max_x + 1)
        )
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


def print_execution_plan_stage_cost(
    execution_plan: ExecutionPlan,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    virtual_transitions: tuple[VirtualTransition, ...],
) -> None:
    """Print the combined worst-stage compute and physical IO estimate.

    Compute is evaluated from the final virtual layouts. Physical IO is
    evaluated from the separate spatial placements and ownership maps. The
    displayed total is the sum of the greatest stage compute and IO bottlenecks.
    """

    worst_stage_compute = max(
        (
            worst_tile_stage_compute(
                stage_nodes=plan.nodes,
                node_output_layouts=plan.node_output_layouts,
                submesh=virtual_submesh(plan),
                device_names=plan.device_names,
            )
            for plan in stage_plans.values()
        ),
        default=0,
    )
    evaluation = evaluate_placement(
        execution_plan.mesh,
        stage_plans,
        placements,
        virtual_transitions,
    )
    worst_stage_io = max(
        (breakdown.total for breakdown in evaluation.stage_breakdowns.values()),
        default=0,
    )
    print(
        "[planner] execution_plan_stage_cost="
        f"{worst_stage_compute + worst_stage_io} "
        f"(worst_compute={worst_stage_compute} worst_io={worst_stage_io})"
    )
