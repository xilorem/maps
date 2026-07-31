"""Human-readable diagnostics for workload-balancing results."""

from __future__ import annotations

from MAPS.core.graph import Node
from MAPS.planner.contracts.stages import StageSelection
from MAPS.planner.workload.metrics import SelectionEvaluation


def print_stage_metric_breakdown(
    enabled: bool,
    stage_selection: StageSelection,
    evaluation: SelectionEvaluation,
) -> None:
    """Print the canonical final compute and communication bottlenecks."""

    if not enabled:
        return
    print("[workload_balancing] final_stage_metric_breakdown:")
    for stage_id, stage_nodes in stage_selection.items():
        breakdown = evaluation.stage_breakdowns[stage_id]
        print(
            f"  stage={stage_id} nodes={_stage_label(stage_nodes)} "
            f"compute={breakdown.compute_cycles} "
            f"communication={breakdown.communication_cycles}"
        )
        for label in dict.fromkeys(
            getattr(node.payload.cost_model, "diagnostic_label", None)
            for node in stage_nodes
        ):
            if label is not None:
                print(f"    cost_diagnostic={label}")


def _stage_label(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact stage label for diagnostics."""

    return "+".join(node.name for node in stage_nodes)
