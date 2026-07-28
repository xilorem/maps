"""Lower inter-stage tensor dependencies into physical transitions."""

from __future__ import annotations

from dataclasses import replace

from MAPS.planner.lowering.context import PipelineLoweringContext
from MAPS.planner.contracts.queries import (
    node_output_index,
    node_output_layouts,
    required_input_slices,
)
from MAPS.planner.contracts.stages import StagePlacement, StagePlan
from MAPS.ops.common.layout_relation import find_layout_relation
from MAPS.transitions import build_transition
from MAPS.transitions.model import Transition, TransitionFragment
from MAPS.transitions.model import TransitionMode


def build_transitions(
    context: PipelineLoweringContext,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
) -> tuple[tuple[Transition, ...], dict[tuple[int, int, int], int]]:
    """Build every cross-stage transfer and index it by destination input.

    Each graph dependency whose producer and consumer belong to different
    stages becomes one ``Transition``.  Transition fragments are first derived
    from the virtual tensor layouts and then translated to physical tile ids.
    The accompanying index lets layer lowering bind each destination input to
    its transition without repeating graph ownership analysis.
    """

    transitions: list[Transition] = []
    transition_id_by_stage_layer_input: dict[tuple[int, int, int], int] = {}

    for destination_stage_id, stage_nodes in context.stage_selection.items():
        destination_plan = stage_plans[destination_stage_id]
        destination_placement = placements[destination_stage_id]
        for destination_layer_idx, destination_node in enumerate(stage_nodes):
            output_layouts = node_output_layouts(destination_plan, destination_node)
            for destination_input_idx, tensor in enumerate(destination_node.inputs):
                source_node = context.producer_by_tensor.get(tensor)
                if source_node is None:
                    continue
                source_stage_id = context.node_stage_ids[id(source_node)]
                if source_stage_id == destination_stage_id:
                    continue

                source_plan = stage_plans[source_stage_id]
                source_output_idx = node_output_index(source_node, tensor)
                destination_input_layout = output_layouts[0]
                relation = find_layout_relation(
                    destination_node.payload,
                    input_index=destination_input_idx,
                    output_index=0,
                )
                if relation is not None:
                    destination_input_layout = (
                        relation.input_layout_from_output_layout(output_layouts[0])
                    )
                transition = build_transition(
                    name=(
                        f"transition_{source_node.name}_to_"
                        f"{destination_node.name}_{tensor.name}"
                    ),
                    tensor=tensor,
                    tensor_id=context.tensor_id_by_tensor[tensor],
                    src_layer_id=source_stage_id,
                    src_output_idx=source_output_idx,
                    dst_layer_id=destination_stage_id,
                    dst_input_idx=destination_input_idx,
                    src_layout=node_output_layouts(source_plan, source_node)[source_output_idx],
                    dst_layout=destination_input_layout,
                    dst_required_slices=required_input_slices(
                        tensor=tensor,
                        destination_node=destination_node,
                        destination_output_layouts=output_layouts,
                    ),
                    mode=getattr(
                        destination_node.payload,
                        "input_transition_mode",
                        TransitionMode.DIRECT_REMAP,
                    ),
                )
                transition_id = len(transitions)
                transitions.append(
                    _bind_transition_to_physical_tiles(
                        transition,
                        placements[source_stage_id],
                        destination_placement,
                    )
                )
                transition_id_by_stage_layer_input[
                    (destination_stage_id, destination_layer_idx, destination_input_idx)
                ] = transition_id

    return tuple(transitions), transition_id_by_stage_layer_input


def _bind_transition_to_physical_tiles(
    transition: Transition,
    source_placement: StagePlacement,
    destination_placement: StagePlacement,
) -> Transition:
    """Translate all virtual fragment endpoints to physical tile ids."""

    return replace(
        transition,
        fragments=tuple(
            TransitionFragment(
                src_hartid=source_placement.physical_tile_id(fragment.src_hartid),
                dst_hartid=destination_placement.physical_tile_id(fragment.dst_hartid),
                src_subslice=fragment.src_subslice,
                dst_subslice=fragment.dst_subslice,
            )
            for fragment in transition.fragments
        ),
    )
