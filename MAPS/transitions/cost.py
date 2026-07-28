"""Transition-level aggregation built on top of primitive transfer legs."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.arch import Mesh
from MAPS.core.tensor import Tensor
from MAPS.transitions.model import Transition, TransitionFragment, TransitionMode

from .transport import TransferKind, TransferLeg, TransportCostModel


@dataclass(frozen=True)
class TransitionCost:
    """Aggregated cost of one transition."""

    mode: TransitionMode
    total_bytes: int
    legs: tuple[TransferLeg, ...] = field(default_factory=tuple)
    producer_loads: dict[int, int] = field(default_factory=dict)
    consumer_loads: dict[int, int] = field(default_factory=dict)
    resource_loads: dict[str, int] = field(default_factory=dict)
    total_cost: int = 0


def _transition_fragment_num_bytes(fragment: TransitionFragment, tensor: Tensor) -> int:
    return fragment.src_subslice.num_elements * tensor.elem_bytes


def _transition_fragment_dma_rows(
    fragment: TransitionFragment,
    tensor: Tensor,
) -> tuple[int | None, int]:
    """Return native-2D row geometry when either local view has row gaps."""
    if fragment.src_subslice.rank < 2:
        return None, 1

    src_inner = fragment.src_subslice.dims[-1]
    dst_inner = fragment.dst_subslice.dims[-1]
    if src_inner.length != dst_inner.length:
        return None, 1

    src_gapped = src_inner.length != fragment.src_subslice.parent.dims[-1].length
    dst_gapped = dst_inner.length != fragment.dst_subslice.parent.dims[-1].length
    if not (src_gapped or dst_gapped):
        return None, 1

    return src_inner.length * tensor.elem_bytes, fragment.src_subslice.num_elements // src_inner.length


def _aggregate_transition(
    legs: tuple[TransferLeg, ...],
    model: TransportCostModel,
    mode: TransitionMode,
) -> TransitionCost:
    producer_loads: dict[int, int] = {}
    consumer_loads: dict[int, int] = {}
    resource_loads: dict[str, int] = {}

    for leg in legs:
        estimate = model.estimate(leg)
        cost = estimate.total_cost

        if leg.src_tile is not None:
            producer_loads[leg.src_tile.tile_id] = producer_loads.get(leg.src_tile.tile_id, 0) + cost
        if leg.dst_tile is not None:
            consumer_loads[leg.dst_tile.tile_id] = consumer_loads.get(leg.dst_tile.tile_id, 0) + cost
        for resource_id, load in estimate.resource_loads.items():
            resource_loads[resource_id] = resource_loads.get(resource_id, 0) + load

    total_cost = max(
        max(producer_loads.values(), default=0),
        max(consumer_loads.values(), default=0),
        max(resource_loads.values(), default=0),
    )

    return TransitionCost(
        mode=mode,
        total_bytes=sum(leg.bytes for leg in legs),
        legs=legs,
        producer_loads=producer_loads,
        consumer_loads=consumer_loads,
        resource_loads=resource_loads,
        total_cost=total_cost,
    )


def _build_direct_remap_legs(
    transition: Transition,
    tensor: Tensor,
    mesh: Mesh,
) -> tuple[TransferLeg, ...]:
    legs: list[TransferLeg] = []

    for fragment in transition.fragments:
        row_bytes, rows = _transition_fragment_dma_rows(fragment, tensor)
        legs.append(
            TransferLeg(
                kind=TransferKind.L1_TO_L1,
                bytes=_transition_fragment_num_bytes(fragment, tensor),
                src_tile=mesh.tile_by_id(fragment.src_hartid),
                dst_tile=mesh.tile_by_id(fragment.dst_hartid),
                row_bytes=row_bytes,
                rows=rows,
            )
        )

    return tuple(legs)


def estimate_transition_cost(
    transition: Transition,
    tensor: Tensor,
    mesh: Mesh,
    model: TransportCostModel,
) -> TransitionCost:
    """Estimate the cost of one transition."""

    transition.validate_for(tensor)

    if transition.mode in (
        TransitionMode.DIRECT_REMAP,
        TransitionMode.PERMUTED_REMAP,
    ):
        legs = _build_direct_remap_legs(transition, tensor, mesh)
        return _aggregate_transition(legs, model, transition.mode)

    raise ValueError(f"unsupported transition mode: {transition.mode}")
