"""Canonical transition compilation plus legacy transition costing."""

from .build import build_transition
from .compile import bind_transitions, build_virtual_transitions
from .contracts import (
    InputDestination,
    InputTransition,
    IntermediateTransition,
    OutputSource,
    OutputTransition,
    Transfer,
    Transition,
    VirtualInputDestination,
    VirtualInputTransition,
    VirtualIntermediateTransition,
    VirtualOutputSource,
    VirtualOutputTransition,
    VirtualTransfer,
    VirtualTransition,
)
from .cost import TransitionCost, estimate_transition_cost
from .model import TransitionFragment, TransitionMode
from .remap import build_direct_remap_fragments, tile_owned_slices
from .transport import TransferKind, TransferLeg, TransportCostModel

__all__ = [
    "InputDestination",
    "InputTransition",
    "IntermediateTransition",
    "OutputSource",
    "OutputTransition",
    "Transfer",
    "TransferKind",
    "TransferLeg",
    "TransportCostModel",
    "Transition",
    "TransitionCost",
    "TransitionFragment",
    "TransitionMode",
    "VirtualInputDestination",
    "VirtualInputTransition",
    "VirtualIntermediateTransition",
    "VirtualOutputSource",
    "VirtualOutputTransition",
    "VirtualTransfer",
    "VirtualTransition",
    "bind_transitions",
    "build_direct_remap_fragments",
    "build_transition",
    "build_virtual_transitions",
    "estimate_transition_cost",
    "tile_owned_slices",
]
