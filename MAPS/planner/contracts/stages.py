"""Data contracts exchanged between planner passes."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.graph import Node
from MAPS.core.submesh import Submesh

StageSelection = dict[int, tuple[Node, ...]]


@dataclass(frozen=True)
class StagePlan:
    """The workload-balancing result for one selected stage.

    ``tile_count`` and ``logical_shape`` describe virtual execution. ``nodes``
    and ``node_output_layouts`` have matching order and preserve the complete
    selected stage. Physical placement is deliberately absent and is represented
    by the separate ``StagePlacement`` contract produced by spatial mapping.
    """

    stage_id: int
    tile_count: int
    logical_shape: tuple[int, int]
    nodes: tuple[Node, ...]
    node_output_layouts: tuple[tuple, ...]
    device_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.device_names) != len(self.nodes):
            raise ValueError("Stage Plan must retain one Device name per Layer")
        if any(not device_name for device_name in self.device_names):
            raise ValueError("Stage Plan Device names must not be empty")


@dataclass(frozen=True)
class StagePlacement:
    """Bind one virtual stage layout to a connected physical tile region.

    ``virtual_to_physical`` must be a bijection covering both submeshes.  The
    physical region may be non-rectangular; connectivity is established by the
    spatial mapper before this contract is constructed.
    """

    stage_id: int
    virtual_submesh: Submesh
    physical_submesh: Submesh
    virtual_to_physical: dict[int, int]

    def __post_init__(self) -> None:
        """Validate complete bijective coverage of virtual and physical tiles."""

        virtual_tile_ids = {tile.tile_id for tile in self.virtual_submesh.tiles}
        physical_tile_ids = set(self.physical_submesh.tile_ids)
        if set(self.virtual_to_physical) != virtual_tile_ids:
            raise ValueError(
                f"placement for stage {self.stage_id} does not cover all virtual tiles"
            )
        if set(self.virtual_to_physical.values()) != physical_tile_ids:
            raise ValueError(
                f"placement for stage {self.stage_id} does not cover all physical tiles"
            )

    def physical_tile_id(self, virtual_tile_id: int) -> int:
        """Return the physical tile assigned to one virtual tile."""

        return self.virtual_to_physical[virtual_tile_id]


def virtual_submesh(plan: StagePlan):
    """Return the virtual submesh shared by a stage's chosen layouts."""

    for layouts in plan.node_output_layouts:
        if layouts:
            return layouts[0].submesh
    raise ValueError(f"stage {plan.stage_id} has no virtual layouts")
