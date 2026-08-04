"""Shared contracts for logical Operations and their planned Tile Work."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from maps.planning.mapping import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorSlice,
    TensorSliceRef,
)
from maps.planning.mapping import Submesh
from maps.graph import Tensor
from maps.hardware import Device, Tile

if TYPE_CHECKING:
    from maps.graph import Node


class OperationPayload(ABC):
    """Source-independent logical Operation stored on a Graph Node."""


class TileWork(ABC):
    """Concrete work produced for one tile from one logical Operation."""

    @property
    @abstractmethod
    def input_slices(self) -> tuple[TensorSliceRef, ...]: ...

    @property
    @abstractmethod
    def output_slices(self) -> tuple[TensorSliceRef, ...]: ...

    @property
    def l1_bytes(self) -> int:
        return sum(ref.num_bytes for ref in self.input_slices + self.output_slices)

    def fits_l1(self, tile: Tile) -> bool:
        return self.l1_bytes <= tile.memory.size


class OpCostModel(ABC):
    """Cost contract evaluated from Tile Work and its assigned Device."""

    @abstractmethod
    def cost(self, tile_work: TileWork, tile: Tile, assigned_device: Device) -> int:
        """Return non-negative cycles spent on one tile's local work."""

    def placement_cost(
        self,
        *,
        node: Node,
        output_layouts: tuple[TensorLayout, ...],
    ) -> int:
        del node, output_layouts
        return 0

    def collective_cost(
        self,
        tile_work: TileWork,
        tile: Tile,
        assigned_device: Device,
        participants: tuple[Tile, ...],
    ) -> int:
        """Return latency for one synchronous participant group."""

        del tile_work, tile, assigned_device, participants
        raise ValueError("operation is not a synchronous collective")


def input_slices_for_tensor(
    tile_work: TileWork,
    tensor: Tensor,
) -> tuple[TensorSlice, ...]:
    """Return every non-empty input region required for one Tensor identity."""

    return tuple(
        reference.tensor_slice
        for reference in tile_work.input_slices
        if reference.tensor is tensor
        and reference.tensor_slice.num_elements > 0
    )


class OpPayload(OperationPayload):
    """Planning behavior shared by primitive logical Operations."""

    @property
    @abstractmethod
    def cost_model(self) -> OpCostModel: ...

    @abstractmethod
    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]: ...

    @abstractmethod
    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> TileWork: ...

    @staticmethod
    def single_output_layout(
        output_layouts: tuple[TensorLayout, ...],
    ) -> TensorLayout:
        if len(output_layouts) != 1:
            raise ValueError(
                f"operation expects exactly one output layout, got {len(output_layouts)}"
            )
        return output_layouts[0]


class CompositeOpPayload(OperationPayload):
    """Logical Operation that must decompose before Planning."""

    @abstractmethod
    def decompose(
        self,
        node: Node,
    ) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]: ...


@dataclass(frozen=True)
class LayoutRelation:
    """One indexed, bidirectional input/output layout relationship."""

    input_index: int
    output_index: int
    input_axis_for_output_axis: tuple[int, ...]
    guarantees_slice_containment: bool
    replicated_output_axes: frozenset[int] = frozenset()
    partial_output_axes: frozenset[int] = frozenset()
    resolves_partial_values: bool = False

    @classmethod
    def exact(
        cls,
        *,
        input_index: int,
        output_index: int,
        tensor: Tensor,
    ) -> LayoutRelation:
        return cls(
            input_index=input_index,
            output_index=output_index,
            input_axis_for_output_axis=tuple(range(tensor.rank)),
            guarantees_slice_containment=True,
        )

    def output_layout_from_input_layout(
        self,
        input_layout: TensorLayout,
    ) -> TensorLayout:
        output_axis_for_input_axis = {
            input_axis: output_axis
            for output_axis, input_axis in enumerate(self.input_axis_for_output_axis)
        }
        return TensorLayout(
            submesh=input_layout.submesh,
            mesh_x=self._output_axis(
                input_layout.mesh_x,
                output_axis_for_input_axis,
            ),
            mesh_y=self._output_axis(
                input_layout.mesh_y,
                output_axis_for_input_axis,
            ),
            logical_width=input_layout.logical_width,
            logical_height=input_layout.logical_height,
        )

    def _output_axis(
        self,
        axis: LayoutAxis,
        output_axis_for_input_axis: dict[int, int],
    ) -> LayoutAxis:
        retargeted = self._retarget_axis(axis, output_axis_for_input_axis)
        if self.resolves_partial_values and retargeted.mode is LayoutAxisMode.PARTIAL:
            return LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        if (
            retargeted.tensor_axis in self.partial_output_axes
            and retargeted.mode is LayoutAxisMode.SHARD
        ):
            return LayoutAxis(
                mode=LayoutAxisMode.PARTIAL,
                tensor_axis=retargeted.tensor_axis,
            )
        if retargeted.tensor_axis in self.replicated_output_axes:
            return LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        return retargeted

    def input_layout_from_output_layout(
        self,
        output_layout: TensorLayout,
    ) -> TensorLayout:
        mapping = {
            output_axis: input_axis
            for output_axis, input_axis in enumerate(self.input_axis_for_output_axis)
        }
        return TensorLayout(
            submesh=output_layout.submesh,
            mesh_x=self._retarget_axis(output_layout.mesh_x, mapping),
            mesh_y=self._retarget_axis(output_layout.mesh_y, mapping),
            logical_width=output_layout.logical_width,
            logical_height=output_layout.logical_height,
        )

    @staticmethod
    def _retarget_axis(axis: LayoutAxis, mapping: dict[int, int]) -> LayoutAxis:
        if axis.tensor_axis is None:
            return axis
        if axis.tensor_axis not in mapping:
            raise ValueError(
                f"layout relation does not map sharded tensor axis {axis.tensor_axis}"
            )
        return LayoutAxis(mode=axis.mode, tensor_axis=mapping[axis.tensor_axis])


def payload_layout_relations(payload: object) -> tuple[LayoutRelation, ...]:
    relations = getattr(payload, "layout_relations", ())
    return tuple(relations)


def find_layout_relation(
    payload: object,
    *,
    input_index: int,
    output_index: int,
) -> LayoutRelation | None:
    matches = tuple(
        relation
        for relation in payload_layout_relations(payload)
        if relation.input_index == input_index
        and relation.output_index == output_index
    )
    if len(matches) > 1:
        raise ValueError(
            f"payload declares duplicate layout relation for input {input_index}, "
            f"output {output_index}"
        )
    return matches[0] if matches else None


def sharded_layout(
    tensor: Tensor,
    submesh: Submesh,
    logical_shape: tuple[int, int] | None,
    *,
    mesh_x_axis: int | None = None,
    mesh_y_axis: int | None = None,
) -> TensorLayout:
    logical_width = None
    logical_height = None
    if logical_shape is not None:
        logical_width, logical_height = logical_shape

    if mesh_x_axis is None:
        mesh_x_axis = tensor.rank - 1
    if mesh_y_axis is None and tensor.rank >= 2:
        mesh_y_axis = tensor.rank - 2
    if mesh_x_axis < 0 or mesh_x_axis >= tensor.rank:
        raise ValueError("mesh_x_axis must be within tensor rank")
    if mesh_y_axis is not None and (mesh_y_axis < 0 or mesh_y_axis >= tensor.rank):
        raise ValueError("mesh_y_axis must be within tensor rank")
    if mesh_y_axis is not None and mesh_x_axis == mesh_y_axis:
        raise ValueError("mesh_x_axis and mesh_y_axis must be different when both shard")

    mesh_y = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
    if mesh_y_axis is not None:
        mesh_y = LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=mesh_y_axis)
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=mesh_x_axis),
        mesh_y=mesh_y,
        logical_width=logical_width,
        logical_height=logical_height,
    )


def require_tile_device(tile: Tile, assigned_device: Device) -> Device:
    if assigned_device not in tile.devices:
        raise ValueError(
            f"assigned device {assigned_device.name} is not present on tile "
            f"{tile.tile_id}"
        )
    return assigned_device
