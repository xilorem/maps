"""Collective operation semantics, Tile Work, and costing."""

from __future__ import annotations

from dataclasses import dataclass

from maps.hardware import Device, Tile, WorkKind
from maps.planning.mapping import (
    tensor_slice_num_bytes,
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from maps.planning.mapping import Submesh
from maps.planning.transitions.transport import TransportCostModel
from maps.graph import Node, Tensor
from .contracts import OpCostModel, OpPayload, TileWork, sharded_layout
from .contracts import LayoutRelation


@dataclass(frozen=True)
class CollectiveTileWork(TileWork):
    """Concrete collective slices associated with one tile."""

    x: Tensor
    output: Tensor
    input_slice: TensorSlice
    output_slice: TensorSlice

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.x, tensor_slice=self.input_slice),)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)


@dataclass(frozen=True)
class AllReducePayload(OpPayload):
    """Configured intra-stage allreduce collective."""

    op_name: str
    x: Tensor
    output: Tensor
    reduction: str
    collective_axis: str = "x"

    @property
    def work_kind(self) -> WorkKind:
        return WorkKind.GROUP_REDUCE

    def __post_init__(self) -> None:
        if self.reduction not in {"sum", "max"}:
            raise ValueError("AllReducePayload reduction must be 'sum' or 'max'")
        if self.collective_axis not in {"x", "y"}:
            raise ValueError("AllReducePayload collective_axis must be 'x' or 'y'")
        self.validate_shapes()

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return (
            LayoutRelation.exact(input_index=0, output_index=0, tensor=self.x),
        )

    @property
    def cost_model(self) -> OpCostModel:
        return AllReduceCostModel(
            reduction=self.reduction,
            collective_axis=self.collective_axis,
        )

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (self._collective_layout(self.output, submesh, logical_shape),)

    def _collective_layout(
        self,
        tensor: Tensor,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None,
    ) -> TensorLayout:
        layout = sharded_layout(tensor, submesh, logical_shape)
        mesh_x = layout.mesh_x
        mesh_y = layout.mesh_y
        if self.collective_axis == "x":
            mesh_x = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        else:
            mesh_y = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        return TensorLayout(
            submesh=submesh,
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            logical_width=layout.logical_width,
            logical_height=layout.logical_height,
        )

    def _input_layout_from_output_layout(self, output_layout: TensorLayout) -> TensorLayout:
        return output_layout

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> CollectiveTileWork:
        output_layout = self.single_output_layout(output_layouts)
        input_layout = self._input_layout_from_output_layout(output_layout)
        return CollectiveTileWork(
            x=self.x,
            output=self.output,
            input_slice=tile_tensor_slice(self.x, input_layout, tile),
            output_slice=tile_tensor_slice(self.output, output_layout, tile),
        )

    def validate_shapes(self) -> None:
        if self.x.rank != self.output.rank or self.x.dims != self.output.dims:
            raise ValueError(f"{self.op_name} input and output shapes must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError(f"{self.op_name} input and output element sizes must match")


@dataclass(frozen=True)
class AllReduceCostModel(OpCostModel):
    """Placement-sensitive allreduce latency model."""

    reduction: str
    collective_axis: str = "x"

    def __post_init__(self) -> None:
        if self.reduction not in {"sum", "max"}:
            raise ValueError("AllReduceCostModel reduction must be 'sum' or 'max'")
        if self.collective_axis not in {"x", "y"}:
            raise ValueError("AllReduceCostModel collective_axis must be 'x' or 'y'")

    def cost(
        self,
        tile_work: object,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        del assigned_device
        del tile_work, tile
        return 0

    def placement_cost(
        self,
        *,
        node: Node,
        output_layouts: tuple[TensorLayout, ...],
    ) -> int:
        if len(output_layouts) != 1:
            raise ValueError("AllReduceCostModel expects exactly one output layout")

        output_layout = output_layouts[0]
        output_tensor = node.outputs[0]
        model = TransportCostModel(mesh=output_layout.submesh.mesh)
        groups = _logical_collective_groups(output_layout, self.collective_axis)
        group_costs = []
        for group_tiles in groups:
            payload_tiles = tuple(
                tile
                for tile in group_tiles
                if tensor_slice_num_bytes(
                    output_tensor,
                    tile_tensor_slice(output_tensor, output_layout, tile),
                ) > 0
            )
            if len(payload_tiles) <= 1:
                group_costs.append(0)
                continue

            root_tile = payload_tiles[0]
            reduce_phase = max(
                (
                    model.l1_to_l1(
                        tile,
                        root_tile,
                        tensor_slice_num_bytes(
                            output_tensor,
                            tile_tensor_slice(output_tensor, output_layout, tile),
                        ),
                    )
                    for tile in payload_tiles[1:]
                ),
                default=0,
            )
            broadcast_phase = max(
                (
                    model.l1_to_l1(
                        root_tile,
                        tile,
                        tensor_slice_num_bytes(
                            output_tensor,
                            tile_tensor_slice(output_tensor, output_layout, tile),
                        ),
                    )
                    for tile in payload_tiles[1:]
                ),
                default=0,
            )
            group_costs.append(reduce_phase + broadcast_phase)
        return max(group_costs, default=0)


def _logical_collective_groups(
    layout: TensorLayout,
    collective_axis: str,
) -> tuple[tuple[Tile, ...], ...]:
    logical_width = layout.effective_logical_width
    logical_height = layout.effective_logical_height
    groups: dict[int, list[Tile]] = {}
    for tile_ordinal, tile in enumerate(layout.submesh.tiles):
        logical_x = tile_ordinal % logical_width
        logical_y = tile_ordinal // logical_width
        group_key = logical_y if collective_axis == "x" else logical_x
        groups.setdefault(group_key, []).append(tile)

    return tuple(
        tuple(group_tiles)
        for _, group_tiles in sorted(groups.items())
    )
