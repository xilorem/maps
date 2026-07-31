"""Collective communication op payloads."""

from __future__ import annotations

from dataclasses import dataclass
from MAPS.arch import WorkKind
from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.payload import OpPayload, sharded_layout
from MAPS.ops.common.layout_relation import LayoutRelation
from MAPS.ops.common.tile_work import TileWork
from MAPS.ops.common.cost import OpCostModel


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
        from MAPS.ops.costs.collective_cost import AllReduceCostModel

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
