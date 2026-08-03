"""Collective operation semantics, Tile Work, and costing."""

from __future__ import annotations

from dataclasses import dataclass

from maps.hardware import Device, Tile, WorkKind
from maps.planning.mapping import (
    TensorLayout,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from maps.planning.mapping import Submesh
from maps.graph import Tensor
from .contracts import OpCostModel, OpPayload, TileWork, sharded_layout
from .contracts import LayoutRelation


ALL_REDUCE_WORK_KINDS = {
    "sum": WorkKind.ALL_REDUCE_SUM,
    "max": WorkKind.ALL_REDUCE_MAX,
}


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
    @property
    def work_kind(self) -> WorkKind:
        return ALL_REDUCE_WORK_KINDS[self.reduction]

    def __post_init__(self) -> None:
        if self.reduction not in {"sum", "max"}:
            raise ValueError("AllReducePayload reduction must be 'sum' or 'max'")
        self.validate_shapes()

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return (
            LayoutRelation(
                input_index=0,
                output_index=0,
                input_axis_for_output_axis=tuple(range(self.x.rank)),
                guarantees_slice_containment=True,
                resolves_partial_values=True,
            ),
        )

    @property
    def cost_model(self) -> OpCostModel:
        return AllReduceCostModel(reduction=self.reduction)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (sharded_layout(self.output, submesh, logical_shape),)

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
    """Generic collective cost pending Device-owned latency modeling."""

    reduction: str

    def __post_init__(self) -> None:
        if self.reduction not in {"sum", "max"}:
            raise ValueError("AllReduceCostModel reduction must be 'sum' or 'max'")

    def cost(
        self,
        tile_work: object,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        del assigned_device
        del tile_work, tile
        return 0
