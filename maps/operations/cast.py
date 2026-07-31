"""Cast semantics, layout, Tile Work, and Device costing."""

from __future__ import annotations

from dataclasses import dataclass

from maps.planning.layouts import TensorLayout, TensorSlice, TensorSliceRef, tile_tensor_slice
from maps.planning.submesh import Submesh
from maps.graph import Tensor
from maps.hardware import Device, Tile, WorkKind

from .contracts import (
    LayoutRelation,
    OpCostModel,
    OpPayload,
    TileWork,
    require_tile_device,
    sharded_layout,
)


@dataclass(frozen=True)
class CastTileWork(TileWork):
    """Concrete Cast slices associated with one tile."""

    x: Tensor
    output: Tensor
    tensor_slice: TensorSlice

    @property
    def work_kind(self) -> WorkKind:
        return WorkKind.CAST

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.x, tensor_slice=self.tensor_slice),)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.tensor_slice),)

    def operation_count(self) -> int:
        return self.tensor_slice.num_elements


@dataclass(frozen=True)
class CastCostModel(OpCostModel):
    """Cast cycles evaluated by the Device assigned to one tile."""

    def cost(
        self,
        tile_work: TileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        return require_tile_device(tile, assigned_device).cycles(tile_work)


@dataclass(frozen=True)
class CastPayload(OpPayload):
    """Convert one Tensor dtype while preserving its logical shape."""

    x: Tensor
    output: Tensor
    work_kind: WorkKind = WorkKind.CAST

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.CAST:
            raise ValueError("Cast must use CAST work")
        if self.x.dims != self.output.dims:
            raise ValueError("Cast input and output shapes must match")

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return (LayoutRelation.exact(input_index=0, output_index=0, tensor=self.x),)

    @property
    def cost_model(self) -> OpCostModel:
        return CastCostModel()

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (sharded_layout(self.output, submesh, logical_shape),)

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> CastTileWork:
        tensor_slice = tile_tensor_slice(
            self.output,
            self.single_output_layout(output_layouts),
            tile,
        )
        return CastTileWork(x=self.x, output=self.output, tensor_slice=tensor_slice)


__all__ = ["CastCostModel", "CastPayload", "CastTileWork"]
