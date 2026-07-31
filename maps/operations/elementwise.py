"""Elementwise semantics, layouts, Tile Work, and Device costing."""

from __future__ import annotations

from dataclasses import dataclass

from maps.planning.layouts import TensorLayout, TensorSlice, TensorSliceRef, tile_tensor_slice
from maps.planning.submesh import Submesh
from maps.graph import Tensor
from maps.hardware import Device, Tile, WorkKind

from .broadcasting import broadcast_input_slice, validate_broadcast_output
from .contracts import (
    LayoutRelation,
    OpCostModel,
    OpPayload,
    TileWork,
    require_tile_device,
    sharded_layout,
)

UNARY_ELEMENTWISE_OPS: dict[str, WorkKind] = {
    "abs": WorkKind.ABS,
    "exp": WorkKind.EXP,
    "log": WorkKind.LOG,
    "neg": WorkKind.NEG,
    "relu": WorkKind.RELU,
    "sigmoid": WorkKind.SIGMOID,
    "sqrt": WorkKind.SQRT,
}

BINARY_ELEMENTWISE_OPS: dict[str, WorkKind] = {
    "add": WorkKind.ADD,
    "div": WorkKind.DIV,
    "mul": WorkKind.MUL,
    "pow": WorkKind.POW,
    "sub": WorkKind.SUB,
}


@dataclass(frozen=True)
class ElementwiseTileWork(TileWork):
    """Concrete elementwise slices associated with one tile."""

    work_kind: WorkKind
    output: Tensor
    output_slice: TensorSlice
    inputs: tuple[Tensor, ...]
    input_tile_slices: tuple[TensorSlice, ...]

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return tuple(
            TensorSliceRef(tensor=tensor, tensor_slice=tensor_slice)
            for tensor, tensor_slice in zip(self.inputs, self.input_tile_slices)
        )

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)

    def operation_count(self) -> int:
        return self.output_slice.num_elements


@dataclass(frozen=True)
class ElementwiseCostModel(OpCostModel):
    """Elementwise cycles evaluated by the assigned Device."""

    work_kind: WorkKind

    def cost(
        self,
        tile_work: TileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        if getattr(tile_work, "work_kind", None) is not self.work_kind:
            raise ValueError(
                f"elementwise cost expects {self.work_kind.name} work, got "
                f"{getattr(tile_work, 'work_kind', None)}"
            )
        return require_tile_device(tile, assigned_device).cycles(tile_work)


@dataclass(frozen=True)
class UnaryElementwisePayload(OpPayload):
    """One configured unary elementwise Operation."""

    op_name: str
    x: Tensor
    output: Tensor
    work_kind: WorkKind = WorkKind.ELEMENTWISE

    def __post_init__(self) -> None:
        operation_name = self.op_name.lower()
        expected = UNARY_ELEMENTWISE_OPS.get(operation_name)
        if expected is None:
            raise ValueError(f"unsupported unary elementwise operation: {self.op_name}")
        if self.work_kind not in (WorkKind.ELEMENTWISE, expected):
            raise ValueError(
                f"{self.op_name} must use work kind {expected.name}, "
                f"got {self.work_kind.name}"
            )
        object.__setattr__(self, "op_name", operation_name)
        object.__setattr__(self, "work_kind", expected)
        if self.x.rank != self.output.rank or self.x.dims != self.output.dims:
            raise ValueError(f"{self.op_name} input and output shapes must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError(f"{self.op_name} input and output element sizes must match")

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return (LayoutRelation.exact(input_index=0, output_index=0, tensor=self.x),)

    @property
    def cost_model(self) -> OpCostModel:
        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (sharded_layout(self.output, submesh, logical_shape),)

    def required_input_slices(self, output_slice: TensorSlice) -> tuple[TensorSlice, ...]:
        if output_slice.rank != self.x.rank:
            raise ValueError(f"{self.op_name} output slice rank must match input rank")
        return (output_slice,)

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ElementwiseTileWork:
        output_slice = tile_tensor_slice(
            self.output,
            self.single_output_layout(output_layouts),
            tile,
        )
        return ElementwiseTileWork(
            work_kind=self.work_kind,
            output=self.output,
            output_slice=output_slice,
            inputs=(self.x,),
            input_tile_slices=self.required_input_slices(output_slice),
        )


@dataclass(frozen=True)
class BinaryElementwisePayload(OpPayload):
    """One configured binary elementwise Operation with broadcasting."""

    op_name: str
    lhs: Tensor
    rhs: Tensor
    output: Tensor
    work_kind: WorkKind = WorkKind.ELEMENTWISE

    def __post_init__(self) -> None:
        operation_name = self.op_name.lower()
        expected = BINARY_ELEMENTWISE_OPS.get(operation_name)
        if expected is None:
            raise ValueError(f"unsupported binary elementwise operation: {self.op_name}")
        if self.work_kind not in (WorkKind.ELEMENTWISE, expected):
            raise ValueError(
                f"{self.op_name} must use work kind {expected.name}, "
                f"got {self.work_kind.name}"
            )
        object.__setattr__(self, "op_name", operation_name)
        object.__setattr__(self, "work_kind", expected)
        validate_broadcast_output((self.lhs, self.rhs), self.output, self.op_name)
        if (
            self.lhs.elem_bytes != self.output.elem_bytes
            or self.rhs.elem_bytes != self.output.elem_bytes
        ):
            raise ValueError(f"{self.op_name} input and output element sizes must match")

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return tuple(
            LayoutRelation.exact(
                input_index=input_index,
                output_index=0,
                tensor=tensor,
            )
            for input_index, tensor in enumerate((self.lhs, self.rhs))
            if tensor.dims == self.output.dims
        )

    @property
    def cost_model(self) -> OpCostModel:
        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (sharded_layout(self.output, submesh, logical_shape),)

    def required_input_slices(self, output_slice: TensorSlice) -> tuple[TensorSlice, ...]:
        return (
            broadcast_input_slice(self.lhs, self.output, output_slice, self.op_name),
            broadcast_input_slice(self.rhs, self.output, output_slice, self.op_name),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ElementwiseTileWork:
        output_slice = tile_tensor_slice(
            self.output,
            self.single_output_layout(output_layouts),
            tile,
        )
        return ElementwiseTileWork(
            work_kind=self.work_kind,
            output=self.output,
            output_slice=output_slice,
            inputs=(self.lhs, self.rhs),
            input_tile_slices=self.required_input_slices(output_slice),
        )


__all__ = [
    "BINARY_ELEMENTWISE_OPS",
    "UNARY_ELEMENTWISE_OPS",
    "BinaryElementwisePayload",
    "ElementwiseCostModel",
    "ElementwiseTileWork",
    "UnaryElementwisePayload",
]
