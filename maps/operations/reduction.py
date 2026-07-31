"""Reduction semantics, decomposition, Tile Work, and costing."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from maps.hardware import Device, Tile, WorkKind
from maps.graph import Node, OpKind, Tensor
from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from MAPS.core.submesh import Submesh
from .contracts import (
    CompositeOpPayload,
    LayoutRelation,
    OpCostModel,
    OpPayload,
    TileWork,
    require_tile_device,
    sharded_layout,
)
from .collective import AllReducePayload
from .elementwise import ElementwiseCostModel, ElementwiseTileWork


REDUCTION_WORK_KINDS: dict[str, WorkKind] = {
    "ReduceMax": WorkKind.REDUCE_MAX,
    "ReduceSum": WorkKind.REDUCE_SUM,
}


@dataclass(frozen=True)
class ReductionTileWork(TileWork):
    """Concrete reduction slices associated with one tile."""

    work_kind: WorkKind
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

    def operation_count(self) -> int:
        return self.input_slice.num_elements


@dataclass(frozen=True)
class ReductionPayload(OpPayload):
    """Configured tile-local reduction operation."""

    op_name: str
    x: Tensor
    output: Tensor
    axis: int
    work_kind: WorkKind

    def __post_init__(self) -> None:
        expected = REDUCTION_WORK_KINDS.get(self.op_name)
        if expected is None:
            raise ValueError(f"unsupported reduction operation: {self.op_name}")
        if self.work_kind is not expected:
            raise ValueError(
                f"{self.op_name} must use work kind {expected.name}, "
                f"got {self.work_kind.name}"
            )
        if self.axis < 0 or self.axis >= self.x.rank:
            raise ValueError("ReductionPayload axis must be in input tensor rank")
        self.validate_shapes()

    @property
    def cost_model(self) -> OpCostModel:
        return ReductionCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        input_layout = sharded_layout(self.x, submesh, logical_shape)
        mesh_x = input_layout.mesh_x
        mesh_y = input_layout.mesh_y
        if mesh_x.tensor_axis == self.axis:
            mesh_x = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        if mesh_y.tensor_axis == self.axis:
            mesh_y = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        return (
            TensorLayout(
                submesh=submesh,
                mesh_x=mesh_x,
                mesh_y=mesh_y,
                logical_width=input_layout.logical_width,
                logical_height=input_layout.logical_height,
            ),
        )

    def _input_layout_from_output_layout(self, output_layout: TensorLayout) -> TensorLayout:
        return sharded_layout(
            self.x,
            output_layout.submesh,
            (
                output_layout.effective_logical_width,
                output_layout.effective_logical_height,
            ),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ReductionTileWork:
        output_layout = self.single_output_layout(output_layouts)
        input_layout = self._input_layout_from_output_layout(output_layout)
        return ReductionTileWork(
            work_kind=self.work_kind,
            x=self.x,
            output=self.output,
            input_slice=tile_tensor_slice(self.x, input_layout, tile),
            output_slice=tile_tensor_slice(self.output, output_layout, tile),
        )

    def validate_shapes(self) -> None:
        if self.x.rank != self.output.rank:
            raise ValueError(f"{self.op_name} input and output ranks must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError(f"{self.op_name} input and output element sizes must match")
        for axis, (input_dim, output_dim) in enumerate(zip(self.x.dims, self.output.dims)):
            expected_output_dim = 1 if axis == self.axis else input_dim
            if output_dim != expected_output_dim:
                raise ValueError(
                    f"{self.op_name} output dim {axis} must be "
                    f"{expected_output_dim}, got {output_dim}"
                )


@dataclass(frozen=True)
class ScalarMultiplyPayload(OpPayload):
    """Elementwise multiplication by a compile-time scalar."""

    x: Tensor
    output: Tensor
    factor: float
    work_kind: WorkKind = WorkKind.MUL

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.MUL:
            raise ValueError("ScalarMultiply must use MUL work")
        if self.x.dims != self.output.dims:
            raise ValueError("ScalarMultiply input and output shapes must match")
        if self.x.elem_bytes != self.output.elem_bytes or self.x.dtype != self.output.dtype:
            raise ValueError(
                "ScalarMultiply input and output element representations must match"
            )

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return (
            LayoutRelation.exact(input_index=0, output_index=0, tensor=self.x),
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

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ElementwiseTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        return ElementwiseTileWork(
            work_kind=self.work_kind,
            output=self.output,
            output_slice=output_slice,
            inputs=(self.x,),
            input_tile_slices=(output_slice,),
        )


@dataclass(frozen=True)
class ReduceSumPayload(CompositeOpPayload):
    """Static single-axis ReduceSum with retained dimensions."""

    x: Tensor
    output: Tensor
    axis: int

    def __post_init__(self) -> None:
        ReductionPayload(
            op_name="ReduceSum",
            x=self.x,
            output=self.output,
            axis=self.axis,
            work_kind=WorkKind.REDUCE_SUM,
        )

    def decompose(self, node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
        collective_axis = _collective_axis(self.x, self.axis)
        if collective_axis is None:
            return (), (
                _reduction_node(
                    name=node.name,
                    x=self.x,
                    output=self.output,
                    axis=self.axis,
                    attributes=node.attributes,
                ),
            )

        local = _tensor_like(f"{node.name}__local", self.output)
        stage_group_id = f"{node.name}::reduce_sum"
        attributes = {**node.attributes, "stage_group_id": stage_group_id}
        return (local,), (
            _reduction_node(
                name=f"{node.name}__local",
                x=self.x,
                output=local,
                axis=self.axis,
                attributes={**attributes, "reduce_sum_step": "local"},
            ),
            Node(
                name=f"{node.name}__allreduce",
                kind=OpKind.CUSTOM,
                inputs=(local,),
                outputs=(self.output,),
                payload=AllReducePayload(
                    op_name="AllReduceSum",
                    x=local,
                    output=self.output,
                    reduction="sum",
                    collective_axis=collective_axis,
                ),
                attributes={**attributes, "reduce_sum_step": "allreduce"},
            ),
        )


@dataclass(frozen=True)
class GlobalAveragePoolPayload(CompositeOpPayload):
    """Global average pooling over all NCHW spatial dimensions."""

    x: Tensor
    output: Tensor

    def __post_init__(self) -> None:
        if self.x.rank != 4 or self.output.rank != 4:
            raise ValueError("GlobalAveragePool requires rank-four NCHW tensors")
        if self.output.dims != self.x.dims[:2] + (1, 1):
            raise ValueError("GlobalAveragePool output shape must be N,C,1,1")
        if self.x.elem_bytes != self.output.elem_bytes or self.x.dtype != self.output.dtype:
            raise ValueError(
                "GlobalAveragePool input and output element representations must match"
            )

    def decompose(self, node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
        width_sum_local = _reduced_tensor(f"{node.name}__width_sum_local", self.x, 3)
        width_sum = _tensor_like(f"{node.name}__width_sum", width_sum_local)
        spatial_sum_local = _reduced_tensor(
            f"{node.name}__spatial_sum_local",
            width_sum,
            2,
        )
        spatial_sum = _tensor_like(f"{node.name}__spatial_sum", spatial_sum_local)
        stage_group_id = f"{node.name}::global_average_pool"
        attributes = {**node.attributes, "stage_group_id": stage_group_id}
        tensors = (width_sum_local, width_sum, spatial_sum_local, spatial_sum)
        nodes = (
            _reduction_node(
                f"{node.name}__reduce_width",
                self.x,
                width_sum_local,
                3,
                {**attributes, "global_average_pool_step": "reduce_width"},
            ),
            _allreduce_node(
                f"{node.name}__allreduce_width",
                width_sum_local,
                width_sum,
                "x",
                {**attributes, "global_average_pool_step": "allreduce_width"},
            ),
            _reduction_node(
                f"{node.name}__reduce_height",
                width_sum,
                spatial_sum_local,
                2,
                {**attributes, "global_average_pool_step": "reduce_height"},
            ),
            _allreduce_node(
                f"{node.name}__allreduce_height",
                spatial_sum_local,
                spatial_sum,
                "y",
                {**attributes, "global_average_pool_step": "allreduce_height"},
            ),
            Node(
                name=f"{node.name}__scale",
                kind=OpKind.ELEMENTWISE,
                inputs=(spatial_sum,),
                outputs=(self.output,),
                payload=ScalarMultiplyPayload(
                    x=spatial_sum,
                    output=self.output,
                    factor=1.0 / prod(self.x.dims[2:]),
                ),
                attributes={**attributes, "global_average_pool_step": "scale"},
            ),
        )
        return tensors, nodes


def _tensor_like(name: str, reference: Tensor) -> Tensor:
    return Tensor(
        name=name,
        rank=reference.rank,
        dims=reference.dims,
        elem_bytes=reference.elem_bytes,
        dtype=reference.dtype,
    )


def _reduced_tensor(name: str, reference: Tensor, axis: int) -> Tensor:
    dims = list(reference.dims)
    dims[axis] = 1
    return Tensor(
        name=name,
        rank=reference.rank,
        dims=tuple(dims),
        elem_bytes=reference.elem_bytes,
        dtype=reference.dtype,
    )


def _collective_axis(tensor: Tensor, axis: int) -> str | None:
    if axis == tensor.rank - 1:
        return "x"
    if tensor.rank >= 2 and axis == tensor.rank - 2:
        return "y"
    return None


def _reduction_node(
    name: str,
    x: Tensor,
    output: Tensor,
    axis: int,
    attributes: dict[str, object],
) -> Node:
    return Node(
        name=name,
        kind=OpKind.REDUCTION,
        inputs=(x,),
        outputs=(output,),
        payload=ReductionPayload(
            op_name="ReduceSum",
            x=x,
            output=output,
            axis=axis,
            work_kind=WorkKind.REDUCE_SUM,
        ),
        attributes=attributes,
    )


def _allreduce_node(
    name: str,
    x: Tensor,
    output: Tensor,
    collective_axis: str,
    attributes: dict[str, object],
) -> Node:
    return Node(
        name=name,
        kind=OpKind.CUSTOM,
        inputs=(x,),
        outputs=(output,),
        payload=AllReducePayload(
            op_name="AllReduceSum",
            x=x,
            output=output,
            reduction="sum",
            collective_axis=collective_axis,
        ),
        attributes=attributes,
    )


@dataclass(frozen=True)
class ReductionCostModel(OpCostModel):
    """Tile-local reduction cycle model backed by tile devices."""

    work_kind: WorkKind

    def __post_init__(self) -> None:
        if self.work_kind not in (WorkKind.REDUCE_SUM, WorkKind.REDUCE_MAX):
            raise ValueError("ReductionCostModel work_kind must be REDUCE_SUM or REDUCE_MAX")

    def cost(
        self,
        tile_work: TileWork,
        tile: Tile,
        assigned_device: Device,
    ) -> int:
        return require_tile_device(tile, assigned_device).cycles(tile_work)
