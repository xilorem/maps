"""Group-normalization semantics, decomposition, Tile Work, and costing."""

from __future__ import annotations

from dataclasses import dataclass

from maps.hardware import Tile, WorkKind
from maps.graph import Node, OpKind, Tensor
from maps.planning.mapping import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
    TensorSliceRef,
    partial_axis_to_shard,
    tile_tensor_slice,
)
from maps.planning.mapping import Submesh
from .contracts import (
    CompositeOpPayload,
    LayoutRelation,
    OpCostModel,
    OpPayload,
    TileWork,
    sharded_layout,
)
from .collective import AllReducePayload
from .elementwise import BinaryElementwisePayload, ElementwiseCostModel


@dataclass(frozen=True)
class GroupReduceTileWork(TileWork):
    x: Tensor
    output: Tensor
    input_slice: TensorSlice
    output_slice: TensorSlice
    work_kind: WorkKind = WorkKind.GROUP_REDUCE

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.x, self.input_slice),)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.output, self.output_slice),)

    def operation_count(self) -> int:
        return self.input_slice.num_elements


@dataclass(frozen=True)
class GroupReducePayload(OpPayload):
    """Compute per-instance, per-group partial sums over owned values."""

    x: Tensor
    output: Tensor
    num_groups: int
    work_kind: WorkKind = WorkKind.GROUP_REDUCE

    def __post_init__(self) -> None:
        expected = (
            self.x.dims[0],
            self.num_groups,
            *(1 for _ in self.x.dims[2:]),
        )
        if self.num_groups <= 0 or self.x.dims[1] % self.num_groups:
            raise ValueError("GroupReduce groups must divide input channels")
        if self.output.dims != expected:
            raise ValueError("GroupReduce output must contain one value per instance and group")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("GroupReduce tensors must agree on element size")

    @property
    def cost_model(self) -> OpCostModel:
        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        input_layout = sharded_layout(self.x, submesh, logical_shape)

        def partial(axis: LayoutAxis) -> LayoutAxis:
            if axis.mode is LayoutAxisMode.SHARD:
                return LayoutAxis(LayoutAxisMode.PARTIAL, tensor_axis=axis.tensor_axis)
            return axis

        return (
            TensorLayout(
                submesh=submesh,
                mesh_x=partial(input_layout.mesh_x),
                mesh_y=partial(input_layout.mesh_y),
                logical_width=input_layout.logical_width,
                logical_height=input_layout.logical_height,
            ),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> GroupReduceTileWork:
        output_layout = self.single_output_layout(output_layouts)
        input_layout = TensorLayout(
            submesh=output_layout.submesh,
            mesh_x=partial_axis_to_shard(output_layout.mesh_x),
            mesh_y=partial_axis_to_shard(output_layout.mesh_y),
            logical_width=output_layout.logical_width,
            logical_height=output_layout.logical_height,
        )
        return GroupReduceTileWork(
            x=self.x,
            output=self.output,
            input_slice=tile_tensor_slice(self.x, input_layout, tile),
            output_slice=tile_tensor_slice(self.output, output_layout, tile),
        )


@dataclass(frozen=True)
class GroupNormalizeTileWork(TileWork):
    x: Tensor
    sum_value: Tensor
    sumsq_value: Tensor
    scale: Tensor
    bias: Tensor
    output: Tensor
    input_tile_slices: tuple[TensorSlice, ...]
    output_slice: TensorSlice
    work_kind: WorkKind = WorkKind.GROUP_NORMALIZE

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return tuple(
            TensorSliceRef(tensor, tensor_slice)
            for tensor, tensor_slice in zip(
                (self.x, self.sum_value, self.sumsq_value, self.scale, self.bias),
                self.input_tile_slices,
            )
        )

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.output, self.output_slice),)

    def operation_count(self) -> int:
        return self.output_slice.num_elements


@dataclass(frozen=True)
class GroupNormalizeFromMomentsPayload(OpPayload):
    x: Tensor
    sum_value: Tensor
    sumsq_value: Tensor
    scale: Tensor
    bias: Tensor
    output: Tensor
    num_groups: int
    epsilon: float
    work_kind: WorkKind = WorkKind.GROUP_NORMALIZE

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("GroupNormalize epsilon must be > 0")
        if self.x.dims != self.output.dims:
            raise ValueError("GroupNormalize input and output shapes must match")
        channels = self.x.dims[1]
        if channels % self.num_groups:
            raise ValueError("GroupNormalize groups must divide channels")
        if self.scale.dims != (channels,) or self.bias.dims != (channels,):
            raise ValueError("GroupNormalize scale and bias must match channels")
        expected_stats = (self.x.dims[0], self.num_groups, *(1 for _ in self.x.dims[2:]))
        if self.sum_value.dims != expected_stats or self.sumsq_value.dims != expected_stats:
            raise ValueError("GroupNormalize moment shapes do not match groups")
        if any(
            tensor.elem_bytes != self.x.elem_bytes
            for tensor in (
                self.sum_value,
                self.sumsq_value,
                self.scale,
                self.bias,
                self.output,
            )
        ):
            raise ValueError("GroupNormalize tensors must agree on element size")

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return (
            LayoutRelation.exact(input_index=0, output_index=0, tensor=self.x),
        )

    @property
    def element_count_per_group(self) -> int:
        count = self.x.dims[1] // self.num_groups
        for dimension in self.x.dims[2:]:
            count *= dimension
        return count

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
    ) -> GroupNormalizeTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        stats_slice = TensorSlice(
            rank=self.sum_value.rank,
            dims=tuple(TensorRange(0, dimension) for dimension in self.sum_value.dims),
        )
        channel_slice = output_slice.dims[1]
        return GroupNormalizeTileWork(
            x=self.x,
            sum_value=self.sum_value,
            sumsq_value=self.sumsq_value,
            scale=self.scale,
            bias=self.bias,
            output=self.output,
            input_tile_slices=(
                output_slice,
                stats_slice,
                stats_slice,
                TensorSlice(1, (channel_slice,)),
                TensorSlice(1, (channel_slice,)),
            ),
            output_slice=output_slice,
        )


@dataclass(frozen=True)
class GroupNormalizationPayload(CompositeOpPayload):
    x: Tensor
    scale: Tensor
    bias: Tensor
    output: Tensor
    num_groups: int
    epsilon: float = 1e-5
    stash_type: int = 1

    def __post_init__(self) -> None:
        if self.x.rank < 2:
            raise ValueError("GroupNormalization input rank must be at least 2")
        if self.num_groups <= 0 or self.x.dims[1] % self.num_groups:
            raise ValueError("GroupNormalization groups must divide channels")
        if self.x.dims != self.output.dims:
            raise ValueError("GroupNormalization input and output shapes must match")
        if self.scale.dims != (self.x.dims[1],) or self.bias.dims != (self.x.dims[1],):
            raise ValueError("GroupNormalization scale and bias must match channels")
        if any(
            tensor.elem_bytes != self.x.elem_bytes
            for tensor in (self.scale, self.bias, self.output)
        ):
            raise ValueError("GroupNormalization tensors must agree on element size")
        if self.epsilon <= 0:
            raise ValueError("GroupNormalization epsilon must be > 0")
        if self.stash_type != 1:
            raise NotImplementedError("GroupNormalization only supports FLOAT stash_type")

    def decompose(self, node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
        return decompose_group_normalization_node(node)


def _stats_tensor(name: str, op: GroupNormalizationPayload) -> Tensor:
    return Tensor(
        name=name,
        rank=op.x.rank,
        dims=(op.x.dims[0], op.num_groups, *(1 for _ in op.x.dims[2:])),
        elem_bytes=op.x.elem_bytes,
        dtype=op.x.dtype,
    )


def _same_shape_tensor(name: str, reference: Tensor) -> Tensor:
    return Tensor(
        name,
        reference.rank,
        reference.dims,
        reference.elem_bytes,
        dtype=reference.dtype,
    )


def decompose_group_normalization_node(
    node: Node,
) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
    if not isinstance(node.payload, GroupNormalizationPayload):
        raise TypeError("expected GroupNormalizationPayload")
    op = node.payload
    attributes = dict(node.attributes)

    squared = _same_shape_tensor(f"{node.name}__squared", op.x)
    sum_local = _stats_tensor(f"{node.name}__sum_local", op)
    sumsq_local = _stats_tensor(f"{node.name}__sumsq_local", op)
    sum_global = _same_shape_tensor(f"{node.name}__sum_global", sum_local)
    sumsq_global = _same_shape_tensor(f"{node.name}__sumsq_global", sumsq_local)

    def attrs(step: str) -> dict[str, object]:
        return {**attributes, "group_norm_step": step}

    nodes = (
        Node(
            f"{node.name}__square",
            OpKind.ELEMENTWISE,
            (op.x, op.x),
            (squared,),
            BinaryElementwisePayload("Mul", op.x, op.x, squared, WorkKind.MUL),
            attrs("square"),
        ),
        Node(
            f"{node.name}__reduce_sum",
            OpKind.REDUCTION,
            (op.x,),
            (sum_local,),
            GroupReducePayload(op.x, sum_local, op.num_groups),
            attrs("reduce_sum"),
        ),
        Node(
            f"{node.name}__reduce_sumsq",
            OpKind.REDUCTION,
            (squared,),
            (sumsq_local,),
            GroupReducePayload(squared, sumsq_local, op.num_groups),
            attrs("reduce_sumsq"),
        ),
        Node(
            f"{node.name}__allreduce_sum",
            OpKind.CUSTOM,
            (sum_local,),
            (sum_global,),
            AllReducePayload("AllReduceSum", sum_local, sum_global, "sum"),
            attrs("allreduce_sum"),
        ),
        Node(
            f"{node.name}__allreduce_sumsq",
            OpKind.CUSTOM,
            (sumsq_local,),
            (sumsq_global,),
            AllReducePayload("AllReduceSum", sumsq_local, sumsq_global, "sum"),
            attrs("allreduce_sumsq"),
        ),
        Node(
            f"{node.name}__normalize",
            OpKind.ELEMENTWISE,
            (op.x, sum_global, sumsq_global, op.scale, op.bias),
            (op.output,),
            GroupNormalizeFromMomentsPayload(
                op.x,
                sum_global,
                sumsq_global,
                op.scale,
                op.bias,
                op.output,
                op.num_groups,
                op.epsilon,
            ),
            attrs("normalize"),
        ),
    )
    return (
        (squared, sum_local, sumsq_local, sum_global, sumsq_global),
        nodes,
    )
