"""Group-normalization semantics, decomposition, Tile Work, and costing."""

from __future__ import annotations

from dataclasses import dataclass, field

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
from .elementwise import ElementwiseCostModel


def _group_stats_dims(x: Tensor, num_groups: int) -> tuple[int, ...]:
    return (x.dims[0], num_groups, *(1 for _ in x.dims[2:]))


def _group_element_count(x: Tensor, num_groups: int) -> int:
    count = x.dims[1] // num_groups
    for dimension in x.dims[2:]:
        count *= dimension
    return count


def _partial_group_layout(
    x: Tensor,
    submesh: Submesh,
    logical_shape: tuple[int, int] | None,
) -> TensorLayout:
    input_layout = sharded_layout(x, submesh, logical_shape)

    def partial(axis: LayoutAxis) -> LayoutAxis:
        if axis.mode is LayoutAxisMode.SHARD:
            return LayoutAxis(LayoutAxisMode.PARTIAL, tensor_axis=axis.tensor_axis)
        return axis

    return TensorLayout(
        submesh=submesh,
        mesh_x=partial(input_layout.mesh_x),
        mesh_y=partial(input_layout.mesh_y),
        logical_width=input_layout.logical_width,
        logical_height=input_layout.logical_height,
    )


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
    """Compute stable per-group partial sums, scaled by the global group size."""

    x: Tensor
    output: Tensor
    num_groups: int
    work_kind: WorkKind = WorkKind.GROUP_REDUCE
    element_count_per_group: int = field(init=False)
    channel_count_per_group: int = field(init=False)

    def __post_init__(self) -> None:
        expected = _group_stats_dims(self.x, self.num_groups)
        if self.num_groups <= 0 or self.x.dims[1] % self.num_groups:
            raise ValueError("GroupReduce groups must divide input channels")
        if self.output.dims != expected:
            raise ValueError("GroupReduce output must contain one value per instance and group")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("GroupReduce tensors must agree on element size")
        object.__setattr__(
            self, "element_count_per_group", _group_element_count(self.x, self.num_groups)
        )
        object.__setattr__(
            self, "channel_count_per_group", self.x.dims[1] // self.num_groups
        )

    @property
    def cost_model(self) -> OpCostModel:
        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (_partial_group_layout(self.x, submesh, logical_shape),)

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
class GroupCenteredReducePayload(OpPayload):
    """Compute partial ``(x - mean)^2 / group_size`` values for each group."""

    x: Tensor
    mean: Tensor
    output: Tensor
    num_groups: int
    work_kind: WorkKind = WorkKind.GROUP_CENTERED_REDUCE
    element_count_per_group: int = field(init=False)
    channel_count_per_group: int = field(init=False)

    def __post_init__(self) -> None:
        expected = _group_stats_dims(self.x, self.num_groups)
        if self.num_groups <= 0 or self.x.dims[1] % self.num_groups:
            raise ValueError("GroupCenteredReduce groups must divide input channels")
        if self.mean.dims != expected or self.output.dims != expected:
            raise ValueError("GroupCenteredReduce statistics must match groups")
        if any(
            tensor.elem_bytes != self.x.elem_bytes
            for tensor in (self.mean, self.output)
        ):
            raise ValueError("GroupCenteredReduce tensors must agree on element size")
        object.__setattr__(
            self, "element_count_per_group", _group_element_count(self.x, self.num_groups)
        )
        object.__setattr__(
            self, "channel_count_per_group", self.x.dims[1] // self.num_groups
        )

    @property
    def cost_model(self) -> OpCostModel:
        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (_partial_group_layout(self.x, submesh, logical_shape),)

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
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        input_slice = tile_tensor_slice(self.x, input_layout, tile)
        return GroupCenteredReduceTileWork(
            x=self.x,
            mean=self.mean,
            output=self.output,
            input_slice=input_slice,
            stats_slice=output_slice,
            output_slice=output_slice,
        )


@dataclass(frozen=True)
class GroupCenteredReduceTileWork(TileWork):
    x: Tensor
    mean: Tensor
    output: Tensor
    input_slice: TensorSlice
    stats_slice: TensorSlice
    output_slice: TensorSlice
    work_kind: WorkKind = WorkKind.GROUP_CENTERED_REDUCE

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (
            TensorSliceRef(self.x, self.input_slice),
            TensorSliceRef(self.mean, self.stats_slice),
        )

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.output, self.output_slice),)

    def operation_count(self) -> int:
        return self.input_slice.num_elements


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
class GroupNormalizeFromStatsPayload(OpPayload):
    x: Tensor
    mean: Tensor
    variance: Tensor
    scale: Tensor
    bias: Tensor
    output: Tensor
    num_groups: int
    epsilon: float
    work_kind: WorkKind = WorkKind.GROUP_NORMALIZE
    element_count_per_group: int = field(init=False)
    channel_count_per_group: int = field(init=False)

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
        expected_stats = _group_stats_dims(self.x, self.num_groups)
        if self.mean.dims != expected_stats or self.variance.dims != expected_stats:
            raise ValueError("GroupNormalize statistic shapes do not match groups")
        if any(
            tensor.elem_bytes != self.x.elem_bytes
            for tensor in (
                self.mean,
                self.variance,
                self.scale,
                self.bias,
                self.output,
            )
        ):
            raise ValueError("GroupNormalize tensors must agree on element size")
        object.__setattr__(
            self, "element_count_per_group", _group_element_count(self.x, self.num_groups)
        )
        object.__setattr__(
            self, "channel_count_per_group", self.x.dims[1] // self.num_groups
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
    ) -> GroupNormalizeTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        stats_slice = TensorSlice(
            rank=self.mean.rank,
            dims=tuple(TensorRange(0, dimension) for dimension in self.mean.dims),
        )
        channel_slice = output_slice.dims[1]
        return GroupNormalizeTileWork(
            x=self.x,
            sum_value=self.mean,
            sumsq_value=self.variance,
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
        dims=_group_stats_dims(op.x, op.num_groups),
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

    sum_local = _stats_tensor(f"{node.name}__sum_local", op)
    mean = _same_shape_tensor(f"{node.name}__mean", sum_local)
    variance_local = _stats_tensor(f"{node.name}__variance_local", op)
    variance = _same_shape_tensor(f"{node.name}__variance", variance_local)

    def attrs(step: str) -> dict[str, object]:
        return {**attributes, "group_norm_step": step}

    nodes = (
        Node(
            f"{node.name}__scaled_sum",
            OpKind.REDUCTION,
            (op.x,),
            (sum_local,),
            GroupReducePayload(op.x, sum_local, op.num_groups),
            attrs("scaled_sum"),
        ),
        Node(
            f"{node.name}__allreduce_sum",
            OpKind.CUSTOM,
            (sum_local,),
            (mean,),
            AllReducePayload("AllReduceSum", sum_local, mean, "sum"),
            attrs("allreduce_sum"),
        ),
        Node(
            f"{node.name}__scaled_centered_sumsq",
            OpKind.REDUCTION,
            (op.x, mean),
            (variance_local,),
            GroupCenteredReducePayload(op.x, mean, variance_local, op.num_groups),
            attrs("scaled_centered_sumsq"),
        ),
        Node(
            f"{node.name}__allreduce_variance",
            OpKind.CUSTOM,
            (variance_local,),
            (variance,),
            AllReducePayload("AllReduceSum", variance_local, variance, "sum"),
            attrs("allreduce_variance"),
        ),
        Node(
            f"{node.name}__normalize",
            OpKind.ELEMENTWISE,
            (op.x, mean, variance, op.scale, op.bias),
            (op.output,),
            GroupNormalizeFromStatsPayload(
                op.x,
                mean,
                variance,
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
        (sum_local, mean, variance_local, variance),
        nodes,
    )


# Compatibility for callers that imported the old unstable payload name.
GroupNormalizeFromMomentsPayload = GroupNormalizeFromStatsPayload
