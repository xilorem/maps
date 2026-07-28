"""GroupNormalization semantic op and collective decomposition."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Tile, WorkKind
from MAPS.core.graph import Node, OpKind
from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.cost import OpCostModel
from MAPS.ops.common.payload import CompositeOpPayload, OpPayload, sharded_layout
from MAPS.ops.common.layout_relation import LayoutRelation
from MAPS.ops.common.tile_work import TileWork
from MAPS.ops.defs.collective import AllReducePayload
from MAPS.ops.defs.elementwise import BinaryElementwisePayload
from MAPS.ops.registry import register_op
from MAPS.ops.spec import OpSpec


def _replicated_layout(
    submesh: Submesh,
    logical_shape: tuple[int, int] | None,
) -> TensorLayout:
    logical_width = logical_shape[0] if logical_shape is not None else None
    logical_height = logical_shape[1] if logical_shape is not None else None
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        logical_width=logical_width,
        logical_height=logical_height,
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
        from MAPS.ops.costs.elementwise_cost import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (_replicated_layout(submesh, logical_shape),)

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> GroupReduceTileWork:
        output_layout = self.single_output_layout(output_layouts)
        input_layout = sharded_layout(
            self.x,
            output_layout.submesh,
            (
                output_layout.effective_logical_width,
                output_layout.effective_logical_height,
            ),
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
        from MAPS.ops.costs.elementwise_cost import ElementwiseCostModel

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
    )


def _same_shape_tensor(name: str, reference: Tensor) -> Tensor:
    return Tensor(name, reference.rank, reference.dims, reference.elem_bytes)


def decompose_group_normalization_node(
    node: Node,
) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
    if not isinstance(node.payload, GroupNormalizationPayload):
        raise TypeError("expected GroupNormalizationPayload")
    op = node.payload
    attributes = dict(node.attributes)
    attributes["stage_group_id"] = f"{node.name}::group_norm"

    squared = _same_shape_tensor(f"{node.name}__squared", op.x)
    sum_local = _stats_tensor(f"{node.name}__sum_local", op)
    sumsq_local = _stats_tensor(f"{node.name}__sumsq_local", op)
    sum_x = _same_shape_tensor(f"{node.name}__sum_x", sum_local)
    sum_global = _same_shape_tensor(f"{node.name}__sum_global", sum_local)
    sumsq_x = _same_shape_tensor(f"{node.name}__sumsq_x", sumsq_local)
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
            f"{node.name}__allreduce_sum_x",
            OpKind.CUSTOM,
            (sum_local,),
            (sum_x,),
            AllReducePayload("AllReduceSum", sum_local, sum_x, "sum", "x"),
            attrs("allreduce_sum_x"),
        ),
        Node(
            f"{node.name}__allreduce_sum_y",
            OpKind.CUSTOM,
            (sum_x,),
            (sum_global,),
            AllReducePayload("AllReduceSum", sum_x, sum_global, "sum", "y"),
            attrs("allreduce_sum_y"),
        ),
        Node(
            f"{node.name}__allreduce_sumsq_x",
            OpKind.CUSTOM,
            (sumsq_local,),
            (sumsq_x,),
            AllReducePayload("AllReduceSum", sumsq_local, sumsq_x, "sum", "x"),
            attrs("allreduce_sumsq_x"),
        ),
        Node(
            f"{node.name}__allreduce_sumsq_y",
            OpKind.CUSTOM,
            (sumsq_x,),
            (sumsq_global,),
            AllReducePayload("AllReduceSum", sumsq_x, sumsq_global, "sum", "y"),
            attrs("allreduce_sumsq_y"),
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
        (squared, sum_local, sumsq_local, sum_x, sum_global, sumsq_x, sumsq_global),
        nodes,
    )


def lower_group_normalization_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, GroupNormalizationPayload]:
    if len(inputs) != 3 or len(outputs) != 1:
        raise ValueError(
            f"GroupNormalization node '{node_name}' must have 3 inputs and 1 output"
        )
    if "num_groups" not in attributes:
        raise ValueError("GroupNormalization num_groups attribute is required")
    return (
        OpKind.CUSTOM,
        GroupNormalizationPayload(
            x=inputs[0],
            scale=inputs[1],
            bias=inputs[2],
            output=outputs[0],
            num_groups=int(attributes["num_groups"]),
            epsilon=float(attributes.get("epsilon", 1e-5)),
            stash_type=int(attributes.get("stash_type", 1)),
        ),
    )


register_op(
    OpSpec(
        name="group_normalization",
        onnx_names=("GroupNormalization",),
        lower_onnx=lower_group_normalization_node,
        work_kinds=(
            WorkKind.MUL,
            WorkKind.GROUP_REDUCE,
            WorkKind.GROUP_NORMALIZE,
        ),
    )
)
