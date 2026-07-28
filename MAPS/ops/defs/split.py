"""Static split canonicalization and rectangular slice execution."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Tile, WorkKind
from MAPS.core.dtype import TensorDType
from MAPS.core.graph import Node, OpKind
from MAPS.core.layout import (
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
from MAPS.ops.common.tile_work import TileWork
from MAPS.ops.registry import register_op
from MAPS.ops.spec import LoweredOperation, OpSpec, STATIC_INPUT_VALUES


@dataclass(frozen=True)
class StaticSliceTileWork(TileWork):
    """One tile's materialized copy from an offset rectangular input region."""

    x: Tensor
    output: Tensor
    input_slice: TensorSlice
    output_slice: TensorSlice
    work_kind: WorkKind = WorkKind.SLICE

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.x, self.input_slice),)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.output, self.output_slice),)

    def operation_count(self) -> int:
        return self.output_slice.num_elements


@dataclass(frozen=True)
class StaticSlicePayload(OpPayload):
    """Materialize a unit-stride rectangular slice at fixed input offsets."""

    x: Tensor
    output: Tensor
    offsets: tuple[int, ...]
    work_kind: WorkKind = WorkKind.SLICE

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.SLICE:
            raise ValueError("StaticSlice must use SLICE work")
        if self.x.rank != self.output.rank:
            raise ValueError("StaticSlice input and output ranks must match")
        if len(self.offsets) != self.x.rank:
            raise ValueError("StaticSlice offsets must match input rank")
        if any(offset < 0 for offset in self.offsets):
            raise ValueError("StaticSlice offsets must be nonnegative")
        for offset, input_dim, output_dim in zip(
            self.offsets,
            self.x.dims,
            self.output.dims,
        ):
            if offset + output_dim > input_dim:
                raise ValueError("StaticSlice output region must fit inside input")
        if self.x.elem_bytes != self.output.elem_bytes or self.x.dtype != self.output.dtype:
            raise ValueError(
                "StaticSlice input and output element representations must match"
            )

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

    def required_input_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.output.rank:
            raise ValueError("StaticSlice output slice rank must match output tensor rank")
        return TensorSlice(
            rank=self.x.rank,
            dims=tuple(
                TensorRange(
                    start=offset + output_range.start,
                    length=output_range.length,
                )
                for offset, output_range in zip(self.offsets, output_slice.dims)
            ),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> StaticSliceTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        return StaticSliceTileWork(
            x=self.x,
            output=self.output,
            input_slice=self.required_input_slice(output_slice),
            output_slice=output_slice,
        )


@dataclass(frozen=True)
class SplitPayload(CompositeOpPayload):
    """Static multi-output split normalized independently of ONNX syntax."""

    x: Tensor
    outputs: tuple[Tensor, ...]
    axis: int
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.axis < 0 or self.axis >= self.x.rank:
            raise ValueError("Split axis must be within input tensor rank")
        if not self.outputs:
            raise ValueError("Split must have at least one output")
        if len(self.sizes) != len(self.outputs):
            raise ValueError("Split sizes must match output count")
        if any(size <= 0 for size in self.sizes):
            raise ValueError("Split sizes must be positive")
        if sum(self.sizes) != self.x.dims[self.axis]:
            raise ValueError("Split sizes must sum to the split input dimension")

        for output, size in zip(self.outputs, self.sizes):
            if output.rank != self.x.rank:
                raise ValueError("Split input and output ranks must match")
            expected_dims = list(self.x.dims)
            expected_dims[self.axis] = size
            if output.dims != tuple(expected_dims):
                raise ValueError("Split output shape does not match its size")
            if output.elem_bytes != self.x.elem_bytes or output.dtype != self.x.dtype:
                raise ValueError(
                    "Split input and output element representations must match"
                )

    def decompose(self, node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
        offsets = [0] * self.x.rank
        split_offset = 0
        nodes = []
        for output_idx, (output, size) in enumerate(zip(self.outputs, self.sizes)):
            offsets[self.axis] = split_offset
            nodes.append(
                Node(
                    name=f"{node.name}__slice_{output_idx}",
                    kind=OpKind.TRANSFORM,
                    inputs=(self.x,),
                    outputs=(output,),
                    payload=StaticSlicePayload(
                        x=self.x,
                        output=output,
                        offsets=tuple(offsets),
                    ),
                    attributes={
                        **node.attributes,
                        "split_output_index": output_idx,
                    },
                )
            )
            split_offset += size
        return (), tuple(nodes)


def _normalized_axis(node_name: str, x: Tensor, attributes: dict[str, object]) -> int:
    axis = int(attributes.get("axis", 0))
    if axis < 0:
        axis += x.rank
    if axis < 0 or axis >= x.rank:
        raise ValueError(f"Split node '{node_name}' axis must be within input rank")
    return axis


def _num_output_sizes(node_name: str, dimension: int, num_outputs: int) -> tuple[int, ...]:
    if num_outputs <= 0:
        raise ValueError(f"Split node '{node_name}' num_outputs must be positive")
    chunk_size = (dimension + num_outputs - 1) // num_outputs
    sizes = (chunk_size,) * (num_outputs - 1) + (
        dimension - chunk_size * (num_outputs - 1),
    )
    if any(size <= 0 for size in sizes):
        raise ValueError(f"Split node '{node_name}' produces a zero-sized output")
    return sizes


def lower_split_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> LoweredOperation:
    """Normalize the supported static ONNX Split forms."""

    if len(inputs) not in (1, 2):
        raise ValueError(f"Split node '{node_name}' must have 1 or 2 inputs")
    if not outputs:
        raise ValueError(f"Split node '{node_name}' must have at least one output")

    unknown_attributes = set(attributes) - {
        "axis",
        "num_outputs",
        STATIC_INPUT_VALUES,
    }
    if unknown_attributes:
        attribute = sorted(unknown_attributes)[0]
        raise NotImplementedError(f"Split attribute '{attribute}' is not implemented")

    x = inputs[0]
    axis = _normalized_axis(node_name, x, attributes)
    has_split_input = len(inputs) == 2
    has_num_outputs = "num_outputs" in attributes
    if has_split_input == has_num_outputs:
        raise ValueError(
            f"Split node '{node_name}' must provide exactly one of "
            "a split initializer or num_outputs"
        )

    if has_split_input:
        split = inputs[1]
        if not split.is_initializer:
            raise NotImplementedError(
                f"Split node '{node_name}' requires a static split initializer"
            )
        if (
            split.dtype is not TensorDType.INT64
            or split.rank != 1
            or split.dims != (len(outputs),)
        ):
            raise ValueError(
                f"Split node '{node_name}' split initializer must be a rank-one "
                "INT64 tensor with one value per output"
            )
        sizes = tuple(output.dims[axis] for output in outputs)
    else:
        num_outputs = int(attributes["num_outputs"])
        if num_outputs != len(outputs):
            raise ValueError(
                f"Split node '{node_name}' num_outputs must match output count"
            )
        sizes = _num_output_sizes(node_name, x.dims[axis], num_outputs)

    payload = SplitPayload(x=x, outputs=outputs, axis=axis, sizes=sizes)
    return LoweredOperation(
        kind=OpKind.TRANSFORM,
        payload=payload,
        inputs=(x,),
        outputs=outputs,
    )


register_op(
    OpSpec(
        name="split",
        onnx_names=("Split",),
        lower_onnx=lower_split_node,
        work_kinds=(WorkKind.SLICE,),
    )
)
