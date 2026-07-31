"""Static reshape and semantic transpose operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from MAPS.arch import Tile, WorkKind
from MAPS.core.graph import OpKind
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
from MAPS.ops.common.payload import OpPayload, sharded_layout
from MAPS.ops.common.layout_relation import LayoutRelation
from MAPS.ops.common.tile_work import TileWork
from MAPS.ops.registry import register_op
from MAPS.ops.spec import LoweredOperation, OpSpec


def _full_slice(tensor: Tensor) -> TensorSlice:
    return TensorSlice(
        rank=tensor.rank,
        dims=tuple(TensorRange(0, dimension) for dimension in tensor.dims),
    )


@dataclass(frozen=True)
class RearrangeTileWork(TileWork):
    x: Tensor
    output: Tensor
    input_slice: TensorSlice
    output_slice: TensorSlice
    work_kind: WorkKind

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.x, self.input_slice),)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(self.output, self.output_slice),)

    def operation_count(self) -> int:
        return self.output_slice.num_elements


class _RearrangePayload(OpPayload):
    work_kind: WorkKind

    @property
    def cost_model(self) -> OpCostModel:
        from MAPS.ops.costs.elementwise_cost import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)


@dataclass(frozen=True)
class ReshapePayload(_RearrangePayload):
    """Static row-major reshape with a provably rectangular sharding axis."""

    x: Tensor
    output: Tensor
    work_kind: WorkKind = WorkKind.RESHAPE

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.RESHAPE:
            raise ValueError("Reshape must use RESHAPE work")
        if self.x.num_elements != self.output.num_elements:
            raise ValueError("Reshape input and output element counts must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("Reshape input and output element sizes must match")

    def _preserved_axis(self) -> tuple[int, int] | None:
        candidates = []
        for input_axis, input_dim in enumerate(self.x.dims):
            if input_dim <= 1:
                continue
            for output_axis, output_dim in enumerate(self.output.dims):
                if (
                    input_dim == output_dim
                    and prod(self.x.dims[:input_axis])
                    == prod(self.output.dims[:output_axis])
                    and prod(self.x.dims[input_axis + 1 :])
                    == prod(self.output.dims[output_axis + 1 :])
                ):
                    candidates.append((input_axis, output_axis))
        return max(candidates, key=lambda axes: self.x.dims[axes[0]], default=None)

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        preserved = self._preserved_axis()
        if preserved is None:
            return ()
        input_axis, output_axis = preserved
        axis_mapping = list(range(self.output.rank))
        axis_mapping[output_axis] = input_axis
        return (
            LayoutRelation(
                input_index=0,
                output_index=0,
                input_axis_for_output_axis=tuple(axis_mapping),
                guarantees_slice_containment=False,
            ),
        )

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        logical_width = logical_shape[0] if logical_shape is not None else None
        logical_height = logical_shape[1] if logical_shape is not None else None
        preserved = self._preserved_axis()
        mesh_x = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        if preserved is not None:
            mesh_x = LayoutAxis(
                mode=LayoutAxisMode.SHARD,
                tensor_axis=preserved[1],
            )
        return (
            TensorLayout(
                submesh=submesh,
                mesh_x=mesh_x,
                mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
                logical_width=logical_width,
                logical_height=logical_height,
            ),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> RearrangeTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        input_slice = _full_slice(self.x)
        preserved = self._preserved_axis()
        if preserved is not None:
            input_axis, output_axis = preserved
            input_dims = list(input_slice.dims)
            input_dims[input_axis] = output_slice.dims[output_axis]
            input_slice = TensorSlice(rank=self.x.rank, dims=tuple(input_dims))
        if input_slice.num_elements != output_slice.num_elements:
            raise ValueError("Reshape tile ownership is not row-major compatible")
        return RearrangeTileWork(
            x=self.x,
            output=self.output,
            input_slice=input_slice,
            output_slice=output_slice,
            work_kind=self.work_kind,
        )

@dataclass(frozen=True)
class TransposePayload(_RearrangePayload):
    """Semantic transpose whose input ownership may require a collective remap."""

    x: Tensor
    output: Tensor
    permutation: tuple[int, ...]
    work_kind: WorkKind = WorkKind.TRANSPOSE

    def __post_init__(self) -> None:
        if self.work_kind is not WorkKind.TRANSPOSE:
            raise ValueError("Transpose must use TRANSPOSE work")
        if sorted(self.permutation) != list(range(self.x.rank)):
            raise ValueError("Transpose permutation must cover every input axis")
        if self.output.dims != tuple(self.x.dims[axis] for axis in self.permutation):
            raise ValueError("Transpose output shape does not match permutation")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("Transpose input and output element sizes must match")

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return (
            LayoutRelation(
                input_index=0,
                output_index=0,
                input_axis_for_output_axis=self.permutation,
                guarantees_slice_containment=False,
            ),
        )

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
    ) -> RearrangeTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        input_dims: list[TensorRange | None] = [None] * self.x.rank
        for output_axis, input_axis in enumerate(self.permutation):
            input_dims[input_axis] = output_slice.dims[output_axis]
        input_slice = TensorSlice(
            rank=self.x.rank,
            dims=tuple(dimension for dimension in input_dims if dimension is not None),
        )
        return RearrangeTileWork(
            x=self.x,
            output=self.output,
            input_slice=input_slice,
            output_slice=output_slice,
            work_kind=self.work_kind,
        )

def lower_reshape_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> LoweredOperation:
    if len(inputs) != 2 or len(outputs) != 1:
        raise ValueError(f"Reshape node '{node_name}' must have 2 inputs and 1 output")
    if not inputs[1].is_initializer:
        raise NotImplementedError(
            f"Reshape node '{node_name}' requires a static shape initializer"
        )
    if int(attributes.get("allowzero", 0)) != 0:
        raise NotImplementedError("Reshape allowzero is not implemented")
    payload = ReshapePayload(x=inputs[0], output=outputs[0])
    return LoweredOperation(
        kind=OpKind.TRANSFORM,
        payload=payload,
        inputs=(inputs[0],),
        outputs=outputs,
    )


def lower_transpose_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, TransposePayload]:
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Transpose node '{node_name}' must have 1 input and 1 output")
    permutation = tuple(attributes.get("perm", tuple(reversed(range(inputs[0].rank)))))
    return (
        OpKind.TRANSFORM,
        TransposePayload(inputs[0], outputs[0], permutation),
    )


def lower_flatten_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, ReshapePayload]:
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Flatten node '{node_name}' must have 1 input and 1 output")
    unknown_attributes = set(attributes) - {"axis"}
    if unknown_attributes:
        attribute = sorted(unknown_attributes)[0]
        raise NotImplementedError(f"Flatten attribute '{attribute}' is not implemented")

    x = inputs[0]
    axis = int(attributes.get("axis", 1))
    if axis < 0:
        axis += x.rank
    if axis < 0 or axis > x.rank:
        raise ValueError(f"Flatten node '{node_name}' axis must be in [0, input rank]")

    expected_dims = (prod(x.dims[:axis]), prod(x.dims[axis:]))
    if outputs[0].dims != expected_dims:
        raise ValueError(
            f"Flatten node '{node_name}' output shape must be {expected_dims}"
        )
    return OpKind.TRANSFORM, ReshapePayload(x=x, output=outputs[0])


register_op(
    OpSpec(
        name="reshape",
        onnx_names=("Reshape",),
        lower_onnx=lower_reshape_node,
        work_kinds=(WorkKind.RESHAPE,),
    )
)
register_op(
    OpSpec(
        name="flatten",
        onnx_names=("Flatten",),
        lower_onnx=lower_flatten_node,
        work_kinds=(WorkKind.RESHAPE,),
    )
)
register_op(
    OpSpec(
        name="transpose",
        onnx_names=("Transpose",),
        lower_onnx=lower_transpose_node,
        work_kinds=(WorkKind.TRANSPOSE,),
    )
)
