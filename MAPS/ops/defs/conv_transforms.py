"""Planner payloads used by the explicit Conv-to-GEMM decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.arch import Tile, WorkKind
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
from maps.operations import OpCostModel, OpPayload, TileWork
from maps.operations.elementwise import BinaryElementwisePayload
from maps.operations.gemm import GemmPayload


CONV_TRANSFORM_WORK_KINDS: dict[str, WorkKind] = {
    "Im2Col": WorkKind.IM2COL,
    "WeightPack": WorkKind.WEIGHT_PACK,
    "OutputReformat": WorkKind.OUTPUT_REFORMAT,
}


def _full_slice(tensor: Tensor) -> TensorSlice:
    return TensorSlice(
        rank=tensor.rank,
        dims=tuple(TensorRange(start=0, length=dim) for dim in tensor.dims),
    )


def _logical_dims(logical_shape: tuple[int, int] | None) -> tuple[int | None, int | None]:
    if logical_shape is None:
        return None, None
    return logical_shape


def _replicated_layout(
    submesh: Submesh,
    logical_shape: tuple[int, int] | None,
) -> TensorLayout:
    logical_width, logical_height = _logical_dims(logical_shape)
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        logical_width=logical_width,
        logical_height=logical_height,
    )


def _channel_sharded_layout(
    tensor: Tensor,
    submesh: Submesh,
    logical_shape: tuple[int, int] | None,
) -> TensorLayout:
    if tensor.rank not in (2, 4):
        raise ValueError("Conv decomposition layouts require rank-2 or rank-4 tensors")
    channel_axis = 1
    logical_width, logical_height = _logical_dims(logical_shape)
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=channel_axis),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        logical_width=logical_width,
        logical_height=logical_height,
    )


@dataclass(frozen=True)
class TransformTileWork(TileWork):
    """Tile-local data movement associated with one Conv transform."""

    output: Tensor
    output_slice: TensorSlice
    inputs: tuple[Tensor, ...]
    input_tile_slices: tuple[TensorSlice, ...]
    work_kind: WorkKind

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


class _TransformPayload(OpPayload):
    @property
    def work_kind(self) -> WorkKind:
        return CONV_TRANSFORM_WORK_KINDS[self.op_name]

    @property
    def cost_model(self) -> OpCostModel:
        from MAPS.ops.costs.conv_transform_cost import ConvTransformCostModel

        return ConvTransformCostModel()


@dataclass(frozen=True)
class Im2ColPayload(_TransformPayload):
    """Extract flattened convolution patches for one local matrix multiply."""

    x: Tensor
    output: Tensor
    kernel_shape: tuple[int, int]
    strides: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilations: tuple[int, int] = (1, 1)
    op_name: str = field(default="Im2Col", init=False)

    def __post_init__(self) -> None:
        if self.x.rank != 4 or self.output.rank != 2:
            raise ValueError("Im2Col requires an NCHW input and rank-2 output")
        if len(self.kernel_shape) != 2:
            raise ValueError("Im2Col kernel_shape must have length 2")
        if len(self.strides) != 2 or len(self.pads) != 4 or len(self.dilations) != 2:
            raise ValueError("Im2Col convolution attributes have invalid rank")
        if any(value <= 0 for value in self.kernel_shape + self.strides + self.dilations):
            raise ValueError("Im2Col kernel, stride, and dilation values must be > 0")
        if any(value < 0 for value in self.pads):
            raise ValueError("Im2Col pads must be >= 0")

        n, c, h, w = self.x.dims
        kh, kw = self.kernel_shape
        sh, sw = self.strides
        pt, pl, pb, pr = self.pads
        dh, dw = self.dilations
        oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
        ow = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
        if self.output.dims != (n * oh * ow, c * kh * kw):
            raise ValueError("Im2Col output shape does not match convolution geometry")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("Im2Col tensors must agree on element size")

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
    ) -> TransformTileWork:
        output_layout = self.single_output_layout(output_layouts)
        return TransformTileWork(
            output=self.output,
            output_slice=tile_tensor_slice(self.output, output_layout, tile),
            inputs=(self.x,),
            input_tile_slices=(_full_slice(self.x),),
            work_kind=self.work_kind,
        )


@dataclass(frozen=True)
class WeightPackPayload(_TransformPayload):
    """Flatten OIHW filters into the K-by-OC GEMM operand."""

    w: Tensor
    output: Tensor
    op_name: str = field(default="WeightPack", init=False)

    def __post_init__(self) -> None:
        if self.w.rank != 4 or self.output.rank != 2:
            raise ValueError("WeightPack requires an OIHW input and rank-2 output")
        oc, c, kh, kw = self.w.dims
        if self.output.dims != (c * kh * kw, oc):
            raise ValueError("WeightPack output shape does not match filter shape")
        if self.w.elem_bytes != self.output.elem_bytes:
            raise ValueError("WeightPack tensors must agree on element size")

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (_channel_sharded_layout(self.output, submesh, logical_shape),)

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> TransformTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        oc_slice = output_slice.dims[1]
        input_slice = TensorSlice(
            rank=self.w.rank,
            dims=(
                oc_slice,
                TensorRange(start=0, length=self.w.dims[1]),
                TensorRange(start=0, length=self.w.dims[2]),
                TensorRange(start=0, length=self.w.dims[3]),
            ),
        )
        return TransformTileWork(
            output=self.output,
            output_slice=output_slice,
            inputs=(self.w,),
            input_tile_slices=(input_slice,),
            work_kind=self.work_kind,
        )


@dataclass(frozen=True)
class ChannelShardedGemmPayload(GemmPayload):
    """GEMM used by Conv decomposition with rows kept local and replicated."""

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (_channel_sharded_layout(self.output, submesh, logical_shape),)


@dataclass(frozen=True)
class ChannelShardedBiasAddPayload(BinaryElementwisePayload):
    """Conv bias add using the same OC-only sharding as its GEMM."""

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (_channel_sharded_layout(self.output, submesh, logical_shape),)


@dataclass(frozen=True)
class OutputReformatPayload(_TransformPayload):
    """Map row-major spatial GEMM results back to NCHW output order."""

    x: Tensor
    output: Tensor
    op_name: str = field(default="OutputReformat", init=False)

    def __post_init__(self) -> None:
        if self.x.rank != 2 or self.output.rank != 4:
            raise ValueError("OutputReformat requires rank-2 input and NCHW output")
        n, oc, oh, ow = self.output.dims
        if self.x.dims != (n * oh * ow, oc):
            raise ValueError("OutputReformat input shape does not match NCHW output")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("OutputReformat tensors must agree on element size")

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (_channel_sharded_layout(self.output, submesh, logical_shape),)

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> TransformTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        input_slice = TensorSlice(
            rank=self.x.rank,
            dims=(
                TensorRange(start=0, length=self.x.dims[0]),
                output_slice.dims[1],
            ),
        )
        return TransformTileWork(
            output=self.output,
            output_slice=output_slice,
            inputs=(self.x,),
            input_tile_slices=(input_slice,),
            work_kind=self.work_kind,
        )
