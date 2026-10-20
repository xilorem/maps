"""NCHW MaxPool semantics and channel-sharded tile work."""
from dataclasses import dataclass
from maps.graph import Tensor
from maps.hardware import Tile, WorkKind
from maps.planning.mapping import (LayoutAxis, LayoutAxisMode, TensorLayout,
    TensorRange, TensorSlice, TensorSliceRef, Submesh, tile_tensor_slice)
from .contracts import OpPayload, TileWork
from .elementwise import ElementwiseCostModel

@dataclass(frozen=True)
class MaxPoolTileWork(TileWork):
    x: Tensor
    output: Tensor
    input_slice: TensorSlice
    output_slice: TensorSlice
    kernel_shape: tuple[int, int]
    work_kind: WorkKind = WorkKind.MAX_POOL

    @property
    def input_slices(self):
        return (TensorSliceRef(tensor=self.x, tensor_slice=self.input_slice),)

    @property
    def output_slices(self):
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)

    def operation_count(self):
        return self.output_slice.num_elements * self.kernel_shape[0] * self.kernel_shape[1]

@dataclass(frozen=True)
class MaxPoolPayload(OpPayload):
    x: Tensor
    output: Tensor
    kernel_shape: tuple[int, int]
    strides: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilations: tuple[int, int] = (1, 1)
    ceil_mode: int = 0
    work_kind: WorkKind = WorkKind.MAX_POOL

    def __post_init__(self):
        if self.x.rank != 4 or self.output.rank != 4:
            raise ValueError("MaxPool requires rank-4 NCHW tensors")
        if len(self.kernel_shape)!=2 or len(self.strides)!=2 or len(self.dilations)!=2 or len(self.pads)!=4:
            raise ValueError("MaxPool attributes have invalid rank")
        if any(v<=0 for v in self.kernel_shape+self.strides+self.dilations):
            raise ValueError("MaxPool kernel, strides and dilations must be positive")
        if self.dilations != (1, 1):
            raise NotImplementedError("MaxPool currently supports unit dilation only")
        if self.ceil_mode not in (0,1) or any(v<0 for v in self.pads):
            raise ValueError("Invalid MaxPool padding or ceil_mode")
        if any(self.pads[i]>=self.kernel_shape[i%2] for i in range(4)):
            raise ValueError("MaxPool padding must be smaller than kernel")
        if self.x.dims[:2]!=self.output.dims[:2] or self.x.dtype!=self.output.dtype:
            raise ValueError("MaxPool batch, channels and dtype must match")
        expected=[]
        for size,k,s,b,e in zip(self.x.dims[2:],self.kernel_shape,self.strides,self.pads[:2],self.pads[2:]):
            length=(size+b+e-k+(s-1 if self.ceil_mode else 0))//s+1
            if self.ceil_mode and (length-1)*s>=size+b: length-=1
            expected.append(length)
        if tuple(expected)!=self.output.dims[2:]:
            raise ValueError("MaxPool output shape does not match attributes")

    @property
    def cost_model(self):
        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(self, submesh: Submesh, logical_shape=None):
        return (TensorLayout(submesh=submesh,
            mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD,tensor_axis=1),
            mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
            logical_width=logical_shape[0] if logical_shape else None,
            logical_height=logical_shape[1] if logical_shape else None),)

    def build_tile_work(self, output_layouts, tile: Tile):
        output=tile_tensor_slice(self.output,self.single_output_layout(output_layouts),tile)
        input_slice=TensorSlice(rank=4,dims=(output.dims[0],output.dims[1],
            TensorRange(start=0,length=self.x.dims[2]),TensorRange(start=0,length=self.x.dims[3])))
        return MaxPoolTileWork(self.x,self.output,input_slice,output,self.kernel_shape)
