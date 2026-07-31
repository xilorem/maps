from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Tile
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorRange, TensorSlice, TensorSliceRef, tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from maps.operations import OpPayload
from maps.operations import TileWork
from maps.operations.gemm import GemmCostModel

@dataclass(frozen=True)
class TutorialPayload(OpPayload):
    input_1: Tensor
    input_2: Tensor
    output: Tensor   


    @property
    def cost_model(self) -> object:
        return GemmCostModel()
    

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:

        logical_width = None
        logical_height = None
        if logical_shape is not None:
            logical_width, logical_height = logical_shape

        return (
            TensorLayout(
                submesh=submesh,
                mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=self.output.rank - 1),
                mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=self.output.rank - 2),
                logical_width=logical_width,
                logical_height=logical_height,
            ),
        )
    

    def required_input_1_slice(self, output_slice: TensorSlice) -> TensorSlice:
        dims = list(output_slice.dims[:-2])
        dims.append(output_slice.dims[-2])
        dims.append(_full_range(self.input_1.dims[-1]))
        return TensorSlice(rank=self.input_1.rank, dims=tuple(dims))

    def required_input_2_slice(self, output_slice: TensorSlice) -> TensorSlice:
        dims = list(output_slice.dims[:-2])
        dims.append(_full_range(self.input_2.dims[-2]))
        dims.append(output_slice.dims[-1])
        return TensorSlice(rank=self.input_2.rank, dims=tuple(dims))

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> TutorialTileWork:

        output_slice = tile_tensor_slice(
            tensor=self.output,
            layout=output_layouts[0],
            tile=tile,
        )

        return TutorialTileWork(
            output_slice=output_slice,
            input_1_slice=self.required_input_1_slice(output_slice),
            input_2_slice=self.required_input_2_slice(output_slice),
            input_1=self.input_1,
            input_2=self.input_2,
            output=self.output,
        )

    def operation_count(self) -> int:
        return self.output_slice.num_elements * self.x_slice.dims[-1].length

    def dimensions(self) -> tuple[int, int, int, int]:
        batch_volume = 1
        for dim in self.output_slice.dims[:-2]:
            batch_volume *= dim.length
        m_size = self.output_slice.dims[-2].length
        n_size = self.output_slice.dims[-1].length
        k_size = self.x_slice.dims[-1].length
        return batch_volume, m_size, n_size, k_size


def _full_range(dim: int) -> TensorRange:
    return TensorRange(start=0, length=dim)



@dataclass(frozen=True)
class TutorialTileWork(TileWork):
    output_slice: TensorSlice
    input_1_slice: TensorSlice
    input_2_slice: TensorSlice
    input_1: Tensor
    input_2: Tensor
    output: Tensor

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (
            TensorSliceRef(tensor=self.input_1, tensor_slice=self.input_1_slice),
            TensorSliceRef(tensor=self.input_2, tensor_slice=self.input_2_slice),
        )

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (
            TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),
        )
