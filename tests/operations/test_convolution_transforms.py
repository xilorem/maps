from maps.graph import Tensor
from maps.operations.convolution_transforms import (
    ChannelShardedGemmPayload,
    Im2ColPayload,
    OutputReformatPayload,
)
from maps.planning.mapping import Submesh, TensorRange
from maps.target.magia import build_mesh as magia_mesh


def test_conv_to_gemm_tile_work_shards_channels_and_complete_output_rows() -> None:
    mesh = magia_mesh(width=2, height=3)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=3)
    x = Tensor("x", 4, (1, 3, 10, 9), 2)
    patches = Tensor("patches", 2, (30, 18), 2)
    packed_weight = Tensor("packed_weight", 2, (18, 7), 2, is_initializer=True)
    bias = Tensor("bias", 1, (7,), 2, is_initializer=True)
    gemm_result = Tensor("gemm_result", 2, (30, 7), 2)
    output = Tensor("output", 4, (1, 7, 5, 6), 2)
    im2col = Im2ColPayload(
        x=x,
        output=patches,
        kernel_shape=(3, 2),
        strides=(2, 2),
        pads=(2, 1, 2, 2),
        dilations=(2, 1),
    )
    gemm = ChannelShardedGemmPayload(
        x=patches,
        w=packed_weight,
        y=bias,
        output=gemm_result,
        row_granularity=6,
    )
    reformat = OutputReformatPayload(x=gemm_result, output=output)

    im2col_layout = im2col.output_layouts(submesh)[0]
    gemm_layout = gemm.output_layouts(submesh)[0]
    reformat_layout = reformat.output_layouts(submesh)[0]

    assert im2col_layout.mesh_y.shard_granularity == 6
    assert im2col_layout.mesh_x.tensor_axis is None
    assert gemm_layout.mesh_y.shard_granularity == 6
    assert reformat_layout.mesh_y.shard_granularity == 1

    expected_rows = (
        TensorRange(0, 12),
        TensorRange(0, 12),
        TensorRange(12, 12),
        TensorRange(12, 12),
        TensorRange(24, 6),
        TensorRange(24, 6),
    )
    expected_heights = (
        TensorRange(0, 2),
        TensorRange(0, 2),
        TensorRange(2, 2),
        TensorRange(2, 2),
        TensorRange(4, 1),
        TensorRange(4, 1),
    )
    expected_halos = (
        TensorRange(0, 5),
        TensorRange(0, 5),
        TensorRange(2, 7),
        TensorRange(2, 7),
        TensorRange(6, 4),
        TensorRange(6, 4),
    )
    expected_channels = (
        TensorRange(0, 4),
        TensorRange(4, 3),
        TensorRange(0, 4),
        TensorRange(4, 3),
        TensorRange(0, 4),
        TensorRange(4, 3),
    )

    for tile, rows, heights, halo, channels in zip(
        submesh.tiles,
        expected_rows,
        expected_heights,
        expected_halos,
        expected_channels,
    ):
        im2col_work = im2col.build_tile_work((im2col_layout,), tile)
        gemm_work = gemm.build_tile_work((gemm_layout,), tile)
        reformat_work = reformat.build_tile_work((reformat_layout,), tile)

        assert im2col_work.output_slice.dims == (rows, TensorRange(0, 18))
        assert im2col_work.input_tile_slices[0].dims == (
            TensorRange(0, 1),
            TensorRange(0, 3),
            halo,
            TensorRange(0, 9),
        )
        assert gemm_work.x_slice == im2col_work.output_slice
        assert gemm_work.output_slice.dims == (rows, channels)
        assert gemm_work.w_slice.dims == (TensorRange(0, 18), channels)
        assert gemm_work.y_slice is not None
        assert gemm_work.y_slice.dims == (channels,)
        assert reformat_work.input_tile_slices[0] == gemm_work.output_slice
        assert reformat_work.output_slice.dims == (
            TensorRange(0, 1),
            channels,
            heights,
            TensorRange(0, 6),
        )
