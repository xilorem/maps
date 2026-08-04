import pytest

from maps.planning.mapping import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
    _apply_layout_axis,
    tensor_slice_num_bytes,
    tile_tensor_slice,
)
from maps.target.magia import build_mesh as magia_mesh
from maps.planning.mapping import Submesh
from maps.planning.stages import derive_virtual_collective_groups
from maps.graph import Tensor


def _format_slice_ranges(ranges: tuple[TensorRange, ...]) -> str:
    return ", ".join(
        f"axis{axis}=[{dim.start}:{dim.start + dim.length})"
        for axis, dim in enumerate(ranges)
    )


def test_tensor_slice_num_elements_and_bytes_use_slice_shape() -> None:
    tensor = Tensor(name="x", rank=3, dims=(8, 8, 12), elem_bytes=2)
    tensor_slice = TensorSlice(
        rank=3,
        dims=(
            TensorRange(start=0, length=8),
            TensorRange(start=2, length=3),
            TensorRange(start=6, length=4),
        ),
    )

    assert tensor_slice.num_elements == 8 * 3 * 4
    assert tensor_slice_num_bytes(tensor, tensor_slice) == tensor_slice.num_elements * 2


def test_tensor_and_slice_support_rank_six() -> None:
    dims = (1, 2, 3, 4, 5, 6)
    tensor = Tensor(name="x", rank=6, dims=dims, elem_bytes=2)
    tensor_slice = TensorSlice(
        rank=6,
        dims=tuple(TensorRange(start=0, length=dimension) for dimension in dims),
    )

    assert tensor.num_elements == 720
    assert tensor_slice.num_elements == 720


def test_tensor_rejects_rank_above_six() -> None:
    with pytest.raises(ValueError, match=r"rank must be in \[1, 6\]"):
        Tensor(name="x", rank=7, dims=(1, 1, 1, 1, 1, 1, 1), elem_bytes=2)


def test_submesh_tile_helpers_use_global_tile_ids() -> None:
    mesh = magia_mesh(width=4, height=3)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=1, y0=1, width=2, height=2)

    assert submesh.tile_mask == (
        (1 << mesh.tile_id(1, 1))
        | (1 << mesh.tile_id(2, 1))
        | (1 << mesh.tile_id(1, 2))
        | (1 << mesh.tile_id(2, 2))
    )
    assert submesh.intersects_tile_ids({mesh.tile_id(0, 0), mesh.tile_id(2, 1)})
    assert not submesh.intersects_tile_ids({mesh.tile_id(0, 0), mesh.tile_id(3, 2)})
    assert submesh.global_to_local(mesh.tile_id(2, 1)) == (1, 0)
    assert submesh.local_to_global(1, 0) == mesh.tile_id(2, 1)


def test_tile_tensor_slice_shards_both_axes() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=3)
    tensor = Tensor(name="x", rank=3, dims=(8, 8, 12), elem_bytes=2)
    target_tile = mesh.tile(1, 2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=2),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
    )
    expected = (
        TensorRange(start=0, length=8),
        TensorRange(start=6, length=2),
        TensorRange(start=6, length=6),
    )

    result = tile_tensor_slice(
        tensor=tensor,
        layout=layout,
        tile=target_tile,
    )

    assert result.rank == 3
    assert result.dims == expected


def test_tile_tensor_slice_balances_indivisible_shard_units() -> None:
    mesh = magia_mesh(width=2, height=1)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids={0, 1})
    tensor = Tensor(name="flattened_rows", rank=1, dims=(9,), elem_bytes=2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(
            mode=LayoutAxisMode.SHARD,
            tensor_axis=0,
            shard_granularity=3,
        ),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )

    assert tile_tensor_slice(tensor, layout, mesh.tiles[0]).dims == (
        TensorRange(start=0, length=6),
    )
    assert tile_tensor_slice(tensor, layout, mesh.tiles[1]).dims == (
        TensorRange(start=6, length=3),
    )


def test_default_shard_granularity_preserves_element_partitioning() -> None:
    mesh = magia_mesh(width=2, height=1)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids={0, 1})
    tensor = Tensor(name="elements", rank=1, dims=(9,), elem_bytes=2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )

    assert layout.mesh_x.shard_granularity == 1
    assert tile_tensor_slice(tensor, layout, mesh.tiles[0]).dims == (
        TensorRange(start=0, length=5),
    )
    assert tile_tensor_slice(tensor, layout, mesh.tiles[1]).dims == (
        TensorRange(start=5, length=4),
    )


def test_layout_rejects_invalid_shard_granularity() -> None:
    mesh = magia_mesh(width=1, height=1)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids={0})
    tensor = Tensor(name="elements", rank=1, dims=(10,), elem_bytes=2)

    with pytest.raises(ValueError, match="shard_granularity must be positive"):
        LayoutAxis(
            mode=LayoutAxisMode.SHARD,
            tensor_axis=0,
            shard_granularity=0,
        )

    for mode in (
        LayoutAxisMode.NONE,
        LayoutAxisMode.PARTIAL,
        LayoutAxisMode.REPLICATE,
    ):
        layout = TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(mode=mode, tensor_axis=0, shard_granularity=2),
            mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        )
        with pytest.raises(
            ValueError,
            match="shard_granularity must be one for non-sharded axes",
        ):
            layout.validate_for(tensor)

    non_divisible = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(
            mode=LayoutAxisMode.SHARD,
            tensor_axis=0,
            shard_granularity=3,
        ),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )
    with pytest.raises(ValueError, match="must divide tensor axis length"):
        non_divisible.validate_for(tensor)


def test_layout_equality_includes_shard_granularity() -> None:
    assert LayoutAxis(LayoutAxisMode.SHARD, tensor_axis=0) != LayoutAxis(
        LayoutAxisMode.SHARD,
        tensor_axis=0,
        shard_granularity=3,
    )


def test_tile_tensor_slice_uses_logical_shape_not_physical_shape() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=6, height=1)
    tensor = Tensor(name="x", rank=2, dims=(6, 12), elem_bytes=2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
        logical_width=3,
        logical_height=2,
    )

    result = tile_tensor_slice(
        tensor=tensor,
        layout=layout,
        tile=mesh.tile(4, 0),
    )

    assert result.dims == (
        TensorRange(start=3, length=3),
        TensorRange(start=4, length=4),
    )


def test_partial_layout_keeps_equal_slices_as_unresolved_contributions() -> None:
    mesh = magia_mesh(width=2, height=1)
    submesh = Submesh(mesh, 0, frozenset((0, 1)))
    tensor = Tensor("partial", rank=1, dims=(1,), elem_bytes=2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(LayoutAxisMode.PARTIAL, tensor_axis=0),
        mesh_y=LayoutAxis(LayoutAxisMode.REPLICATE),
    )

    assert tile_tensor_slice(tensor, layout, mesh.tiles[0]) == tile_tensor_slice(
        tensor, layout, mesh.tiles[1]
    )


def test_collective_groups_follow_partial_ownership_and_keep_singletons() -> None:
    mesh = magia_mesh(width=2, height=2)
    submesh = Submesh(mesh, 0, frozenset((0, 1, 2, 3)))
    tensor = Tensor("partial", rank=2, dims=(4, 1), elem_bytes=2)
    row_groups = derive_virtual_collective_groups(
        tensor,
        TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(LayoutAxisMode.PARTIAL, tensor_axis=1),
            mesh_y=LayoutAxis(LayoutAxisMode.SHARD, tensor_axis=0),
        ),
    )
    singleton_groups = derive_virtual_collective_groups(
        tensor,
        TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(LayoutAxisMode.REPLICATE),
            mesh_y=LayoutAxis(LayoutAxisMode.SHARD, tensor_axis=0),
        ),
    )

    assert tuple(group.virtual_tile_ids for group in row_groups) == (
        (0, 1),
        (2, 3),
    )
    assert tuple(group.virtual_tile_ids for group in singleton_groups) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )
