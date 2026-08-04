import pytest

from maps.hardware import WorkKind
from maps.graph import TensorDType
from maps.planning.mapping import TensorRange
from maps.planning.mapping import Submesh
from maps.graph import Tensor
from maps.target.magia import build_mesh as magia_mesh
from maps.target.magia import CORE_DEVICE as MAGIA_CORE_DEVICE
from maps.operations.split import SplitPayload, StaticSlicePayload
from maps.planning.allocation.candidates import permanent_l1_allocation_for_tile_work


def test_static_slice_maps_sharded_output_to_offset_input_region() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    x = Tensor("x", 3, (4, 6, 8), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 3, (2, 3, 8), 2, dtype=TensorDType.FLOAT16)
    payload = StaticSlicePayload(x, output, offsets=(1, 2, 0))

    tile_work = payload.build_tile_work(
        payload.output_layouts(submesh),
        submesh.tiles[1],
    )

    assert tile_work.work_kind is WorkKind.SLICE
    assert tile_work.output_slice.dims == (
        TensorRange(0, 2),
        TensorRange(0, 3),
        TensorRange(4, 4),
    )
    assert tile_work.input_slice.dims == (
        TensorRange(1, 2),
        TensorRange(2, 3),
        TensorRange(4, 4),
    )
    assert tile_work.operation_count() == 24
    assert payload.cost_model.cost(
        tile_work,
        submesh.tiles[1],
        MAGIA_CORE_DEVICE,
    ) == 24


def test_static_slice_validates_offsets_bounds_and_element_representation() -> None:
    x = Tensor("x", 2, (4, 6), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 2, (2, 3), 2, dtype=TensorDType.FLOAT16)

    with pytest.raises(ValueError, match="offsets must match input rank"):
        StaticSlicePayload(x, output, offsets=(0,))
    with pytest.raises(ValueError, match="offsets must be nonnegative"):
        StaticSlicePayload(x, output, offsets=(-1, 0))
    with pytest.raises(ValueError, match="region must fit inside input"):
        StaticSlicePayload(x, output, offsets=(3, 0))

    float32_output = Tensor(
        "float32_output",
        2,
        (2, 3),
        4,
        dtype=TensorDType.FLOAT32,
    )
    with pytest.raises(ValueError, match="element representations must match"):
        StaticSlicePayload(x, float32_output, offsets=(0, 0))


def test_split_payload_validates_axis_sizes_and_output_geometry() -> None:
    x = Tensor("x", 2, (4, 6), 2, dtype=TensorDType.FLOAT16)
    output0 = Tensor("output0", 2, (2, 6), 2, dtype=TensorDType.FLOAT16)
    output1 = Tensor("output1", 2, (2, 6), 2, dtype=TensorDType.FLOAT16)

    with pytest.raises(ValueError, match="axis must be within input tensor rank"):
        SplitPayload(x, (output0, output1), axis=2, sizes=(2, 2))
    with pytest.raises(ValueError, match="must sum to the split input dimension"):
        SplitPayload(x, (output0, output1), axis=0, sizes=(1, 2))

    bad_output = Tensor("bad", 2, (2, 5), 2, dtype=TensorDType.FLOAT16)
    with pytest.raises(ValueError, match="output shape does not match"):
        SplitPayload(x, (output0, bad_output), axis=0, sizes=(2, 2))


def test_split_maps_every_sharded_output_to_its_offset_input_region() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=4, height=1)
    x = Tensor("x", 2, (2, 7), 2, dtype=TensorDType.FLOAT16)
    outputs = (
        Tensor("q", 2, (2, 1), 2, dtype=TensorDType.FLOAT16),
        Tensor("k", 2, (2, 3), 2, dtype=TensorDType.FLOAT16),
        Tensor("v", 2, (2, 3), 2, dtype=TensorDType.FLOAT16),
    )
    payload = SplitPayload(x, outputs, axis=1, sizes=(1, 3, 3))

    layouts = payload.output_layouts(submesh, logical_shape=(4, 1))
    tile_work = payload.build_tile_work(layouts, submesh.tiles[2])

    assert len(layouts) == 3
    assert all(layout.submesh is submesh for layout in layouts)
    assert all(layout.logical_width == 4 for layout in layouts)
    assert tile_work.work_kind is WorkKind.SPLIT
    assert tuple(reference.tensor for reference in tile_work.output_slices) == outputs
    assert tuple(
        reference.tensor_slice.dims for reference in tile_work.output_slices
    ) == (
        (TensorRange(0, 2), TensorRange(1, 0)),
        (TensorRange(0, 2), TensorRange(2, 1)),
        (TensorRange(0, 2), TensorRange(2, 1)),
    )
    assert tuple(
        reference.tensor_slice.dims for reference in tile_work.input_slices
    ) == (
        (TensorRange(0, 2), TensorRange(1, 0)),
        (TensorRange(0, 2), TensorRange(3, 1)),
        (TensorRange(0, 2), TensorRange(6, 1)),
    )
    assert tile_work.operation_count() == 4
    assert payload.cost_model.cost(
        tile_work,
        submesh.tiles[2],
        MAGIA_CORE_DEVICE,
    ) == 4
    assert permanent_l1_allocation_for_tile_work(
        (tile_work,),
        frozenset(),
        num_token_slots=2,
    ) == 56
    assert permanent_l1_allocation_for_tile_work(
        (tile_work,),
        frozenset({x}),
        num_token_slots=2,
    ) == 40
