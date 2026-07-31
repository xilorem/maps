import pytest

from MAPS.arch import WorkKind
from MAPS.core.dtype import TensorDType
from MAPS.core.layout import TensorRange
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.hw.chips import magia_mesh
from MAPS.hw.chips.magia import MAGIA_CORE_DEVICE
from maps.operations.split import SplitPayload, StaticSlicePayload


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
