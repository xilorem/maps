import pytest

from maps.hardware import Device, DeviceKind, L1Memory, MatrixDevice, ScalarDevice, SystolicDevice, Tile, VectorDevice, WorkKind, WorkSignature
from MAPS.hw.chips import magia_mesh
from MAPS.hw.chips.n300d import wormhole_n300d_mesh
from MAPS.hw.chips.magia import MAGIA_CORE_DEVICE, MAGIA_REDMULE_DEVICE
from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.core.dtype import TensorDType
from MAPS.core.tensor import Tensor
from MAPS.ops.costs.elementwise_cost import ElementwiseCostModel
from MAPS.ops.costs.gemm_cost import GemmCostModel
from MAPS.hw.devices.generic import GENERIC_SCALAR_DEVICE
from MAPS.hw.devices.redmule import REDMULE_ARRAY_HEIGHT, REDMULE_ARRAY_WIDTH
from MAPS.hw.devices.spatz import (
    SPATZ_DEVICE,
    SPATZ_FREQUENCY_HZ,
    SPATZ_LANES,
    SPATZ_LANE_WIDTH_BYTES,
    SPATZ_TCDM_BANKS,
    SPATZ_TCDM_BANK_WIDTH_BYTES,
    SPATZ_VLEN_BITS,
    SPATZ_VLSU_PORTS,
    SPATZ_VLSU_SWITCH_LOAD_TO_STORE_CYCLES,
    SPATZ_VLSU_SWITCH_SAME_CYCLES,
    SPATZ_VLSU_SWITCH_STORE_TO_LOAD_CYCLES,
    SpatzDevice,
)
from MAPS.hw.devices.tensix_tile import TENSIX_MATRIX_DEVICE
from MAPS.ops.defs.gemm import GemmTileWork
from MAPS.ops.defs.elementwise import ElementwiseTileWork
from MAPS.ops.defs.reduction import ReductionTileWork


def _tile_work(m_size: int = 4, n_size: int = 8, k_size: int = 16) -> GemmTileWork:
    x = Tensor(name="x", rank=2, dims=(m_size, k_size), elem_bytes=4)
    w = Tensor(name="w", rank=2, dims=(k_size, n_size), elem_bytes=4)
    output = Tensor(name="output", rank=2, dims=(m_size, n_size), elem_bytes=4)
    output_slice = TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=m_size),
            TensorRange(start=0, length=n_size),
        ),
    )
    return GemmTileWork(
        output_slice=output_slice,
        x_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=m_size),
                TensorRange(start=0, length=k_size),
            ),
        ),
        w_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=k_size),
                TensorRange(start=0, length=n_size),
            ),
        ),
        y_slice=None,
        x=x,
        w=w,
        output=output,
    )


def test_device_base_class_is_not_directly_instantiable() -> None:
    try:
        Device(
            name="base",
            kind=DeviceKind.SCALAR,
            throughput={WorkKind.ELEMENTWISE: 1},
        )
    except TypeError as exc:
        assert "concrete device type" in str(exc)
    else:
        raise AssertionError("expected Device base class construction to fail")


def test_tile_can_use_generic_scalar_device() -> None:
    tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=1),
        devices=(GENERIC_SCALAR_DEVICE,),
    )

    assert tile.devices[0].name == "core"
    assert tile.devices[0].kind is DeviceKind.SCALAR
    assert isinstance(tile.devices[0], ScalarDevice)
    assert tile.devices[0].supports(
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT32, TensorDType.FLOAT32),
            (TensorDType.FLOAT32,),
        )
    )


def test_tile_rejects_empty_devices() -> None:
    try:
        Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=1), devices=())
    except ValueError as exc:
        assert "tile devices must not be empty" in str(exc)
    else:
        raise AssertionError("expected Tile construction to fail")


def test_redmule_is_a_named_systolic_device() -> None:
    device = MAGIA_REDMULE_DEVICE

    assert device.name == "redmule"
    assert device.kind is DeviceKind.SYSTOLIC
    assert isinstance(device, SystolicDevice)
    assert device.array_width == REDMULE_ARRAY_WIDTH
    assert device.array_height == REDMULE_ARRAY_HEIGHT
    assert device.supports(
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT16, TensorDType.FLOAT16),
            (TensorDType.FLOAT16,),
        )
    )
    assert device.throughput[WorkKind.GEMM] == REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT


def test_gemm_cost_uses_systolic_device_when_available() -> None:
    scalar_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=1),
        devices=(GENERIC_SCALAR_DEVICE,),
    )
    redmule_tile = magia_mesh().tile(0, 0)
    model = GemmCostModel()
    tile_work = _tile_work()

    assert model.cost(tile_work, redmule_tile, MAGIA_REDMULE_DEVICE) < model.cost(
        tile_work, scalar_tile, GENERIC_SCALAR_DEVICE
    )


def test_redmule_gemm_cost_accounts_for_array_shape() -> None:
    redmule_tile = magia_mesh().tile(0, 0)
    model = GemmCostModel()

    compact_work = _tile_work(m_size=4, n_size=8, k_size=16)
    wide_work = _tile_work(m_size=1, n_size=32, k_size=16)

    assert model.cost(compact_work, redmule_tile, MAGIA_REDMULE_DEVICE) == 46
    assert model.cost(wide_work, redmule_tile, MAGIA_REDMULE_DEVICE) == 92


def test_tensix_matrix_device_uses_matrix_kind() -> None:
    assert TENSIX_MATRIX_DEVICE.kind is DeviceKind.MATRIX
    assert isinstance(TENSIX_MATRIX_DEVICE, MatrixDevice)
    assert TENSIX_MATRIX_DEVICE.supports(
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT32, TensorDType.FLOAT32),
            (TensorDType.FLOAT32,),
        )
    )


def test_wormhole_tile_exposes_matrix_device_for_gemm() -> None:
    tile = wormhole_n300d_mesh().tile(0, 0)

    matrix_devices = tuple(device for device in tile.devices if device.kind is DeviceKind.MATRIX)

    assert len(matrix_devices) == 1
    assert matrix_devices[0].name == "tensix_matrix"


def test_gemm_cost_prefers_matrix_device_when_available() -> None:
    scalar_gemm_device = ScalarDevice(
        name="scalar_gemm",
        kind=DeviceKind.SCALAR,
        throughput={WorkKind.GEMM: 1},
    )
    matrix_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=1),
        devices=(scalar_gemm_device, TENSIX_MATRIX_DEVICE),
    )
    model = GemmCostModel()
    tile_work = _tile_work(m_size=64, n_size=64, k_size=64)

    assert model.cost(
        tile_work, matrix_tile, TENSIX_MATRIX_DEVICE
    ) == TENSIX_MATRIX_DEVICE.cycles(tile_work)


def test_gemm_cost_rejects_a_device_outside_the_tile() -> None:
    with pytest.raises(ValueError, match="is not present on tile 0"):
        GemmCostModel().cost(
            _tile_work(),
            magia_mesh(width=1, height=1).tile(0, 0),
            GENERIC_SCALAR_DEVICE,
        )


def test_vector_device_rejects_zero_length() -> None:
    with pytest.raises(ValueError, match="vector_length must be > 0"):
        VectorDevice(
            name="vector",
            kind=DeviceKind.VECTOR,
            throughput={WorkKind.ELEMENTWISE: 1},
            vector_length=0,
        )


def test_scalar_device_uses_operation_specific_throughput() -> None:
    device = ScalarDevice(
        name="scalar",
        kind=DeviceKind.SCALAR,
        throughput={WorkKind.ADD: 4, WorkKind.DIV: 1},
    )
    tensor = Tensor(name="x", rank=1, dims=(8,), elem_bytes=4)
    output_slice = TensorSlice(rank=1, dims=(TensorRange(start=0, length=8),))

    add_work = ElementwiseTileWork(
        work_kind=WorkKind.ADD,
        output=tensor,
        output_slice=output_slice,
        inputs=(tensor,),
        input_tile_slices=(output_slice,),
    )
    div_work = ElementwiseTileWork(
        work_kind=WorkKind.DIV,
        output=tensor,
        output_slice=output_slice,
        inputs=(tensor,),
        input_tile_slices=(output_slice,),
    )

    assert device.cycles(add_work) == 2
    assert device.cycles(div_work) == 8


def _elementwise_work(
    work_kind: WorkKind,
    count: int,
    elem_bytes: int,
    input_count: int = 1,
) -> ElementwiseTileWork:
    output = Tensor(name="output", rank=1, dims=(count,), elem_bytes=elem_bytes)
    output_slice = TensorSlice(rank=1, dims=(TensorRange(start=0, length=count),))
    inputs = tuple(
        Tensor(name=f"input_{index}", rank=1, dims=(count,), elem_bytes=elem_bytes)
        for index in range(input_count)
    )
    return ElementwiseTileWork(
        work_kind=work_kind,
        output=output,
        output_slice=output_slice,
        inputs=inputs,
        input_tile_slices=(output_slice,) * input_count,
    )


def test_spatz_has_named_magia_gvsoc_configuration() -> None:
    device = SPATZ_DEVICE

    assert isinstance(device, SpatzDevice)
    assert device.name == "spatz"
    assert device.kind is DeviceKind.VECTOR
    assert device.lanes == SPATZ_LANES == 4
    assert device.lane_width_bytes == SPATZ_LANE_WIDTH_BYTES == 4
    assert device.vlen_bits == SPATZ_VLEN_BITS == 512
    assert device.frequency_hz == SPATZ_FREQUENCY_HZ == 200_000_000
    assert device.vlsu_ports == SPATZ_VLSU_PORTS == 4
    assert device.tcdm_banks == SPATZ_TCDM_BANKS == 32
    assert device.tcdm_bank_width_bytes == SPATZ_TCDM_BANK_WIDTH_BYTES == 4
    assert device.vlsu_switch_same == SPATZ_VLSU_SWITCH_SAME_CYCLES == 2
    assert device.vlsu_switch_store_to_load == SPATZ_VLSU_SWITCH_STORE_TO_LOAD_CYCLES == 3
    assert device.vlsu_switch_load_to_store == SPATZ_VLSU_SWITCH_LOAD_TO_STORE_CYCLES == 7


@pytest.mark.parametrize(
    ("elem_bytes", "expected_cycles"),
    (
        (2, 10),
        (4, 13),
        (8, 19),
    ),
)
def test_spatz_cost_scales_with_element_width(
    elem_bytes: int,
    expected_cycles: int,
) -> None:
    work = _elementwise_work(WorkKind.RELU, count=8, elem_bytes=elem_bytes)

    assert SPATZ_DEVICE.cycles(work) == expected_cycles


def test_spatz_strip_mines_at_lmul8_and_accounts_for_tail_switches() -> None:
    full_vector = _elementwise_work(WorkKind.RELU, count=256, elem_bytes=2)
    one_element_tail = _elementwise_work(WorkKind.RELU, count=257, elem_bytes=2)

    assert SPATZ_DEVICE.cycles(full_vector) == 103
    assert SPATZ_DEVICE.cycles(one_element_tail) == 116


def test_spatz_binary_profile_adds_second_load_and_same_direction_gap() -> None:
    unary = _elementwise_work(WorkKind.RELU, count=8, elem_bytes=2)
    binary = _elementwise_work(WorkKind.ADD, count=8, elem_bytes=2, input_count=2)

    assert SPATZ_DEVICE.cycles(unary) == 10
    assert SPATZ_DEVICE.cycles(binary) == 13


def test_spatz_reduction_uses_latency_three_chunk_spacing() -> None:
    x = Tensor(name="x", rank=1, dims=(32,), elem_bytes=2)
    output = Tensor(name="output", rank=1, dims=(1,), elem_bytes=2)
    input_slice = TensorSlice(rank=1, dims=(TensorRange(start=0, length=32),))
    output_slice = TensorSlice(rank=1, dims=(TensorRange(start=0, length=1),))
    work = ReductionTileWork(
        work_kind=WorkKind.REDUCE_SUM,
        x=x,
        output=output,
        input_slice=input_slice,
        output_slice=output_slice,
    )

    assert SPATZ_DEVICE.cycles(work) == 24


def test_spatz_leaves_unprofiled_work_on_scalar_core() -> None:
    tile = magia_mesh(width=1, height=1).tile(0, 0)
    work = _elementwise_work(WorkKind.LOG, count=8, elem_bytes=2)

    assert work.work_kind is WorkKind.LOG
    assert ElementwiseCostModel(work_kind=WorkKind.LOG).cost(
        work, tile, MAGIA_CORE_DEVICE
    ) == (
        MAGIA_CORE_DEVICE.cycles(work)
    )


def test_profiled_elementwise_work_can_select_spatz_over_scalar() -> None:
    tile = magia_mesh(width=1, height=1).tile(0, 0)
    work = _elementwise_work(WorkKind.ADD, count=8, elem_bytes=2, input_count=2)

    assert SPATZ_DEVICE.cycles(work) < MAGIA_CORE_DEVICE.cycles(work)
    assert ElementwiseCostModel(work_kind=WorkKind.ADD).cost(
        work, tile, SPATZ_DEVICE
    ) == (
        SPATZ_DEVICE.cycles(work)
    )


def test_magia_gemm_still_uses_redmule_with_spatz_present() -> None:
    tile = magia_mesh(width=1, height=1).tile(0, 0)
    work = _tile_work()

    assert GemmCostModel().cost(
        work, tile, MAGIA_REDMULE_DEVICE
    ) == MAGIA_REDMULE_DEVICE.cycles(work)
