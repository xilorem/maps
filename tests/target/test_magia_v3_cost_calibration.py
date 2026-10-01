from dataclasses import dataclass

import pytest

from maps.graph import Tensor
from maps.hardware import WorkKind
from maps.planning.mapping import TensorRange, TensorSlice, TensorSliceRef
from maps.target import magia_v3


def _slice_ref(name: str, dims: tuple[int, ...]) -> TensorSliceRef:
    tensor = Tensor(
        name=name,
        rank=len(dims),
        dims=tuple(max(1, length) for length in dims),
        elem_bytes=2,
    )
    tensor_slice = TensorSlice(
        rank=len(dims),
        dims=tuple(TensorRange(start=0, length=length) for length in dims),
    )
    return TensorSliceRef(tensor, tensor_slice)


@dataclass(frozen=True)
class _MeasuredWork:
    work_kind: WorkKind
    amount: int
    input_slices: tuple[TensorSliceRef, ...]
    output_slices: tuple[TensorSliceRef, ...]
    gemm_dimensions: tuple[int, int, int, int] = (1, 1, 1, 1)

    def operation_count(self) -> int:
        return self.amount

    def dimensions(self) -> tuple[int, int, int, int]:
        return self.gemm_dimensions


@pytest.mark.parametrize(
    ("work_kind", "amount", "input_dims", "output_dims", "measured"),
    (
        (WorkKind.ADD, 2_048, ((2_048,), (2_048,)), (2_048,), 898),
        (WorkKind.RELU, 8_192, ((8_192,),), (8_192,), 2_411),
        (WorkKind.SUB, 64, ((1, 1, 4, 16), (1, 1, 4, 1)), (1, 1, 4, 16), 301),
        (WorkKind.DIV, 64, ((1, 1, 4, 16), (1, 1, 4, 1)), (1, 1, 4, 16), 301),
        (WorkKind.SOFTMAX_EXP, 64, ((64,),), (64,), 1_710),
        (WorkKind.GROUP_REDUCE, 8_192, ((8_192,),), (1,), 2_928),
        (WorkKind.GROUP_CENTERED_REDUCE, 8_192, ((8_192,), (1,)), (1,), 5_973),
        (WorkKind.GROUP_NORMALIZE, 8_192, ((8_192,), (1,), (1,), (128,), (128,)), (8_192,), 13_938),
        (WorkKind.REDUCE_SUM, 8_192, ((8_192,),), (512,), 78_426),
        (WorkKind.REDUCE_MAX, 64, ((64,),), (4,), 1_321),
    ),
)
def test_magia_v3_spatz_costs_track_mobilevit_gvsoc_measurements(
    work_kind: WorkKind,
    amount: int,
    input_dims: tuple[tuple[int, ...], ...],
    output_dims: tuple[int, ...],
    measured: int,
) -> None:
    work = _MeasuredWork(
        work_kind,
        amount,
        tuple(_slice_ref(f"input_{index}", dims) for index, dims in enumerate(input_dims)),
        (_slice_ref("output", output_dims),),
    )

    assert magia_v3.SPATZ_DEVICE.cycles(work) == pytest.approx(measured, rel=0.06)


@pytest.mark.parametrize(
    ("amount", "output_dims", "measured"),
    (
        (64, (1, 1, 4, 1), 704),
        (8_192, (1, 128, 4, 1), 78_426),
    ),
)
def test_magia_v3_reducesum_cost_tracks_both_measured_scales(
    amount: int,
    output_dims: tuple[int, ...],
    measured: int,
) -> None:
    work = _MeasuredWork(
        WorkKind.REDUCE_SUM,
        amount,
        (_slice_ref("input", (*output_dims[:-1], 16)),),
        (_slice_ref("output", output_dims),),
    )

    assert magia_v3.SPATZ_DEVICE.cycles(work) == pytest.approx(measured, rel=0.01)


@pytest.mark.parametrize(
    ("amount", "output_elements", "measured_8x8"),
    (
        (8, 1, 181),
        (256, 128, 5_340),
    ),
)
def test_magia_v3_reducesum_tracks_input_and_output_work(
    amount: int,
    output_elements: int,
    measured_8x8: int,
) -> None:
    work = _MeasuredWork(
        WorkKind.REDUCE_SUM,
        amount,
        (_slice_ref("input", (amount,)),),
        (_slice_ref("output", (output_elements,)),),
    )

    predicted = magia_v3.SPATZ_DEVICE.cycles(work)

    assert predicted == pytest.approx(measured_8x8, rel=0.03)


@pytest.mark.parametrize(
    ("m_size", "n_size", "measured"),
    (
        (64, 32, 82_117),
        (85, 32, 108_938),
        (86, 32, 110_145),
    ),
)
def test_magia_v3_gemm_cost_tracks_output_width(
    m_size: int,
    n_size: int,
    measured: int,
) -> None:
    k_size = 128
    work = _MeasuredWork(
        WorkKind.GEMM,
        m_size * n_size * k_size,
        (_slice_ref("weights", (m_size, k_size)), _slice_ref("input", (k_size, n_size))),
        (_slice_ref("output", (m_size, n_size)),),
        gemm_dimensions=(1, m_size, n_size, k_size),
    )

    assert magia_v3.SPATZ_DEVICE.cycles(work) == pytest.approx(measured, rel=0.04)


def test_magia_v3_gemm_cost_charges_fixed_loop_work_per_vector_block() -> None:
    def cycles(n_size: int) -> int:
        work = _MeasuredWork(
            WorkKind.GEMM,
            8 * n_size * 128,
            (_slice_ref("weights", (8, 128)), _slice_ref("input", (128, n_size))),
            (_slice_ref("output", (8, n_size)),),
            gemm_dimensions=(1, 8, n_size, 128),
        )
        return magia_v3.SPATZ_DEVICE.cycles(work)

    per_fma_increment = round(8 * 128 / 7.45)

    assert cycles(128) - cycles(127) == pytest.approx(per_fma_increment, abs=1)
    assert cycles(129) - cycles(128) > 5_000


def test_magia_v3_empty_gemm_shard_has_no_cost() -> None:
    work = _MeasuredWork(
        WorkKind.GEMM,
        0,
        (_slice_ref("weights", (0, 128)), _slice_ref("input", (128, 16))),
        (_slice_ref("output", (0, 16)),),
        gemm_dimensions=(1, 0, 16, 128),
    )

    assert magia_v3.SPATZ_DEVICE.cycles(work) == 0


@pytest.mark.parametrize(
    ("broadcast_dims", "measured"),
    (
        ((1, 1, 4, 16), 4_522),
        ((1, 128, 4, 1), 10_296),
    ),
)
def test_magia_v3_mul_distinguishes_sdk_broadcast_paths(
    broadcast_dims: tuple[int, ...],
    measured: int,
) -> None:
    output_dims = (1, 128, 4, 16)
    work = _MeasuredWork(
        WorkKind.MUL,
        8_192,
        (_slice_ref("input", output_dims), _slice_ref("broadcast", broadcast_dims)),
        (_slice_ref("output", output_dims),),
    )

    assert magia_v3.SPATZ_DEVICE.cycles(work) == pytest.approx(measured, rel=0.03)


@pytest.mark.parametrize(
    ("output_dims", "broadcast_dims", "expected_compute"),
    (
        ((1, 128, 1, 5), (1, 1, 1, 5), 11_180),
        ((1, 128, 1, 6), (1, 1, 1, 6), 2_621),
        ((1, 128, 4, 4), (1, 128, 4, 1), 10_296),
    ),
)
def test_magia_v3_mul_tracks_eight_by_eight_kernel_geometry(
    output_dims: tuple[int, ...],
    broadcast_dims: tuple[int, ...],
    expected_compute: int,
) -> None:
    amount = _slice_ref("output", output_dims).tensor_slice.num_elements
    work = _MeasuredWork(
        WorkKind.MUL,
        amount,
        (_slice_ref("input", output_dims), _slice_ref("broadcast", broadcast_dims)),
        (_slice_ref("output", output_dims),),
    )

    assert magia_v3.SPATZ_DEVICE.cycles(work) == pytest.approx(
        expected_compute, rel=0.01
    )


@pytest.mark.parametrize(
    ("work_kind", "row_len", "expected_compute"),
    (
        (WorkKind.SUB, 2, 111),
        (WorkKind.SUB, 3, 1_773),
        (WorkKind.DIV, 2, 111),
        (WorkKind.DIV, 3, 1_792),
    ),
)
def test_magia_v3_binary_broadcast_models_odd_row_scalar_fallback(
    work_kind: WorkKind,
    row_len: int,
    expected_compute: int,
) -> None:
    output_dims = (1, 1, 1, row_len)
    work = _MeasuredWork(
        work_kind,
        row_len,
        (
            _slice_ref("input", output_dims),
            _slice_ref("broadcast", (1, 1, 1, 1)),
        ),
        (_slice_ref("output", output_dims),),
    )

    assert magia_v3.SPATZ_DEVICE.cycles(work) == pytest.approx(
        expected_compute, rel=0.01
    )


def test_magia_v3_mul_charges_vector_blocks_instead_of_active_lanes() -> None:
    def cycles(row_len: int) -> int:
        output_dims = (1, 128, 1, row_len)
        work = _MeasuredWork(
            WorkKind.MUL,
            128 * row_len,
            (
                _slice_ref("input", output_dims),
                _slice_ref("broadcast", (1, 1, 1, row_len)),
            ),
            (_slice_ref("output", output_dims),),
        )
        return magia_v3.SPATZ_DEVICE.cycles(work)

    assert cycles(64) > cycles(4)
    assert cycles(130) > cycles(128)


@pytest.mark.parametrize(
    ("work_kind", "input_dims", "output_dims", "compute_estimate"),
    (
        (WorkKind.ADD, ((512,), (512,)), (512,), 277),
        (WorkKind.SOFTMAX_EXP, ((5,),), (5,), 198),
        (WorkKind.SOFTMAX_EXP, ((6,),), (6,), 223),
        (WorkKind.GROUP_REDUCE, ((2_048,),), (1,), 2_001),
        (WorkKind.GROUP_CENTERED_REDUCE, ((2_048,), (1,)), (1,), 2_757),
        (
            WorkKind.GROUP_NORMALIZE,
            ((1, 128, 4, 4), (1,), (1,), (128,), (128,)),
            (1, 128, 4, 4),
            9_330,
        ),
    ),
)
def test_magia_v3_compute_formulas_remain_shape_based_for_small_shards(
    work_kind: WorkKind,
    input_dims: tuple[tuple[int, ...], ...],
    output_dims: tuple[int, ...],
    compute_estimate: int,
) -> None:
    amount = _slice_ref("output", output_dims).tensor_slice.num_elements
    if work_kind in {WorkKind.GROUP_REDUCE, WorkKind.GROUP_CENTERED_REDUCE}:
        amount = _slice_ref("input", input_dims[0]).tensor_slice.num_elements
    work = _MeasuredWork(
        work_kind,
        amount,
        tuple(_slice_ref(f"input_{index}", dims) for index, dims in enumerate(input_dims)),
        (_slice_ref("output", output_dims),),
    )

    assert magia_v3.SPATZ_DEVICE.cycles(work) == compute_estimate


@pytest.mark.parametrize(
    ("elements", "measured"),
    (
        (2_048, 2_457),
        (4_096, 2_841),
    ),
)
def test_magia_v3_im2col_conservatively_tracks_movement_measurements(
    elements: int,
    measured: int,
) -> None:
    work = _MeasuredWork(
        WorkKind.IM2COL,
        elements,
        (_slice_ref("input", (elements,)),),
        (_slice_ref("output", (elements,)),),
    )

    assert magia_v3.IDMA_WRITE_DEVICE.cycles(work) == pytest.approx(measured, rel=0.20)


@pytest.mark.parametrize(
    ("work_kind", "elements", "measured"),
    (
        (WorkKind.ALL_REDUCE_SUM, 1, 594),
        (WorkKind.ALL_REDUCE_SUM, 4, 669),
        (WorkKind.ALL_REDUCE_SUM, 512, 730),
        (WorkKind.ALL_REDUCE_MAX, 4, 769),
    ),
)
def test_magia_v3_singleton_collective_conservatively_tracks_movement_measurements(
    work_kind: WorkKind,
    elements: int,
    measured: int,
) -> None:
    tile = magia_v3.build_mesh(width=1, height=1).tiles[0]

    assert magia_v3.SPATZ_DEVICE.collective_cycles(
        work_kind, elements, (tile,)
    ) == pytest.approx(measured, rel=0.20)
