import pytest

from maps.graph import TensorDType
from maps.hardware import WorkKind, WorkSignature
from maps.target import magia_v3


@pytest.mark.parametrize(
    ("work_kind", "input_count"),
    (
        (WorkKind.GROUP_REDUCE, 1),
        (WorkKind.ALL_REDUCE_SUM, 1),
        (WorkKind.GROUP_CENTERED_REDUCE, 2),
        (WorkKind.GROUP_NORMALIZE, 5),
        (WorkKind.GEMM, 3),
        (WorkKind.REDUCE_MAX, 1),
        (WorkKind.ALL_REDUCE_MAX, 1),
        (WorkKind.SUB, 2),
        (WorkKind.SOFTMAX_EXP, 1),
        (WorkKind.REDUCE_SUM, 1),
        (WorkKind.DIV, 2),
        (WorkKind.MUL, 2),
        (WorkKind.RELU, 1),
        (WorkKind.ADD, 2),
    ),
)
def test_mobilevit_compute_is_assigned_to_spatz(
    work_kind: WorkKind,
    input_count: int,
) -> None:
    signature = WorkSignature(
        work_kind,
        (TensorDType.FLOAT16,) * input_count,
        (TensorDType.FLOAT16,),
    )

    assert magia_v3.DEVICE_ASSIGNMENT.assignments[signature] == "spatz"
    assert magia_v3.SPATZ_DEVICE.supports(signature)
    assert not magia_v3.CORE_DEVICE.supports(signature)
