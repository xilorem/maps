"""RedMulE systolic device model."""

from __future__ import annotations

from MAPS.arch import DeviceKind, SystolicDevice, WorkKind, WorkSignature
from MAPS.core.dtype import TensorDType

REDMULE_ARRAY_WIDTH = 24
REDMULE_ARRAY_HEIGHT = 8


REDMULE_DEVICE = SystolicDevice(
    name="redmule",
    kind=DeviceKind.SYSTOLIC,
    throughput={
        WorkKind.GEMM: REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT,
        WorkKind.CONV2D: REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT,
    },
    capabilities=frozenset(
        {
            WorkSignature(
                work_kind=WorkKind.GEMM,
                input_dtypes=(TensorDType.FLOAT16, TensorDType.FLOAT16),
                output_dtypes=(TensorDType.FLOAT16,),
            ),
            WorkSignature(
                work_kind=WorkKind.GEMM,
                input_dtypes=(
                    TensorDType.FLOAT16,
                    TensorDType.FLOAT16,
                    TensorDType.FLOAT16,
                ),
                output_dtypes=(TensorDType.FLOAT16,),
            ),
        }
    ),
    array_width=REDMULE_ARRAY_WIDTH,
    array_height=REDMULE_ARRAY_HEIGHT,
)
