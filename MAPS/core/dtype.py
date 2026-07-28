
from enum import Enum

class TensorDType(Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"

_DTYPE_ELEM_BYTES = {
    TensorDType.FLOAT16: 2,
    TensorDType.FLOAT32: 4,
    TensorDType.INT32: 4,
    TensorDType.INT64: 8,
    TensorDType.UINT8: 1,
    TensorDType.BOOL: 1,
}


def dtype_elem_bytes(dtype: TensorDType) -> int:
    return _DTYPE_ELEM_BYTES[dtype]
