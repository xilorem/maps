"""Deterministic conversion and packing of deployment constants."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import prod

import numpy as np

from MAPS.core.constants import Constant, ConstantStore
from MAPS.core.dtype import TensorDType
from MAPS.pipeline.execution_plan import ExecutionPlan


DEFAULT_WEIGHT_ALIGNMENT = 16

_NUMPY_DTYPES = {
    TensorDType.FLOAT16: np.dtype("<f2"),
    TensorDType.FLOAT32: np.dtype("<f4"),
    TensorDType.INT32: np.dtype("<i4"),
    TensorDType.INT64: np.dtype("<i8"),
    TensorDType.UINT8: np.dtype("u1"),
    TensorDType.BOOL: np.dtype("?"),
}


@dataclass(frozen=True)
class PackedInitializer:
    tensor_id: int
    name: str
    offset: int
    byte_size: int
    dtype: TensorDType
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class PackedWeights:
    data: bytes
    initializers: tuple[PackedInitializer, ...]
    alignment: int = DEFAULT_WEIGHT_ALIGNMENT

    @property
    def sha256(self) -> str:
        return sha256(self.data).hexdigest()


def _deployment_bytes(constant: Constant) -> tuple[TensorDType, bytes]:
    source_dtype = _NUMPY_DTYPES[constant.dtype]
    expected_size = prod(constant.shape) * source_dtype.itemsize
    if len(constant.data) != expected_size:
        raise ValueError(
            f"constant '{constant.name}' has {len(constant.data)} bytes; "
            f"expected {expected_size}"
        )
    return constant.dtype, constant.data


def pack_weights(
    execution_plan: ExecutionPlan,
    constants: ConstantStore,
    *,
    alignment: int = DEFAULT_WEIGHT_ALIGNMENT,
) -> PackedWeights:
    """Convert and pack constants in final Execution Plan tensor-id order."""

    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("weight alignment must be a positive power of two")

    tensor_ids: dict[str, int] = {}
    tensors_by_name = {}
    for tensor_id, tensor in enumerate(execution_plan.tensors):
        if tensor.name in tensor_ids:
            raise ValueError(
                f"execution plan tensor name '{tensor.name}' is not unique"
            )
        tensor_ids[tensor.name] = tensor_id
        tensors_by_name[tensor.name] = tensor

    missing = sorted(
        constant.name for constant in constants.constants
        if constant.name not in tensor_ids
    )
    if missing:
        raise ValueError(
            f"constants have no Execution Plan tensor ID: {', '.join(missing)}"
        )
    for constant in constants.constants:
        tensor = tensors_by_name[constant.name]
        if not tensor.is_initializer:
            raise ValueError(
                f"constant '{constant.name}' maps to a non-initializer "
                "Execution Plan tensor"
            )
        if tensor.dtype is not constant.dtype:
            raise ValueError(
                f"constant '{constant.name}' dtype does not match "
                "Execution Plan tensor"
            )
        if tensor.dims != constant.shape:
            raise ValueError(
                f"constant '{constant.name}' shape does not match "
                "Execution Plan tensor"
            )

    ordered = sorted(
        constants.constants,
        key=lambda constant: (tensor_ids[constant.name], constant.name),
    )
    image = bytearray()
    metadata: list[PackedInitializer] = []
    for constant in ordered:
        padding = (-len(image)) % alignment
        image.extend(b"\x00" * padding)
        offset = len(image)
        dtype, data = _deployment_bytes(constant)
        image.extend(data)
        metadata.append(PackedInitializer(
            tensor_id=tensor_ids[constant.name],
            name=constant.name,
            offset=offset,
            byte_size=len(data),
            dtype=dtype,
            shape=constant.shape,
            sha256=sha256(data).hexdigest(),
        ))

    packed = PackedWeights(bytes(image), tuple(metadata), alignment)
    validate_packed_weights(packed)
    return packed


def validate_packed_weights(packed: PackedWeights) -> None:
    """Validate alignment, ranges, ordering, and per-tensor checksums."""

    previous_end = 0
    previous_key = (-1, "")
    for initializer in packed.initializers:
        key = (initializer.tensor_id, initializer.name)
        if key <= previous_key:
            raise ValueError("packed initializers are not in deterministic tensor order")
        if initializer.offset % packed.alignment:
            raise ValueError(f"initializer '{initializer.name}' offset is not aligned")
        if initializer.offset < previous_end:
            raise ValueError(f"initializer '{initializer.name}' overlaps its predecessor")
        end = initializer.offset + initializer.byte_size
        if end > len(packed.data):
            raise ValueError(f"initializer '{initializer.name}' exceeds weight image")
        tensor_data = packed.data[initializer.offset:end]
        if sha256(tensor_data).hexdigest() != initializer.sha256:
            raise ValueError(f"initializer '{initializer.name}' checksum mismatch")
        if any(packed.data[previous_end:initializer.offset]):
            raise ValueError("weight image padding must contain only zero bytes")
        previous_end = end
        previous_key = key
