"""Logical tensor IR matching the runtime-side `tensor_t`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .dtype import TensorDType, dtype_elem_bytes

if TYPE_CHECKING:
    from MAPS.core.layout import TensorSlice

TENSOR_MAX_DIMS = 6


@dataclass(frozen=True)
class Tensor:
    """Logical tensor metadata only."""

    name: str
    rank: int
    dims: tuple[int, ...]
    elem_bytes: int
    is_initializer: bool = False
    dtype: TensorDType | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if self.rank <= 0 or self.rank > TENSOR_MAX_DIMS:
            raise ValueError(f"rank must be in [1, {TENSOR_MAX_DIMS}]")
        if len(self.dims) != self.rank:
            raise ValueError("dims length must match rank")
        if any(dim <= 0 for dim in self.dims):
            raise ValueError("all tensor dimensions must be > 0")
        if self.elem_bytes <= 0:
            raise ValueError("elem_bytes must be > 0")
        if self.dtype is not None and self.elem_bytes != dtype_elem_bytes(self.dtype):
            raise ValueError("elem_bytes must match dtype")

    @property
    def num_elements(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim
        return total

    def slice_num_bytes(self, tensor_slice: TensorSlice) -> int:
        return tensor_slice.num_elements * self.elem_bytes
