
from dataclasses import dataclass
from math import prod
from typing import Callable

from maps.graph.dtype import TensorDType
from maps.graph.dtype import dtype_elem_bytes
from maps.graph.graph import Graph


@dataclass(frozen=True)
class Constant:
    name: str
    dtype: TensorDType
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constant name must not be empty")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"constant '{self.name}' dimensions must be > 0")
        object.__setattr__(self, "data", bytes(self.data))


@dataclass(frozen=True)
class ConstantStore:
    constants: tuple[Constant, ...]

    def __post_init__(self) -> None:
        names = [constant.name for constant in self.constants]
        if len(names) != len(set(names)):
            raise ValueError("constant names must be unique")

    def get(self, name: str) -> Constant:
        for constant in self.constants:
            if constant.name == name:
                return constant
        raise KeyError(name)

    def replace(self, constant: Constant) -> "ConstantStore":
        if not any(item.name == constant.name for item in self.constants):
            raise KeyError(constant.name)
        return ConstantStore(tuple(
            constant if item.name == constant.name else item
            for item in self.constants
        ))

    def remove(self, name: str) -> "ConstantStore":
        if not any(item.name == name for item in self.constants):
            raise KeyError(name)
        return ConstantStore(tuple(item for item in self.constants if item.name != name))

    def transform(
        self,
        name: str,
        transform: Callable[[Constant], Constant],
    ) -> "ConstantStore":
        return self.replace(transform(self.get(name)))


def validate_constants(graph: Graph, constants: ConstantStore) -> None:
    """Validate that graph initializer metadata and owned bytes agree exactly."""

    initializer_by_name = {tensor.name: tensor for tensor in graph.initializers}
    if len(initializer_by_name) != len(graph.initializers):
        raise ValueError("graph initializer names must be unique")

    constant_by_name = {constant.name: constant for constant in constants.constants}
    missing = sorted(set(initializer_by_name) - set(constant_by_name))
    if missing:
        raise ValueError(f"missing constants for graph initializers: {', '.join(missing)}")
    orphaned = sorted(set(constant_by_name) - set(initializer_by_name))
    if orphaned:
        raise ValueError(f"orphaned constants: {', '.join(orphaned)}")

    for name, tensor in initializer_by_name.items():
        constant = constant_by_name[name]
        if tuple(tensor.dims) != constant.shape:
            raise ValueError(f"constant '{name}' shape does not match graph tensor")
        if tensor.dtype is None:
            raise ValueError(f"initializer tensor '{name}' has no dtype")
        if tensor.dtype is not constant.dtype:
            raise ValueError(f"constant '{name}' dtype does not match graph tensor")
        expected_size = prod(constant.shape) * dtype_elem_bytes(constant.dtype)
        if len(constant.data) != expected_size:
            raise ValueError(
                f"constant '{name}' has {len(constant.data)} bytes; expected {expected_size}"
            )
