"""Hardware-independent graph, tensor, and constant model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import prod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from maps.operations import OperationPayload

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


class OpKind(IntEnum):
    GEMM = 0
    ELEMENTWISE = 1
    REDUCTION = 2
    CONV = 3
    TRANSFORM = 4
    CUSTOM = 255


@dataclass(frozen=True)
class Node:
    """One logical compute node in the graph."""

    name: str
    kind: OpKind
    inputs: tuple[Tensor, ...] = field(default_factory=tuple)
    outputs: tuple[Tensor, ...] = field(default_factory=tuple)
    payload: "OperationPayload | None" = None
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("node name must not be empty")


@dataclass(frozen=True)
class Edge:
    """One tensor dependency between two graph nodes."""

    tensor: Tensor
    src: Node | None
    dst: Node | None

    def __post_init__(self) -> None:
        if self.src is None and self.dst is None:
            raise ValueError("edge must connect at least one endpoint")


@dataclass(frozen=True)
class Graph:
    """One logical graph ready to be consumed by the scheduler."""

    name: str
    tensors: tuple[Tensor, ...] = field(default_factory=tuple)
    nodes: tuple[Node, ...] = field(default_factory=tuple)
    edges: tuple[Edge, ...] = field(default_factory=tuple)
    inputs: tuple[Tensor, ...] = field(default_factory=tuple)
    outputs: tuple[Tensor, ...] = field(default_factory=tuple)
    initializers: tuple[Tensor, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("graph name must not be empty")


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


@dataclass(frozen=True)
class ImportedModel:
    graph: Graph
    constants: ConstantStore

    def validate(self) -> None:
        """Validate that initializer metadata and immutable bytes agree."""

        validate_constants(self.graph, self.constants)


def validate_imported_model(model: ImportedModel) -> None:
    model.validate()


def build_graph_edges_from_nodes(
    nodes: tuple[Node, ...],
    tensors: dict[str, Tensor],
    graph_output_names: tuple[str, ...],
) -> tuple[Edge, ...]:
    """Build explicit graph edges from an already-lowered node sequence."""

    producers: dict[str, Node] = {}
    for node in nodes:
        for tensor in node.outputs:
            if tensor.name in producers:
                raise ValueError(f"tensor '{tensor.name}' has multiple producers")
            producers[tensor.name] = node

    edges: list[Edge] = []

    for node in nodes:
        for tensor in node.inputs:
            edges.append(
                Edge(
                    tensor=tensors[tensor.name],
                    src=producers.get(tensor.name),
                    dst=node,
                )
            )

    for tensor_name in graph_output_names:
        src_node = producers.get(tensor_name)
        if src_node is None:
            continue
        edges.append(Edge(tensor=tensors[tensor_name], src=src_node, dst=None))

    return tuple(edges)
