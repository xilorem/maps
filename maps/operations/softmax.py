"""Softmax semantics and deterministic primitive decomposition."""

from __future__ import annotations

from dataclasses import dataclass

from maps.hardware import WorkKind
from maps.graph import Node, OpKind, Tensor
from .contracts import CompositeOpPayload
from .collective import AllReducePayload
from .elementwise import BinaryElementwisePayload, UnaryElementwisePayload
from .reduction import ReductionPayload


@dataclass(frozen=True)
class SoftmaxPayload(CompositeOpPayload):
    """High-level softmax payload that must be decomposed before planning."""

    x: Tensor
    output: Tensor
    axis: int

    def __post_init__(self) -> None:
        self.validate_shapes()

    def validate_shapes(self) -> None:
        if self.axis < 0 or self.axis >= self.x.rank:
            raise ValueError("Softmax axis must be within input tensor rank")
        if self.x.rank != self.output.rank or self.x.dims != self.output.dims:
            raise ValueError("Softmax input and output shapes must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("Softmax input and output element sizes must match")

    def decompose(self, node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
        return decompose_softmax_node(node)


def decompose_softmax_node(node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
    """Lower one high-level softmax node into grouped primitive planner nodes."""

    if not isinstance(node.payload, SoftmaxPayload):
        raise TypeError("decompose_softmax_node expects a Node with SoftmaxPayload payload")

    op = node.payload
    x = op.x
    output = op.output
    axis = op.axis

    attributes = dict(node.attributes)

    max_local = _reduced_tensor(f"{node.name}__max_local", x, axis)
    max_global = _same_shape_tensor(f"{node.name}__max_global", max_local)
    max_value = max_global
    new_tensors: list[Tensor] = [max_local, max_global]
    nodes: list[Node] = [
        Node(
            name=f"{node.name}__reduce_max",
            kind=OpKind.REDUCTION,
            inputs=(x,),
            outputs=(max_local,),
            payload=ReductionPayload(
                op_name="ReduceMax",
                x=x,
                output=max_local,
                axis=axis,
                work_kind=WorkKind.REDUCE_MAX,
            ),
            attributes={**attributes, "softmax_step": "reduce_max"},
        )
    ]

    nodes.append(
        Node(
            name=f"{node.name}__allreduce_max",
            kind=OpKind.CUSTOM,
            inputs=(max_local,),
            outputs=(max_global,),
            payload=AllReducePayload(
                op_name="AllReduceMax",
                x=max_local,
                output=max_global,
                reduction="max",
            ),
            attributes={**attributes, "softmax_step": "allreduce_max"},
        )
    )

    shifted = _same_shape_tensor(f"{node.name}__shifted", x)
    exp = _same_shape_tensor(f"{node.name}__softmax_exp", x)
    sum_local = _reduced_tensor(f"{node.name}__sum_local", x, axis)
    new_tensors.extend((shifted, exp, sum_local))
    nodes.extend(
        (
            Node(
                name=f"{node.name}__sub",
                kind=OpKind.ELEMENTWISE,
                inputs=(x, max_value),
                outputs=(shifted,),
                payload=BinaryElementwisePayload(
                    op_name="Sub",
                    lhs=x,
                    rhs=max_value,
                    output=shifted,
                    work_kind=WorkKind.SUB,
                ),
                attributes={**attributes, "softmax_step": "sub"},
            ),
            Node(
                name=f"{node.name}__softmax_exp",
                kind=OpKind.ELEMENTWISE,
                inputs=(shifted,),
                outputs=(exp,),
                payload=UnaryElementwisePayload(
                    op_name="SoftmaxExp",
                    x=shifted,
                    output=exp,
                    work_kind=WorkKind.SOFTMAX_EXP,
                ),
                attributes={**attributes, "softmax_step": "exp"},
            ),
            Node(
                name=f"{node.name}__reduce_sum",
                kind=OpKind.REDUCTION,
                inputs=(exp,),
                outputs=(sum_local,),
                payload=ReductionPayload(
                    op_name="ReduceSum",
                    x=exp,
                    output=sum_local,
                    axis=axis,
                    work_kind=WorkKind.REDUCE_SUM,
                ),
                attributes={**attributes, "softmax_step": "reduce_sum"},
            ),
        )
    )

    sum_global = _same_shape_tensor(f"{node.name}__sum_global", sum_local)
    new_tensors.append(sum_global)
    nodes.append(
        Node(
            name=f"{node.name}__allreduce_sum",
            kind=OpKind.CUSTOM,
            inputs=(sum_local,),
            outputs=(sum_global,),
            payload=AllReducePayload(
                op_name="AllReduceSum",
                x=sum_local,
                output=sum_global,
                reduction="sum",
            ),
            attributes={
                **attributes,
                "softmax_step": "allreduce_sum",
            },
        )
    )
    sum_value = sum_global

    nodes.append(
        Node(
            name=f"{node.name}__div",
            kind=OpKind.ELEMENTWISE,
            inputs=(exp, sum_value),
            outputs=(output,),
            payload=BinaryElementwisePayload(
                op_name="Div",
                lhs=exp,
                rhs=sum_value,
                output=output,
                work_kind=WorkKind.DIV,
            ),
            attributes={**attributes, "softmax_step": "div"},
        )
    )

    return tuple(new_tensors), tuple(nodes)


def _same_shape_tensor(name: str, reference: Tensor) -> Tensor:
    return Tensor(
        name=name,
        rank=reference.rank,
        dims=reference.dims,
        elem_bytes=reference.elem_bytes,
        dtype=reference.dtype,
    )


def _reduced_tensor(name: str, reference: Tensor, axis: int) -> Tensor:
    dims = list(reference.dims)
    dims[axis] = 1
    return Tensor(
        name=name,
        rank=reference.rank,
        dims=tuple(dims),
        elem_bytes=reference.elem_bytes,
        dtype=reference.dtype,
    )
