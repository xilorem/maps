"""Softmax high-level op plus decomposition into primitive MAPS ops."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import WorkKind
from MAPS.core.graph import Node, OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.common.payload import CompositeOpPayload
from MAPS.ops.defs.collective import AllReducePayload
from MAPS.ops.defs.elementwise import BinaryElementwisePayload, UnaryElementwisePayload
from MAPS.ops.defs.reduction import ReductionPayload
from MAPS.ops.registry import register_op
from MAPS.ops.spec import OpSpec


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


def lower_softmax_onnx(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, SoftmaxPayload]:
    """Lower one ONNX Softmax node into one high-level MAPS softmax op."""

    if len(inputs) != 1:
        raise ValueError(f"Softmax node '{node_name}' must have exactly 1 input")
    if len(outputs) != 1:
        raise ValueError(f"Softmax node '{node_name}' must have exactly 1 output")

    x = inputs[0]
    output = outputs[0]
    axis = int(attributes.get("axis", -1))
    if axis < 0:
        axis += x.rank
    return OpKind.CUSTOM, SoftmaxPayload(x=x, output=output, axis=axis)


def decompose_softmax_node(node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
    """Lower one high-level softmax node into grouped primitive planner nodes."""

    if not isinstance(node.payload, SoftmaxPayload):
        raise TypeError("decompose_softmax_node expects a Node with SoftmaxPayload payload")

    op = node.payload
    x = op.x
    output = op.output
    axis = op.axis

    max_attributes = {
        **node.attributes,
        "stage_group_id": f"{node.name}::softmax:max",
    }
    normalization_attributes = {
        **node.attributes,
        "stage_group_id": f"{node.name}::softmax:normalize",
    }

    max_local = _reduced_tensor(f"{node.name}__max_local", x, axis)
    collective_axis = _collective_axis_for_softmax(x, axis)
    max_value = max_local
    new_tensors: list[Tensor] = [max_local]
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
            attributes={**max_attributes, "softmax_step": "reduce_max"},
        )
    ]

    if collective_axis is not None:
        max_global = _same_shape_tensor(f"{node.name}__max_global", max_local)
        new_tensors.append(max_global)
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
                    collective_axis=collective_axis,
                ),
                attributes={**max_attributes, "softmax_step": "allreduce_max"},
            )
        )
        max_value = max_global

    shifted = _same_shape_tensor(f"{node.name}__shifted", x)
    exp = _same_shape_tensor(f"{node.name}__exp", x)
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
                attributes={**normalization_attributes, "softmax_step": "sub"},
            ),
            Node(
                name=f"{node.name}__exp",
                kind=OpKind.ELEMENTWISE,
                inputs=(shifted,),
                outputs=(exp,),
                payload=UnaryElementwisePayload(
                    op_name="Exp",
                    x=shifted,
                    output=exp,
                    work_kind=WorkKind.EXP,
                ),
                attributes={**normalization_attributes, "softmax_step": "exp"},
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
                attributes={**normalization_attributes, "softmax_step": "reduce_sum"},
            ),
        )
    )

    sum_value = sum_local
    if collective_axis is not None:
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
                    collective_axis=collective_axis,
                ),
                attributes={
                    **normalization_attributes,
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
            attributes={**normalization_attributes, "softmax_step": "div"},
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


def _collective_axis_for_softmax(tensor: Tensor, axis: int) -> str | None:
    if axis == tensor.rank - 1:
        return "x"
    if tensor.rank >= 2 and axis == tensor.rank - 2:
        return "y"
    return None


register_op(
    OpSpec(
        name="softmax",
        onnx_names=("Softmax",),
        lower_onnx=lower_softmax_onnx,
        work_kinds=(
            WorkKind.REDUCE_MAX,
            WorkKind.EXP,
            WorkKind.REDUCE_SUM,
            WorkKind.SUB,
            WorkKind.DIV,
        ),
    )
)
