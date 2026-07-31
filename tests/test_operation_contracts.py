from dataclasses import dataclass

import pytest

from MAPS.arch import WorkKind
from MAPS.core.graph import OpKind
from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.core.tensor import Tensor
from MAPS.ops.common import CompositeOpPayload, OperationPayload
from MAPS.ops.defs.conv import ConvPayload
from MAPS.ops.defs.elementwise import BinaryElementwisePayload, UnaryElementwisePayload
from MAPS.ops.defs.gemm import GemmPayload, lower_gemm_node, lower_matmul_node
from MAPS.ops.registry import get_op, register_op
from MAPS.ops.spec import OpSpec


def _tensor(name: str, dims: tuple[int, ...]) -> Tensor:
    return Tensor(name=name, rank=len(dims), dims=dims, elem_bytes=2)


def test_binary_elementwise_requires_exact_broadcast_result() -> None:
    lhs = _tensor("lhs", (1,))
    rhs = _tensor("rhs", (1,))

    with pytest.raises(ValueError, match="broadcast result"):
        BinaryElementwisePayload(
            op_name="Add",
            lhs=lhs,
            rhs=rhs,
            output=_tensor("output", (8,)),
        )


def test_elementwise_payload_derives_and_validates_work_kind() -> None:
    x = _tensor("x", (4, 8))
    output = _tensor("output", (4, 8))

    payload = UnaryElementwisePayload(op_name="Exp", x=x, output=output)
    assert payload.work_kind is WorkKind.EXP

    with pytest.raises(ValueError, match="must use work kind EXP"):
        UnaryElementwisePayload(
            op_name="Exp",
            x=x,
            output=output,
            work_kind=WorkKind.LOG,
        )


def test_gemm_bias_broadcasts_to_the_owned_output_slice() -> None:
    payload = GemmPayload(
        x=_tensor("x", (4, 6)),
        w=_tensor("w", (6, 8)),
        y=_tensor("bias", (8,)),
        output=_tensor("output", (4, 8)),
    )
    output_slice = TensorSlice(
        rank=2,
        dims=(TensorRange(0, 4), TensorRange(4, 4)),
    )

    assert payload.required_y_slice(output_slice) == TensorSlice(
        rank=1,
        dims=(TensorRange(4, 4),),
    )


@pytest.mark.parametrize(
    ("attribute", "value"),
    (("alpha", 0.5), ("beta", 0.0), ("transA", 1)),
)
def test_onnx_gemm_rejects_unrepresented_attributes(attribute: str, value: object) -> None:
    inputs = (_tensor("x", (4, 6)), _tensor("w", (6, 8)))
    outputs = (_tensor("output", (4, 8)),)

    with pytest.raises(NotImplementedError, match=attribute):
        lower_gemm_node("gemm", inputs, outputs, {attribute: value})


def test_onnx_gemm_represents_transposed_weight_storage() -> None:
    inputs = (_tensor("x", (4, 6)), _tensor("w", (8, 6)))
    outputs = (_tensor("output", (4, 8)),)

    _, payload = lower_gemm_node("gemm", inputs, outputs, {"transB": 1})

    assert payload.transpose_w
    output_slice = TensorSlice(
        rank=2,
        dims=(TensorRange(0, 4), TensorRange(2, 3)),
    )
    assert payload.required_w_slice(output_slice) == TensorSlice(
        rank=2,
        dims=(TensorRange(2, 3), TensorRange(0, 6)),
    )


def test_onnx_matmul_rejects_batch_broadcasting_explicitly() -> None:
    inputs = (_tensor("x", (2, 4, 6)), _tensor("w", (1, 6, 8)))
    outputs = (_tensor("output", (2, 4, 8)),)

    with pytest.raises(NotImplementedError, match="broadcasted batch dimensions"):
        lower_matmul_node("matmul", inputs, outputs, {})


def test_conv_is_only_a_composite_contract() -> None:
    payload = ConvPayload(
        x=_tensor("x", (1, 3, 5, 5)),
        w=_tensor("w", (8, 3, 3, 3)),
        b=None,
        output=_tensor("output", (1, 8, 3, 3)),
    )

    assert isinstance(payload, CompositeOpPayload)
    assert not hasattr(payload, "cost_model")
    assert not hasattr(payload, "build_tile_work")


@dataclass(frozen=True)
class _RegistryPayload(OperationPayload):
    pass


def _lower_registry_payload(node_name, inputs, outputs, attributes):
    del node_name, inputs, outputs, attributes
    return OpKind.CUSTOM, _RegistryPayload()


def test_registry_rejects_collisions_without_partial_registration() -> None:
    # Explicit Graph converters reserve their external names from the registry.
    with pytest.raises(ValueError, match="duplicate ONNX op mapping"):
        register_op(
            OpSpec(
                name="atomic_registration_test",
                onnx_names=("MatMul",),
                lower_onnx=_lower_registry_payload,
            )
        )

    with pytest.raises(ValueError, match="unknown op spec"):
        get_op("atomic_registration_test")
