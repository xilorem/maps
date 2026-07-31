from dataclasses import FrozenInstanceError

import pytest

from maps.hardware import (
    DeviceKind,
    FixedDeviceAssignment,
    L1Memory,
    ScalarDevice,
    Tile,
    WorkKind,
    WorkSignature,
)
from maps.graph import Node, OpKind, Tensor, TensorDType
from maps.operations.gemm import GemmPayload
from MAPS.hw.devices.redmule import REDMULE_DEVICE
from MAPS.hw.devices.spatz import SPATZ_DEVICE


def _tensor(
    name: str,
    dtype: TensorDType,
    dims: tuple[int, ...] = (2, 2),
) -> Tensor:
    return Tensor(
        name=name,
        rank=len(dims),
        dims=dims,
        elem_bytes=2 if dtype is TensorDType.FLOAT16 else 4,
        dtype=dtype,
    )


def _gemm_node(dtype: TensorDType, *, with_bias: bool) -> Node:
    x = _tensor("x", dtype)
    weight = _tensor("weight", dtype)
    output = _tensor("output", dtype)
    bias = _tensor("bias", dtype, (2,)) if with_bias else None
    inputs = (x, weight) + ((bias,) if bias is not None else ())
    return Node(
        name="gemm",
        kind=OpKind.GEMM,
        inputs=inputs,
        outputs=(output,),
        payload=GemmPayload(x=x, w=weight, y=bias, output=output),
    )


def test_work_signature_preserves_node_operand_order_and_optional_bias() -> None:
    without_bias = WorkSignature.from_node(
        _gemm_node(TensorDType.FLOAT16, with_bias=False)
    )
    with_bias = WorkSignature.from_node(
        _gemm_node(TensorDType.FLOAT16, with_bias=True)
    )

    assert without_bias == WorkSignature(
        work_kind=WorkKind.GEMM,
        input_dtypes=(TensorDType.FLOAT16, TensorDType.FLOAT16),
        output_dtypes=(TensorDType.FLOAT16,),
    )
    assert with_bias.input_dtypes == (
        TensorDType.FLOAT16,
        TensorDType.FLOAT16,
        TensorDType.FLOAT16,
    )


def test_work_signature_is_immutable() -> None:
    signature = WorkSignature.from_node(
        _gemm_node(TensorDType.FLOAT32, with_bias=False)
    )

    with pytest.raises(FrozenInstanceError):
        setattr(signature, "work_kind", WorkKind.ADD)


def _signature(
    work_kind: WorkKind,
    inputs: tuple[TensorDType, ...],
    outputs: tuple[TensorDType, ...],
) -> WorkSignature:
    return WorkSignature(
        work_kind=work_kind,
        input_dtypes=inputs,
        output_dtypes=outputs,
    )


def test_typed_capability_distinguishes_same_width_types_from_throughput() -> None:
    fp32_add = _signature(
        WorkKind.ADD,
        (TensorDType.FLOAT32, TensorDType.FLOAT32),
        (TensorDType.FLOAT32,),
    )
    int32_add = _signature(
        WorkKind.ADD,
        (TensorDType.INT32, TensorDType.INT32),
        (TensorDType.INT32,),
    )
    device = ScalarDevice(
        name="typed_core",
        kind=DeviceKind.SCALAR,
        throughput={WorkKind.ELEMENTWISE: 1},
        capabilities=frozenset({fp32_add}),
    )

    assert device.supports(fp32_add)
    assert not device.supports(int32_add)


def test_redmule_declares_only_fp16_gemm_with_optional_bias() -> None:
    fp16_gemm = _signature(
        WorkKind.GEMM,
        (TensorDType.FLOAT16, TensorDType.FLOAT16),
        (TensorDType.FLOAT16,),
    )
    fp16_gemm_with_bias = _signature(
        WorkKind.GEMM,
        (TensorDType.FLOAT16,) * 3,
        (TensorDType.FLOAT16,),
    )
    fp32_gemm = _signature(
        WorkKind.GEMM,
        (TensorDType.FLOAT32, TensorDType.FLOAT32),
        (TensorDType.FLOAT32,),
    )
    fp16_conv = _signature(
        WorkKind.CONV2D,
        (TensorDType.FLOAT16, TensorDType.FLOAT16),
        (TensorDType.FLOAT16,),
    )

    assert REDMULE_DEVICE.supports(fp16_gemm)
    assert REDMULE_DEVICE.supports(fp16_gemm_with_bias)
    assert not REDMULE_DEVICE.supports(fp32_gemm)
    assert not REDMULE_DEVICE.supports(fp16_conv)


def test_spatz_capabilities_are_explicit_for_fp16_and_fp32() -> None:
    fp16_relu = _signature(
        WorkKind.RELU,
        (TensorDType.FLOAT16,),
        (TensorDType.FLOAT16,),
    )
    fp32_relu = _signature(
        WorkKind.RELU,
        (TensorDType.FLOAT32,),
        (TensorDType.FLOAT32,),
    )
    int32_relu = _signature(
        WorkKind.RELU,
        (TensorDType.INT32,),
        (TensorDType.INT32,),
    )
    fp16_to_fp32 = _signature(
        WorkKind.CAST,
        (TensorDType.FLOAT16,),
        (TensorDType.FLOAT32,),
    )
    fp32_to_fp16 = _signature(
        WorkKind.CAST,
        (TensorDType.FLOAT32,),
        (TensorDType.FLOAT16,),
    )

    assert SPATZ_DEVICE.supports(fp16_relu)
    assert SPATZ_DEVICE.supports(fp32_relu)
    assert SPATZ_DEVICE.supports(fp16_to_fp32)
    assert SPATZ_DEVICE.supports(fp32_to_fp16)
    assert not SPATZ_DEVICE.supports(int32_relu)


def _capable_core(name: str, signature: WorkSignature) -> ScalarDevice:
    return ScalarDevice(
        name=name,
        kind=DeviceKind.SCALAR,
        throughput={signature.work_kind: 1},
        capabilities=frozenset({signature}),
    )


def _tile(
    devices: tuple[ScalarDevice, ...],
    assignment: FixedDeviceAssignment = FixedDeviceAssignment(),
) -> Tile:
    return Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=1),
        devices=devices,
        device_assignment=assignment,
    )


def test_tile_retains_valid_fixed_device_assignment() -> None:
    signature = _signature(
        WorkKind.ADD,
        (TensorDType.FLOAT16, TensorDType.FLOAT16),
        (TensorDType.FLOAT16,),
    )
    core = _capable_core("core", signature)
    assignment = FixedDeviceAssignment({signature: "core"})

    tile = _tile((core,), assignment)

    assert tile.device_assignment.assignments[signature] == "core"
    assert tile.assigned_device(signature) is core


def test_tile_rejects_duplicate_device_names() -> None:
    signature = _signature(
        WorkKind.RELU,
        (TensorDType.FLOAT16,),
        (TensorDType.FLOAT16,),
    )

    with pytest.raises(ValueError, match="duplicate device name: core"):
        _tile((_capable_core("core", signature), _capable_core("core", signature)))


def test_tile_rejects_assignment_to_missing_device() -> None:
    signature = _signature(
        WorkKind.RELU,
        (TensorDType.FLOAT16,),
        (TensorDType.FLOAT16,),
    )

    with pytest.raises(ValueError, match="unknown device missing"):
        _tile(
            (_capable_core("core", signature),),
            FixedDeviceAssignment({signature: "missing"}),
        )


def test_tile_rejects_assignment_to_incapable_device() -> None:
    fp16_relu = _signature(
        WorkKind.RELU,
        (TensorDType.FLOAT16,),
        (TensorDType.FLOAT16,),
    )
    fp32_relu = _signature(
        WorkKind.RELU,
        (TensorDType.FLOAT32,),
        (TensorDType.FLOAT32,),
    )

    with pytest.raises(ValueError, match="does not declare capability"):
        _tile(
            (_capable_core("core", fp16_relu),),
            FixedDeviceAssignment({fp32_relu: "core"}),
        )
