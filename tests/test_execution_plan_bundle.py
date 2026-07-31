from dataclasses import replace
import json

import numpy as np
import pytest

from MAPS.arch import L1Memory, L2Memory, Mesh
from MAPS.core import Constant, ConstantStore, Graph, Node, OpKind, Tensor, TensorDType
from maps.planning.layouts import TensorRange, TensorSlice
from maps.planning.submesh import Submesh
from MAPS.deployment import (
    DeploymentBundle,
    validate_execution_plan_bundle_files,
    write_execution_plan_bundle,
)
from maps.planning import ExecutionPlan, Layer, LayerInput, Stage
from maps.operations.elementwise import UnaryElementwisePayload
from maps.planning.transitions import InputDestination
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _bundle(l2_size: int = 4096) -> DeploymentBundle:
    weight = Tensor("weight", 2, (2, 2), 4, True, TensorDType.FLOAT32)
    output = Tensor("output", 2, (2, 2), 4, dtype=TensorDType.FLOAT32)
    node = Node(
        "consume_weight",
        OpKind.ELEMENTWISE,
        inputs=(weight,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Relu", weight, output),
    )
    graph = Graph(
        "model",
        tensors=(weight, output),
        nodes=(node,),
        initializers=(weight,),
    )
    mesh = _mesh(l2_size)
    full_slice = TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=2),
            TensorRange(start=0, length=2),
        ),
    )
    stage = Stage(
        "stage",
        Submesh(mesh, 0, frozenset((0,))),
        layers=(
            Layer(
                node,
                inputs=(
                    LayerInput.initializer(
                        tensor_id=0,
                        destinations=(InputDestination(0, full_slice),),
                    ),
                ),
                device_name="core",
            ),
        ),
    )
    execution_plan = ExecutionPlan(
        "model",
        mesh,
        tensors=(weight,),
        stages=(stage,),
    )
    constants = ConstantStore((Constant(
        "weight",
        TensorDType.FLOAT32,
        (2, 2),
        np.array([[1.0, -2.0], [3.5, 4.0]], dtype="<f4").tobytes(),
    ),))
    return DeploymentBundle(execution_plan, graph, constants)


def _mesh(l2_size: int) -> Mesh:
    return Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=l2_size, bandwidth=1),
        noc=rectangular_test_noc(1, 1),
        tiles=rectangular_test_tiles(
            1,
            1,
            memory=L1Memory(size=4096, bandwidth=1),
        ),
    )


def test_write_execution_plan_bundle_is_deterministic_and_preserves_fp32(
    tmp_path,
) -> None:
    first_json, first_weights = write_execution_plan_bundle(
        _bundle(), tmp_path / "first" / "model.json", tmp_path / "first" / "model.weights.bin"
    )
    second_json, second_weights = write_execution_plan_bundle(
        _bundle(), tmp_path / "second" / "model.json", tmp_path / "second" / "model.weights.bin"
    )

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_weights.read_bytes() == second_weights.read_bytes()
    payload = json.loads(first_json.read_text())
    initializer = payload["tensors"][0]["initializer"]
    assert payload["bundle"]["schema_version"] == 1
    assert initializer["dtype"] == "float32"
    assert initializer["shape"] == [2, 2]
    assert payload["stages"][0]["layers"][0]["inputs"][0]["source"] == {
        "kind": "INITIALIZER",
        "destinations": [
            {
                "tile_id": 0,
                "tensor_slice": {
                    "rank": 2,
                    "dims": [
                        {"start": 0, "length": 2},
                        {"start": 0, "length": 2},
                    ],
                },
            },
        ],
    }
    assert "initializations" not in payload
    assert "finalizations" not in payload
    assert payload["transitions"] == []
    start = initializer["offset"]
    end = start + initializer["byte_size"]
    values = np.frombuffer(first_weights.read_bytes()[start:end], dtype="<f4")
    np.testing.assert_array_equal(values, np.array([1.0, -2.0, 3.5, 4.0], dtype="<f4"))


def test_write_execution_plan_bundle_rejects_l2_capacity_failure(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires 16 L2 bytes"):
        write_execution_plan_bundle(
            _bundle(l2_size=15), tmp_path / "model.json", tmp_path / "model.weights.bin"
        )


def test_serialized_bundle_validation_detects_weight_corruption(tmp_path) -> None:
    json_path, weights_path = write_execution_plan_bundle(
        _bundle(), tmp_path / "model.json", tmp_path / "model.weights.bin"
    )
    corrupted = bytearray(weights_path.read_bytes())
    corrupted[0] ^= 0xFF
    weights_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_execution_plan_bundle_files(json_path, weights_path)


def test_write_execution_plan_bundle_rejects_initializer_tensor_dtype_mismatch(
    tmp_path,
) -> None:
    bundle = _bundle()
    weight = replace(
        bundle.execution_plan.tensors[0],
        elem_bytes=2,
        dtype=TensorDType.FLOAT16,
    )
    execution_plan = replace(bundle.execution_plan, tensors=(weight,))

    with pytest.raises(
        ValueError,
        match="constant 'weight' dtype does not match Execution Plan tensor",
    ):
        write_execution_plan_bundle(
            replace(bundle, execution_plan=execution_plan),
            tmp_path / "model.json",
            tmp_path / "model.weights.bin",
        )
