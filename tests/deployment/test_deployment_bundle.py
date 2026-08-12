from dataclasses import replace
import json

import numpy as np
import pytest

from maps.hardware import L1Memory, L2Memory, Mesh
from maps.graph import Constant, ConstantStore, Graph, Node, OpKind, Tensor, TensorDType
from maps.graph import ImportedModel
from maps.planning.mapping import TensorRange, TensorSlice
from maps.planning.mapping import Submesh
from maps.deployment import (
    DeploymentBundle,
    PackedInitializer,
    PackedWeights,
    build_deployment_bundle,
    pack_weights,
    validate_deployment_bundle,
    write_deployment_bundle,
    write_execution_plan,
)
from maps.deployment.serialization import (
    execution_plan_payload,
)
from maps.planning import ExecutionPlan, Layer, LayerInput, Stage
from maps.operations.elementwise import UnaryElementwisePayload
from maps.planning.transitions import InputDestination
from maps.target import SpecializationResult
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def test_deployment_owns_bundle_packing_and_serialization() -> None:
    assert DeploymentBundle.__module__ == "maps.deployment.bundle"
    assert build_deployment_bundle.__module__ == "maps.deployment.bundle"
    assert PackedInitializer.__module__ == "maps.deployment.weights"
    assert PackedWeights.__module__ == "maps.deployment.weights"
    assert pack_weights.__module__ == "maps.deployment.weights"
    assert execution_plan_payload.__module__ == "maps.deployment.serialization"
    assert write_execution_plan.__module__ == "maps.deployment.serialization"


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
    return build_deployment_bundle(
        SpecializationResult(ImportedModel(graph, constants)),
        execution_plan,
    )


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


def test_write_deployment_bundle_is_deterministic_and_preserves_fp32(
    tmp_path,
) -> None:
    first_bundle, first_initializers = write_deployment_bundle(
        _bundle(), tmp_path / "first" / "model.bundle.json", tmp_path / "first" / "model.initializers.bin"
    )
    second_bundle, second_initializers = write_deployment_bundle(
        _bundle(), tmp_path / "second" / "model.bundle.json", tmp_path / "second" / "model.initializers.bin"
    )

    assert first_bundle.read_bytes() == second_bundle.read_bytes()
    assert first_initializers.read_bytes() == second_initializers.read_bytes()
    payload = json.loads(first_bundle.read_text())
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
    values = np.frombuffer(first_initializers.read_bytes()[start:end], dtype="<f4")
    np.testing.assert_array_equal(values, np.array([1.0, -2.0, 3.5, 4.0], dtype="<f4"))


def test_plain_execution_plan_is_distinct_from_deployment_bundle(tmp_path) -> None:
    bundle = _bundle()
    plan_path = write_execution_plan(bundle.execution_plan, tmp_path / "model.plan.json")
    bundle_path, _ = write_deployment_bundle(
        bundle,
        tmp_path / "model.bundle.json",
        tmp_path / "model.initializers.bin",
    )

    plan_payload = json.loads(plan_path.read_text())
    bundle_payload = json.loads(bundle_path.read_text())

    assert "bundle" not in plan_payload
    assert "provenance" not in plan_payload
    assert "initializer" not in plan_payload["tensors"][0]
    assert bundle_payload["bundle"]["alignment"] == 16
    assert bundle_payload["provenance"] == {"rewrite_report": []}
    assert bundle_payload["tensors"][0]["initializer"]["sha256"]


def test_write_deployment_bundle_rejects_l2_capacity_failure(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires 16 L2 bytes"):
        write_deployment_bundle(
            _bundle(l2_size=15), tmp_path / "model.bundle.json", tmp_path / "model.initializers.bin"
        )


def test_serialized_bundle_validation_detects_weight_corruption(tmp_path) -> None:
    bundle_path, initializers_path = write_deployment_bundle(
        _bundle(), tmp_path / "model.bundle.json", tmp_path / "model.initializers.bin"
    )
    corrupted = bytearray(initializers_path.read_bytes())
    corrupted[0] ^= 0xFF
    initializers_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_deployment_bundle(bundle_path, initializers_path)


def test_serialized_bundle_validation_enforces_l2_capacity(tmp_path) -> None:
    bundle_path, initializers_path = write_deployment_bundle(
        _bundle(), tmp_path / "model.bundle.json", tmp_path / "model.initializers.bin"
    )

    with pytest.raises(ValueError, match="exceeds L2 capacity"):
        validate_deployment_bundle(bundle_path, initializers_path, l2_capacity=15)


def test_write_deployment_bundle_rejects_initializer_tensor_dtype_mismatch(
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
        write_deployment_bundle(
            replace(bundle, execution_plan=execution_plan),
            tmp_path / "model.bundle.json",
            tmp_path / "model.initializers.bin",
        )
