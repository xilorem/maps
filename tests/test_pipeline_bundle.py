import json

import numpy as np
import pytest

from MAPS.arch import L2Memory, Mesh
from MAPS.core import Constant, ConstantStore, Graph, Tensor, TensorDType
from MAPS.deployment import (
    DeploymentBundle,
    validate_pipeline_bundle_files,
    write_pipeline_bundle,
)
from MAPS.pipeline import Pipeline
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _bundle(l2_size: int = 4096) -> DeploymentBundle:
    weight = Tensor("weight", 2, (2, 2), 4, True, TensorDType.FLOAT32)
    graph = Graph("model", tensors=(weight,), initializers=(weight,))
    pipeline = Pipeline("model", _mesh(l2_size), tensors=(weight,))
    constants = ConstantStore((Constant(
        "weight",
        TensorDType.FLOAT32,
        (2, 2),
        np.array([[1.0, -2.0], [3.5, 4.0]], dtype="<f4").tobytes(),
    ),))
    return DeploymentBundle(pipeline, graph, constants)


def _mesh(l2_size: int) -> Mesh:
    return Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=l2_size, bandwidth=1),
        noc=rectangular_test_noc(1, 1),
        tiles=rectangular_test_tiles(1, 1),
    )


def test_write_pipeline_bundle_is_deterministic_and_preserves_fp32(tmp_path) -> None:
    first_json, first_weights = write_pipeline_bundle(
        _bundle(), tmp_path / "first" / "model.json", tmp_path / "first" / "model.weights.bin"
    )
    second_json, second_weights = write_pipeline_bundle(
        _bundle(), tmp_path / "second" / "model.json", tmp_path / "second" / "model.weights.bin"
    )

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_weights.read_bytes() == second_weights.read_bytes()
    payload = json.loads(first_json.read_text())
    initializer = payload["tensors"][0]["initializer"]
    assert payload["bundle"]["schema_version"] == 1
    assert initializer["dtype"] == "float32"
    assert initializer["shape"] == [2, 2]
    start = initializer["offset"]
    end = start + initializer["byte_size"]
    values = np.frombuffer(first_weights.read_bytes()[start:end], dtype="<f4")
    np.testing.assert_array_equal(values, np.array([1.0, -2.0, 3.5, 4.0], dtype="<f4"))


def test_write_pipeline_bundle_rejects_l2_capacity_failure(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires 16 L2 bytes"):
        write_pipeline_bundle(
            _bundle(l2_size=15), tmp_path / "model.json", tmp_path / "model.weights.bin"
        )


def test_serialized_bundle_validation_detects_weight_corruption(tmp_path) -> None:
    json_path, weights_path = write_pipeline_bundle(
        _bundle(), tmp_path / "model.json", tmp_path / "model.weights.bin"
    )
    corrupted = bytearray(weights_path.read_bytes())
    corrupted[0] ^= 0xFF
    weights_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_pipeline_bundle_files(json_path, weights_path)
