from hashlib import sha256

import numpy as np
import pytest

from MAPS.arch import L2Memory, Mesh
from MAPS.core import Constant, ConstantStore, Tensor, TensorDType
from maps.deployment.weights import pack_weights
from maps.planning import ExecutionPlan
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _mesh(l2_size: int = 4096) -> Mesh:
    return Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=l2_size, bandwidth=1),
        noc=rectangular_test_noc(1, 1),
        tiles=rectangular_test_tiles(1, 1),
    )


def test_pack_weights_orders_by_execution_plan_id_preserves_dtypes_and_aligns() -> None:
    later = Tensor("later", 1, (2,), 4, True, TensorDType.FLOAT32)
    first = Tensor("first", 1, (3,), 2, True, TensorDType.FLOAT16)
    execution_plan = ExecutionPlan("model", _mesh(), tensors=(first, later))
    constants = ConstantStore((
        Constant("later", TensorDType.FLOAT32, (2,), np.array([4.0, 5.0], dtype="<f4").tobytes()),
        Constant("first", TensorDType.FLOAT16, (3,), np.array([1.0, 2.0, 3.0], dtype="<f2").tobytes()),
    ))

    packed = pack_weights(execution_plan, constants)

    assert [item.name for item in packed.initializers] == ["first", "later"]
    assert [item.offset for item in packed.initializers] == [0, 16]
    assert packed.data[6:16] == bytes(10)
    assert np.frombuffer(packed.data[16:24], dtype="<f4").tolist() == [4.0, 5.0]
    assert packed.initializers[0].dtype is TensorDType.FLOAT16
    assert packed.initializers[1].dtype is TensorDType.FLOAT32
    assert packed.initializers[1].sha256 == sha256(packed.data[16:24]).hexdigest()


def test_pack_weights_rejects_constant_without_execution_plan_tensor_id() -> None:
    execution_plan = ExecutionPlan("model", _mesh())
    constants = ConstantStore((
        Constant("missing", TensorDType.UINT8, (1,), b"\x01"),
    ))

    with pytest.raises(ValueError, match="no Execution Plan tensor ID"):
        pack_weights(execution_plan, constants)
