from hashlib import sha256
from pathlib import Path

import numpy as np

from examples.mobilevit_oracle import (
    ABSOLUTE_LOGIT_TOLERANCE,
    INPUT_SHA256,
    INPUT_SHAPE,
    MAPS_LOGITS_SHA256,
    RELATIVE_LOGIT_TOLERANCE,
    SDK_LOGITS_SHA256,
    TOP1_CLASS,
    mobilevit_input,
    mobilevit_logits,
)


PROJECT_ROOT = Path(__file__).parents[2]
SOURCE_MODEL = PROJECT_ROOT / "examples/mobilenet.onnx"


def test_mobilevit_oracle_reproduces_the_sdk_reference_input() -> None:
    first = mobilevit_input()
    second = mobilevit_input()

    assert first.shape == INPUT_SHAPE
    assert first.dtype == np.float16
    assert first.tobytes() == second.tobytes()
    assert sha256(first.tobytes()).hexdigest() == INPUT_SHA256


def test_mobilevit_oracle_freezes_logits_and_sdk_tolerances() -> None:
    sdk_logits = mobilevit_logits(SOURCE_MODEL, maps_contract=False)
    maps_logits = mobilevit_logits(SOURCE_MODEL, maps_contract=True)

    assert sdk_logits.shape == maps_logits.shape == (1, 1000)
    assert sdk_logits.dtype == maps_logits.dtype == np.float16
    assert np.isfinite(sdk_logits).all()
    assert np.isfinite(maps_logits).all()
    assert sha256(sdk_logits.tobytes()).hexdigest() == SDK_LOGITS_SHA256
    assert sha256(maps_logits.tobytes()).hexdigest() == MAPS_LOGITS_SHA256
    assert int(np.argmax(sdk_logits)) == int(np.argmax(maps_logits)) == TOP1_CLASS
    np.testing.assert_allclose(
        maps_logits,
        sdk_logits,
        atol=ABSOLUTE_LOGIT_TOLERANCE,
        rtol=RELATIVE_LOGIT_TOLERANCE,
    )
    assert ABSOLUTE_LOGIT_TOLERANCE == 0.03125
    assert RELATIVE_LOGIT_TOLERANCE == 0.025
