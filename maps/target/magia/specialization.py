"""MAGIA-owned Target Specialization policy."""

from __future__ import annotations

from maps.graph import ImportedModel, TensorDType
from maps.hardware import Mesh, WorkKind, WorkSignature
from maps.target.contracts import (
    PrecisionLoweringRecipe,
    RewriteEvent,
    RewriteReport,
    SpecializationOptions,
    SpecializationResult,
)

from .convolution import lower_convolutions
from .devices import REDMULE_DEVICE
from .effects import RewriteEffect
from .precision import precision_lower_model


PRECISION_LOWERING_RECIPES = tuple(
    PrecisionLoweringRecipe(
        source_signature=WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT32,) * input_count,
            (TensorDType.FLOAT32,),
        ),
        target_signature=WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT16,) * input_count,
            (TensorDType.FLOAT16,),
        ),
        device_name=REDMULE_DEVICE.name,
    )
    for input_count in (2, 3)
)


def _events(
    rewrite_name: str,
    effects: tuple[RewriteEffect, ...],
) -> tuple[RewriteEvent, ...]:
    return tuple(
        RewriteEvent(
            rewrite_name=rewrite_name,
            source_node=effect.source_node,
            original_signature=effect.original_signature,
            resulting_signatures=effect.resulting_signatures,
            converted_initializers=effect.converted_initializers,
        )
        for effect in effects
    )


def specialize(
    model: ImportedModel,
    mesh: Mesh,
    options: SpecializationOptions | None = None,
) -> SpecializationResult:
    """Specialize one hardware-independent Imported Model for MAGIA."""

    options = options or SpecializationOptions(enable_precision_lowering=True)
    model.validate()
    convolution = lower_convolutions(model)
    events = list(_events("conv_to_gemm", convolution.effects))
    rewritten = convolution.model
    if options.enable_precision_lowering:
        precision = precision_lower_model(
            rewritten,
            mesh,
            PRECISION_LOWERING_RECIPES,
        )
        rewritten = precision.model
        events.extend(_events("precision_lowering", precision.effects))
    rewritten.validate()
    return SpecializationResult(rewritten, RewriteReport(tuple(events)))


__all__ = ["PRECISION_LOWERING_RECIPES", "specialize"]
