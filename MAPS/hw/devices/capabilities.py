"""Helpers for declaring exact typed Device Capability catalogs."""

from MAPS.arch import WorkKind, WorkSignature
from MAPS.core.dtype import TensorDType


def same_dtype_signatures(
    work_kinds: tuple[WorkKind, ...],
    input_counts: tuple[int, ...],
    dtypes: tuple[TensorDType, ...],
) -> frozenset[WorkSignature]:
    """Build exact signatures whose operands share one stored TensorDType."""

    return frozenset(
        WorkSignature(
            work_kind=work_kind,
            input_dtypes=(dtype,) * input_count,
            output_dtypes=(dtype,),
        )
        for work_kind in work_kinds
        for input_count in input_counts
        for dtype in dtypes
    )


__all__ = ["same_dtype_signatures"]
