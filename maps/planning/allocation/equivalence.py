"""Conservative equivalence keys for representative tile analysis.

Only the exact built-in model/device implementations below may share compute
costs. Unknown implementations can depend on coordinates or arbitrary work
fields and therefore retain per-tile evaluation. L1 equivalence instead follows
its allocation contract: shapes and relative input positions, grouped by Tensor
identity, determine the resident bounds and aligned allocation sequence.
"""

from functools import cache

from maps.hardware import WorkKind


@cache
def _builtin_models():
    from maps.operations.cast import CastCostModel
    from maps.operations.collective import AllReduceCostModel
    from maps.operations.convolution import Conv2DCostModel
    from maps.operations.convolution_transforms import ConvTransformCostModel
    from maps.operations.elementwise import ElementwiseCostModel
    from maps.operations.gemm import GemmCostModel
    from maps.operations.reduction import ReductionCostModel
    from maps.hardware.device import DMADevice, MatrixDevice, ScalarDevice, SystolicDevice, VectorDevice
    from maps.target.magia.devices import SpatzDevice
    from maps.target.magia_v3.devices import MagiaV3SpatzDevice

    return (
        frozenset((CastCostModel, Conv2DCostModel, ElementwiseCostModel, GemmCostModel, ReductionCostModel)),
        {
            DMADevice: "amount", ScalarDevice: "amount", VectorDevice: "amount",
            MatrixDevice: "dimensions", SystolicDevice: "dimensions",
            SpatzDevice: "streams", MagiaV3SpatzDevice: "calibrated",
        },
        AllReduceCostModel,
        ConvTransformCostModel,
    )


def compute_equivalence_key(model, work, tile, device, inputs, outputs):
    models, devices, collective, transform = _builtin_models()
    if type(model) is collective:
        return ()  # Local collective work is zero; group latency stays separate.
    if type(model) is transform:
        return (tile.memory.bandwidth, sum(ref.num_bytes for ref in inputs + outputs))
    if type(model) not in models or type(device) not in devices:
        return None
    kind = getattr(work, "work_kind", None)
    if kind is None:
        return None
    base = (id(device), kind)
    mode = devices[type(device)]
    if mode == "dimensions":
        return base + (tuple(work.dimensions()),)
    amount = work.operation_count()
    if mode == "amount":
        return base + (amount,)
    if mode == "calibrated" and kind in device.kernel_startup_cycles:
        if amount == 0:
            return base + (0,)
        if kind is WorkKind.GEMM:
            return base + (amount, tuple(work.dimensions()))
        if kind in (WorkKind.MUL, WorkKind.SUB, WorkKind.DIV):
            return base + (amount, device._broadcast_geometry(work))
        if kind is WorkKind.REDUCE_SUM:
            return base + (amount, outputs[0].tensor_slice.num_elements)
        return base + (amount,)
    # Uncalibrated Spatz work depends on ordered stream sizes and output dtype.
    return base + (
        amount, outputs[0].tensor.elem_bytes,
        tuple(ref.num_bytes for ref in inputs + outputs),
    )


def l1_equivalence_key(work_slices):
    input_ids = [id(ref.tensor) for inputs, _ in work_slices for ref in inputs]
    if len(set(input_ids)) == len(input_ids):
        # One read per tensor needs no bounding geometry: only allocation bytes.
        return (0, tuple(
            (tuple((id(ref.tensor), ref.num_bytes) for ref in inputs),
             tuple((id(ref.tensor), ref.num_bytes) for ref in outputs))
            for inputs, outputs in work_slices
        ))
    origins = {}
    key = []
    for inputs, outputs in work_slices:
        input_key = []
        for ref in inputs:
            identity = id(ref.tensor)
            dims = ref.tensor_slice.dims
            origin = origins.setdefault(identity, tuple(dim.start for dim in dims))
            input_key.append((identity, tuple(
                (dim.start - start, dim.length) for dim, start in zip(dims, origin)
            )))
        key.append((tuple(input_key), tuple(
            (id(ref.tensor), tuple(dim.length for dim in ref.tensor_slice.dims))
            for ref in outputs
        )))
    return (1, tuple(key))
