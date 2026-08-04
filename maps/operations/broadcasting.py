"""Source-independent multidirectional broadcasting behavior."""

from maps.planning.mapping import TensorRange, TensorSlice
from maps.graph import Tensor


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        raise ValueError("at least one shape is required for broadcasting")
    result: list[int] = []
    max_rank = max(len(shape) for shape in shapes)
    padded_shapes = ((1,) * (max_rank - len(shape)) + shape for shape in shapes)
    for dimensions in zip(*padded_shapes):
        non_unit = {dimension for dimension in dimensions if dimension != 1}
        if len(non_unit) > 1:
            raise ValueError(f"shapes are not broadcast-compatible: {shapes}")
        result.append(next(iter(non_unit), 1))
    return tuple(result)


def validate_broadcast_output(
    inputs: tuple[Tensor, ...],
    output: Tensor,
    operation_name: str,
) -> None:
    expected = broadcast_shape(*(tensor.dims for tensor in inputs))
    if output.dims != expected:
        raise ValueError(
            f"{operation_name} output shape must be the broadcast result {expected}, "
            f"got {output.dims}"
        )


def validate_broadcastable_to(
    input_tensor: Tensor,
    output: Tensor,
    operation_name: str,
) -> None:
    if input_tensor.rank > output.rank:
        raise ValueError(f"{operation_name} input rank cannot exceed output rank")
    padded = (1,) * (output.rank - input_tensor.rank) + input_tensor.dims
    if any(
        input_dim not in (1, output_dim)
        for input_dim, output_dim in zip(padded, output.dims)
    ):
        raise ValueError(f"{operation_name} input shape is not broadcastable to output")


def broadcast_input_slice(
    input_tensor: Tensor,
    output: Tensor,
    output_slice: TensorSlice,
    operation_name: str,
) -> TensorSlice:
    validate_broadcastable_to(input_tensor, output, operation_name)
    rank_offset = output.rank - input_tensor.rank
    dims = tuple(
        TensorRange(start=0, length=1)
        if input_dim == 1 and output.dims[input_axis + rank_offset] != 1
        else output_slice.dims[input_axis + rank_offset]
        for input_axis, input_dim in enumerate(input_tensor.dims)
    )
    return TensorSlice(rank=input_tensor.rank, dims=dims)
