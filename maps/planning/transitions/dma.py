"""Descriptor-aware costs for virtual runtime transfers.

Copy geometry follows the SDK's contiguous, 2D, 3D and block-copy paths.
Starts affect addresses, not descriptor counts; strides come from the resident
parent Slice, rather than from the copied SubSlice's lengths.
"""

from dataclasses import dataclass
from math import gcd, prod

from maps.hardware import Mesh
from maps.planning.mapping import TensorSlice, TensorSubSlice
from .contracts import (
    VirtualInputTransition,
    VirtualIntermediateTransition,
    VirtualOutputTransition,
    VirtualTransition,
)


@dataclass(frozen=True)
class CopyGeometry:
    lengths: tuple[int, ...]
    strides: tuple[int, ...]

    @property
    def elements(self) -> int:
        return prod(self.lengths)

    def squeezed(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (length, stride)
            for length, stride in zip(self.lengths, self.strides)
            if length != 1
        )

    def normalized(self) -> tuple[tuple[int, int], ...]:
        dimensions: list[tuple[int, int]] = []
        for length, stride in self.squeezed():
            if dimensions and dimensions[-1][1] == length * stride:
                outer_length, _ = dimensions.pop()
                dimensions.append((outer_length * length, stride))
            else:
                dimensions.append((length, stride))
        return tuple(dimensions)


def packed_geometry(lengths: tuple[int, ...], element_bytes: int) -> CopyGeometry:
    strides = []
    stride = element_bytes
    for length in reversed(lengths):
        strides.append(stride)
        stride *= length
    return CopyGeometry(lengths, tuple(reversed(strides)))


def subslice_geometry(subslice: TensorSubSlice, element_bytes: int) -> CopyGeometry:
    return CopyGeometry(
        tuple(dim.length for dim in subslice.dims),
        packed_geometry(
            tuple(dim.length for dim in subslice.parent.dims), element_bytes
        ).strides,
    )


def slice_geometry(
    tensor_dims: tuple[int, ...], tensor_slice: TensorSlice, element_bytes: int
) -> CopyGeometry:
    return CopyGeometry(
        tuple(dim.length for dim in tensor_slice.dims),
        packed_geometry(tensor_dims, element_bytes).strides,
    )


def dma_job_count(
    source: CopyGeometry,
    destination: CopyGeometry,
    element_bytes: int,
    *,
    to_packed: bool = False,
) -> int:
    """Count hardware submissions, including the SDK's blocking fallback.

    FIFO's asynchronous producer path first attempts submit_to_packed. If its
    normalized geometry is unsupported, it falls back to the blocking mover.
    """
    if source.elements == 0:
        return 0
    source_normalized = source.normalized()
    destination_normalized = destination.normalized()
    if to_packed and (
        len(source_normalized) <= 2
        or (len(source_normalized) == 3 and source_normalized[-1][1] == element_bytes)
    ):
        return 1

    source_squeezed = source.squeezed()
    destination_squeezed = destination.squeezed()
    source_rank = len(source_squeezed)
    destination_rank = len(destination_squeezed)
    same_lengths = tuple(d[0] for d in source_squeezed) == tuple(
        d[0] for d in destination_squeezed
    )
    if source_rank == destination_rank == 3 and same_lengths:
        if (
            source_squeezed[-1][1] == element_bytes
            and destination_squeezed[-1][1] == element_bytes
        ):
            return 1
    if source_rank == destination_rank == 2 and same_lengths:
        if (
            source_squeezed[-1][1] != element_bytes
            or destination_squeezed[-1][1] != element_bytes
        ):
            return 1
    for shaped, linear in (
        (source_squeezed, destination_squeezed),
        (destination_squeezed, source_squeezed),
    ):
        if len(linear) == 1 and linear[0][1] == element_bytes:
            if len(shaped) == 3 and shaped[-1][1] == element_bytes:
                return 1
            if len(shaped) == 2 and shaped[-1][1] != element_bytes:
                return 1

    source_rank = len(source_normalized)
    destination_rank = len(destination_normalized)
    if source_rank == destination_rank == 1:
        if (
            source_normalized[0][1] == element_bytes
            and destination_normalized[0][1] == element_bytes
        ):
            return 1
    if source_rank == destination_rank == 2:
        if (
            source_normalized[-1][1] == element_bytes
            and destination_normalized[-1][1] == element_bytes
        ):
            return 1
    for shaped, linear in (
        (source_normalized, destination_normalized),
        (destination_normalized, source_normalized),
    ):
        if len(shaped) == 2 and len(linear) == 1:
            if shaped[-1][1] == element_bytes and linear[0][1] == element_bytes:
                return 1

    source_run = (
        source_normalized[-1][0]
        if source_normalized and source_normalized[-1][1] == element_bytes else 1
    )
    destination_run = (
        destination_normalized[-1][0]
        if destination_normalized and destination_normalized[-1][1] == element_bytes else 1
    )
    return source.elements // gcd(source_run, destination_run)


def intermediate_copy_costs(
    mesh: Mesh, source: CopyGeometry, destination: CopyGeometry,
    element_bytes: int, destination_bandwidth: int,
) -> tuple[int, int]:
    """Return producer overhead and consumer unpack service for one transfer.

    The caller prices network payload separately. FIFO unpack adds local payload
    service on the destination DMA engine, not a second network transfer.
    """
    runtime = mesh.dma_runtime_cost
    if not runtime.packed_intermediates:
        jobs = dma_job_count(source, destination, element_bytes)
        return runtime.overhead(jobs, publish=True), 0
    packed = packed_geometry(source.lengths, element_bytes)
    send_jobs = dma_job_count(source, packed, element_bytes, to_packed=True)
    unpack_jobs = dma_job_count(packed, destination, element_bytes)
    unpack_bytes = source.elements * element_bytes
    unpack_cycles = (
        (unpack_bytes + destination_bandwidth - 1) // destination_bandwidth
        + runtime.overhead(unpack_jobs)
    ) if unpack_jobs else 0
    return runtime.overhead(send_jobs, publish=True), unpack_cycles


def virtual_communication_cycles(
    mesh: Mesh, transitions: tuple[VirtualTransition, ...],
    tile_ids: dict[int, tuple[int, ...]],
) -> dict[int, dict[int, int]]:
    """Accumulate each tile's runtime communication service independently."""
    cycles = {stage: dict.fromkeys(ids, 0) for stage, ids in tile_ids.items()}
    runtime = mesh.dma_runtime_cost
    for transition in transitions:
        element_bytes = transition.tensor.elem_bytes
        if isinstance(transition, VirtualIntermediateTransition):
            for transfer in transition.transfers:
                source_tile = mesh.tile_by_id(transfer.source_virtual_tile_id)
                destination_tile = mesh.tile_by_id(transfer.destination_virtual_tile_id)
                source = subslice_geometry(transfer.source_subslice, element_bytes)
                destination = subslice_geometry(transfer.destination_subslice, element_bytes)
                byte_count = source.elements * element_bytes
                bandwidth = min(source_tile.memory.bandwidth, destination_tile.memory.bandwidth)
                overhead, unpack = intermediate_copy_costs(
                    mesh, source, destination, element_bytes,
                    destination_tile.memory.bandwidth,
                )
                cycles[transition.source_stage_id][source_tile.tile_id] += (
                    (byte_count + bandwidth - 1) // bandwidth + overhead
                )
                cycles[transition.destination_stage_id][destination_tile.tile_id] += unpack
        elif isinstance(transition, VirtualInputTransition):
            for destination in transition.destinations:
                source = slice_geometry(transition.tensor.dims, destination.tensor_slice, element_bytes)
                target = packed_geometry(source.lengths, element_bytes)
                tile = mesh.tile_by_id(destination.virtual_tile_id)
                bandwidth = min(tile.memory.bandwidth, mesh.l2_memory.bandwidth)
                byte_count = source.elements * element_bytes
                cycles[transition.destination_stage_id][tile.tile_id] += (
                    (byte_count + bandwidth - 1) // bandwidth
                    + runtime.overhead(dma_job_count(source, target, element_bytes))
                )
        elif isinstance(transition, VirtualOutputTransition):
            for source_ref in transition.sources:
                target = slice_geometry(transition.tensor.dims, source_ref.tensor_slice, element_bytes)
                source = packed_geometry(target.lengths, element_bytes)
                tile = mesh.tile_by_id(source_ref.virtual_tile_id)
                bandwidth = min(tile.memory.bandwidth, mesh.l2_memory.bandwidth)
                byte_count = source.elements * element_bytes
                cycles[transition.source_stage_id][tile.tile_id] += (
                    (byte_count + bandwidth - 1) // bandwidth
                    + runtime.overhead(dma_job_count(source, target, element_bytes))
                )
    return cycles
