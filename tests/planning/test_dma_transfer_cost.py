"""Runtime DMA service charged during virtual allocation."""

from dataclasses import replace

from maps.graph import Tensor, TensorDType
from maps.hardware import DMARuntimeCost
from maps.planning.mapping import TensorRange, TensorSlice, TensorSubSlice
from maps.planning.transitions.contracts import (
    VirtualIntermediateTransition, VirtualTransfer,
)
from maps.planning.transitions.dma import (
    CopyGeometry, dma_job_count, intermediate_copy_costs,
    virtual_communication_cycles,
)
from maps.target.magia_v3 import build_mesh


def test_strided_2d_copy_uses_one_descriptor() -> None:
    source = CopyGeometry((4, 8), (32, 2))
    packed = CopyGeometry((4, 8), (16, 2))
    assert dma_job_count(source, packed, 2, to_packed=True) == 1
    assert dma_job_count(packed, source, 2) == 1


def test_irregular_4d_copy_counts_fallback_blocks() -> None:
    source = CopyGeometry((2, 3, 4, 5), (150, 50, 10, 2))
    packed = CopyGeometry((2, 3, 4, 5), (120, 40, 10, 2))
    # The last two dimensions coalesce, so the producer uses one 3D job.
    assert dma_job_count(source, packed, 2, to_packed=True) == 1
    # Four non-coalescing dimensions force one fallback job per inner run.
    source = CopyGeometry((2, 3, 4, 5), (800, 200, 30, 2))
    assert dma_job_count(source, packed, 2, to_packed=True) == 24


def test_equal_bytes_with_more_transfers_cost_more_on_producer_and_receiver() -> None:
    mesh = build_mesh(width=2, height=1)
    tensor = Tensor("value", 1, (16,), 2, dtype=TensorDType.FLOAT16)
    parent = TensorSlice(1, (TensorRange(0, 16),))

    def transfer(start: int, length: int) -> VirtualTransfer:
        subslice = TensorSubSlice(parent, (TensorRange(start, length),))
        return VirtualTransfer(0, 1, subslice, subslice)

    def score(transfers: tuple[VirtualTransfer, ...]) -> dict[int, dict[int, int]]:
        transition = VirtualIntermediateTransition(tensor, 0, 0, 1, transfers)
        return virtual_communication_cycles(mesh, (transition,), {0: (0,), 1: (1,)})

    one = score((transfer(0, 16),))
    two = score((transfer(0, 8), transfer(8, 8)))
    assert two[0][0] - one[0][0] == 2_001
    assert two[1][1] - one[1][1] == 2_001


def test_fifo_unpack_is_charged_to_consumer() -> None:
    mesh = build_mesh(width=2, height=1)
    source = CopyGeometry((16,), (2,))
    destination = CopyGeometry((16,), (4,))
    producer, consumer = intermediate_copy_costs(mesh, source, destination, 2, 128)
    assert producer == 2_000
    assert consumer > 2_000
    direct_mesh = replace(mesh, dma_runtime_cost=DMARuntimeCost(submission_cycles=2_000))
    producer, consumer = intermediate_copy_costs(direct_mesh, source, source, 2, 128)
    assert producer == 2_000
    assert consumer == 0
