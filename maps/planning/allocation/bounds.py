"""Admissible communication floors without constructing pairwise transfers."""

from maps.planning.mapping import TensorLayout
from maps.hardware import DMARuntimeCost
from maps.planning.stages import node_output_index, node_output_layouts


def _ceil_div(amount: int, divisor: int) -> int:
    return (amount + divisor - 1) // divisor


class TransferLowerBounds:
    """Bound outgoing total service and per-destination unpack service.

    Canonical layouts cover the whole tensor domain, so every non-empty demand
    inside that domain needs at least one producer copy. Copies may split across
    sources or duplicate replicated data; both can only increase these floors.
    Legacy byte-only pricing aggregates tensors before rounding, so its floor
    aggregates demanded bytes by destination tile too.
    """

    def __init__(self, mesh, compiler):
        self.mesh = mesh
        self.compiler = compiler
        runtime = mesh.dma_runtime_cost
        tensors = tuple(tensor for node in compiler.graph.nodes for tensor in node.inputs + node.outputs)
        self.supported = type(runtime) is DMARuntimeCost and all(type(value) is int for value in (
            runtime.submission_cycles, runtime.setup_cycles, runtime.publication_cycles,
            mesh.l2_memory.bandwidth,
            *(tile.memory.bandwidth for tile in mesh.tiles),
            *(tensor.elem_bytes for tensor in tensors),
            *(size for tensor in tensors for size in tensor.dims),
        ))
        self.maximum_bandwidth = max(tile.memory.bandwidth for tile in mesh.tiles)
        self.runtime_pricing = any((runtime.submission_cycles, runtime.setup_cycles, runtime.publication_cycles))
        self._floors = {}

    def edge(self, source, destination):
        """Return a sender total floor and sparse receiver tile floors."""
        key = (id(source), id(destination))
        if key in self._floors:
            return self._floors[key]
        runtime = self.mesh.dma_runtime_cost
        total = 0
        receiver = {}
        legacy_bytes = {}
        for tensor, destinations in self.compiler.demands(destination):
            producer = self.compiler.producers.get(id(tensor))
            if producer is None or self.compiler.stage_ids[id(producer)] != source.stage_id:
                continue
            layout = node_output_layouts(source, producer)[node_output_index(producer, tensor)]
            # Unknown layout implementations retain an uninformative zero floor.
            if type(layout) is not TensorLayout:
                continue
            layout.validate_for(tensor)
            for tile, slice_ in destinations:
                if slice_.rank != tensor.rank or any(
                    type(dim.start) is not int or type(dim.length) is not int for dim in slice_.dims
                ):
                    continue
                covered = 1
                for dim, size in zip(slice_.dims, tensor.dims):
                    covered *= max(0, min(dim.start + dim.length, size) - max(dim.start, 0))
                byte_count = covered * tensor.elem_bytes
                if not byte_count:
                    continue
                if not self.runtime_pricing:
                    legacy_bytes[tile.tile_id] = legacy_bytes.get(tile.tile_id, 0) + byte_count
                    continue
                bandwidth = min(self.maximum_bandwidth, tile.memory.bandwidth)
                total += _ceil_div(byte_count, bandwidth) + runtime.overhead(1, publish=True)
                if runtime.packed_intermediates:
                    receiver[tile.tile_id] = receiver.get(tile.tile_id, 0) + (
                        _ceil_div(byte_count, tile.memory.bandwidth) + runtime.overhead(1)
                    )
        if not self.runtime_pricing:
            total = sum(
                _ceil_div(amount, min(self.maximum_bandwidth, self.mesh.tile_by_id(tile).memory.bandwidth))
                for tile, amount in legacy_bytes.items()
            )
        self._floors[key] = (total, receiver)
        return total, receiver
