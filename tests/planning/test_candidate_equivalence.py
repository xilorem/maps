"""Representative facts must match independent evaluation of every tile."""

from dataclasses import dataclass, replace

import pytest

from maps.graph import Graph, Node, OpKind, Tensor, TensorDType
from maps.hardware import WorkSignature
from maps.operations.contracts import TileWork
from maps.operations.elementwise import BinaryElementwisePayload
from maps.operations.gemm import GemmPayload
from maps.planning.allocation import candidates as candidates_module
from maps.planning.allocation import selection
from maps.planning.allocation.candidates import (
    StageCandidateAnalyzer, StageTileFacts, assigned_device_name,
    logical_shape_options, permanent_l1_allocation_for_tile_work,
    representative_connected_submesh, resolve_stage_layouts, _stage_collective_groups,
)
from maps.planning.allocation.equivalence import l1_equivalence_key
from maps.planning.mapping import TensorRange, TensorSlice, TensorSliceRef
from maps.planning.stage_latency import estimate_stage_latency
from maps.target.magia_v3 import build_mesh
from tests.planning.test_stage_candidates import (
    _CountingUnaryPayload, _FixedCostPayload, _collective_mesh,
    _mesh, _scratch_stage_nodes, _unary_node,
)


def _reference_facts(nodes, mesh, initializers, count, shape, slots):
    submesh = representative_connected_submesh(mesh, 0, count)
    layouts = resolve_stage_layouts(nodes, submesh, shape)
    names = tuple(assigned_device_name(node, mesh.tiles) for node in nodes)
    works = tuple(tuple(node.payload.build_tile_work(layout, tile) for tile in submesh.tiles)
                  for node, layout in zip(nodes, layouts))
    facts = []
    for index, tile in enumerate(submesh.tiles):
        facts.append(StageTileFacts(
            tile_id=tile.tile_id,
            local_cycles=sum(
                node.payload.cost_model.cost(work[index], tile, tile.device_by_name(name))
                + int(node.payload.cost_model.placement_cost(node=node, output_layouts=layout))
                for node, name, work, layout in zip(nodes, names, works, layouts)
            ),
            permanent_l1_bytes=permanent_l1_allocation_for_tile_work(
                tuple(work[index] for work in works), initializers, slots,
            ),
            scratch_l1_bytes=max(tile.device_by_name(name).temporary_l1_bytes(WorkSignature.from_node(node))
                                 for node, name in zip(nodes, names)),
        ))
    if any(fact.total_l1_bytes > tile.memory.size for fact, tile in zip(facts, submesh.tiles)):
        return None
    latency = estimate_stage_latency(
        stage_nodes=nodes, node_output_layouts=layouts, virtual_tiles=submesh.tiles,
        device_names=names, virtual_collective_groups=_stage_collective_groups(nodes, layouts),
        node_tile_work=works,
    )
    return tuple(facts), latency


def _assert_full_analysis(nodes, mesh, initializers, counts, slots=2):
    analyzer = StageCandidateAnalyzer({0: nodes}, mesh, initializers, slots)
    for count in counts:
        expected = {
            shape: result for shape in logical_shape_options(count)
            if (result := _reference_facts(nodes, mesh, initializers, count, shape, slots)) is not None
        }
        actual = {candidate.plan.logical_shape: (candidate.tile_facts, candidate.stage_latency)
                  for candidate in analyzer.candidates(0, count)}
        assert actual == expected


@pytest.mark.parametrize("slots", (1, 3))
@pytest.mark.parametrize("transpose", (False, True))
def test_uneven_batched_gemm_with_bias_and_empty_tiles_matches_full_analysis(slots, transpose):
    x = Tensor("x", 3, (2, 7, 5), 2, dtype=TensorDType.FLOAT16)
    w = Tensor("w", 3, (2, 11, 5) if transpose else (2, 5, 11), 2, dtype=TensorDType.FLOAT16)
    bias = Tensor("bias", 1, (11,), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("out", 3, (2, 7, 11), 2, dtype=TensorDType.FLOAT16)
    node = Node("gemm", OpKind.GEMM, (x, w, bias), (output,),
                payload=GemmPayload(x, w, bias, output, transpose_w=transpose))
    mesh = build_mesh(width=8, height=8)
    _assert_full_analysis((node,), mesh, frozenset((w, bias)), (1, 2, 3, 4, 6, 8, 12, 16, 64), slots)


@pytest.mark.parametrize("op", ("mul", "sub", "div"))
def test_broadcast_geometry_and_odd_vector_lengths_match_full_analysis(op):
    x = Tensor("x", 3, (3, 7, 11), 2, dtype=TensorDType.FLOAT16)
    rhs = Tensor("rhs", 3, (1, 7, 1), 2, dtype=TensorDType.FLOAT16)
    output = replace(x, name="out")
    node = Node(op, OpKind.ELEMENTWISE, (x, rhs), (output,),
                payload=BinaryElementwisePayload(op, x, rhs, output))
    _assert_full_analysis((node,), build_mesh(width=8, height=8), frozenset(), (1, 2, 3, 4, 6, 8, 16, 64))


def test_coordinate_dependent_custom_costs_and_heterogeneous_devices_are_not_merged():
    node = _unary_node("coordinate", 16)
    node = replace(node, payload=_FixedCostPayload(node.inputs[0], node.outputs[0], (1, 9, 3, 17), 5))
    mesh = _mesh(4)
    _assert_full_analysis((node,), mesh, frozenset(), (1, 2, 4))
    ordinary = _unary_node("heterogeneous", 16)
    tiles = tuple(replace(tile, devices=tuple(
        replace(device, startup_cycles=device.startup_cycles + tile.tile_id * 7)
        for device in tile.devices
    )) for tile in mesh.tiles)
    _assert_full_analysis((ordinary,), replace(mesh, tiles=tiles), frozenset(), (1, 2, 4))


def test_fused_stage_scratch_and_memory_feasibility_match_full_analysis():
    nodes = _scratch_stage_nodes()
    for size in (95, 96, 128):
        _assert_full_analysis(nodes, _collective_mesh(l1_size=size), frozenset(), (1, 2))


@dataclass(frozen=True)
class _SliceWork(TileWork):
    inputs: tuple[TensorSliceRef, ...]
    outputs: tuple[TensorSliceRef, ...]

    @property
    def input_slices(self):
        return self.inputs

    @property
    def output_slices(self):
        return self.outputs


def test_l1_equivalence_preserves_gaps_between_reads_of_the_same_tensor():
    tensor = Tensor("shared", 1, (64,), 2, dtype=TensorDType.FLOAT16)

    def work(shift, gap):
        return _SliceWork(tuple(TensorSliceRef(tensor, TensorSlice(1, (TensorRange(start, 4),)))
                                for start in (shift, shift + gap)), ())

    first, translated, wider = work(0, 8), work(7, 8), work(7, 9)
    key = lambda item: l1_equivalence_key(((item.input_slices, item.output_slices),))
    assert key(first) == key(translated)
    assert key(first) != key(wider)
    memory = lambda item: permanent_l1_allocation_for_tile_work((item,), frozenset())
    assert memory(first) == memory(translated) == 48
    assert memory(wider) == 52


def test_equivalent_tiles_share_costs_and_communication_reuses_work(monkeypatch):
    node = _unary_node("regular", 64)
    node = replace(node, payload=_CountingUnaryPayload("Relu", node.inputs[0], node.outputs[0]))
    graph = Graph("regular", nodes=(node,), tensors=node.inputs + node.outputs,
                  inputs=node.inputs, outputs=node.outputs)
    mesh = _mesh(16)
    calls = {"compute": 0, "l1": 0}
    original_cost = candidates_module._node_cost
    original_l1 = candidates_module.permanent_l1_allocation_for_tile_work

    def counted_cost(*args):
        calls["compute"] += 1
        return original_cost(*args)

    def counted_l1(*args):
        calls["l1"] += 1
        return original_l1(*args)

    monkeypatch.setattr(candidates_module, "_node_cost", counted_cost)
    monkeypatch.setattr(candidates_module, "permanent_l1_allocation_for_tile_work", counted_l1)
    _CountingUnaryPayload.build_calls = 0
    analyzer = StageCandidateAnalyzer({0: (node,)}, mesh, frozenset())
    candidates = analyzer.candidates(0, 16)
    assert calls == {"compute": 5, "l1": 5}  # Five shapes, instead of 80 tile evaluations.
    assert _CountingUnaryPayload.build_calls == 80
    cache = selection.AllocationCommunicationCache(graph, mesh, {0: candidates[0].plan})
    for candidate in candidates:
        cached = selection.evaluate_candidate_selection({0: candidate}, mesh, 1, 1, graph, cache)
        before = _CountingUnaryPayload.build_calls
        full = selection.evaluate_candidate_selection({0: candidate}, mesh, 1, 1, graph)
        assert cached == full
        assert _CountingUnaryPayload.build_calls == before + 16
    assert _CountingUnaryPayload.build_calls == 160


def test_custom_dimension_work_without_a_kind_retains_device_evaluation():
    from maps.operations.gemm import GemmCostModel
    from maps.planning.allocation.equivalence import compute_equivalence_key

    class DimensionWork(_SliceWork):
        def dimensions(self):
            return [1, 4, 4, 4]

    work = DimensionWork((), ())
    tile = build_mesh(width=1, height=1).tiles[0]
    device = tile.device_by_name("redmule")
    model = GemmCostModel()
    # A custom TileWork may satisfy the matrix device's dimensions contract
    # without declaring the extra attributes used by other devices.
    assert model.cost(work, tile, device) >= 0
    assert compute_equivalence_key(model, work, tile, device, (), ()) is None
