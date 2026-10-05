"""Communication caching must preserve complete-plan service and dependency reuse."""

from dataclasses import replace
from itertools import product

import pytest

from maps.graph import Graph
from maps.hardware import DMARuntimeCost, L1Memory, L2Memory
from maps.operations.gemm import GemmPayload
from maps.planning.allocation import selection
from maps.planning.allocation.candidates import StageCandidateAnalyzer
from tests.planning.test_allocation import _gemm_node, _mesh_with_l1


def _cache_graph():
    first = _gemm_node("first", 8, 8, 8)
    second = _gemm_node("second", 8, 8, 8)

    def consumer(name, x, w):
        node = _gemm_node(name, 8, 8, 8)
        return replace(node, inputs=(x, w), payload=GemmPayload(x=x, w=w, y=None, output=node.outputs[0]))

    joined = consumer("joined", first.outputs[0], second.outputs[0])
    final = consumer("final", first.outputs[0], joined.outputs[0])
    isolated = _gemm_node("isolated", 8, 8, 8)
    nodes = (first, second, joined, final, isolated)
    graph = Graph(
        "cache", nodes=nodes,
        tensors=tuple(dict.fromkeys(t for node in nodes for t in node.inputs + node.outputs)),
        inputs=first.inputs + (second.inputs[0],) + isolated.inputs,
        initializers=(second.inputs[1],),
        outputs=first.outputs + final.outputs + isolated.outputs,
    )
    return graph, {0: (first, second), 1: (joined,), 2: (final,), 3: (isolated,)}


@pytest.mark.parametrize("runtime", (
    DMARuntimeCost(),
    DMARuntimeCost(submission_cycles=13, setup_cycles=7, publication_cycles=3),
    DMARuntimeCost(submission_cycles=13, setup_cycles=7, publication_cycles=3, packed_intermediates=True),
))
def test_cached_service_matches_full_rebuild_across_layouts_and_edges(runtime):
    graph, stages = _cache_graph()
    mesh = _mesh_with_l1(4, 4, 65536)
    mesh = replace(
        mesh, dma_runtime_cost=runtime, l2_memory=L2Memory(size=65536, bandwidth=11),
        tiles=tuple(replace(tile, memory=L1Memory(size=65536, bandwidth=7)) for tile in mesh.tiles),
    )
    analyzer = StageCandidateAnalyzer(stages, mesh, frozenset(graph.initializers))
    options = {
        stage: tuple(c for count in (1, 2, 4) for c in analyzer.candidates(stage, count))
        for stage in stages
    }
    seed = {stage: options[stage][0].plan for stage in stages}
    cache = selection.AllocationCommunicationCache(graph, mesh, seed)
    # Exercise multiple tensors on an edge, fan-out, graph I/O, initializers,
    # fused stage demands, shrinking, growth and same-count layout changes.
    for candidates in product(*(options[stage] for stage in stages)):
        plans = {stage: candidate.plan for stage, candidate in zip(stages, candidates)}
        assert cache.cycles(plans) == selection._virtual_communication_cycles(graph, mesh, plans)
    before = len(cache.edges), len(cache.boundaries)
    repeated = cache.cycles(seed)
    repeated[0][0] += 1
    assert cache.cycles(seed) == selection._virtual_communication_cycles(graph, mesh, seed)
    assert (len(cache.edges), len(cache.boundaries)) == before


def test_changing_candidate_reuses_unaffected_edges(monkeypatch):
    graph, stages = _cache_graph()
    mesh = _mesh_with_l1(4, 4, 65536)
    analyzer = StageCandidateAnalyzer(stages, mesh, frozenset(graph.initializers))
    plans = {stage: analyzer.candidate(stage, 1).plan for stage in stages}
    cache = selection.AllocationCommunicationCache(graph, mesh, plans)
    calls = []
    original = cache.compiler.intermediates

    def counted(source, destination):
        calls.append((source.stage_id, destination.stage_id))
        return original(source, destination)

    monkeypatch.setattr(cache.compiler, "intermediates", counted)
    cache.cycles(plans)
    assert calls == [(0, 1), (0, 2), (1, 2)]
    calls.clear()
    cache.cycles(plans | {3: analyzer.candidate(3, 2).plan})
    assert calls == []
    cache.cycles(plans | {0: analyzer.candidate(0, 2).plan})
    assert calls == [(0, 1), (0, 2)]
    calls.clear()
    cache.cycles(plans)
    assert calls == []


@pytest.mark.parametrize("packed", (False, True))
def test_cached_search_selects_the_same_allocation(monkeypatch, packed):
    graph, stages = _cache_graph()
    mesh = replace(_mesh_with_l1(4, 4, 65536), dma_runtime_cost=DMARuntimeCost(
        submission_cycles=13, packed_intermediates=packed,
    ))
    cached = selection.allocate(graph, mesh, stages)
    monkeypatch.setattr(selection.AllocationCommunicationCache, "cycles",
                        lambda self, plans, reject=None: selection._virtual_communication_cycles(graph, mesh, plans))
    assert selection.allocate(graph, mesh, stages) == cached
