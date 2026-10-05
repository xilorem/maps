"""Pruning must use admissible bounds and preserve the complete search result."""

from dataclasses import replace
from itertools import product

import pytest

from maps.hardware import DMARuntimeCost, L1Memory
from maps.planning.allocation import selection
from maps.planning.allocation.candidates import StageCandidateAnalyzer
from tests.planning.test_allocation import _mesh_with_l1
from tests.planning.test_allocation_cache import _cache_graph


RUNTIMES = (
    DMARuntimeCost(),
    DMARuntimeCost(submission_cycles=13, setup_cycles=7, publication_cycles=3),
    DMARuntimeCost(submission_cycles=13, setup_cycles=7, publication_cycles=3,
                   packed_intermediates=True),
)


def _fixture(runtime):
    graph, stages = _cache_graph()
    mesh = _mesh_with_l1(4, 4, 65536)
    mesh = replace(mesh, dma_runtime_cost=runtime, tiles=tuple(
        replace(tile, memory=L1Memory(size=65536, bandwidth=3 + tile.tile_id % 5))
        for tile in mesh.tiles
    ))
    analyzer = StageCandidateAnalyzer(stages, mesh, frozenset(graph.initializers))
    options = {stage: tuple(c for count in (1, 2, 4)
                            for c in analyzer.candidates(stage, count)) for stage in stages}
    seed = {stage: values[0] for stage, values in options.items()}
    return graph, stages, mesh, analyzer, options, seed


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_edge_and_partial_selection_bounds_never_exceed_exact_cost(runtime):
    graph, stages, mesh, analyzer, options, seed = _fixture(runtime)
    plans = {stage: candidate.plan for stage, candidate in seed.items()}
    cache = selection.AllocationCommunicationCache(graph, mesh, plans)
    for source, destination in cache.compiler.edges:
        for src, dst in product(options[source], options[destination]):
            sender, receiver = cache.transfer_bounds.edge(src.plan, dst.plan)
            exact = selection._communication_cycles_for_transitions(
                mesh, {source: src.plan, destination: dst.plan},
                cache.compiler.intermediates(src.plan, dst.plan),
            )
            assert sender <= sum(exact[source].values())
            assert all(cost <= exact[destination][tile] for tile, cost in receiver.items())
    # Keep the same cache to exercise both cold edges and partially warm trials,
    # including simultaneous replacements on both sides of an edge.
    for candidates in product(*(options[stage] for stage in stages)):
        trial = {stage: candidate.plan for stage, candidate in zip(stages, candidates)}
        bounds = []
        actual = cache.cycles(trial, lambda lower: bounds.append(lower) or False)
        expected = selection._virtual_communication_cycles(graph, mesh, trial)
        assert actual == expected
        assert bounds
        for lower in bounds:
            assert all(value <= max(expected[stage].values()) for stage, value in lower.items())


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("weights", ((1, 1), (0, 1), (1, 0), (0.01, 3), (-1, 1)))
def test_pruned_search_matches_unpruned_search(runtime, weights):
    graph, stages, mesh, analyzer, options, seed = _fixture(runtime)
    context = selection.build_allocation_context(graph, stages)
    kwargs = dict(context=context, mesh=mesh, selected_candidates=seed, analyzer=analyzer,
                  stage_latency_weight=weights[0], communication_weight=weights[1], debug=False)
    pruned = selection.grow_stage_candidates(**kwargs)
    full = selection.grow_stage_candidates(**kwargs, prune_candidates=False)
    assert pruned == full


def test_rejected_trial_can_be_evaluated_again_and_ties_prefer_fewer_tiles(monkeypatch):
    graph, stages, mesh, analyzer, options, seed = _fixture(RUNTIMES[2])
    cache = selection.AllocationCommunicationCache(
        graph, mesh, {stage: candidate.plan for stage, candidate in seed.items()},
    )
    kwargs = dict(candidates=seed, mesh=mesh, graph=graph, stage_latency_weight=1,
                  communication_weight=1, communication_cache=cache)
    calls = []
    original = cache.compiler.intermediates

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(cache.compiler, "intermediates", counted)
    assert selection.evaluate_candidate_selection(**kwargs, objective_limit=((0,) * 4, 4)) is None
    assert not calls
    assert not cache.edges
    full = selection.evaluate_candidate_selection(**kwargs)
    assert full is not None and calls
    objective = selection.selection_objective(full.metrics)
    assert selection.evaluate_candidate_selection(**kwargs, objective_limit=(objective, 4)) is None
    # A score tie can still improve the secondary tile-count objective.
    assert selection.evaluate_candidate_selection(**kwargs, objective_limit=(objective, 5)) == full
    assert full == selection.evaluate_candidate_selection(**(kwargs | {"communication_cache": None}))


def test_nonintegral_runtime_costs_disable_bounds():
    graph, stages, mesh, analyzer, options, seed = _fixture(DMARuntimeCost(submission_cycles=0.5))
    cache = selection.AllocationCommunicationCache(graph, mesh,
        {stage: candidate.plan for stage, candidate in seed.items()})
    assert not cache.transfer_bounds.supported
    result = selection.evaluate_candidate_selection(seed, mesh, 1, 1, graph, cache,
                                                     objective_limit=((0,) * 4, 4))
    assert result is not None
    assert cache.pruned_selections == 0


def test_fanout_floor_can_reject_before_any_pairwise_transfer_is_built(monkeypatch):
    graph, stages, mesh, analyzer, options, seed = _fixture(RUNTIMES[2])
    plans = {stage: values[-1].plan for stage, values in options.items()}
    cache = selection.AllocationCommunicationCache(graph, mesh, plans)

    def unexpected(*args):
        pytest.fail("a rejected fan-out trial should not build pairwise transfers")

    monkeypatch.setattr(cache.compiler, "intermediates", unexpected)
    observed = []

    def reject(lower):
        observed.append(lower)
        # Stage 1 has neither graph input nor output traffic. Its positive
        # service floor comes entirely from unpriced cross-stage copies.
        return lower[1] > 0

    assert cache.cycles(plans, reject) is None
    assert observed[0] == dict.fromkeys(plans, 0)
    assert observed[-1][1] > 0
    assert cache.priced_edges == 0
    assert cache.pruned_selections == 1
