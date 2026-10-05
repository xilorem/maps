"""Measure allocation alone; run each mesh in a fresh process for comparable RSS.

Example:
  .venv/bin/python benchmarks/allocation_scaling.py --side 16
  .venv/bin/python benchmarks/allocation_scaling.py --side 16 --uncached
  .venv/bin/python benchmarks/allocation_scaling.py --side 16 --unpruned

Use --checkout to benchmark an unmodified checkout with this same script.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[2] / "maps-experiments/workloads/mobilevit_170-179/model/mobilevit-nodes-170-179.onnx")
    parser.add_argument("--uncached", action="store_true")
    parser.add_argument("--unpruned", action="store_true", help="disable safe trial pruning")
    parser.add_argument("--token-slots", type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))

    from maps.graph import import_onnx_model, run_graph_rewrites_with_effects
    from maps.planning.allocation import selection
    from maps.planning.stages import form_stages
    from maps.target import SpecializationOptions, magia_v3

    mesh = magia_v3.build_mesh(width=args.side, height=args.side)
    rewritten, _ = run_graph_rewrites_with_effects(import_onnx_model(args.model))
    graph = magia_v3.specialize(
        rewritten, mesh, SpecializationOptions(enable_precision_lowering=False),
    ).model.graph
    stages = form_stages(graph)
    if args.uncached:
        selection.AllocationCommunicationCache.cycles = (
            lambda self, plans, reject=None: selection._virtual_communication_cycles(graph, mesh, plans)
        )
    if args.unpruned:
        original_evaluate = selection.evaluate_candidate_selection

        def unpruned_evaluate(*values, **kwargs):
            kwargs.pop("objective_limit", None)
            return original_evaluate(*values, **kwargs)

        selection.evaluate_candidate_selection = unpruned_evaluate
    caches = []
    original_cache = selection.AllocationCommunicationCache

    class CapturingCache(original_cache):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            caches.append(self)

    selection.AllocationCommunicationCache = CapturingCache
    analyzers = []
    original_analyzer = selection.StageCandidateAnalyzer

    class CapturingAnalyzer(original_analyzer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            analyzers.append(self)

    selection.StageCandidateAnalyzer = CapturingAnalyzer
    start = time.perf_counter()
    plans = selection.allocate(graph, mesh, stages, num_token_slots=args.token_slots)
    elapsed = time.perf_counter() - start
    allocation_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    communication = selection._virtual_communication_cycles(graph, mesh, plans)
    digest = hashlib.sha256()
    for (stage, count), candidates in sorted(analyzers[0]._cache.items()):
        digest.update(json.dumps((stage, count, tuple(
            (candidate.plan.logical_shape, candidate.stage_latency, tuple(
                (fact.tile_id, fact.local_cycles, fact.permanent_l1_bytes, fact.scratch_l1_bytes)
                for fact in candidate.tile_facts
            )) for candidate in candidates
        )), separators=(",", ":")).encode())
    print(json.dumps({
        "mesh_side": args.side, "tiles": mesh.num_tiles, "stages": len(stages),
        "seconds": elapsed, "peak_rss_mib": allocation_peak,
        "candidate_facts_sha256": digest.hexdigest(),
        "analyzed_stage_counts": len(analyzers[0]._cache),
        "pruned_selections": sum(getattr(cache, "pruned_selections", 0) for cache in caches),
        "completed_selections": (
            sum(cache.completed_selections for cache in caches)
            if all(hasattr(cache, "completed_selections") for cache in caches) else None
        ),
        "priced_edges": sum(getattr(cache, "priced_edges", len(cache.edges)) for cache in caches),
        "plans": {stage: {"tiles": plan.tile_count, "shape": plan.logical_shape}
                  for stage, plan in plans.items()},
        "communication": communication,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
