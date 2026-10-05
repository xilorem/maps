Allocation communication cache measurements, 2026-10-05.

These historical measurements cover the first cache-only change. Follow-up candidate-analysis measurements are in [allocation_candidate_scaling.md](allocation_candidate_scaling.md).

The workload is the existing MobileViT nodes 170–179 ONNX slice (eight formed stages), using the MAGIA-v3 target, two token slots, and unit latency/communication weights. Each measurement runs in a fresh process, sequentially on the same machine, without a profiler. Time covers `allocate()` only; model import, rewrites, specialization, mesh construction, placement, and code generation are excluded. Peak RSS covers the entire Python process up to the end of allocation, including preparation. These are single-run observations, not averaged benchmarks.

| Mesh | Tiles | Baseline seconds | Cached seconds | Speedup | Baseline peak MiB | Cached peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4×4 | 16 | 1.29 | 0.32 | 4.08× | 57.7 | 58.7 |
| 8×8 | 64 | 14.04 | 3.17 | 4.44× | 69.3 | 79.2 |
| 16×16 | 256 | 33.69 | 14.18 | 2.38× | 84.1 | 116.7 |
| 32×32 | 1,024 | 127.72 | 78.03 | 1.64× | 149.2 | 265.0 |

Baseline and cached runs select identical stage tile counts and logical shapes at every mesh size, and their final per-tile communication cycles match exactly. Raw results, including those fingerprints, are in [allocation_scaling_results.jsonl](allocation_scaling_results.jsonl).

The cache is local to one allocation invocation. It retains resident input demands per candidate, boundary costs per candidate, and per-tile communication costs per connected candidate pair. Pairwise transfer lists are discarded after costing. Bandwidth-only scoring retains the original aggregation and rounding across tensors; DMA scoring retains descriptor overhead and receiver unpack service. Candidate generation, count/layout neighborhoods, objective ordering, and tile exchanges remain unchanged.

Caching substantially reduces repeated work, but growth is still steep: increasing tiles from 64 to 256 increases cached time 4.48×, and increasing from 256 to 1,024 increases it 5.50×. Larger trial layouts and previously unseen candidate pairs still need analysis. Gains depend on graph topology and candidate reuse; the table does not establish linear scaling or a universal speedup. The additional cache also increases peak memory, particularly on larger meshes.

A separate cProfile run of the cached 16×16 case identifies candidate analysis as the largest remaining cost: 22.19 of 41.64 profiled allocation seconds (about 53%). Communication evaluation accounts for about 46%, including previously unseen candidate-pair transfers and DMA scoring. These profiled times include instrumentation overhead and are excluded from the table. The next optimization target is per-tile layout/work/L1 analysis in `StageCandidateAnalyzer._analyze`; statistics are in [allocation_cached_profile.txt](allocation_cached_profile.txt).

Validation: 494 tests passed, three skipped with the project virtual environment on PATH. Dedicated cache tests compare every per-tile cost across multiple layouts, fused stages, multi-tensor edges, fan-out, graph I/O, initializers, direct DMA, and packed DMA; they also verify unchanged allocation results and reuse of unaffected edges.

Baseline commit: `a932552049fb49ea0bccae31faabf69dea832c43`. Python: 3.13.11. CPU: Intel Core i7-10700 at 2.90 GHz. Model SHA-256: `e84ac346f3f5cefb7e8829dff559e107b8f163d9b9897cbd57f8ba0c05be169d`.

Reproduce from the MAPS checkout, with the sibling `maps-experiments` workload available:

```sh
mkdir -p /tmp/maps-cache-baseline
git archive a932552049fb49ea0bccae31faabf69dea832c43 maps | tar -x -C /tmp/maps-cache-baseline
.venv/bin/python benchmarks/allocation_scaling.py --side 16 --checkout /tmp/maps-cache-baseline
.venv/bin/python benchmarks/allocation_scaling.py --side 16
```

Repeat with `--side 4`, `8`, or `32`. Pass `--model PATH` for another ONNX workload. The last output line is a JSON measurement. `--uncached` disables communication caching in the modified checkout; the published baseline measurements above instead use the original commit.
