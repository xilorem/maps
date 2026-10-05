Safe trial pruning measurements, 2026-10-05.

This follow-up compares [representative candidate analysis](allocation_candidate_scaling.md) with the same allocator plus admissible communication bounds. It uses the same eight-stage MobileViT nodes 170–179 workload, MAGIA-v3, two token slots, unit weights, Python 3.13.11, and Intel i7-10700. The model SHA-256 is `e84ac346f3f5cefb7e8829dff559e107b8f163d9b9897cbd57f8ba0c05be169d`. Each measurement runs allocation alone in a fresh process, sequentially, without profiling or concurrent tests. These are single-run observations rather than statistical estimates.

| Mesh | Tiles | Before seconds | Pruned seconds | Additional speedup | Before peak MiB | Pruned peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4×4 | 16 | 0.267 | 0.131 | 2.04× | 59.1 | 56.5 |
| 8×8 | 64 | 2.555 | 1.411 | 1.81× | 79.6 | 66.6 |
| 16×16 | 256 | 10.909 | 6.676 | 1.63× | 115.9 | 94.8 |
| 32×32 | 1,024 | 56.104 | 41.065 | 1.37× | 258.0 | 199.9 |

| Mesh | Exact edge pairs before | Exact edge pairs after | Rejected trial evaluations | Completed trial evaluations after |
| --- | ---: | ---: | ---: | ---: |
| 4×4 | 657 | 54 | 1,007 | 15 |
| 8×8 | 2,115 | 260 | 5,281 | 30 |
| 16×16 | 3,175 | 360 | 6,846 | 30 |
| 32×32 | 4,499 | 456 | 8,844 | 30 |

Rejected counts include repeat trials because partial evaluations are never memoized as complete results. Exact edge pairs count cache misses that construct and price pairwise transfer lists. The preceding implementation did not record completed evaluations; its raw counter is null. All final stage tile counts and logical layouts, every per-tile communication cost, and fingerprints of all analyzed candidate facts match. Analyzed stage/count entries remain 72, 177, 225, and 273. Raw data: [allocation_pruning_scaling_results.jsonl](allocation_pruning_scaling_results.jsonl).

The search first compares intrinsic latency against the current best objective. If necessary, it adds exact graph-boundary and already-cached edge costs. For an unseen edge, each positive destination demand inside the tensor domain requires at least one producer copy. Canonical layouts cover that domain; partitioning or replication cannot reduce the required bytes or minimum one-copy DMA overhead. Maximum source bandwidth gives a conservative payload floor, and packed intermediates additionally have a per-destination unpack floor. Legacy bandwidth-only pricing aggregates bytes before rounding, including across tensors on an edge. A stage's maximum is at least its known per-tile floor and the ceiling of its total service floor divided by its tile count.

These bounds reject only trials whose ordered stage metrics and tile-count tie-break cannot strictly improve the current best. Surviving edges are priced exactly, with another bound check after each edge. Negative/non-finite weights, fractional transfer-cost parameters, and custom runtime pricing use the original complete evaluation path. Unknown tensor-layout implementations contribute no speculative edge floor. There is no tile-count cap or change to the search neighborhood, L1 feasibility, objective, or allocation quality.

Candidate generation and intrinsic analysis still run for every explored layout. The remaining scaling cost is substantial: from 64 to 256 tiles time grows 4.73×, and from 256 to 1,024 tiles it grows 6.15×. At 32×32, the cumulative improvement relative to the original uncached 127.72-second observation is approximately 3.11×. This change removes most new communication-pair pricing but does not make candidate analysis scale linearly.

Validation: 527 tests passed, three skipped. Tests cover independent exact transfer costs for all combinations of small layouts under legacy, direct DMA, and packed DMA pricing with heterogeneous bandwidths and multiple tensors per edge. They check cold and partially cached whole-selection bounds, simultaneous replacements, pruned versus unpruned searches with five weight combinations, tie-breaking toward fewer tiles, re-evaluation after rejection, fractional-cost fallback, and fan-out rejection before transfer construction.

Source snapshots use SHA-256 over sorted `maps/**/*.py`, with each path relative to `maps` and its contents separated by NUL bytes. Baseline: `122bb494f3ef2bc49ecc98253486dc405210d7ac0382e5a066ab2de5dd45cfb9`; optimized: `6c3eab9d71171121f2f1f20e9c54af8782f1a55830111b765c6bca263cd2d355`. The baseline snapshot remains at `/tmp/maps-pruning-baseline`.

Reproduce either mode on the current checkout:

```sh
.venv/bin/python benchmarks/allocation_scaling.py --side 32
.venv/bin/python benchmarks/allocation_scaling.py --side 32 --unpruned
```

Use `--side 4`, `8`, or `16` for smaller meshes. To use the exact previous source snapshot, add `--checkout /tmp/maps-pruning-baseline` instead of `--unpruned`.
