Representative candidate analysis measurements, 2026-10-05.

Subsequent results for safe trial pruning are in [allocation_pruning_scaling.md](allocation_pruning_scaling.md).

This follow-up compares the previous communication-cache implementation with duplicate-work reuse, representative compute/L1 costing, and constant-time ownership lookups. The workload and measurement method match [the earlier cache study](allocation_scaling.md): eight-stage MobileViT nodes 170–179, MAGIA-v3, two token slots, unit weights, fresh sequential Python processes, allocation-only wall time, and no profiler. These are single-run observations. The cache-only baseline was preserved before these edits at `/tmp/maps-candidates-baseline`; optimized measurements were rerun after the final compatibility guard.

| Mesh | Tiles | Cache-only seconds | Optimized seconds | Additional speedup | Cache-only peak MiB | Optimized peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4×4 | 16 | 0.33 | 0.27 | 1.21× | 58.7 | 59.1 |
| 8×8 | 64 | 3.18 | 2.56 | 1.24× | 79.2 | 79.6 |
| 16×16 | 256 | 14.12 | 11.00 | 1.28× | 116.7 | 115.8 |
| 32×32 | 1,024 | 77.80 | 55.48 | 1.40× | 264.0 | 257.5 |

Every analyzed candidate has identical logical shape, stage latency, and per-tile compute/permanent-L1/scratch-L1 facts. SHA-256 fingerprints of the entire analyzed candidate corpus match at every mesh size, including infeasible tile-count entries. Final stage tile counts, layouts, and every per-tile communication cycle also match. The number of analyzed stage/count combinations remains 72, 177, 225, and 273, respectively: search neighborhoods and allocation quality are unchanged. Raw measurements and fingerprints are in [allocation_candidate_scaling_results.jsonl](allocation_candidate_scaling_results.jsonl).

Candidate analysis now reuses resident input demands when pricing communication, shares already calculated operation cycles with barrier-aware stage latency, and resolves scratch requirements once per device/signature. Equivalent built-in compute work shares costs according to the actual device inputs: operation counts, matrix dimensions, stream sizes, or calibrated broadcast geometry. Unknown model/device implementations and custom work without the required attributes use the existing per-tile path. Physical collective group latency is still calculated separately.

L1 equivalence preserves tensor identities, allocation order, sizes, and relative positions between repeated reads of a tensor, so bounding slices and alignment remain exact. Single-read tensors need only allocation byte counts. Single-slice bounds reuse their existing immutable slice, and tile slicing now uses the cached ordinal dictionary instead of scanning a tuple of tile IDs.

Concrete tile work is still generated for every tile for correct ownership, locality checks, and collective analysis; only equivalent compute and L1 evaluations use representatives. In a regular 16-tile test, five layouts require five compute and five L1 evaluations instead of 80 of each, and communication does not rebuild their tile work. Larger candidates and unseen communication pairs still require substantial work. Scaling remains steep: optimized time increases about 4.30× from 64 to 256 tiles and 5.04× from 256 to 1,024 tiles. The combined 32×32 result is approximately 2.30× faster than the original uncached 127.72-second baseline.

Validation: 506 tests passed, three skipped. New tests independently recompute all per-tile facts and stage latency for uneven batched GEMM, transposed weights, bias, empty tiles, odd broadcast/vector lengths, custom coordinate-dependent costs, heterogeneous devices, scratch limits, translated and differently spaced reads, and custom matrix work. Existing collective-barrier and allocation/cache tests pass.

Baseline source-tree SHA-256: `4709af4af25bee9bcfe65b70248fe442b7b4d5d5d3c315b4059db7b52983cb78`. Optimized source-tree SHA-256: `122bb494f3ef2bc49ecc98253486dc405210d7ac0382e5a066ab2de5dd45cfb9` (sorted `maps/**/*.py`, each relative path and contents separated by a NUL byte). Hardware, Python version, and model hash are unchanged from the earlier study.

Reproduce the optimized run from the MAPS checkout:

```sh
.venv/bin/python benchmarks/allocation_scaling.py --side 32
```

The last line includes timing, process peak RSS, final plans/costs, analyzed-count total, and the candidate-facts fingerprint. Use `--side 4`, `8`, or `16` for smaller meshes, or `--model PATH` for another workload. To rerun the cache-only baseline while the local snapshot remains available, add `--checkout /tmp/maps-candidates-baseline`.
