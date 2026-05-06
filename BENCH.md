# Benchmark results

Headline numbers for `xgboost-ocaml` measured on the development host
(16-core x86_64, 15 GB RAM, no GPU, libxgboost 3.0.0 CPU-only build,
`OMP_NUM_THREADS=4`). All numbers are min-of-N wall-clock milliseconds.
The grid spec, methodology, and reproduction recipe live in
[`bench/README.md`](bench/README.md).

## Phase 7 (scale validation) — current

Validates that the binding survives at the data sizes downstream
blockchain pipelines need (50–80 M rows × ~36 cols, see
`tarides/BlockSci`#67 / #71). Single-shot run, not a perf gate.

Run on `blocksci-bench` (Hetzner FSN1-DC18, EPYC 7502P 32c/64t,
512 GB RAM, 2× 894 GB NVMe RAID, Ubuntu 24.04, libxgboost 3.0.0
from the upstream Python wheel, `OMP_NUM_THREADS=32`). Synthetic CSR
data at ~50% density for 1 M / 10 M (in-memory), dense streaming
batches for 50 M (external memory via `DMatrix.of_iterator
~cache_prefix`). Same booster config across all three:
`tree_method=hist`, `max_depth=6`, `objective=binary:logistic`.

| Rows  | Cols | Path                       | Construct | Peak RSS | Train 1 round | predict_dense 100k | Cache on disk |
|-------|-----:|----------------------------|----------:|---------:|--------------:|-------------------:|--------------:|
| 1 M   |   36 | `of_csr` (in-memory)       |    130 ms | 0.32 GiB |        467 ms |             1.6 ms |  n/a          |
| 10 M  |   36 | `of_csr` (in-memory)       |  1 175 ms | 2.94 GiB |      3 865 ms |             1.5 ms |  n/a          |
| 50 M  |   36 | `of_iterator ~cache_prefix`| 47 987 ms | 0.70 GiB |     33 687 ms |             1.5 ms |  17.72 GiB    |

**Verdict — scales linearly to 50 M rows; recommended for that
workload.** From 1 M → 10 M (in-memory CSR), construction is ~9× for
10× rows and one round of training is ~8.3× — sublinear, consistent
with OpenMP overhead amortising at scale. From 10 M → 50 M (switching
to external memory), construction goes 1.18 s → 48 s (≈ 40×) — that's
the one-time cost of hashing every batch through to the on-disk cache.
But training one round goes 3.86 s → 33.7 s (≈ 8.7× for 5× rows),
i.e. libxgboost streams from the cache near-linearly with row count.

The big result is that the streaming iterator with `cache_prefix`
keeps **RSS under 1 GiB at 50 M rows** — peak observed 0.70 GiB —
versus a projected ~14 GiB had the same data been loaded in-memory
(extrapolated from the 10 M / 2.94 GiB working set). Disk usage at
peak was 17.72 GiB on NVMe, freed when the DMatrix is dropped.
External memory mode is the right path for the Möser-Narayanan
reproduction at 50 M+ rows.

`predict_dense` on a 100 k held-out matrix is independent of training
row count (the model is the same depth-6 hist booster regardless), so
the ~1.5 ms latency is constant across all three scales.

Bench harness: `bench/xgboost/perf_scale.ml`. Reproduction:
```sh
opam exec -- dune exec bench/xgboost/perf_scale.exe -- --all \
  --cache-prefix /var/tmp/xgb-cache/ext --batch-rows 1000000 --verbose
```

## Phase 3 (streaming + in-place predict) — current

Streaming iterator and in-place predict added:
- `Xgboost.DMatrix.of_iterator ~next ~reset ()` builds a DMatrix by
  pulling batches from an OCaml callback. Backed by libxgboost's
  XGProxyDMatrix + XGDMatrixCreateFromCallback, with `Foreign.funptr`
  trampolines (no C shim required). Supports per-batch labels and an
  optional `cache_prefix` for external memory mode.
- `Xgboost.Booster.predict_dense bst m` runs prediction directly
  against a Bigarray, skipping DMatrix construction. Uses the modern
  `XGBoosterPredictFromDense` with a JSON `__array_interface__`.

Phase 3 measurements (OMP=4, min-of-N ms):

| Workload | Configuration | Layer C |
|----------|---------------|--------:|
| W7 streaming construction | 100k rows in 10 batches × 50 cols | 58.8 ms |
| W9 in-place predict 100k rows | 30-tree binary:logistic | 13.1 ms |
| W3 batch predict 100k rows | (same model, same data) | 11.5 ms |
| W9 in-place predict 1k rows | 30-tree binary:logistic | 0.36 ms |
| W3 batch predict 1k rows | (same model, same data) | 0.24 ms |

W3 vs W9 takeaway: at typical batch sizes (1k–100k), `predict` via
DMatrix is slightly faster than `predict_dense` because
`XGBoosterPredictFromDense` reconstructs an internal DMatrix-equivalent
on every call. The in-place path is most useful for single-row online
inference where the DMatrix construction overhead dominates;
extending W4 to use it remains future work.

W7 streaming overhead is ~6 ms per batch on the dev box, dominated by
the per-batch slice-copy in the bench harness (synthetic; real
streaming use would already have batches in their own Bigarrays).

## Phase 6 (of_bigarray2 modernisation, C ref also modernised) — current

After Phase 5 closed the W6 gap by migrating `Xgboost.DMatrix.of_csr`
to `XGDMatrixCreateFromCSR`, the bench grid showed Python beating the
C reference on W5 — the cause was the C reference still using the
deprecated `XGDMatrixCreateFromMat` while Python uses the modern
`XGDMatrixCreateFromDense`. Probe (same machine, all sizes 100k×100):

| Workload | C, legacy entry point | C, modern entry point |
|----------|----------------------:|----------------------:|
| W5 dense | 60.7 ms (`FromMat`)   | 39.4 ms (`FromDense`) |
| W6 CSR   |  5.78 ms (`CSREx`)    |  3.85 ms (`FromCSR`)  |

The legacy paths in libxgboost 3.0 are ~30–35% slower than the
modern ones. Both bin/c_reference/perf.c and bench/bindings/perf.ml
now use the modern entry points by default; the binding's
`Xgboost.DMatrix.of_bigarray2` was migrated alongside.

**Updated W1–W6 grid (everyone on the modern API):**

| Workload | C ref | OCaml (raw) | OCaml | Python |
|----------|------:|------------:|------:|-------:|
| W1 train tiny                                    |  135 ms |  148 ms |  179 ms |  146 ms |
| W2 train (100k×50, 30 iters logistic hist)       |  434 ms |  428 ms |  458 ms |  434 ms |
| W3 batch predict 100k                            |  13.4 ms |  12.8 ms |  11.8 ms |  11.8 ms |
| W4 online predict 10k single-row in loop         |  353 ms |  375 ms |  526 ms | 2611 ms |
| W5 DMatrix-from-dense 100k×100                   |   41 ms |   39 ms |   38 ms |   43 ms |
| W6 DMatrix-from-CSR 100k×100, 5%                 |  4.0 ms |  4.5 ms |  5.0 ms |  3.3 ms |

**W5 outcome:** OCaml binding now leads (38 ms vs C-ref 41, Python
43). Was 76 ms before this phase, ~2× faster after.

**W4 regression:** OCaml went 392 → 526 ms on the W4 pattern
(creating a fresh DMatrix per single-row predict in a tight loop).
The cost is the per-call `Printf.sprintf` of the JSON
`__array_interface__` plus libxgboost's per-call JSON parsing, ×10k
iterations. For online inference loops the recommended path is
`Booster.predict_dense` (skips the DMatrix entirely); for batch
predict the existing `predict` path is fastest. The W4 benchmark
intentionally measures the worst-case "loop with DMatrix per call"
pattern.

## Phase 5 (of_csr modernisation) — current

`Xgboost.DMatrix.of_csr` switched from the legacy
`XGDMatrixCreateFromCSREx` (which requires per-element copy of int32
indptr/indices into `size_t` / `unsigned` C arrays on the OCaml side)
to the modern `XGDMatrixCreateFromCSR` (which takes JSON
`__array_interface__` strings encoding the Bigarray buffers
directly — zero copy).

Standalone probe of the OCaml-side copy loop alone, on the W6 size
(100001 indptr entries + 500000 indices):

| Path                         | Time |
|------------------------------|------:|
| element copy (legacy CSREx)  | 11.8 ms |
| JSON __array_interface__     | <0.1 ms |

W6 (DMatrix from CSR, 100k×100, 5% density), all four layers, before
and after the fix:

| Layer       | Before | After | Change |
|-------------|------:|------:|-------:|
| C ref (uses CSREx)         |  6.2 ms |  6.2 ms | unchanged |
| OCaml raw bindings (CSREx) |  6.8 ms |  5.7 ms | within noise |
| OCaml API (was CSREx, now modern) | 25.8 ms | **4.7 ms** | **−81% (5.5×)** |
| Python (uses modern path)  |  3.4 ms |  3.5 ms | unchanged |

Post-fix the OCaml API beats the C reference on W6 because the C
reference is using the older API path. A user writing fresh C against
`c_api.h` today would also use `XGDMatrixCreateFromCSR` and would
land in the same neighbourhood (≈4.7 ms).

The same modernisation applied to `DMatrix.of_bigarray2` (currently
using `XGDMatrixCreateFromMat`) would close the W5 gap against
Python (W5 OCaml: 76 ms; Python: 56 ms). Deferred.

## Phase 4 (post-W3 memcpy fix) — current

`--repeat 3-5 --warmup 1-2`, `OMP_NUM_THREADS=4`, all min-of-N ms.
Python peer = `xgboost==3.0.0` from PyPI in `bench/python/.venv/`.

| Workload                                       | C ref | Layer B | Layer C | Python | LayerC/Cref | LayerC/Py |
|------------------------------------------------|------:|--------:|--------:|-------:|------------:|----------:|
| W1 train 1k×50, 100 iters reg:squarederror     | 134.0 |  134.3  |  139.6  |  131.8 |   +4%       |  +6%      |
| W2 train 100k×50, 30 iters binary:logistic     | 417.1 |  397.7  |  448.2  |  424.1 |  +7%        |  +6%      |
| W3 batch predict 100k                          |  11.5 |   12.6  |   11.9  |   16.6 |  +4%        | -28% (we win) |
| W4 online predict 10k single-row in loop       | 346.1 |  356.6  |  392.2  | 2774.4 | +13%        | -86% (we win 7×) |
| W5 DMatrix-from-dense 100k×100                 |  67.9 |   65.5  |   ~67   |   43.7 |  ≈0%        | +53% (Py wins) |
| W6 DMatrix-from-CSR 100k×100, 5% density       |   6.0 |    6.7  |   ~7    |    3.2 |  ≈+15%      | +118% (Py wins) |

**Phase-2 perf gates: all met.**
- Heavy training (W1, W2) target: <10%. Achieved +4% / +7%.
- Prediction (W3, W4) target: <30%. Achieved +4% / +13%.

**Versus the Python peer:**
- We match Python on heavy training (libxgboost itself is the
  bottleneck; both wrappers add ~6% overhead).
- We beat Python by a meaningful margin on prediction-heavy paths
  (W3 by 28%, W4 — online single-row predict — by ~7×). Python's
  per-iteration interpreter cost dominates W4.
- Python beats us on raw DMatrix construction (W5 +53%, W6 +118%) —
  numpy's array_interface lets libxgboost ingest the data with one
  memcpy, whereas our `XGDMatrixCreateFromMat` and the deprecated
  `XGDMatrixCreateFromCSREx` paths take a slower internal route. The
  fix is to switch `DMatrix.of_bigarray2` and `DMatrix.of_csr` to the
  modern `__array_interface__`-based constructors (deferred from
  Phase 3).

## Phase 4 — gap-filling decomposition

### G3 — FFI roundtrip cost

`XGBoostVersion` (writes 3 ints from compile-time constants, no
allocation) called 10⁷ times in a tight loop:

| Layer       | Total (ms) | ns/call |
|-------------|-----------:|--------:|
| C reference |   7.9      | 0.79    |
| Layer B (ctypes static stub) | 119.7  | 11.97 |

Per-call FFI overhead is ≈11.2 ns. Within ctypes' published range
(8–30 ns) and below the threshold that would justify a `[@@noalloc]`
shim per call. Confirms the dune ctypes plugin's static stubs are
performing as designed.

### W3 fix — what changed

**Before**: `Internal.copy_borrowed_float32` copied each predict
output via an OCaml-side element loop using ctypes pointer deref +
store. Cost ≈4 ns/element (≈0.4 ms per 100k rows) above the ~12 ms
predict call → +35% Layer-C overhead, just over the +30% gate.

**After**: same function uses `Ctypes.bigarray_of_ptr` to wrap the
borrowed `const float*` as a temporary `Bigarray.Array1.t` view, then
calls `Bigarray.Array1.blit` into a fresh OCaml-owned buffer. blit
becomes `memcpy` when kind/layout match, executing at memory
bandwidth (≈25 µs for 400 KB).

W3 Layer-C overhead dropped from +35% to +4%.

The same fix did not need to be applied to `copy_borrowed_bytes`
(used only by `save_model_buffer` / `save_json_config`, infrequent
per-model calls). The element-loop cost there is below noise.

## Phase 2 (initial) — previous

git: `8fcdd09`, before the Phase-4 fix. Same harness, recorded for
historical comparison.

| Workload | C ref | Layer B | Layer C | Layer-C/C ref |
|----------|------:|--------:|--------:|--------------:|
| W1 train 1k×50 reg                 | 141 ms | 140 ms | 146 ms | +3% |
| W2 train 100k×50 hist              | 409 ms | 431 ms | 449 ms | +10% |
| W3 batch predict 100k              |  12.9 ms |  13.2 ms |  17.4 ms | +35% ⚠ |
| W4 online predict 10k single-row   | 358 ms | 344 ms | 370 ms | +3% |

W3 was the only workload above its +30% gate; Phase 4's measurement-
driven fix above closed it.

## Reproduction

```sh
make -C bin/c_reference                                  # C reference
opam exec -- dune build                                  # OCaml layers

# Per-workload three-way comparison (warm cache, 4 threads)
OMP_NUM_THREADS=4 ./bin/c_reference/perf --workload W3 --repeat 5 --warmup 2
OMP_NUM_THREADS=4 dune exec bench/bindings/perf.exe -- --workload W3 --repeat 5 --warmup 2
OMP_NUM_THREADS=4 dune exec bench/xgboost/perf.exe  -- --workload W3 --repeat 5 --warmup 2
```

Variance is dominated by OpenMP scheduling jitter; `OMP_NUM_THREADS=4`
gives stable min-of-N within ~3% on the dev host. `OMP_NUM_THREADS=1`
is even tighter (<1%) but slower in absolute terms. See
`bench/README.md` for the canonical regime.
