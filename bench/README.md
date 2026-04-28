# Benchmark grid

The grid is the substrate the rest of the project's design decisions are
measured against. Every workload is run from every applicable layer; no cells
are skipped unless the strategy genuinely does not apply (and that is recorded
in the cell, not silently omitted).

## Workloads

| ID | Description | Default size | Times |
|----|-------------|--------------|-------|
| W1 | Train tiny — `reg:squarederror` | 1k × 50, 100 iters | per-iter overhead |
| W2 | Train large — `binary:logistic`, `tree_method=hist` | 1M × 100, 100 iters | amortised throughput |
| W3 | Batch predict — single `XGBoosterPredict` | 1M rows | output marshalling |
| W4 | Online predict — 100k single-row in tight loop | 100k × 100 | per-call FFI overhead |
| W5 | DMatrix-from-dense via `XGDMatrixCreateFromMat` | 100k × 100 | dense ingestion |
| W6 | DMatrix-from-CSR via `XGDMatrixCreateFromCSREx` | 100k × 100, 5% density | sparse ingestion |
| W7 | (Phase 3) Streaming external memory | 10M rows × 100, 100 batches | iterator + cache page-in |
| W8 | (Phase 3) `predict_dense` in-place vs DMatrix-then-predict | 1M × 100 | DMatrix construction tax |

## Layers

| Layer | Path | Built by |
|-------|------|----------|
| C reference (ceiling) | `bin/c_reference/perf` | `make -C bin/c_reference` |
| OCaml low-level (layer B) | `bench/bindings/perf.exe` | `dune build bench/bindings` (Phase 1) |
| OCaml high-level (layer C) | `bench/xgboost/perf.exe` | `dune build bench/xgboost` (Phase 2) |
| Python xgboost (peer floor) | `bench/python/perf.py` | `pip install xgboost==3.0.0` |

## Cache regime

**Warm by default.** XGBoost workloads are compute-bound, not I/O-bound; cold
caches measure mmap'd-model load, which is once-per-session. The harness does
a warm-up call before timing wherever ambiguity exists.

W2 is also reported cold (drop caches between runs) once for completeness, in
`bench/results/<sha>-W2-cold.csv`. Everything else: warm.

## Acceptance per workload

OCaml high-level (layer C) overhead vs C reference:

| Workload class | Target |
|----------------|--------|
| Heavy training (W1, W2, W7) | < 10% |
| Prediction (W3, W4, W8) | < 30% |
| Ingestion (W5, W6) | diagnostic only — no fixed target |

Targets that fail at any phase block the gate to the next phase. Failures are
investigated via the G1–G4 decomposition (see `architecture.md` §6.3) before
optimisation work begins.

## Reproducing

Quick run, all workloads, default sizes:

```
make -C bin/c_reference
./scripts/run-bench.sh c
```

Specific workload at custom size:

```
./bin/c_reference/perf --workload W2 --rows 1000000 --cols 100 --iters 100 --repeat 5
```

Each run emits a single CSV line on stdout with `min_ms`, `mean_ms`, `max_ms`
over the repeats. Headers are suppressed when `NO_HEADER=1`. Aggregated
results land in `bench/results/<git-sha>.csv`.

## Ground rules

- **Repeat count**: `--repeat 5` minimum; the lowest minimum-of-5 is the
  reported number, mean and max are diagnostic.
- **Stable variance**: numbers must be within 2% of the median across 5 runs
  before any decision is made on them. Otherwise, increase `--repeat` and
  re-measure on a quieter machine.
- **No skipped cells**: a workload that genuinely cannot be expressed in a
  layer (e.g. `predict_dense` does not exist before Phase 3) is recorded
  literally as `n/a`, never silently dropped from the table.
- **Layer-parallel reporting**: when the BENCH.md report is regenerated, all
  layers are run on the same git SHA on the same machine on the same day.
