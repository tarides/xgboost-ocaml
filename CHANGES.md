# Changelog

## 0.2.0 (2026-08-20)

First opam release.

- Native dependency is now **hybrid**: the build links a system
  libxgboost ≥ 3.0 when `pkg-config` finds one (installed via opam
  `depexts` on platforms that package it), and otherwise compiles a
  pinned, vendored copy of XGBoost from source (bundled under
  `vendor/xgboost`, CMake, static link, no network required). This lets
  `opam install xgboost` work on any platform, including those without a
  distro package.
- `Xgboost.Eval` — parse `eval_one_iter` metric strings; compute AUC and
  ROC directly from prediction/label Bigarrays.
- `Xgboost.Cv.k_fold` — k-fold cross validation with optional
  group-coherent splitting (`?group_ids`) and `summarise`.
- `Xgboost.DMatrix.slice` — row-subset a DMatrix.

## 0.1.0

Initial release. Native OCaml bindings to libxgboost ≥ 3.0 covering:

- DMatrix construction
  - dense (Bigarray.Array2 of float32, zero-copy via the modern
    `XGDMatrixCreateFromDense` array_interface path)
  - sparse CSR (modern `XGDMatrixCreateFromCSR`)
  - streaming iterator (`DMatrix.of_iterator`) backed by
    `XGProxyDMatrixCreate` + `XGDMatrixCreateFromCallback`, with
    optional external-memory caching via `cache_prefix`
- Booster lifecycle, training, prediction
  - `update_one_iter`, `boost_one_iter` (custom gradient/hessian),
    `eval_one_iter`, `reset`
  - `predict` (eager copy), `predict_dense` (in-place from Bigarray),
    `Unsafe.predict_borrowed` (no-copy expert variant)
  - `feature_score` with selectable importance type
- Persistence: model save/load via path or buffer; JSON config
  save/load
- GC-safe lifetime model: `Gc.finalise_last`d handles, scoped
  combinators (`with_`), idempotent `free`
- Comprehensive test suite (alcotest + qcheck), cross-layer fixture
  parity oracle, clean run under AddressSanitizer
- Benchmarks against pure-C reference and Python xgboost on a fixed
  workload grid; see BENCH.md
