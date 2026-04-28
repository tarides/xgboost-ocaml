# Changelog

## unreleased

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
