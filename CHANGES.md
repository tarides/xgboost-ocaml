# Changelog

## 0.2.5 (2026-10-26)

- Pin `OMP_NUM_THREADS=1` (and `OMP_THREAD_LIMIT=1`) for the test suite
  via a `test/dune` `(env …)` stanza. The tests train/predict on tiny
  matrices, so a single OpenMP thread is faster; more importantly, under
  opam-repo-ci's `dune … -j <host-cpus>` the parallel test executables
  each spawned host-cpu-count OMP threads, oversubscribing large CI
  workers into a ~40-minute thrash (and an occasional spurious
  property-test failure). Bounding threads per executable removes the
  blow-up regardless of `-j`, and keeps libxgboost's FP reductions
  deterministic across workers. (Makes the assumption already documented
  in `config/discover.ml` actually hold.)

## 0.2.4 (2026-10-25)

- Link `libexecinfo` on the BSDs in the vendored build. XGBoost's bundled
  `dmlc` logging references `backtrace`/`backtrace_symbols`, which are in
  libc on glibc and in libSystem on macOS but in a separate `libexecinfo`
  on FreeBSD/OpenBSD/NetBSD/DragonFly. The library linked there but the
  test executables failed with `undefined symbol: backtrace`; the vendored
  link flags now add `-lexecinfo` on those platforms.
- Mark the package unavailable on `riscv64`
  (`available: ... & arch != "riscv64"`). Its property-test executable
  crashes at runtime on riscv64 (opam-repository CI, debian-13/riscv64) —
  a separate ABI issue from the build — so exclude the arch until it is
  diagnosed.
- Lower the minimum OCaml version from 5.1 to 4.11. The library and its
  FFI stack were already 4.x-compatible; the only OCaml 5-only use was
  `Domain.recommended_domain_count` in the build configurator
  (`config/discover.ml`), which picked the `-j` count for the vendored
  CMake fallback. It now shells out to `getconf _NPROCESSORS_ONLN`
  (falling back to `nproc --all`), which is equally OMP_NUM_THREADS-
  agnostic and compiler-independent. CI now covers 4.11.

## 0.2.3 (2026-10-25)

- Cap the parallelism of the vendored XGBoost CMake build at 6 jobs.
  It previously used every available CPU, so on opam-repository CI
  workers (80+ CPUs reported, with a much smaller memory budget) it
  started dozens of `cc1plus` processes and was OOM-killed.
- Fix a segmentation fault in `Booster.save_model_buffer` and
  `Booster.save_json_config` on musl-based systems (e.g. Alpine Linux).
  The borrowed C buffer was copied with a per-element ctypes dereference
  loop, which crashed under musl's allocator (glibc tolerated it). It is
  now copied in a single bulk `string_from_ptr`, which is also faster.

## 0.2.2 (2026-10-20)

- Install the vendored XGBoost and dmlc static archives with the bindings
  and resolve them relative to the OCaml archives. Downstream executables
  can link after the package's build directory is removed, including
  after relocating the installation.
- On the system-libxgboost path, do not install the empty vendored
  placeholder archives.
- Fix the macOS vendored static build by skipping OpenMP dylib
  post-processing when no dylib is built.

## 0.2.1 (2026-08-21)

- Fix the build against libxgboost >= 3.1: drop a stray, unused binding
  to `XGDMatrixCreateFromCSREx`, which was removed from the C API after
  3.0 and broke ctypes stub compilation on newer libxgboost (e.g.
  Homebrew's 3.4.1). The sparse-CSR path already uses the modern
  `XGDMatrixCreateFromCSR`.

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
