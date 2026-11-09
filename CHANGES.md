# Changelog

## Unreleased

- Fix use-after-frees across C calls. Two cases:
  - Every OCaml version: functions taking a `Booster.t` or `DMatrix.t`
    passed only the raw handle to C. If the caller held no other
    reference (e.g. `Booster.predict_dense (make_booster ()) m`, or
    `Booster.predict bst (DMatrix.of_bigarray2 x)`), a GC right before
    the call could run the record's finaliser, which frees the
    libxgboost object, and libxgboost then used it: a segfault in
    `InplacePredictImpl` under the new test. Results borrowed from the
    booster (`predict`, `predict_dense`, `save_model_buffer`,
    `save_json_config`, `feature_score`) could likewise be read after
    the booster was freed.
  - OCaml 4.11 and 4.12: `Booster.predict_dense` kept its input
    Bigarray alive with `ignore (Sys.opaque_identity m)` after the C
    call, which the native backend of OCaml < 4.13 treats as pure and
    deletes; a GC before the call then finalised the input, and
    libxgboost read freed memory (valgrind: invalid reads in
    `GBTree::InplacePredict` of a block freed by `caml_ba_finalize`).
    This showed as a flaky qcheck property on 4.11/4.12.
    (`boost_one_iter`'s `ignore (Sys.opaque_identity (grad, hess))` was
    safe: the tuple allocation isn't deleted.)

  Every such function now ends with `Internal.keep_alive` on its
  records and arrays, after copying any borrowed result. `keep_alive` is
  a never-inlined function, so it keeps its argument live on every OCaml
  version. A new test, `test/xgboost/keep_alive.ml`, checks this
  deterministically: the booster, DMatrix or array is only reachable
  through the call (arrays also poison themselves when finalised), and
  `Gc.Memprof` forces a full GC at every allocation during the call. It
  skips itself on OCaml 5.0 and 5.1, which lack `Gc.Memprof`.
- Packaging only: also depend on `conf-libxgboost` on Linux Mint
  releases whose base ships `libxgboost >= 3.0`, so they take the
  system-library fast path instead of compiling the vendored sources.
  That is LMDE >= 7 (Debian 13 base, `os-family = "debian"`) and
  mainline Mint with `VERSION_ID >= 23` (Ubuntu 26.04 base,
  `os-family = "ubuntu"`). Mint 21/22 and LMDE 6 ship libxgboost < 3.0
  and keep using the vendored build.
- Select the native provider by choosing between two definitions of the
  `xgboost_bindings` library (`src/bindings/system` and
  `src/bindings/vendored`), enabled from the decision written by
  `config/provider`. The vendored one bundles the CMake build through
  dune's `foreign_archives`, which replaces the `-L$CAMLORIGIN` link
  flags, the generated install list and the system path's placeholder
  archives. Nothing vendored is built or installed on the system path.
  Plain bytecode programs on the vendored path load shared builds of the
  vendored archives from `lib/stublibs`.
  The system-or-vendored decision, including the `>= 3.0` check, moves
  from `config/discover.ml` to `config/provider/detect.ml`;
  `config/discover.ml` now only builds the vendored archives.

## 0.2.6 (2026-10-28)

- Packaging only (no source or runtime change): split the native
  libxgboost dependency into a separate `conf-libxgboost` virtual
  package. `xgboost` now depends on `conf-libxgboost` only where a
  platform ships `libxgboost >= 3.0` (Debian >= 13, Ubuntu >= 25.10,
  Homebrew, FreeBSD); elsewhere the vendored build is used as before.
  `conf-libxgboost` is a generic presence probe
  (`pkg-config --exists xgboost`), so other libxgboost bindings can
  reuse it; the `>= 3.0` requirement and the vendored fallback stay in
  `config/discover.ml`. The opam-repository template for the conf
  package lives under `packaging/`.

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
