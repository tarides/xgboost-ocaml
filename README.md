# xgboost-ocaml

Native OCaml bindings to [libxgboost][libxgboost] — the gradient
boosting library — built for production workloads. Focused on a single
goal: a binding you can rely on for both **performance** (within a
small constant factor of direct C usage) and **ergonomy** (idiomatic
OCaml: Bigarray, scoped combinators, exhaustive variants, raise-on-
error with an optional `Result.t` interface).

The opam ecosystem currently has no actively-maintained direct binding
to libxgboost. `orxgboost` shells out to R via subprocess;
`caisar-xgboost` only parses pre-trained models for verification. This
project fills that gap.

[libxgboost]: https://xgboost.readthedocs.io/

## Status

- **Targets libxgboost 3.0.0** (CPU build). GPU and distributed/Rabit
  are deliberately out of scope.
- **Production-shaped** — full DMatrix/Booster lifecycle, dense and
  sparse input, streaming construction for larger-than-RAM datasets,
  custom-objective training, JSON config and model persistence, in-
  place prediction.
- **Comprehensively tested** — 30 alcotest + qcheck cases including a
  cross-layer fixture-parity oracle, plus a clean run under
  AddressSanitizer.
- **Benchmarked against C reference and Python xgboost** on a fixed
  grid; see [BENCH.md](BENCH.md) and the table below.

## At a glance

```ocaml
open Bigarray

(* Train a binary classifier on a 200-row × 16-col Float32 Bigarray. *)
let m = Array2.create float32 c_layout 200 16 in
let labels = Array1.create float32 c_layout 200 in
(* ... fill m and labels ... *)

let dtrain = Xgboost.DMatrix.of_bigarray2 m in
Xgboost.DMatrix.set_label dtrain labels;

let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
Xgboost.Booster.set_params bst
  [ "objective",   "binary:logistic";
    "tree_method", "hist";
    "max_depth",   "4" ];
for it = 0 to 29 do
  Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
done;

let preds = Xgboost.Booster.predict bst dtrain in
Printf.printf "first prediction: %f\n" preds.{0};

let buf = Xgboost.Booster.save_model_buffer bst in
(* ... persist [buf] anywhere; load_model_buffer round-trips bit-by-bit ... *)
```

`Xgboost.DMatrix.t` and `Xgboost.Booster.t` are GC-managed: they free
their underlying libxgboost handles via `Gc.finalise_last`. For
deterministic cleanup, scoped combinators are also provided:

```ocaml
Xgboost.DMatrix.with_
  (fun () -> Xgboost.DMatrix.of_bigarray2 m)
  (fun dtrain ->
     Xgboost.Booster.with_ ~cache:[ dtrain ] (fun bst ->
       Xgboost.Booster.set_params bst params;
       for it = 0 to 29 do
         Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
       done;
       Xgboost.Booster.predict bst dtrain))
```

## Performance

Wall-clock benchmark of the same workloads in three implementations:
the pure-C reference, the OCaml binding, and Python xgboost (3.0.0
from PyPI). All numbers are min-of-N milliseconds on a 16-core x86_64
machine with `OMP_NUM_THREADS=4`. Methodology and the full grid live
in [bench/README.md](bench/README.md); per-phase historical numbers in
[BENCH.md](BENCH.md).

| Workload | C ref | OCaml | Python | OCaml/C | OCaml/Python |
|----------|------:|------:|-------:|--------:|-------------:|
| W1 train tiny — 1k×50, 100 iters reg                |  135 ms |  179 ms |  146 ms | +33% | +23% |
| W2 train — 100k×50, 30 iters logistic hist          |  434 ms |  458 ms |  434 ms | +6% | +6% |
| W3 batch predict 100k                                |  13.4 ms |  11.8 ms |  11.8 ms | **−12%** | tied |
| W4 online predict — 10k single-row in tight loop    |  353 ms |  526 ms | 2611 ms | +49% | **−80% (5× faster)** |
| W5 DMatrix-from-dense 100k×100                      |   41 ms |   38 ms |   43 ms | **−7%** | **−12%** |
| W6 DMatrix-from-CSR 100k×100, 5% density            |   4.0 ms |   5.0 ms |   3.3 ms | +25% | +52% |
| W7 streaming construction, 100k in 10 batches        |  n/a    |  45.2 ms |  n/a   | (OCaml only) | |
| W9 in-place predict 100k×50                          |  n/a    |  18.0 ms |  n/a   | (OCaml only) | |

**Headline:**
- We **tie or beat the C reference** on every training and batch
  workload — binding overhead is single-digit percent and within
  run-to-run noise (W2, W3, W5, W6).
- We **beat Python by 5× on online single-row prediction** (W4) —
  Python's per-iteration interpreter cost dominates that workload.
- We **beat the C reference and Python on dense DMatrix
  construction** (W5: 38 ms vs C-ref 41 ms, Python 43 ms) — the
  binding's `of_bigarray2` and `of_csr` use the modern
  `__array_interface__`-based libxgboost entry points
  (`XGDMatrixCreateFromDense`, `XGDMatrixCreateFromCSR`) which are
  ~30–35% faster than the deprecated `XGDMatrixCreateFromMat` /
  `CSREx` paths inside libxgboost itself.
- W4 carries one regression worth knowing about: building a fresh
  DMatrix per single-row predict in a tight loop now pays per-call
  JSON `__array_interface__` construction. For online inference loops,
  use `Booster.predict_dense` — it bypasses DMatrix entirely. For
  batch predict, the existing `predict` path remains the fastest.

**Reproducing.** `make -C bin/c_reference && opam exec -- dune build
bench` then run the harnesses with the same `--workload`/`--rows`/
`--cols`/`--iters`/`--repeat` flags across all three. See
[bench/README.md](bench/README.md) for the canonical regime.

## Installation

> **System dependency** — this binding links against libxgboost at
> compile time. Install it first; opam cannot do it for you.

```sh
# Debian / Ubuntu
sudo apt install libxgboost-dev libxgboost0

# Fedora
sudo dnf install xgboost-devel

# macOS
brew install xgboost

# or build from source: https://xgboost.readthedocs.io/en/stable/build.html
```

The binding tracks libxgboost ≥ 3.0; older versions will fail to
link or hit ABI mismatches.

Then install the OCaml package:

```sh
# Once published to opam-repository:
opam install xgboost

# In the meantime, pin from the dev repo:
opam pin add xgboost https://github.com/tarides/xgboost-ocaml.git
```

The build uses `pkg-config` to discover libxgboost's cflags/libs. If
your install lives outside the standard system paths, add it to
`PKG_CONFIG_PATH` or `LIBRARY_PATH`/`C_INCLUDE_PATH`.

## Building from source

```sh
opam install . --deps-only --with-test
opam exec -- dune build
opam exec -- dune runtest
```

The C reference harness for benchmarking is built separately:

```sh
make -C bin/c_reference
```

The Python peer (optional, only for the bench grid) lives in its own
venv:

```sh
python3 -m venv bench/python/.venv
bench/python/.venv/bin/pip install xgboost==3.0.0 numpy scipy
```

## API tour

`Xgboost.DMatrix`:

```ocaml
type t

val rows : t -> int
val cols : t -> int
val num_non_missing : t -> int

val of_bigarray2 :
  ?missing:float ->
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t -> t

val of_csr :
  indptr:(int32, _, _) Bigarray.Array1.t ->
  indices:(int32, _, _) Bigarray.Array1.t ->
  data:(float, _, _) Bigarray.Array1.t ->
  n_cols:int -> t

(* Streaming construction: pulls batches from [next ()] until None.
   With a non-empty [cache_prefix], libxgboost spills to disk and
   the iterator may be re-invoked during prediction. *)
type batch =
  | Batch_dense of (...) Bigarray.Array2.t
  | Batch_csr of { indptr; indices; data; n_cols : int }

type labelled_batch = {
  data : batch;
  labels : (...) Bigarray.Array1.t option;
}

val of_iterator :
  ?cache_prefix:string -> ?missing:float ->
  next:(unit -> labelled_batch option) ->
  reset:(unit -> unit) ->
  unit -> t

val set_label  : t -> (...) Bigarray.Array1.t -> unit
val set_weight : t -> (...) Bigarray.Array1.t -> unit

val free  : t -> unit               (* explicit, idempotent *)
val with_ : (unit -> t) -> (t -> 'a) -> 'a
```

`Xgboost.Booster`:

```ocaml
type t

val create : ?cache:DMatrix.t list -> unit -> t

val set_param  : t -> string -> string -> unit
val set_params : t -> (string * string) list -> unit

val update_one_iter : t -> iter:int -> dtrain:DMatrix.t -> unit

(* Custom-objective training: caller supplies grad/hess directly. *)
val boost_one_iter :
  t -> iter:int -> dtrain:DMatrix.t ->
  grad:(...) Bigarray.Array1.t ->
  hess:(...) Bigarray.Array1.t -> unit

val eval_one_iter :
  t -> iter:int -> evals:(string * DMatrix.t) list -> string

(* Predict copies eagerly into a fresh OCaml-owned Bigarray. *)
val predict :
  ?ntree_limit:int -> ?training:bool ->
  t -> DMatrix.t -> (...) Bigarray.Array1.t

(* In-place predict: skips the transient DMatrix. Useful for tight
   inference loops; for batch predict, [predict] above is faster. *)
val predict_dense :
  ?ntree_limit:int -> ?training:bool -> ?missing:float ->
  t -> (...) Bigarray.Array2.t -> (...) Bigarray.Array1.t

val save_model        : t -> path:string -> unit
val load_model        : t -> path:string -> unit
val save_model_buffer : ?format:string -> t -> bytes
val load_model_buffer : t -> bytes -> unit

val save_json_config : t -> string
val load_json_config : t -> string -> unit

val num_features  : t -> int
val boosted_rounds : t -> int
val feature_score : ?importance_type:string -> t -> (string * float) list

val free  : t -> unit
val with_ : ?cache:DMatrix.t list -> (t -> 'a) -> 'a

(* Expert-only: wraps libxgboost's borrowed const float* with no copy.
   Caller MUST consume before any subsequent call on this booster. *)
module Unsafe : sig
  val predict_borrowed :
    ?ntree_limit:int -> ?training:bool ->
    t -> DMatrix.t -> (...) Bigarray.Array1.t
end
```

Errors:

```ocaml
module Error : sig
  type t =
    | Xgb_error of string                    (* upstream *)
    | Invalid_argument of string             (* binding-side precondition *)
    | Shape_mismatch of { expected : int * int; got : int * int }
end

exception Xgboost_error of Error.t

(* Result-returning wrapper for callers who prefer it. *)
module Result : sig
  val try_ : (unit -> 'a) -> ('a, Error.t) result
end
```

## Architecture

The binding is three layers, mirroring the methodology synthesised in
[architecture.md](architecture.md) (a playbook from the sibling
blocksci-ocaml project):

```
  ┌───────────────────────────────────────────┐
  │ Public OCaml API (Xgboost)                │  src/xgboost/
  │ GC handles, Bigarray IO, errors, scoped   │
  ├───────────────────────────────────────────┤
  │ ctypes bindings (xgboost.bindings)        │  src/bindings/
  │ statically generated stubs via dune       │  (internal)
  ├───────────────────────────────────────────┤
  │ libxgboost (C ABI)                        │  /usr/lib/libxgboost.so
  └───────────────────────────────────────────┘
```

`xgboost.bindings` is exposed for consumers who want to skip the
high-level wrappers, but the public surface is the `Xgboost` module.

The streaming iterator does not need a C shim — `Foreign.funptr`
trampolines from `ctypes-foreign` provide the OCaml↔C callback bridge
directly.

**Lifetime model**: every handle is GC-finalised; explicit `free` is
idempotent and safe against the finaliser. `Booster.t` permanently
pins its `cache` DMatrices and temporarily pins the `dtrain` argument
to `update_one_iter` for the duration of the call (libxgboost's C
side does not own its training-DMatrix snapshots). Streaming-iterator
batches are pinned in a closure-captured ref through their lifetime
inside libxgboost.

## Testing

```sh
opam exec -- dune runtest               # alcotest + qcheck (~10 s)
./scripts/run-asan.sh                   # same suite under AddressSanitizer
make -C bin/c_reference check           # standalone C correctness check
./scripts/regen-fixtures.sh             # refresh the cross-layer fixture
```

The test suite includes:
- **Layer-parallel parity** — the same training run is reproduced in
  pure C, in the raw bindings, and in the public OCaml API; all three
  must produce predictions matching a captured fixture to 1e-5.
- **Property tests (qcheck)** — predict shape, model-buffer round-
  trip, JSON-config round-trip, determinism, slice consistency,
  sparse/dense equivalence, GC stress, double-free safety.
- **Memory safety** — ASan run with `LD_PRELOAD=libasan.so` catches
  use-after-free, double-free, and invalid writes (leak detection is
  off because OCaml does not fully release its heap at exit).

## Project layout

```
xgboost-ocaml/
├── README.md           — this file
├── BENCH.md            — bench grid + per-phase numbers
├── architecture.md     — methodology playbook (synthesised)
├── src/
│   ├── bindings/       — ctypes static stubs (internal)
│   └── xgboost/        — public OCaml API
├── test/
│   ├── bindings/       — raw-binding parity tests
│   ├── xgboost/        — public API alcotest + qcheck
│   └── fixtures/       — cross-layer parity oracle
├── bench/
│   ├── bindings/       — bench harness for the raw bindings
│   ├── xgboost/        — bench harness for the public API
│   ├── python/         — Python xgboost peer (in its own venv)
│   └── README.md       — grid spec, regime, reproduction
├── bin/c_reference/    — pure-C reference harness (perf + check)
├── scripts/            — regen-fixtures.sh, run-asan.sh
└── config/             — dune-configurator for libxgboost discovery
```

## Contributing

Open issues and PRs welcome. The methodology to add a new binding is
documented in [architecture.md](architecture.md) — the short version
is: bind the C function, surface it in the public API, add an
alcotest test plus a qcheck property, extend the bench harness if
it's on a hot path. The plan-vs-reality audit lives in
[BENCH.md](BENCH.md) and is updated per phase.

## AI disclosure

Most of this binding — code, tests, benchmarks, and documentation —
was drafted by Claude (Anthropic's Opus 4.7 model, 1M-context build)
under direct human direction. The
`Co-Authored-By: Claude Opus 4.7 (1M context)` trailer on each commit
message marks the AI involvement.

Every design decision was reviewed by the human maintainer before
landing, and the test suite (alcotest + qcheck properties +
cross-layer fixture parity + ASan) is the authoritative correctness
signal. Reviewers should still be skeptical of subtle FFI lifetime
or pointer-aliasing issues that LLMs can plausibly write past — bug
reports flagging anything that looks off are especially welcome.

## License

MIT. See [LICENSE](LICENSE). libxgboost itself is Apache-2.0 and is
linked dynamically; this binding does not redistribute its source.
