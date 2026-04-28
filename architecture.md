# blocksci-ocaml: Design and Architecture

A standalone synthesis of the project's goals, structure, design
choices, and benchmarking methodology, written for engineers building
similar bindings — bridges between a tracing-GC language (OCaml,
Java, Go, Python) and a performance-critical C/C++ library.

This document does not duplicate the artefacts it synthesises. For
specifics, follow the cross-references at the end.

## Contents

1. [Project goal and scope](#1-project-goal-and-scope)
2. [Two co-equal requirements: performance and API ergonomy](#2-two-co-equal-requirements-performance-and-api-ergonomy)
3. [Architecture: the four-layer FFI stack](#3-architecture-the-four-layer-ffi-stack)
4. [Lifetime model: the GC ↔ RAII bridge](#4-lifetime-model-the-gc--raii-bridge)
5. [Three iteration strategies, discovered by measurement](#5-three-iteration-strategies-discovered-by-measurement)
6. [Benchmarking and profiling as design discipline](#6-benchmarking-and-profiling-as-design-discipline)
7. [Testing strategy](#7-testing-strategy)
8. [Reproducing this approach on other bindings](#8-reproducing-this-approach-on-other-bindings)
9. [Cross-references and further reading](#9-cross-references-and-further-reading)

---

## 1. Project goal and scope

`blocksci-ocaml` is an OCaml binding for the Tarides fork of
[BlockSci](https://github.com/citp/BlockSci), a high-performance
Bitcoin blockchain analysis platform initially developed at Princeton
CITP (Kalodner et al., USENIX Security 2020 — paper bundled as
`../blocksci-paper.pdf`). It exposes BlockSci's C++ querying engine to
OCaml through a four-layer FFI stack: native C++ library, stable C
ABI wrapper, ctypes bindings, and a GC-integrated OCaml library.

### Why an OCaml binding for a C++ engine

A binding only earns its existence if it serves a genuine need that
neither side covers alone. BlockSci has a C++ API (fast, awkward to
use interactively) and a Python API via blockscipy (idiomatic for
notebook analysis, slow on full-chain scans). The OCaml binding
targets a different point in the design space:

- **Type safety for analysis pipelines.** Bitcoin analysis often
  involves long composed pipelines (block → transactions → outputs →
  addresses → clusters). OCaml's type system catches kind errors
  (passing an `Input.t` where an `Output.t` is expected) at compile
  time, where Python catches them at runtime — sometimes hours into a
  full-chain scan.
- **Competitive performance without rewriting in C++.** Python is the
  ergonomic baseline; C++ is the performance ceiling. The binding
  targets the gap: write in OCaml, run within a small constant factor
  of C++.
- **OCaml 5 domains for parallel scans.** BlockSci's data layer is
  read-only mmap'd flat files plus RocksDB indices — safe for
  concurrent reads from multiple OS threads. OCaml 5 makes this
  exploitable from a single high-level program. Not yet measured;
  flagged as future work in [`../BENCH.md`](../BENCH.md) §Non-goals.

### Non-goals

The project deliberately does **not** address several axes that, if
included, would dilute the optimisation target:

- **Multi-domain benchmarking.** Single-threaded analysis is the
  current performance target. Adding domains conflates FFI overhead
  measurement with scheduler noise.
- **Latency distributions.** BlockSci queries are batch scans. There
  is no request/response, no p99, no tail. Wall-clock for the full
  scan is the only useful metric.
- **Write workloads.** BlockSci data is read-only after parsing.
- **Python interpreter optimisation.** Python is an external
  reference, not an optimisation target.

This scoping decision is itself transferable. **A binding that tries
to be everything will optimise nothing well.** Pick the workload that
matters and refuse the rest.

---

## 2. Two co-equal requirements: performance and API ergonomy

The single most important framing in this document. Most FFI bindings
fail by making one of these subordinate to the other.

The two requirements pull in opposite directions:

| Performance wants… | Ergonomy wants… |
|---|---|
| Stack-allocated value types | GC-managed handles |
| Zero-copy ranges (borrowed pointers) | Safe iteration (no dangling pointers) |
| `[@@noalloc]` C stubs returning scalars | `Seq.t`, `'a list`, idiomatic iterators |
| No callback re-entry in hot loops | Pure-OCaml callbacks for composition |
| Integer enums | Exhaustive variants |
| NULL-on-error for cheap probes | Raise-on-error or `option` for safety |
| `Unsigned.UInt32` to match C widths | `int`, `int64` in user signatures |

Each layer of the binding is a negotiated settlement between these
pressures. **The shape of the settlement is empirically discovered
(measured), not a priori designed.** This thesis frames every later
section.

### The two failure modes

**Performance-only binding.** A 1:1 mirror of the C ABI that exposes
`ptr`, `Unsigned.UInt32`, manual `_free`, NULL pointers, and string
lifetime caveats. Fast. Unusable. Users will write `Gc.finalise`
wrappers themselves, badly.

**Ergonomy-only binding.** Idiomatic OCaml wrappers that allocate per
element, box every `int64`, re-enter the GC on every callback, and
copy strings on every read. Pleasant interactive REPL experience,
5–20× slower than C++ on full-chain scans. Users will work around it
by hot-pathing into the lower layers, badly.

The deliverable in `blocksci-ocaml` is **both at once** — but not via
a single magic abstraction. It is via **per-pattern strategies**
chosen from a small palette, each tuned for a class of access pattern
(see §5).

### Why both layers must exist

Some bindings try to skip the low-level layer ("just write the high-
level API directly"). This works until the first time you need to
debug a memory issue, where the absence of a thin testable layer
makes every bug ambiguous between the C boundary and the GC bridge.

`blocksci-ocaml` keeps the low-level ctypes layer (in `lib/`,
internally named `abi`) as a private, testable boundary. The
high-level layer in `lib/blocksci/` is built on top. This separation
has three concrete benefits:

1. **Bug localisation.** A failure in `lib/test/check.ml` (low-level)
   versus `lib/blocksci/test/check.ml` (high-level) tells you which
   layer broke.
2. **Performance attribution.** Benchmarking the low-level layer
   (`lib/test/perf.ml`) versus the high-level layer
   (`lib/blocksci/test/perf.ml`) shows what the GC integration
   actually costs.
3. **Mock substitution.** The mock library in `lib_mock/` swaps the
   underlying engine without touching either layer above it (see §7).

---

## 3. Architecture: the four-layer FFI stack

```
  ┌──────────────────────────────────────────────────────┐
  │ 4. lib/blocksci/        — high-level OCaml library   │  ergonomy
  │                           GC'd handles, Seq.t, fold, │
  │                           variants, pure-OCaml types │
  ├──────────────────────────────────────────────────────┤
  │ 3. lib/                 — ctypes bindings (auto-gen) │  plumbing
  │                           Data + Fn functors → Stub  │
  ├──────────────────────────────────────────────────────┤
  │ 2. abi/                 — C ABI wrapper              │  boundary
  │                           opaque handles, _free,     │
  │                           NULL-on-failure, try/catch │
  ├──────────────────────────────────────────────────────┤
  │ 1. libblocksci          — upstream C++ engine        │  reference
  │                           value types, mmap files,   │
  │                           RocksDB indices            │
  └──────────────────────────────────────────────────────┘
```

### Layer 1 — `libblocksci` (upstream C++ engine)

**What it owns.** Memory-mapped flat files for the chain data;
RocksDB indices for hash lookup and address lookup; the `blocksci::*`
value types (`Blockchain`, `Block`, `Transaction`, `Input`, `Output`,
`Address`, `Script*`, `Cluster`); the analysis algorithms.

**What it does not own.** Any awareness of the binding. The C++
library is upstream; the binding does not patch it. Constraints
discovered in upstream (e.g. `DataAccess::reload()` is a barrier
operation, see §4) become contracts the binding must respect.

**Concrete artefacts.** Installed via `blocksci` and `blocksci-dev`
deb packages. Headers under `/usr/include/blocksci/`, shared library
at `/usr/lib/libblocksci.so`.

### Layer 2 — `abi/` (stable C ABI wrapper)

**What it owns.** The boundary. Translates C++ value types and
exceptions into a flat C surface that ctypes can call.

- **Opaque handles.** Every C++ type becomes `typedef struct
  abi_<name>_t abi_<name>_t;` with explicit `_free()`. The handle is
  a `reinterpret_cast`'d C++ pointer. (Exception: a few types were
  initially wrapped in inner-pointer structs; see
  [`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md) concern #3 for the
  cleanup.)
- **Exception barrier.** Every function wraps its body in
  `try { … } catch (...) { return nullptr_or_sentinel; }`. C++
  exceptions crossing the FFI boundary are undefined behaviour;
  catching at the boundary is non-negotiable.
- **NULL-on-failure.** Pointer-returning functions return NULL on
  error; the OCaml side lifts this to `option` via `ptr_opt`.
  Inconsistencies (e.g. `abi_block_height` returning 0 on NULL even
  though 0 is the valid genesis height) are documented in
  [`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md) concern #5 and worked
  around in layer 4.
- **Thread-local string buffers.** String returns use
  `thread_local std::string` buffers, one per function. Pointer is
  valid until the next call to the same function on the same thread.
  Caller must copy before further calls.

**What it does not own.** No GC awareness, no closure handling, no
ctypes specifics. Layer 2 is testable from C and C++ alone
(`abi/check.c`, `abi/check.cpp`).

**Concrete artefacts.** `abi/abi.h` (header), `abi/abi.cpp` (impl),
`abi/Makefile` (builds `libabi.so` and `libabi.a`).

### Layer 3 — `lib/` (ctypes bindings)

**What it owns.** Type-level mirror of the C ABI. Two functors per
the dune ctypes plugin convention:

- `lib/data.ml` — `Types` functor producing opaque `Ctypes` structure
  types for each handle.
- `lib/fn.ml` — `Functions` functor binding each `abi_*` C function
  with its exact ctypes signature.

The dune ctypes plugin auto-generates the `Stub` module and C stubs
from these functors. The result is the internal `abi` library.

**What it does not own.** Memory management; lifetime; pretty
errors; user-facing types. This layer is internal plumbing.

**Concrete artefacts.** `lib/dune` (ctypes plugin config), `lib/data.ml`,
`lib/fn.ml`. Tests in `lib/test/`.

### Layer 4 — `lib/blocksci/` (high-level OCaml library)

**What it owns.** The user-facing API. GC-managed lifetimes, OCaml
types in every signature, `Seq.t` and `_range.fold` for iteration,
exhaustive variants for enumerations.

- **GC integration.** `Gc.finalise` registers the appropriate `_free`
  call for every handle. Derived values hold back-references to
  preserve lifetime invariants (see §4).
- **Eager string copies.** Every C string crosses the boundary into
  an OCaml `string` immediately, before any other call could
  invalidate the thread-local buffer.
- **Variant types.** `address_type` is a flat OCaml variant
  representing BlockSci's `AddressType::Enum`; pattern matching is
  exhaustive.
- **Iterators.** `Tx_range`, `Output_range`, `Input_range` for
  zero-allocation indexed iteration. `Output_bag`, `Input_bag`,
  `Tx_bag`, `Address_bag` for borrowed-pointer iteration over
  C++-owned vectors. `Seq.t` wrappers for composition. Specialised
  `_range.fold` and `_bag.fold` for hot loops.

**What it does not own.** Any C-level concept. `Unsigned.UInt32`,
`ptr_opt`, `char_ptr_of_string` never appear in user-visible
signatures. `int`, `int64`, `string`, `option`, `Seq.t`, `bytes` do.

**Concrete artefacts.** `lib/blocksci/blocksci.{ml,mli}` (top-level),
`lib/blocksci/fold_stubs.c` (zero-alloc C stubs marked `[@@noalloc]`),
`lib/blocksci/dune`. Tests in `lib/blocksci/test/`.

### Layer-interaction contracts

Each boundary has a contract and a trade-off:

| Boundary | Contract | Trade-off accepted |
|---|---|---|
| 1 ↔ 2 | Opaque handles via `reinterpret_cast`; exceptions caught at boundary; NULL on failure | One indirection per call; try/catch on every call |
| 2 ↔ 3 | NULL → `ptr_opt`; primitives via `Unsigned.UIntN`; strings via thread-local buffers | ctypes marshalling cost (~8 ns/call literature, ~3 ns hand-written) |
| 3 ↔ 4 | `Gc.finalise` for lifetimes; back-references for ownership graphs | One pointer per OCaml value; one finaliser registration per value |

These contracts are explicit so that any future change can be
evaluated against the layer it actually affects.

### Inheritance flattening

BlockSci's C++ API uses two inheritance hierarchies (chain ranges
and address scripts). Both are erased at layer 2 and re-encoded at
layer 4 — never propagated through layer 3.

- **Range inheritance** (`Blockchain : BlockRange`,
  `Block : TransactionRange`) becomes explicit extraction at layer 2:
  `abi_block_get_tx_range()` returns a separate
  `abi_tx_range_t*`. The C++ "is-a" becomes a C "has-a-method-to-get".
  At layer 4, `Block.txs : t -> Tx.t Seq.t` re-introduces the
  iteration contract idiomatically.
- **Script hierarchy** (`Address ← ScriptBase ← PubkeyAddressBase ←
  ScriptAddress<…>`) becomes type-erased at layer 2: an integer enum
  + type-specific accessors. At layer 4, the integer enum becomes a
  flat OCaml variant (`address_type`) and exhaustive pattern matching
  replaces dynamic dispatch.

This pattern generalises. **C ABIs cannot express inheritance; OCaml
prefers exhaustive variants over dynamic dispatch.** Erase the
hierarchy at the C boundary, re-encode it as a variant on the OCaml
side, and the high-level API ends up cleaner than either source
language would naturally produce.

For the full hierarchy and concern catalogue, see
[`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md) §"C++ Inheritance and How
It Is Handled".

---

## 4. Lifetime model: the GC ↔ RAII bridge

The hardest single design problem in any C++ ↔ tracing-GC binding.
Stated precisely:

> C++ encodes object dependencies in scoping/RAII. OCaml encodes them
> in reachability. The GC has no reason to keep `Chain` alive while
> `Block` is alive unless the dependency is made visible.

In BlockSci, every `Block`, `Transaction`, `Input`, `Output`,
`Address` holds a non-owning `DataAccess*` that points into the
`Blockchain`'s subsystems. If OCaml drops the `Chain` reference but
keeps a `Block`, the GC is free to finalise the chain. The block's
`DataAccess*` becomes dangling. Use-after-free at the next access.

This is not a BlockSci flaw, nor a C ABI flaw. It is inherent to
bridging deterministic-lifetime languages with tracing-GC languages.
Java's JNI, Python's ctypes, Go's cgo, OCaml's ctypes — all face the
same issue.

### Solution: back-references make dependencies GC-visible

Every derived OCaml value carries a back-reference to its owning
chain (and, transitively, to whatever holds the lifetime).

```ocaml
type chain  (* opaque; finaliser frees the C handle *)

type block = {
  ptr : Stub.Block.t Ctypes.structure Ctypes.ptr;
  chain : chain;  (* never dereferenced — exists as a GC edge *)
}
```

The `chain` field is never read by any function. Its sole purpose is
to be a reachability edge from `block` to `chain`, so the GC keeps
`chain` alive as long as any block does. **Zero C-side cost; one
pointer per OCaml value.**

The same pattern applies all the way down: `Tx.t` references the
chain, `Input.t` references the tx, etc. Reachability graphs
mirror the lifetime graph.

### Alternative considered and rejected: C-side reference counting

A reference-counted handle in the C ABI (atomic increment on copy,
decrement in `_free`) would shift the problem to the C layer and
remove the back-reference from the OCaml type. Rejected because:

- Atomic increments in the hot path of a 1B-tx scan are not free.
- Reference cycles would still need OCaml-side tracking.
- The problem originates at the GC boundary; the fix belongs there.

The minimum-cost solution is a single non-traversed pointer per
OCaml value. That is what was implemented.

### `Gc.finalise` as safety net, scoped combinators as primary API

`Gc.finalise` is registered for every handle that owns a C resource.
This is necessary but not sufficient: finalisers run at GC's
discretion, with no ordering guarantee.

Where possible, the API offers scoped combinators
(`with_chain conf f`) that guarantee deterministic free at scope
exit. Users who need predictable resource management use these.
Users who don't can rely on the finaliser net.

This is a transferable principle. **GC finalisers are a safety net,
not the primary API. Provide scoped combinators when scoping makes
sense.**

### Threading model

Layer 2's thread-local string buffers introduce a non-obvious
threading constraint. Re-stating the analysis from
[`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md) §"OCaml 5 Concurrency
Interaction":

- **OCaml 5 domains (OS threads): SAFE.** Each domain has its own
  TLS slot. Domains operating on different blocks do not share string
  buffers.
- **Fibers within a domain: UNSAFE if pointers escape.** Fibers share
  TLS. Holding a raw `const char*` across an `Effect.perform` is a
  use-after-write.
- **Mitigation: layer 4 copies strings to OCaml `string` immediately.**
  No raw pointer escapes the call. Fiber-safe by construction.

The deeper analysis of `DataAccess`, `ChainAccess`, `ScriptAccess`,
`HashIndex`, `AddressIndex` thread-safety (read-only mmap, RocksDB
`Get` thread-safety) lives in
[`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md). Summary: a single
`Chain.t` can be shared across multiple OCaml 5 domains for parallel
read-only analysis, provided no domain calls `reload()` or any write
method. The high-level API does not expose these methods to user
code.

---

## 5. Three iteration strategies, discovered by measurement

The most reusable methodology contribution of this project. Three
iteration patterns emerged from measurement, not from a priori
design. They cover every benchmarked query (Q1–Q6) and they coexist
behind a uniform `_range.fold` / `_bag.fold` / `Seq.t` API surface.

### The three strategies

**1. tx-buffer** — `Tx_range.fold`. OCaml-driven loop over a single
heap-allocated tx buffer. Each iteration calls a `[@@noalloc]` C
stub that *reinitialises* the buffer in place via C++ placement-new
(~3 ns per call per [`../README.md`](../README.md) §Performance).
Zero heap allocation per tx in the OCaml-side loop.

- **When it applies.** Tx-level properties (locktime, fee, version,
  hash, …) where the OCaml callback needs a `Tx.t` handle but does
  not need to retain it past the iteration.
- **C-side requirement.** A reusable buffer of the right size and
  alignment, plus a `[@@noalloc]` stub that runs `placement-new` on
  it from the next index.
- **Best for.** Q1 (locktime > 0), Q3 (fee sum).

**2. zero-alloc** — `Tx_range.fold` plus per-output `[@@noalloc]`
direct accessors. Inner loop never materialises an `Output.t` handle
at all. Stubs construct the C++ `Output` (or `Output` + spending
`Transaction`) on the C stack and return a single scalar.

- **When it applies.** Per-output queries where the callback needs a
  small set of properties (`is_spent`, `value`, `spending_block_height`,
  `spending_locktime`) and not a full `Output.t` handle.
- **C-side requirement.** A `[@@noalloc]` stub for each property,
  plus careful API design so the user does not need a handle.
- **Best for.** Q2 (max output), Q5 (zero-conf), Q6 (locktime
  change). Eliminates both heap traffic *and* `caml_callback` re-entry
  beyond the tx-level loop.

**3. zero-copy** — `Output_bag.fold`. Borrowed-pointer iteration over
a C++ `std::vector<Output>` returned by an address query. The
`Output_bag` is GC-managed and frees the vector when collected;
elements seen in the callback are pointers *into* the vector's
storage.

- **When it applies.** Address lookups and heuristic queries that
  return small C++-owned vectors where copying every element to OCaml
  would dominate the work.
- **C-side requirement.** A vector type with documented stability
  (no realloc during iteration), plus a fold stub that walks the
  vector and invokes the OCaml callback with a borrowed handle.
- **Best for.** Q4 (Satoshi Dice address sum).

### Generalisation

The three patterns recur in every GC ↔ C++ binding under different
names:

- **tx-buffer** generalises to **batched buffer reuse**: amortise
  one heap allocation across many iterations by reinitialising in
  place.
- **zero-alloc** generalises to **scalar-returning accessors**: when
  the callback needs only fields, expose the fields directly, not the
  handle.
- **zero-copy** generalises to **borrowed iteration**: when the
  underlying container guarantees stability, hand out borrowed
  pointers to its elements.

The contribution of this project is naming them and showing they can
coexist behind a single OCaml API surface — the user picks the
strategy by choosing the iteration combinator (`Tx_range.fold` vs
`Output_bag.fold` vs `Seq.t`-wrapped equivalents).

### Companion design: dual access patterns

Every range type provides both a `Seq.t` form (composable, ~10–15 ns
per element overhead for closure allocation, per
[`../BENCH.md`](../BENCH.md) and
[`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md) §"Seq.t vs Indexed
Access — Performance Analysis") and an indexed `_range.fold` form
(zero overhead, less composable).

The implementation cost of providing both is near zero: `Seq.t` is a
thin wrapper over the indexed access. Users writing exploratory
analysis use `Seq.t`; users writing performance-critical full-chain
scans use `_range.fold`. The benchmark in
[`../BENCH.md`](../BENCH.md) reports the best strategy per query;
the API does not force the user to commit before measuring.

This is also transferable. **When a fast path and a composable path
have near-identical implementation cost, expose both.** Forcing one
choice on the user pushes the other choice into user-space
workarounds.

---

## 6. Benchmarking and profiling as design discipline

The methodological core of the document. Benchmarking is not a
post-hoc verification step in this project — it is the discipline
that produced the architecture. This section is organised around the
practices that an engineer starting a similar project should adopt
from day one.

### 6.1 Comprehensive coverage matrix

The benchmark grid is not "OCaml versus C++". It is a 4-language ×
6-query × N-strategy grid with no skipped cells, lifted directly from
[`../BENCH.md`](../BENCH.md):

| Layer | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
|---|---|---|---|---|---|---|
| C++ reference (`abi/perf.cpp`) | + | + | + | + | + | + |
| Python fluent (`lib/test/perf.py`) | + | + | + | + | n/a | n/a |
| Python imperative (`lib/test/perf.py`) | n/a | n/a | n/a | n/a | + | + |
| C ABI by-index (`abi/perf.c`) | + | + | + | + | + | + |
| C ABI callback (`abi/perf.c`) | + | + | + | n/a | + | + |
| OCaml low-level (`lib/test/perf.ml`) | + | + | + | + | + | + |
| OCaml high-level — tx-buffer | + | n/a | + | n/a | n/a | n/a |
| OCaml high-level — zero-alloc | n/a | + | n/a | n/a | + | + |
| OCaml high-level — zero-copy | n/a | n/a | n/a | + | n/a | n/a |

The cells marked `n/a` are not skipped to save effort; they reflect
that the strategy genuinely does not apply (e.g. Python's fluent API
cannot express per-output graph traversal with conditionals, so Q5
and Q6 fall back to imperative).

**Why this matters.** Skipping a cell hides where the abstraction
taxes you. A blank cell on "OCaml high-level for Q4" would let you
plausibly claim the binding is cheap. Filling all cells forces an
honest accounting of what each layer costs at each access pattern.

**Reusable principle.** *For any FFI project, benchmark every layer
at every operation.* The cell that's hardest to fill is usually the
most informative, because it identifies where the abstractions
under-cover the access patterns.

### 6.2 Reference implementations bracket the range

Two outsider implementations bracket the binding's performance
range:

- **C++ direct** (`abi/perf.cpp`) — the absolute ceiling.
  Stack-allocated `Transaction` and `Output` value types, no FFI.
  Anything below this is the binding's overhead.
- **Python fluent + imperative** (`lib/test/perf.py`,
  `run_queries.py`) — the ergonomic peer. What an analyst would
  actually write today. Anything close to this proves the project's
  premise.

The latest results from [`../BENCH.md`](../BENCH.md) (full ~1B txes,
hetzner, AMD Ryzen, NVMe):

| Query | C++ | Best OCaml | OCaml/C++ |
|---|---|---|---|
| Q1 | 0.67 s | 1.55 s | 2.3× |
| Q2 | 0.68 s | 2.51 s | 3.7× |
| Q3 | 0.83 s | 1.72 s | 2.1× |
| Q4 | 0.19 s | 0.41 s | 2.2× |
| Q5 | 4.56 s | 6.83 s | 1.5× |
| Q6 | 6.34 s | 9.66 s | 1.5× |

The README's cold-cache full-chain table shows even tighter ratios
(within a few percent of C++) because the bottleneck shifts from FFI
to disk I/O — see [`../README.md`](../README.md) for those numbers
and the cache-effects discussion.

**Reusable principle.** *Always benchmark against both the absolute
ceiling and the comparable peer.* Without the ceiling, you don't know
how much overhead is fundamental. Without the peer, you don't know if
your win matters.

### 6.3 The gap-filling plan: from "how fast?" to "where does the time go?"

The single most important methodological inflection in the project's
benchmarking history.

**Round 1 — "How fast can OCaml get on this query?"** Answered by
wall-clock comparison. Produces the table above. Insufficient on its
own: a 1.5–3.7× slowdown is not a number you can act on.

**Round 2 — "Why does each layer cost what it costs?"** Requires
decomposition. The four-step gap-filling plan from
[`../BENCH.md`](../BENCH.md) §"Gap-filling plan":

- **G1. GC statistics.** Print `Gc.stat()` before and after each
  query to attribute time to minor collections, major collections,
  and heap words allocated. The zero-alloc path should show near-zero
  GC activity, confirming the allocation hypothesis (or refuting it).
- **G2. Multiple runs.** Each benchmark runs 3 times; report the
  minimum. Removes upward noise from background processes and GC
  jitter.
- **G3. Microbenchmark: empty FFI roundtrip.** Add a trivial
  `abi_noop()` and measure 1 B calls from each layer. Separately
  measure 1 B `abi_tx_new_with_index` + `abi_tx_free` cycles. This
  isolates per-call FFI overhead from per-object allocation overhead.
- **G4. Cold vs warm cache.** Run the same query on cold cache (`echo
  3 > /proc/sys/vm/drop_caches`) and warm. If they differ by >2×,
  the headline number measures memory bandwidth, not the binding
  itself.

The expected output, from the same source:

> Q5 OCaml zero-alloc: 6.52 s = 4.97 s (C++ work) + ~0.8 s
> (`caml_callback2` re-entry × 1 B txes) + ~0.7 s (`int64` boxing
> for accumulator) + ~0.05 s (GC).

This is what every FFI project should aim to produce: a per-layer
cost attribution, not a slowdown ratio. **A ratio is a question. A
decomposition is an answer.**

**Reusable principle.** *Wall-clock numbers are a starting point, not
an endpoint.* Without decomposition, you cannot tell whether closing
the gap means optimising your binding or accepting an unavoidable
cost.

### 6.4 Cold cache as the default benchmark regime

`bench_cold.sh` drops `/proc/sys/vm/drop_caches` before each run.
This is unusual and intentional. Most BlockSci queries scan ~205 GB
of mmap'd flat files — far larger than RAM on most hardware. Real
analyst workflows therefore run cold for the first scan, warm
*within* a session for repeated lookups.

Warm-cache numbers measure memory bandwidth, not the binding. For
fairness across hardware (different RAM sizes, different storage),
cold-cache is the regime where the binding's overhead matters most
relative to the I/O baseline. The README headline numbers reflect
this — they show OCaml within a few percent of C++ on the long
queries (Q1, Q3, Q5, Q6) precisely because storage I/O dominates.

**Reusable principle.** *Match the benchmark cache regime to
production.* Pick the regime that reflects what users do, not the
regime that gives the most flattering numbers.

### 6.5 Three feedback loops between measurement and design

Concrete episodes where measurement *changed* the design. Each
follows the same shape: observation → hypothesis → measurement →
design change → re-measurement.

**Episode 1 — Heap allocation in the inner loop.**

- Observation: low-level OCaml benchmark (`lib/test/perf.ml`) was
  several × slower than the C ABI baseline.
- Hypothesis: per-tx `new`/`delete` on the C++ side plus pointer
  wrapping plus `Gc.finalise` registration dominate.
- Measurement: profiled the inner loop; allocation cost was the
  dominant term.
- Design change: introduced `Tx_range.fold` — one heap-allocated tx
  buffer, reinitialised in place per iteration via placement-new
  (~3 ns).
- Re-measurement: tx-buffer strategy closes most of the gap on Q1
  and Q3.

**Episode 2 — `caml_callback` re-entry cost.**

- Observation: even with zero allocation, per-output callbacks were
  not free.
- Hypothesis: each `caml_callback2` re-entry into the OCaml runtime
  costs ~1.5 ns, multiplied by output count (~2.5 B on the full
  chain).
- Measurement: G3 microbenchmark on a noop callback confirmed the
  per-call cost.
- Design change: per-output `[@@noalloc]` direct accessors that
  return a scalar without re-entering the OCaml runtime.
- Re-measurement: zero-alloc strategy cuts Q2/Q5/Q6 by another large
  fraction.

**Episode 3 — `Seq.t` closure overhead.**

- Observation: `Seq.t`-based iteration on hot paths was measurably
  slower than indexed iteration.
- Hypothesis: the `Cons` node + next-closure allocation cost ~10–15 ns
  per element (per `DESIGN_REVIEW.md` §"Seq.t vs Indexed Access —
  Performance Analysis").
- Measurement: micro-benchmark comparing `Seq.iter` over `Block.txs`
  versus a `for` loop over `Block.tx_count` + `Block.tx`.
- Design change: keep `Seq.t` for composition; expose `_range.fold`
  for hot loops. Both wrap the same C calls.
- Re-measurement: `_range.fold` matches the indexed path, `Seq.t`
  remains the ergonomic default.

**The canonical loop.** Each episode produced both a strategy (now
permanent in the API) and a measurement (now permanent in
[`../BENCH.md`](../BENCH.md)). The strategies stack: a query may use
all three patterns at once (e.g. Q5 uses tx-buffer for the tx loop
plus zero-alloc per-output accessors). The pattern recurs in any
serious FFI binding. **Profile, hypothesise, measure, change,
re-measure. Repeat until the cost decomposition is acceptable.**

---

## 7. Testing strategy

Tests are arranged so that a regression localises to a single layer.
Same operations, different layers — when the C test passes and the
OCaml test fails, the bug is in the binding, not the engine.

| Layer | Test target | Purpose |
|---|---|---|
| C ABI from C | `abi/check.c` | Boundary correctness from a C client. Catches NULL handling, lifetime sequencing. |
| C ABI from C++ | `abi/check.cpp` | Same operations from C++ client. Confirms `reinterpret_cast` and exception barrier behave identically. |
| Memory correctness | `make check-asan-build` | ASan/UBSan rebuild of the C tests. Catches leaks, double-free, use-after-free, buffer overflows. The highest-value testing investment for an FFI project. |
| Low-level OCaml | `lib/test/check.ml` | ctypes correctness without GC integration. |
| High-level OCaml | `lib/blocksci/test/check.ml` | Public API: variants, options, GC-managed lifetimes, string copies. |
| GC stress | (in `lib/blocksci/test/`) | Deliberate GC pressure with deep ownership chains; validates back-references (see [`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md) §1.3.3 and `../PLAN.md` §1.3.3). |
| Mock substitution | `lib_mock/`, `lib/blocksci/test_mock/` | Synthetic ABI mock library exposing the same C ABI but against a deterministic in-memory universe. Hermetic CI without a real BlockSci dataset. |
| Record/replay | `lib/blocksci/test_replay/`, `fixtures/` | Capture real BlockSci answers to `abi_*` calls into a JSON fixture; replay through the mock. See [`record-replay.md`](record-replay.md). |
| Fuzzing | `fuzz/` | afl++ harnesses on string-ingesting entry points (`fuzz_address_of_string`, `fuzz_tx_of_hash`, `fuzz_chain_new`). |

There is no Python test suite in the binding repo; Python is treated
as an external reference (see §6.2), not a code path to validate.

**Reusable principle.** *Test parity at every layer.* C test, C++
test, OCaml test — same operations, same expected results. Turns
"where is the bug?" into "which test failed?" Localises every
regression to a single layer. The cost of writing a parallel test in
each layer is small; the cost of debugging an FFI bug without layer
isolation is large.

---

## 8. Reproducing this approach on other bindings

The transferable conclusion. The audience is an engineer starting,
say, a Rust binding to a Python ML library, a Java binding to a C++
graph database, a Go binding to a C cryptography library — any
binding between a tracing-GC language and a performance-critical
non-GC library.

The numbered checklist below is induced from this project. Items 1,
2, 4, 6 are universal regardless of language pair. Items 3, 5, 7, 8
are universal in shape but the specifics vary. Item 9 is on the
team.

### A nine-step checklist

1. **Identify the boundary's two pressures.** Performance and
   ergonomy in this project; in your project, the names may be the
   same or may differ ("safety", "memory", "concurrency"). Name what
   "idiomatic" looks like on each side. Write it down — when a design
   choice arises later, you will refer back to this list to pick.

2. **Build all four layers, even if some are thin.** Native library →
   stable C ABI → low-level binding → high-level binding. Skipping
   the C ABI layer (e.g. binding C++ directly via SWIG, or using
   `extern "C++"` with a non-stable mangling) couples your binding to
   compiler ABI quirks. The C ABI is a stability boundary that pays
   for itself the first time you ship across a compiler upgrade.

3. **Apply a concern checklist before writing the high-level layer.**
   The ten concerns in [`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md)
   generalise:
   - Ownership graph (who frees what)
   - Error sentinels (NULL, -1, 0, exceptions)
   - String lifetime (returned pointers, buffers, copies)
   - Inheritance flattening
   - Bounds checking on indexed accessors
   - GC integration vs manual free
   - `option` / null safety in user-facing types
   - High-level type re-encoding
   - Lifetime coupling across the GC boundary
   - Const-correctness on returned data

   Walk this list before writing user-facing API. Most of the work
   happens here, not in the implementation.

4. **Build the benchmark grid before optimising.** Every layer ×
   every operation × every strategy, with reference implementations
   on both sides (the absolute ceiling and the comparable peer). The
   grid is the substrate for every later decision.

5. **Run a G1–G4-style decomposition early.** GC stats, multiple
   runs, microbenchmarks for FFI roundtrip and alloc cycle, cold/warm
   cache. Decomposition often identifies easy wins that wall-clock
   comparison would miss — and tells you when a "win" is actually
   measurement noise.

6. **Discover strategies by measurement; don't pre-design them.**
   The tx-buffer / zero-alloc / zero-copy taxonomy emerged from
   access patterns observed in real queries, not from a whiteboard
   session. If the API forces a single iteration model on the user,
   you will discover at the worst time that a query needed something
   else. Provide a small palette and let the user pick.

7. **Make the GC ↔ RAII bridge a load-bearing design artefact.**
   Back-references, finaliser policy, scoped combinators. Document
   it. Test it under GC pressure (see §7). This is the most common
   place for subtle bugs in tracing-GC bindings; it deserves explicit
   design attention.

8. **Test parity at every layer.** Same operations, same fixtures,
   same expected results. Add a memory-safety target (ASan / UBSan /
   Valgrind / MSan as appropriate). Add fuzzing on string-ingesting
   entry points.

9. **Maintain the document.** A design document that doesn't track
   the codebase becomes worse than no document, because readers will
   trust it and act on stale information. Re-review after every
   significant change to the FFI surface.

### What is genuinely transferable

- The four-layer architecture with explicit per-boundary contracts.
- The principle that strategies are discovered, not designed.
- The benchmark grid as a design substrate.
- The decomposition discipline: ratios are questions, decompositions
  are answers.
- Layer-parallel testing.
- Back-references for GC ↔ RAII lifetime bridging.

### What is project-specific

- The exact strategy palette (tx-buffer / zero-alloc / zero-copy)
  works because BlockSci's data model has short-lived value types and
  bounded vectors. A library with infinite streams or large mutable
  graphs would discover a different palette.
- Cold-cache as the default regime is right for BlockSci's
  full-chain scans; it would be wrong for a low-latency RPC binding.
- Thread-local string buffers work because layer 4 copies eagerly. A
  binding that returns lazily-evaluated strings would need a
  different policy.

The right transfer is the methodology, not the specific solutions.
Apply the checklist; let the answers emerge from measurement on your
domain.

---

## 9. Cross-references and further reading

| For… | See… |
|---|---|
| Headline performance results | [`../README.md`](../README.md) §Performance |
| Concrete numbers and run logs | [`../RESULTS.md`](../RESULTS.md), `../bench-*.log` |
| Query-level methodology, coverage matrix, gap-filling plan | [`../BENCH.md`](../BENCH.md) |
| Concern catalogue (10 design concerns + resolutions) | [`../DESIGN_REVIEW.md`](../DESIGN_REVIEW.md) |
| Implementation roadmap (Phase 1 / Phase 2) | [`../PLAN.md`](../PLAN.md) |
| Maturity audit (2026-04-22 baseline) | [`../AUDIT.md`](../AUDIT.md) |
| CIOH clustering shortfall and proposed ABI extension | [`../lonnrot.md`](../lonnrot.md) |
| Record/replay testing infrastructure | [`record-replay.md`](record-replay.md) |
| Original engine paper (Kalodner et al., USENIX Security 2020) | `../blocksci-paper.pdf` |

This document synthesises and links — it does not duplicate. When in
doubt about a number or a specific design decision, follow the
cross-reference to the authoritative source.
