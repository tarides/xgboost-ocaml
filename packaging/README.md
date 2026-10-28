# Packaging templates

## `conf-libxgboost.opam`

`conf-libxgboost` is a virtual (`flags: conf`) opam package: it has no
source of its own, it only probes — through `pkg-config` — for a system
`libxgboost` and declares the `depexts` that install it where a platform
packages it. By opam convention such `conf-*` packages are maintained
**directly in `ocaml/opam-repository`**, not generated from an upstream
source tree.

This copy is kept here only as the versioned source of truth, so the
design travels with the bindings and the opam-repository entry can be
regenerated from it. It is deliberately **not** at the repo root and has
**no** matching `dune-project` package stanza, so neither `dune`,
`opam pin`, nor `opam publish` treats it as a package produced by this
repository.

How it pairs with `xgboost.opam`:

- `xgboost` depends on `conf-libxgboost` only on platforms whose package
  manager ships `libxgboost >= 3.0` (the conservative filter in
  `xgboost.opam`). That dependency is what gates the vendored-vs-system
  decision, so it is kept narrow: an omitted platform merely vendors,
  whereas pulling `conf-libxgboost` where the system package is absent
  would hard-fail with no fallback.
- `conf-libxgboost` itself is a generic, version-agnostic presence probe
  (`pkg-config --exists`). The `>= 3.0` requirement is enforced at build
  time by `config/discover.ml`, which falls back to the vendored build
  when the system library is absent or too old.

When releasing, publish/refresh `conf-libxgboost` in opam-repository
from this template (dropping nothing — conf packages carry no `url`).
