# Vendored XGBoost

`xgboost/` is a pruned copy of the upstream XGBoost C++ sources, used as
the **fallback** native dependency: `config/discover.ml` builds it with
CMake and links it statically when no system libxgboost >= 3.0 is found
(see the "Installation" section of the top-level README). It is bundled
so the build needs no network — which is what lets `opam install xgboost`
work in the opam-repository CI sandbox and on distros without a package.

## Platform coverage

Snapshot as of 2026-10-20. On most opam-repository CI targets the build
takes this vendored path, because no usable system libxgboost exists
there. The fast path needs three things: a packaged libxgboost >= 3.0,
an `xgboost.pc` for pkg-config, and a `depexts` entry in `xgboost.opam`
so that opam installs it.

The targets come from opam-repo-ci (`lib/build.ml`) and
`ocaml-dockerfile`'s active distro list (`src-opam/distro.ml`, `main`
branch). The deployed CI may lag behind the newest entries.

| CI target | Packaged libxgboost | depext | Path |
|---|---|---|---|
| Debian 13 trixie (main distro; also arm64/ppc64le/s390x) | 3.0.0 | `libxgboost-dev` | system |
| Debian 12 bookworm | 1.7.4 (too old) | `libxgboost-dev` | vendored |
| Debian testing (forky) / unstable | 3.3.0 | `libxgboost-dev` | system |
| Ubuntu 22.04 / 24.04 (also riscv64) / 25.04 | 1.5.2 / 2.0.3 / 2.1.4 (too old) | `libxgboost-dev` | vendored |
| Ubuntu 25.10 / 26.04 | 3.0.0 / 3.0.4 | `libxgboost-dev` | system |
| Alpine 3.24 | none | — | vendored |
| Arch | none (AUR only) | — | vendored |
| CentOS Stream 9 / 10 | none (not in EPEL either) | — | vendored |
| Fedora 42 / 43 / 44 | none | — | vendored |
| openSUSE Leap 16.0 | none | — | vendored |
| openSUSE Tumbleweed | 2.0.0 (too old) | — | vendored |
| macOS Homebrew (arm64, x86_64) | 3.4.2 | `xgboost` | system |
| FreeBSD 15.1 | `misc/xgboost` 3.4.1 | none | vendored |

Debian's `libxgboost-dev` ships `xgboost.pc`. opam treats Ubuntu as
`os-family = "debian"`, so on Ubuntu <= 25.04 and Debian 12 the depext
installs a libxgboost that `config/provider/detect.ml` then rejects as < 3.0. FreeBSD
could use the system path with a depext, provided the port installs
`xgboost.pc` (not verified).

Sources: Repology, sources.debian.org and madison, Launchpad, Fedora
mdapi, and the Arch, Alpine and openSUSE package indexes.

## Provenance

- Upstream: https://github.com/dmlc/xgboost
- Tag: **v3.0.4** (with submodules `dmlc-core`, `gputreeshap`)
- Obtained with:

  ```sh
  git clone --recurse-submodules --shallow-submodules --depth 1 \
    --branch v3.0.4 https://github.com/dmlc/xgboost.git
  ```

## Pruning

Everything not needed for a CPU-only static build was removed to keep the
release tarball small (~5 MB). Removed from the XGBoost tree:

    doc demo tests R-package jvm-packages gputreeshap python-package
    dev ops amalgamation .github CITATION SECURITY.md CONTRIBUTORS.md
    README.md NEWS.md

and from `dmlc-core/`:

    test doc example scripts tracker .github appveyor

Kept: `CMakeLists.txt`, `cmake/`, `include/`, `src/`, `plugin/` (referenced
unconditionally by the top-level CMakeLists), `dmlc-core/{src,include,cmake,
CMakeLists.txt,make,...}`, and `LICENSE`. The R-package, jvm-packages, tests
and GPU (gputreeshap/CUDA) subtrees are all gated OFF in the CMake build we
run (`-DUSE_CUDA=OFF`, `-DBUILD_STATIC_LIB=ON`, `-DUSE_OPENMP=ON`).

## Updating

Re-clone at the new tag, re-apply the pruning above, update the tag here,
and rebuild both paths (see the repo README's verification steps).

Local CMake change: the macOS OpenMP dylib post-processing is skipped
when `BUILD_STATIC_LIB` is enabled. Preserve this guard when updating
until the bundled upstream version includes it.
