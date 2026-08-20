# Vendored XGBoost

`xgboost/` is a pruned copy of the upstream XGBoost C++ sources, used as
the **fallback** native dependency: `config/discover.ml` builds it with
CMake and links it statically when no system libxgboost >= 3.0 is found
(see the "Installation" section of the top-level README). It is bundled
so the build needs no network — which is what lets `opam install xgboost`
work in the opam-repository CI sandbox and on distros without a package.

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
