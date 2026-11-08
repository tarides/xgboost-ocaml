#!/bin/sh
# Build, test and check the installed linking of xgboost on the vendored
# path, inside an ocaml/opam image (CI uses debian-12, which packages
# only libxgboost < 3.0 and is outside conf-libxgboost's filter).
#
# Run as the image's opam user, with the checkout mounted read-only at
# /src:
#
#   docker run --rm -v "$PWD:/src:ro" ocaml/opam:debian-12-ocaml-5.5 \
#     sh /src/scripts/check-vendored-container.sh
#
# opam installs the dependencies and the depexts they declare (cmake, a
# C++ compiler, OpenMP, pkg-config) as it would for a user, and nothing
# forces the vendored path: config/provider has to pick it on its own,
# which check-installed-linking.sh asserts. To keep the CMake build
# between runs, set DUNE_CACHE_ROOT to a mounted directory and
# DUNE_CACHE_STORAGE_MODE=copy: the mount is on another filesystem than
# the build, where dune's default hardlink mode silently stores nothing.
#
# Unlike check-distro-install.sh, this doesn't opam install the pinned
# package: that would build XGBoost a second time for
# check-installed-linking.sh, which installs from this build instead.

set -eu

export OPAMYES=1 OPAMCONFIRMLEVEL=unsafe-yes OMP_NUM_THREADS=1

. /etc/os-release
echo "== $ID ${VERSION_ID:-rolling}, $(opam exec -- ocamlc -version)"

# The image's opam-repository is a snapshot; bring it up to date.
git -C "$HOME/opam-repository" pull -q origin master
opam update -q

cp -R /src "$HOME/xgboost"
rm -rf "$HOME/xgboost/_build"
cd "$HOME/xgboost"

opam install ./xgboost.opam --deps-only --with-test

# DUNE_CACHE=enabled also caches user rules (the CMake build is one);
# dune's default mode skips them. check-installed-linking.sh keeps this
# setting in REUSE_BUILD mode, so it doesn't rebuild XGBoost.
export DUNE_CACHE=enabled
opam exec -- dune build -p xgboost -j 4
opam exec -- dune runtest -p xgboost -j 4
REUSE_BUILD=1 bash scripts/check-installed-linking.sh "$HOME/xgboost" vendored
