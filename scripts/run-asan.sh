#!/bin/sh
# Run the OCaml test suite under AddressSanitizer.
#
# Nothing here is compiled with -fsanitize=address: not the OCaml
# runtime, not the ctypes stubs, not libxgboost. LD_PRELOAD only installs
# ASan's runtime, i.e. its malloc/free/new/delete and libc interceptors.
# That catches:
#   - double frees, frees of non-heap or foreign pointers, and
#     malloc/delete (new/free) mismatches;
#   - use-after-free and out-of-bounds heap accesses, but only when they
#     go through an intercepted libc function (memcpy, memset, strlen,
#     ...).
# Direct loads and stores in uninstrumented code are not checked, so a
# plain use-after-free or overflow inside libxgboost or the stubs goes
# unnoticed. Catching those would need libxgboost (e.g. the vendored
# build) compiled with -fsanitize=address.
#
# Leak detection is disabled because OCaml programs do not fully release
# their heap at exit (false positives).
#
# The fixture-parity tests (test/{bindings,xgboost}/check.ml) skip
# themselves under ASan because libxgboost's hist tree training has
# OpenMP-reduction-order non-determinism that depends on heap layout —
# the fixture predictions captured under one regime drift by a few
# percent under another.
#
# Usage:  ./scripts/run-asan.sh

set -eu

LIBASAN=$(gcc --print-file-name=libasan.so)
if [ ! -e "$LIBASAN" ]; then
  echo "libasan.so not found; install gcc or set LIBASAN explicitly" >&2
  exit 1
fi

export LD_PRELOAD="$LIBASAN"
export ASAN_OPTIONS="detect_leaks=0:abort_on_error=0:print_stacktrace=1"

opam exec -- dune runtest --force "$@"
