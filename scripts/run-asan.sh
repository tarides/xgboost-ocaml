#!/bin/sh
# Run the OCaml test suite under AddressSanitizer.
#
# We don't link the OCaml binaries against ASan (would require
# rebuilding OCaml with -fsanitize=address). Instead, LD_PRELOAD
# libasan.so so ASan instruments libxgboost (and our shim if any) at
# load time. Leak detection is disabled because OCaml programs do not
# fully release their heap at exit (false positives); use-after-free,
# double-free, and invalid-write detection remain active.
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
