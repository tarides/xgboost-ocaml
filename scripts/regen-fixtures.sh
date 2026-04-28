#!/bin/sh
# Regenerate the deterministic test fixtures.
#
# Run from the repo root after a libxgboost upgrade or any change to
# bin/c_reference/check.c. The output files (test/fixtures/*) should
# be diffed by hand and the change explained in the commit message.

set -eu

cd "$(dirname "$0")/.."

echo "[regen] building C reference..."
make -C bin/c_reference check >/dev/null

echo "[regen] capturing C reference predictions -> test/fixtures/check_predictions.txt"
./bin/c_reference/check > test/fixtures/check_predictions.txt

echo "[regen] done. Inspect:"
diff -u test/fixtures/check_predictions.txt \
        test/fixtures/check_predictions.txt.prev 2>/dev/null \
  || true
