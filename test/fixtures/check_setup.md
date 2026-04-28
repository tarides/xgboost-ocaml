# Reference fixture: check_predictions.txt

The five floats in `check_predictions.txt` are the predictions of a
deterministic small training run, used as a cross-layer parity oracle.
If the C reference, layer B, and layer C all reproduce these to 1e-5,
the binding's training/predict pipeline agrees with libxgboost itself.

## Setup that produced the fixture

- 200 rows × 16 cols, deterministic xorshift32 PRNG seeded with
  `0xCAFEBABEu` (see `bin/c_reference/bench_common.h`).
- Labels: binary threshold of a fixed linear projection of the first
  4 columns (see `gen_labels_binary`).
- Booster: `binary:logistic`, `tree_method=hist`, `max_depth=4`,
  `seed=0`, `nthread=1`, `verbosity=0`. (`nthread=1` makes the fixture
  reproducible across machines, OMP thread counts, and ASan vs
  no-ASan; hist's OpenMP reduction order is non-deterministic across
  memory layouts otherwise.)
- 30 boost iterations.
- Predict on the training matrix; report the first 5 outputs.

## Regenerate

```sh
make -C bin/c_reference check
./bin/c_reference/check > test/fixtures/check_predictions.txt
```

If the upstream libxgboost version changes its training algorithm
(rare across patch releases), the fixture must be regenerated and any
diff to the prior version flagged in the commit message.

The OCaml-side reference matrix `random_100x10.bin` (Bigarray-format)
plus its predict-output digest will land alongside this when the
`scripts/regen-fixtures.sh` workflow is generalised.
