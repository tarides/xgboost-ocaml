/*
 * check.c — pure-C correctness sanity test for the xgboost-ocaml C reference.
 *
 * Trains a small binary:logistic model on a deterministic dataset and
 * prints the first 5 predictions. The expected outputs are captured as a
 * fixture by `scripts/regen-fixtures.sh` and checked back in test/fixtures/.
 *
 * The OCaml bindings tests call this same model deterministically and
 * compare predictions to ~1e-5; if both diverge from the captured fixture
 * the bug is in the binding (or in libxgboost itself).
 *
 * Usage: ./check
 * Output: 5 floats, one per line, in scientific notation.
 */

#include "bench_common.h"

#include <math.h>

int main(void) {
  const size_t rows = 200;
  const size_t cols = 16;
  const int iters = 30;
  const uint32_t seed = 0xCAFEBABEu;

  xs32 s = seed;
  float *data = malloc(rows * cols * sizeof(float));
  float *labels = malloc(rows * sizeof(float));
  if (!data || !labels) { fprintf(stderr, "check: oom\n"); abort(); }
  gen_dense(data, rows, cols, &s);
  gen_labels_binary(labels, data, rows, cols);

  DMatrixHandle dtrain = NULL;
  XGB_OK(XGDMatrixCreateFromMat(data, (bst_ulong)rows, (bst_ulong)cols, NAN,
                                &dtrain));
  XGB_OK(XGDMatrixSetFloatInfo(dtrain, "label", labels, (bst_ulong)rows));

  BoosterHandle booster = NULL;
  XGB_OK(XGBoosterCreate(&dtrain, 1, &booster));
  XGB_OK(XGBoosterSetParam(booster, "objective", "binary:logistic"));
  XGB_OK(XGBoosterSetParam(booster, "tree_method", "hist"));
  XGB_OK(XGBoosterSetParam(booster, "max_depth", "4"));
  XGB_OK(XGBoosterSetParam(booster, "seed", "0"));
  XGB_OK(XGBoosterSetParam(booster, "verbosity", "0"));
  for (int it = 0; it < iters; ++it) {
    XGB_OK(XGBoosterUpdateOneIter(booster, it, dtrain));
  }

  bst_ulong out_len = 0;
  const float *out = NULL;
  XGB_OK(XGBoosterPredict(booster, dtrain, 0, 0, 0, &out_len, &out));
  if (out_len < 5) {
    fprintf(stderr, "check: predict returned %llu values, expected >= 5\n",
            (unsigned long long)out_len);
    return 1;
  }

  for (int i = 0; i < 5; ++i) {
    printf("%.9e\n", out[i]);
  }

  XGB_OK(XGBoosterFree(booster));
  XGB_OK(XGDMatrixFree(dtrain));
  free(data);
  free(labels);
  return 0;
}
