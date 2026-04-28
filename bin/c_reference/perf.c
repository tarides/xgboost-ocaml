/*
 * perf.c — pure-C reference benchmark harness for xgboost-ocaml.
 *
 * Implements the workload grid documented in bench/README.md:
 *   W1  train tiny      (1k x 50,  100 iters, reg:squarederror)
 *   W2  train large     (1M x 100, 100 iters, binary:logistic, hist)
 *   W3  batch predict   (1M-row predict on a W2-style model)
 *   W4  online predict  (100k single-row predict on a W2-style model)
 *   W5  dmatrix dense   (XGDMatrixCreateFromMat at user-supplied size)
 *   W6  dmatrix csr     (XGDMatrixCreateFromCSREx at user-supplied size, density)
 *
 * Output is one CSV line per run on stdout, header on the first line:
 *   workload,rows,cols,iters,density,repeat,min_ms,mean_ms,max_ms
 *
 * Usage:
 *   ./perf --workload W1                            (default sizes)
 *   ./perf --workload W2 --rows 1000000 --cols 100 --iters 100 --repeat 5
 *   ./perf --workload W6 --rows 100000 --cols 100 --density 0.05
 *
 * The model used by W3 and W4 is trained inside the run() of those workloads
 * (its training cost is excluded from the timed region). Train sizes for the
 * model are the same as W2's defaults.
 */

#include "bench_common.h"

#include <math.h>
#include <stdbool.h>

/* ---------- options ---------- */

typedef struct {
  const char *workload;
  size_t rows;
  size_t cols;
  int iters;
  float density;
  int repeat;
  int warmup;
  uint32_t seed;
  bool verbose;
} opts;

static const opts opts_defaults = {
    .workload = NULL,
    .rows = 0,
    .cols = 0,
    .iters = 0,
    .density = 0.05f,
    .repeat = 5,
    .warmup = 1,
    .seed = 0xC0FFEEu,
    .verbose = false,
};

static void usage(FILE *out) {
  fprintf(out,
          "usage: perf --workload {W1|W2|W3|W4|W5|W6} [--rows N] [--cols N]\n"
          "             [--iters N] [--density F] [--repeat K] [--seed N]\n"
          "             [--verbose]\n");
}

static int parse_int(const char *s, int *out) {
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') return -1;
  *out = (int)v;
  return 0;
}

static int parse_size(const char *s, size_t *out) {
  char *end = NULL;
  unsigned long long v = strtoull(s, &end, 10);
  if (end == s || *end != '\0') return -1;
  *out = (size_t)v;
  return 0;
}

static int parse_args(int argc, char **argv, opts *o) {
  *o = opts_defaults;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      usage(stdout);
      exit(0);
    } else if (!strcmp(a, "--workload") && i + 1 < argc) {
      o->workload = argv[++i];
    } else if (!strcmp(a, "--rows") && i + 1 < argc) {
      if (parse_size(argv[++i], &o->rows)) return -1;
    } else if (!strcmp(a, "--cols") && i + 1 < argc) {
      if (parse_size(argv[++i], &o->cols)) return -1;
    } else if (!strcmp(a, "--iters") && i + 1 < argc) {
      if (parse_int(argv[++i], &o->iters)) return -1;
    } else if (!strcmp(a, "--density") && i + 1 < argc) {
      o->density = strtof(argv[++i], NULL);
    } else if (!strcmp(a, "--repeat") && i + 1 < argc) {
      if (parse_int(argv[++i], &o->repeat)) return -1;
    } else if (!strcmp(a, "--warmup") && i + 1 < argc) {
      if (parse_int(argv[++i], &o->warmup)) return -1;
    } else if (!strcmp(a, "--seed") && i + 1 < argc) {
      int s;
      if (parse_int(argv[++i], &s)) return -1;
      o->seed = (uint32_t)s;
    } else if (!strcmp(a, "--verbose")) {
      o->verbose = true;
    } else {
      fprintf(stderr, "unknown argument: %s\n", a);
      usage(stderr);
      return -1;
    }
  }
  if (!o->workload) {
    fprintf(stderr, "--workload is required\n");
    usage(stderr);
    return -1;
  }
  return 0;
}

/* ---------- defaults per workload ---------- */

static void apply_workload_defaults(opts *o) {
  if (!strcmp(o->workload, "W1")) {
    if (!o->rows) o->rows = 1000;
    if (!o->cols) o->cols = 50;
    if (!o->iters) o->iters = 100;
  } else if (!strcmp(o->workload, "W2") || !strcmp(o->workload, "W3") ||
             !strcmp(o->workload, "W4")) {
    if (!o->rows) o->rows = 1000000;
    if (!o->cols) o->cols = 100;
    if (!o->iters) o->iters = 100;
  } else if (!strcmp(o->workload, "W5")) {
    if (!o->rows) o->rows = 100000;
    if (!o->cols) o->cols = 100;
  } else if (!strcmp(o->workload, "W6")) {
    if (!o->rows) o->rows = 100000;
    if (!o->cols) o->cols = 100;
  } else if (!strcmp(o->workload, "G3")) {
    if (!o->iters) o->iters = 10000000;
  }
}

/* ---------- shared training helper ---------- */

/* Train a binary:logistic hist booster on a fresh dense matrix of the given
 * size. Returns the booster (caller frees) and writes the dtrain handle to
 * *out_dtrain (caller frees). */
static void train_model(size_t rows, size_t cols, int iters, uint32_t seed,
                        BoosterHandle *out_booster, DMatrixHandle *out_dtrain) {
  xs32 s = seed;
  float *data = malloc(rows * cols * sizeof(float));
  float *labels = malloc(rows * sizeof(float));
  if (!data || !labels) {
    fprintf(stderr, "train_model: oom\n");
    abort();
  }
  gen_dense(data, rows, cols, &s);
  gen_labels_binary(labels, data, rows, cols);

  DMatrixHandle dtrain = NULL;
  XGB_OK(XGDMatrixCreateFromMat(data, (bst_ulong)rows, (bst_ulong)cols,
                                NAN, &dtrain));
  XGB_OK(XGDMatrixSetFloatInfo(dtrain, "label", labels, (bst_ulong)rows));

  BoosterHandle booster = NULL;
  XGB_OK(XGBoosterCreate(&dtrain, 1, &booster));
  XGB_OK(XGBoosterSetParam(booster, "objective", "binary:logistic"));
  XGB_OK(XGBoosterSetParam(booster, "tree_method", "hist"));
  XGB_OK(XGBoosterSetParam(booster, "max_depth", "6"));
  XGB_OK(XGBoosterSetParam(booster, "verbosity", "0"));
  for (int it = 0; it < iters; ++it) {
    XGB_OK(XGBoosterUpdateOneIter(booster, it, dtrain));
  }

  free(data);
  free(labels);
  *out_booster = booster;
  *out_dtrain = dtrain;
}

/* ---------- W1: train tiny (reg:squarederror) ---------- */

static uint64_t run_W1(const opts *o) {
  xs32 s = o->seed;
  float *data = malloc(o->rows * o->cols * sizeof(float));
  float *labels = malloc(o->rows * sizeof(float));
  gen_dense(data, o->rows, o->cols, &s);
  gen_labels_reg(labels, data, o->rows, o->cols);

  DMatrixHandle dtrain = NULL;
  XGB_OK(XGDMatrixCreateFromMat(data, (bst_ulong)o->rows, (bst_ulong)o->cols,
                                NAN, &dtrain));
  XGB_OK(XGDMatrixSetFloatInfo(dtrain, "label", labels, (bst_ulong)o->rows));

  BoosterHandle booster = NULL;
  XGB_OK(XGBoosterCreate(&dtrain, 1, &booster));
  XGB_OK(XGBoosterSetParam(booster, "objective", "reg:squarederror"));
  XGB_OK(XGBoosterSetParam(booster, "verbosity", "0"));

  bench_timer t;
  timer_start(&t);
  for (int it = 0; it < o->iters; ++it) {
    XGB_OK(XGBoosterUpdateOneIter(booster, it, dtrain));
  }
  uint64_t ns = timer_elapsed_ns(&t);

  XGB_OK(XGBoosterFree(booster));
  XGB_OK(XGDMatrixFree(dtrain));
  free(data);
  free(labels);
  return ns;
}

/* ---------- W2: train large (binary:logistic, hist) ---------- */

static uint64_t run_W2(const opts *o) {
  xs32 s = o->seed;
  float *data = malloc(o->rows * o->cols * sizeof(float));
  float *labels = malloc(o->rows * sizeof(float));
  if (!data || !labels) { fprintf(stderr, "W2: oom\n"); abort(); }
  gen_dense(data, o->rows, o->cols, &s);
  gen_labels_binary(labels, data, o->rows, o->cols);

  DMatrixHandle dtrain = NULL;
  XGB_OK(XGDMatrixCreateFromMat(data, (bst_ulong)o->rows, (bst_ulong)o->cols,
                                NAN, &dtrain));
  XGB_OK(XGDMatrixSetFloatInfo(dtrain, "label", labels, (bst_ulong)o->rows));

  BoosterHandle booster = NULL;
  XGB_OK(XGBoosterCreate(&dtrain, 1, &booster));
  XGB_OK(XGBoosterSetParam(booster, "objective", "binary:logistic"));
  XGB_OK(XGBoosterSetParam(booster, "tree_method", "hist"));
  XGB_OK(XGBoosterSetParam(booster, "max_depth", "6"));
  XGB_OK(XGBoosterSetParam(booster, "verbosity", "0"));

  bench_timer t;
  timer_start(&t);
  for (int it = 0; it < o->iters; ++it) {
    XGB_OK(XGBoosterUpdateOneIter(booster, it, dtrain));
  }
  uint64_t ns = timer_elapsed_ns(&t);

  XGB_OK(XGBoosterFree(booster));
  XGB_OK(XGDMatrixFree(dtrain));
  free(data);
  free(labels);
  return ns;
}

/* ---------- W3: batch predict on a freshly-trained model ---------- */

static uint64_t run_W3(const opts *o) {
  BoosterHandle booster = NULL;
  DMatrixHandle dtrain = NULL;
  /* train ahead of the timed region */
  train_model(o->rows, o->cols, o->iters, o->seed, &booster, &dtrain);

  /* fresh predict matrix (different seed → different rows) */
  xs32 s = o->seed ^ 0xA5A5A5A5u;
  float *pdata = malloc(o->rows * o->cols * sizeof(float));
  gen_dense(pdata, o->rows, o->cols, &s);
  DMatrixHandle dpred = NULL;
  XGB_OK(XGDMatrixCreateFromMat(pdata, (bst_ulong)o->rows, (bst_ulong)o->cols,
                                NAN, &dpred));

  bst_ulong out_len = 0;
  const float *out = NULL;
  bench_timer t;
  timer_start(&t);
  XGB_OK(XGBoosterPredict(booster, dpred, /*option_mask*/ 0,
                          /*ntree_limit*/ 0, /*training*/ 0, &out_len, &out));
  /* touch the output to defeat any lazy materialisation */
  volatile float sink = 0.f;
  for (bst_ulong i = 0; i < out_len; ++i) sink += out[i];
  (void)sink;
  uint64_t ns = timer_elapsed_ns(&t);

  XGB_OK(XGDMatrixFree(dpred));
  XGB_OK(XGDMatrixFree(dtrain));
  XGB_OK(XGBoosterFree(booster));
  free(pdata);
  return ns;
}

/* ---------- W4: online predict (single-row in tight loop) ---------- */

static uint64_t run_W4(const opts *o) {
  BoosterHandle booster = NULL;
  DMatrixHandle dtrain = NULL;
  /* train on W2 size; predict 1 row at a time, repeated rows times */
  train_model(o->rows, o->cols, o->iters, o->seed, &booster, &dtrain);

  size_t n_pred = (o->rows < 100000 ? o->rows : 100000);
  xs32 s = o->seed ^ 0xA5A5A5A5u;
  float *prow = malloc(o->cols * sizeof(float));
  if (!prow) { fprintf(stderr, "W4: oom\n"); abort(); }

  /* warm: prebuild one DMatrix and predict to ensure caches are hot */
  for (size_t c = 0; c < o->cols; ++c) prow[c] = xs32_uniform(&s);
  {
    DMatrixHandle d1 = NULL;
    XGB_OK(XGDMatrixCreateFromMat(prow, 1, (bst_ulong)o->cols, NAN, &d1));
    bst_ulong ol = 0; const float *o1 = NULL;
    XGB_OK(XGBoosterPredict(booster, d1, 0, 0, 0, &ol, &o1));
    XGB_OK(XGDMatrixFree(d1));
  }

  volatile float sink = 0.f;
  bench_timer t;
  timer_start(&t);
  for (size_t i = 0; i < n_pred; ++i) {
    for (size_t c = 0; c < o->cols; ++c) prow[c] = xs32_uniform(&s);
    DMatrixHandle d1 = NULL;
    XGB_OK(XGDMatrixCreateFromMat(prow, 1, (bst_ulong)o->cols, NAN, &d1));
    bst_ulong ol = 0;
    const float *out = NULL;
    XGB_OK(XGBoosterPredict(booster, d1, 0, 0, 0, &ol, &out));
    sink += out[0];
    XGB_OK(XGDMatrixFree(d1));
  }
  uint64_t ns = timer_elapsed_ns(&t);
  (void)sink;

  XGB_OK(XGDMatrixFree(dtrain));
  XGB_OK(XGBoosterFree(booster));
  free(prow);
  return ns;
}

/* ---------- W5: DMatrix from dense ----------
 * Uses the modern XGDMatrixCreateFromDense (JSON __array_interface__).
 * The deprecated XGDMatrixCreateFromMat is ~30-35% slower in libxgboost
 * 3.0 and is what `Xgboost.DMatrix.of_bigarray2` used to call until
 * we modernised it; matching the binding here keeps the bench grid
 * apples-to-apples. */

static uint64_t run_W5(const opts *o) {
  xs32 s = o->seed;
  float *data = malloc(o->rows * o->cols * sizeof(float));
  if (!data) { fprintf(stderr, "W5: oom\n"); abort(); }
  gen_dense(data, o->rows, o->cols, &s);

  char interface[256];
  snprintf(interface, sizeof interface,
           "{\"data\":[%llu,false],\"shape\":[%zu,%zu],"
           "\"strides\":null,\"typestr\":\"<f4\",\"version\":3}",
           (unsigned long long)(uintptr_t)data, o->rows, o->cols);
  const char *config = "{\"missing\":NaN}";

  DMatrixHandle d = NULL;
  bench_timer t;
  timer_start(&t);
  XGB_OK(XGDMatrixCreateFromDense(interface, config, &d));
  uint64_t ns = timer_elapsed_ns(&t);

  XGB_OK(XGDMatrixFree(d));
  free(data);
  return ns;
}

/* ---------- G3: FFI roundtrip microbench (XGBoostVersion) ----------
 * Per the gap-filling plan in BENCH.md: time --iters calls to the
 * cheapest libxgboost symbol (XGBoostVersion writes 3 ints from compile-
 * time constants, no allocation, no logger). Subtracting this from
 * heavier workload timings exposes per-call FFI overhead. */

static uint64_t run_G3(const opts *o) {
  size_t n = o->iters > 0 ? (size_t)o->iters : 10000000;
  int maj = 0, min = 0, pat = 0;
  /* Force the compiler to keep the loop body live. */
  volatile int sink_maj = 0;
  bench_timer t;
  timer_start(&t);
  for (size_t i = 0; i < n; ++i) {
    XGBoostVersion(&maj, &min, &pat);
    sink_maj = maj;
  }
  uint64_t ns = timer_elapsed_ns(&t);
  (void)sink_maj;
  return ns;
}

/* ---------- W6: DMatrix from CSR ----------
 * Uses the modern XGDMatrixCreateFromCSR (JSON __array_interface__).
 * The deprecated XGDMatrixCreateFromCSREx is the path the binding's
 * `Xgboost.DMatrix.of_csr` used to take before modernisation; matching
 * the binding here keeps the bench grid apples-to-apples. */

static uint64_t run_W6(const opts *o) {
  xs32 s = o->seed;
  uint64_t cap = (uint64_t)o->rows * (uint64_t)o->cols;
  uint64_t *indptr = malloc((o->rows + 1) * sizeof(uint64_t));
  uint32_t *indices = malloc(cap * sizeof(uint32_t));
  float *values = malloc(cap * sizeof(float));
  if (!indptr || !indices || !values) {
    fprintf(stderr, "W6: oom\n");
    abort();
  }
  uint64_t nnz = 0;
  indptr[0] = 0;
  for (size_t r = 0; r < o->rows; ++r) {
    for (size_t c = 0; c < o->cols; ++c) {
      if (xs32_uniform(&s) < o->density) {
        indices[nnz] = (uint32_t)c;
        values[nnz] = xs32_uniform(&s);
        ++nnz;
      }
    }
    indptr[r + 1] = nnz;
  }

  char json_indptr[256], json_indices[256], json_data[256];
  snprintf(json_indptr, sizeof json_indptr,
           "{\"data\":[%llu,false],\"shape\":[%zu],"
           "\"strides\":null,\"typestr\":\"<u8\",\"version\":3}",
           (unsigned long long)(uintptr_t)indptr, (size_t)(o->rows + 1));
  snprintf(json_indices, sizeof json_indices,
           "{\"data\":[%llu,false],\"shape\":[%llu],"
           "\"strides\":null,\"typestr\":\"<u4\",\"version\":3}",
           (unsigned long long)(uintptr_t)indices, (unsigned long long)nnz);
  snprintf(json_data, sizeof json_data,
           "{\"data\":[%llu,false],\"shape\":[%llu],"
           "\"strides\":null,\"typestr\":\"<f4\",\"version\":3}",
           (unsigned long long)(uintptr_t)values, (unsigned long long)nnz);
  const char *config = "{\"missing\":NaN}";

  DMatrixHandle d = NULL;
  bench_timer t;
  timer_start(&t);
  XGB_OK(XGDMatrixCreateFromCSR(json_indptr, json_indices, json_data,
                                (bst_ulong)o->cols, config, &d));
  uint64_t ns = timer_elapsed_ns(&t);

  XGB_OK(XGDMatrixFree(d));
  free(indptr);
  free(indices);
  free(values);
  return ns;
}

/* ---------- dispatch ---------- */

typedef uint64_t (*workload_fn)(const opts *);

static workload_fn lookup_workload(const char *name) {
  if (!strcmp(name, "W1")) return run_W1;
  if (!strcmp(name, "W2")) return run_W2;
  if (!strcmp(name, "W3")) return run_W3;
  if (!strcmp(name, "W4")) return run_W4;
  if (!strcmp(name, "W5")) return run_W5;
  if (!strcmp(name, "W6")) return run_W6;
  if (!strcmp(name, "G3")) return run_G3;
  return NULL;
}

int main(int argc, char **argv) {
  opts o;
  if (parse_args(argc, argv, &o) != 0) return 2;
  apply_workload_defaults(&o);

  workload_fn fn = lookup_workload(o.workload);
  if (!fn) {
    fprintf(stderr, "unknown workload: %s\n", o.workload);
    return 2;
  }

  /* Touch libxgboost once to force lazy initialisation before timing. */
  {
    int maj = 0, min = 0, pat = 0;
    XGBoostVersion(&maj, &min, &pat);
    if (o.verbose) {
      fprintf(stderr, "# libxgboost %d.%d.%d\n", maj, min, pat);
    }
  }

  for (int i = 0; i < o.warmup; ++i) {
    uint64_t ns = fn(&o);
    if (o.verbose) fprintf(stderr, "# warmup %d: %.3f ms (discarded)\n", i, ns / 1e6);
  }

  bench_stats st;
  stats_init(&st);
  for (int i = 0; i < o.repeat; ++i) {
    uint64_t ns = fn(&o);
    stats_record(&st, ns);
    if (o.verbose) fprintf(stderr, "# run %d: %.3f ms\n", i, ns / 1e6);
  }

  /* CSV header + body. The header is suppressed if NO_HEADER=1. */
  const char *no_hdr = getenv("NO_HEADER");
  if (!no_hdr || strcmp(no_hdr, "1") != 0) {
    printf("workload,rows,cols,iters,density,repeat,min_ms,mean_ms,max_ms\n");
  }
  printf("%s,%zu,%zu,%d,%.4f,%d,%.3f,%.3f,%.3f\n", o.workload, o.rows, o.cols,
         o.iters, o.density, o.repeat, stats_min_ms(&st), stats_mean_ms(&st),
         (double)st.max_ns / 1e6);
  return 0;
}
