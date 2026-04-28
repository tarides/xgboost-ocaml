/*
 * bench_common.h — shared utilities for the pure-C reference harness.
 *
 * Provides:
 *   - timer_t / timer_start / timer_elapsed_ns wall-clock timing on CLOCK_MONOTONIC
 *   - xs32 xorshift32 PRNG (deterministic, fast)
 *   - data generators for dense matrices, labels, and CSR sparse layouts
 *   - XGB_OK error-check macro that aborts with the upstream error string on -1
 */
#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <xgboost/c_api.h>

/* ---------- error checking ---------- */

#define XGB_OK(call)                                                          \
  do {                                                                        \
    int _rc = (call);                                                         \
    if (_rc != 0) {                                                           \
      fprintf(stderr, "%s:%d: XGBoost call failed: %s\n", __FILE__, __LINE__, \
              XGBGetLastError());                                             \
      abort();                                                                \
    }                                                                         \
  } while (0)

/* ---------- timing ---------- */

typedef struct {
  struct timespec ts;
} bench_timer;

static inline void timer_start(bench_timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->ts);
}

static inline uint64_t timer_elapsed_ns(const bench_timer *t) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (uint64_t)(now.tv_sec - t->ts.tv_sec) * 1000000000ULL +
         (uint64_t)(now.tv_nsec - t->ts.tv_nsec);
}

/* ---------- PRNG (xorshift32, deterministic) ---------- */

typedef uint32_t xs32;

static inline uint32_t xs32_next(xs32 *s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x ? x : 1u;
  return *s;
}

static inline float xs32_uniform(xs32 *s) {
  /* uniform on [0, 1) */
  return (float)(xs32_next(s) >> 8) * (1.0f / 16777216.0f);
}

/* ---------- data generators ---------- */

static inline void gen_dense(float *data, size_t nrow, size_t ncol, xs32 *s) {
  size_t n = nrow * ncol;
  for (size_t i = 0; i < n; ++i) data[i] = xs32_uniform(s);
}

/* Binary labels via a fixed linear projection of the first 4 columns.
 * Reproducible and weakly informative — enough for hist to converge. */
static inline void gen_labels_binary(float *labels, const float *data,
                                     size_t nrow, size_t ncol) {
  for (size_t r = 0; r < nrow; ++r) {
    float x =
        0.5f * data[r * ncol + 0] - 0.3f * data[r * ncol + 1] +
        0.2f * data[r * ncol + 2] + 0.1f * data[r * ncol + (3 % ncol)];
    labels[r] = (x > 0.25f) ? 1.0f : 0.0f;
  }
}

/* Continuous labels for regression: same linear projection as above. */
static inline void gen_labels_reg(float *labels, const float *data, size_t nrow,
                                  size_t ncol) {
  for (size_t r = 0; r < nrow; ++r) {
    float x =
        0.5f * data[r * ncol + 0] - 0.3f * data[r * ncol + 1] +
        0.2f * data[r * ncol + 2] + 0.1f * data[r * ncol + (3 % ncol)];
    labels[r] = x;
  }
}

/* Generate a CSR sparse matrix with `density` of dense.
 * indptr has size nrow+1, indices and data are sized to nnz. The caller
 * passes pre-sized buffers and receives the actual nnz back via *out_nnz. */
static inline void gen_csr(uint64_t *indptr, uint32_t *indices, float *values,
                           size_t nrow, size_t ncol, float density,
                           uint64_t *out_nnz, xs32 *s) {
  uint64_t nnz = 0;
  indptr[0] = 0;
  for (size_t r = 0; r < nrow; ++r) {
    for (size_t c = 0; c < ncol; ++c) {
      if (xs32_uniform(s) < density) {
        indices[nnz] = (uint32_t)c;
        values[nnz] = xs32_uniform(s);
        ++nnz;
      }
    }
    indptr[r + 1] = nnz;
  }
  *out_nnz = nnz;
}

/* ---------- summary stats over repeats ---------- */

typedef struct {
  uint64_t min_ns, max_ns, sum_ns;
  uint64_t samples[256];
  int n;
} bench_stats;

static inline void stats_init(bench_stats *st) {
  st->min_ns = UINT64_MAX;
  st->max_ns = 0;
  st->sum_ns = 0;
  st->n = 0;
}

static inline void stats_record(bench_stats *st, uint64_t ns) {
  if (st->n < (int)(sizeof(st->samples) / sizeof(st->samples[0]))) {
    st->samples[st->n] = ns;
  }
  if (ns < st->min_ns) st->min_ns = ns;
  if (ns > st->max_ns) st->max_ns = ns;
  st->sum_ns += ns;
  ++st->n;
}

static inline double stats_mean_ms(const bench_stats *st) {
  return (double)st->sum_ns / (double)st->n / 1e6;
}

static inline double stats_min_ms(const bench_stats *st) {
  return (double)st->min_ns / 1e6;
}

#endif /* BENCH_COMMON_H */
