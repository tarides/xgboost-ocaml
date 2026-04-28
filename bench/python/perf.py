#!/usr/bin/env python3
"""Python xgboost peer benchmark harness.

Mirrors bin/c_reference/perf.c, bench/bindings/perf.ml, and
bench/xgboost/perf.ml on the same workload grid (W1..W6) to give the
"ergonomic peer floor" reference for the comparison table in BENCH.md.

Output is one CSV line per run on stdout, header on the first line:
    workload,rows,cols,iters,density,repeat,min_ms,mean_ms,max_ms

Usage (from a venv with xgboost==3.0.0 + numpy installed):
    bench/python/.venv/bin/python bench/python/perf.py --workload W1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import xgboost as xgb


# ---------- deterministic xorshift32 PRNG matching the C / OCaml side ----------


def _xs32_uniform(state):
    """state is a 1-element np.uint32 array; returns float in [0, 1)."""
    x = state[0]
    x ^= np.uint32(x << np.uint32(13))
    x ^= np.uint32(x >> np.uint32(17))
    x ^= np.uint32(x << np.uint32(5))
    if x == 0:
        x = np.uint32(1)
    state[0] = x
    return float(x >> np.uint32(8)) / 16777216.0


def _gen_dense(rows, cols, seed):
    state = np.array([seed], dtype=np.uint32)
    out = np.empty(rows * cols, dtype=np.float32)
    for i in range(rows * cols):
        out[i] = _xs32_uniform(state)
    return out.reshape(rows, cols)


def _labels_binary(m):
    rows, cols = m.shape
    cols_ = max(1, cols)
    v = (
        0.5 * m[:, 0]
        - 0.3 * m[:, min(1, cols_ - 1)]
        + 0.2 * m[:, min(2, cols_ - 1)]
        + 0.1 * m[:, min(3, cols_ - 1)]
    )
    return (v > 0.25).astype(np.float32)


def _labels_reg(m):
    rows, cols = m.shape
    cols_ = max(1, cols)
    return (
        0.5 * m[:, 0]
        - 0.3 * m[:, min(1, cols_ - 1)]
        + 0.2 * m[:, min(2, cols_ - 1)]
        + 0.1 * m[:, min(3, cols_ - 1)]
    ).astype(np.float32)


# ---------- workloads ----------


def run_W1(opts):
    m = _gen_dense(opts.rows, opts.cols, opts.seed)
    labels = _labels_reg(m)
    dtrain = xgb.DMatrix(m, label=labels, missing=float("nan"))
    bst = xgb.Booster(
        params={"objective": "reg:squarederror", "verbosity": 0},
        cache=[dtrain],
    )
    t0 = time.perf_counter_ns()
    for it in range(opts.iters):
        bst.update(dtrain, it)
    return time.perf_counter_ns() - t0


def run_W2(opts):
    m = _gen_dense(opts.rows, opts.cols, opts.seed)
    labels = _labels_binary(m)
    dtrain = xgb.DMatrix(m, label=labels, missing=float("nan"))
    bst = xgb.Booster(
        params={
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 6,
            "verbosity": 0,
        },
        cache=[dtrain],
    )
    t0 = time.perf_counter_ns()
    for it in range(opts.iters):
        bst.update(dtrain, it)
    return time.perf_counter_ns() - t0


def _train_binary(rows, cols, iters, seed):
    m = _gen_dense(rows, cols, seed)
    labels = _labels_binary(m)
    dtrain = xgb.DMatrix(m, label=labels, missing=float("nan"))
    bst = xgb.Booster(
        params={
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 6,
            "verbosity": 0,
        },
        cache=[dtrain],
    )
    for it in range(iters):
        bst.update(dtrain, it)
    return bst, dtrain, m, labels


def run_W3(opts):
    bst, _, _, _ = _train_binary(opts.rows, opts.cols, opts.iters, opts.seed)
    pm = _gen_dense(opts.rows, opts.cols, opts.seed ^ 0xA5A5A5A5)
    dpred = xgb.DMatrix(pm, missing=float("nan"))
    t0 = time.perf_counter_ns()
    out = bst.predict(dpred)
    sink = float(out.sum())
    elapsed = time.perf_counter_ns() - t0
    _ = sink
    return elapsed


def run_W4(opts):
    bst, _, _, _ = _train_binary(opts.rows, opts.cols, opts.iters, opts.seed)
    n_pred = min(opts.rows, 100_000)
    state = np.array([opts.seed ^ 0xA5A5A5A5], dtype=np.uint32)
    prow = np.empty((1, opts.cols), dtype=np.float32)
    # warmup
    for c in range(opts.cols):
        prow[0, c] = _xs32_uniform(state)
    bst.predict(xgb.DMatrix(prow, missing=float("nan")))
    sink = 0.0
    t0 = time.perf_counter_ns()
    for _ in range(n_pred):
        for c in range(opts.cols):
            prow[0, c] = _xs32_uniform(state)
        d1 = xgb.DMatrix(prow, missing=float("nan"))
        out = bst.predict(d1)
        sink += float(out[0])
    elapsed = time.perf_counter_ns() - t0
    _ = sink
    return elapsed


def run_W5(opts):
    m = _gen_dense(opts.rows, opts.cols, opts.seed)
    t0 = time.perf_counter_ns()
    _ = xgb.DMatrix(m, missing=float("nan"))
    return time.perf_counter_ns() - t0


def run_W6(opts):
    # Build a CSR via scipy.sparse if available; otherwise build via xgb's
    # native csr support from raw numpy arrays.
    import scipy.sparse as sp  # type: ignore

    state = np.array([opts.seed], dtype=np.uint32)
    rng = np.random.default_rng(opts.seed)
    dense = rng.random(size=(opts.rows, opts.cols), dtype=np.float32)
    mask = rng.random(size=(opts.rows, opts.cols)) < opts.density
    csr = sp.csr_matrix(dense * mask, dtype=np.float32)
    _ = state  # keep the xs32 PRNG defined for parity with C side
    t0 = time.perf_counter_ns()
    _ = xgb.DMatrix(csr, missing=float("nan"))
    return time.perf_counter_ns() - t0


WORKLOADS = {
    "W1": run_W1,
    "W2": run_W2,
    "W3": run_W3,
    "W4": run_W4,
    "W5": run_W5,
    "W6": run_W6,
}


WORKLOAD_DEFAULTS = {
    "W1": dict(rows=1000, cols=50, iters=100),
    "W2": dict(rows=1_000_000, cols=100, iters=100),
    "W3": dict(rows=1_000_000, cols=100, iters=100),
    "W4": dict(rows=1_000_000, cols=100, iters=100),
    "W5": dict(rows=100_000, cols=100, iters=0),
    "W6": dict(rows=100_000, cols=100, iters=0),
}


def main():
    p = argparse.ArgumentParser(description="xgboost-ocaml Python peer bench")
    p.add_argument("--workload", required=True, choices=list(WORKLOADS))
    p.add_argument("--rows", type=int, default=0)
    p.add_argument("--cols", type=int, default=0)
    p.add_argument("--iters", type=int, default=0)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--verbose", action="store_true")
    opts = p.parse_args()

    defaults = WORKLOAD_DEFAULTS[opts.workload]
    if opts.rows == 0:
        opts.rows = defaults["rows"]
    if opts.cols == 0:
        opts.cols = defaults["cols"]
    if opts.iters == 0:
        opts.iters = defaults["iters"]

    fn = WORKLOADS[opts.workload]

    if opts.verbose:
        print(f"# python {sys.version_info[0]}.{sys.version_info[1]}", file=sys.stderr)
        print(f"# xgboost {xgb.__version__}", file=sys.stderr)

    for i in range(opts.warmup):
        ns = fn(opts)
        if opts.verbose:
            print(f"# warmup {i}: {ns / 1e6:.3f} ms (discarded)", file=sys.stderr)

    samples = []
    for i in range(opts.repeat):
        ns = fn(opts)
        samples.append(ns)
        if opts.verbose:
            print(f"# run {i}: {ns / 1e6:.3f} ms", file=sys.stderr)

    mn = min(samples)
    mx = max(samples)
    mean = sum(samples) / len(samples)

    if os.environ.get("NO_HEADER") != "1":
        print("workload,rows,cols,iters,density,repeat,min_ms,mean_ms,max_ms")
    print(
        f"{opts.workload},{opts.rows},{opts.cols},{opts.iters},"
        f"{opts.density:.4f},{opts.repeat},{mn / 1e6:.3f},{mean / 1e6:.3f},{mx / 1e6:.3f}"
    )


if __name__ == "__main__":
    main()
