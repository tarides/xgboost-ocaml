# Pure-C reference harness for xgboost-ocaml benchmarks.
# Implements W1-W6 (and later W7-W8) using libxgboost directly via c_api.h.
#
# Build: make
# Run:   ./perf --workload W1 [--rows N] [--cols N] [--iters N] [--repeat K]
# Check: ./check
#
# Override compiler / flags from the environment, e.g.
#   CC=clang make
#   CFLAGS_EXTRA=-fsanitize=address make asan

CC      ?= cc
PKGCFG  ?= pkg-config
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -std=c11
LDFLAGS ?=
LDLIBS  ?= -lxgboost -lm
CFLAGS_EXTRA ?=

# Pull cflags / libs from pkg-config when available; otherwise rely on the
# system defaults (libxgboost is on /usr/lib, headers in /usr/include).
PC_CFLAGS := $(shell $(PKGCFG) --cflags xgboost 2>/dev/null)
PC_LIBS   := $(shell $(PKGCFG) --libs   xgboost 2>/dev/null)

ALL_CFLAGS := $(CFLAGS) $(CFLAGS_EXTRA) $(PC_CFLAGS)
ALL_LDLIBS := $(if $(PC_LIBS),$(PC_LIBS),$(LDLIBS)) -lm

BINS := perf check

.PHONY: all asan clean
all: $(BINS)

perf: perf.c bench_common.h
	$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ perf.c $(ALL_LDLIBS)

check: check.c bench_common.h
	$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ check.c $(ALL_LDLIBS)

asan:
	$(MAKE) CFLAGS_EXTRA="-fsanitize=address,undefined -fno-omit-frame-pointer -g"

clean:
	rm -f $(BINS)
