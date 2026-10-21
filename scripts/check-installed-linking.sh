#!/usr/bin/env bash
set -euo pipefail

checkout=$(cd "${1:?Usage: check-installed-linking.sh CHECKOUT vendored|system}" && pwd)
mode=${2:?Select vendored or system}
case "$mode" in
  vendored|system) ;;
  *) echo "Unknown native provider: $mode" >&2; exit 2 ;;
esac

# REUSE_BUILD=1 installs from the checkout's existing _build (as left by
# `dune build -p xgboost`) instead of building a fresh copy, then hides
# that _build while consumers link. CI uses it to avoid building the
# vendored XGBoost a second time. RUN_TESTS=0 skips @runtest in the
# fresh build; REUSE_BUILD implies it.
reuse_build=${REUSE_BUILD:-0}
run_tests=${RUN_TESTS:-1}

scratch=$(mktemp -d "${TMPDIR:-/tmp}/xgboost-installed-linking.XXXXXX")
hidden_build="$checkout/_build.hidden-by-installed-linking"
cleanup() {
  if [[ -d "$hidden_build" ]]; then mv -- "$hidden_build" "$checkout/_build"; fi
  rm -rf -- "$scratch"
}
trap cleanup EXIT
mkdir -p "$scratch/empty-pkgconfig" "$scratch/consumer"
if [[ "$reuse_build" == 1 ]]; then
  root="$checkout"
  test -d "$root/_build" || { echo "REUSE_BUILD=1 needs an existing $root/_build" >&2; exit 2; }
else
  root="$scratch/source"
  mkdir -p "$root"
  # Copy the package inputs, including uncommitted edits, without local build caches.
  tar -C "$checkout" -cf - dune-project xgboost.opam LICENSE CHANGES.md README.md \
    config src vendor test | tar -x -C "$root"
fi
cp "$checkout/scripts/installed_linking_probe.ml" "$scratch/consumer/probe.ml"
export DUNE_CACHE=disabled
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1

if [[ "$mode" == vendored ]]; then
  export PKG_CONFIG_PATH=
  export PKG_CONFIG_LIBDIR="$scratch/empty-pkgconfig"
else
  pkg-config --atleast-version=3.0 xgboost
  pkg-config --modversion xgboost
fi

if [[ "$reuse_build" == 1 ]]; then
  # Same flags as CI's build step, so this is a no-op over its _build.
  # (-p implies --root ., hence the cd.)
  (cd "$root" && opam exec -- dune build -p xgboost -j "${BUILD_JOBS:-2}" @install)
else
  targets=(@install)
  if [[ "$run_tests" == 1 ]]; then targets+=(@runtest); fi
  opam exec -- dune build --root "$root" --only-packages xgboost \
    --profile release -j "${BUILD_JOBS:-2}" "${targets[@]}"
fi

opam exec -- dune install --root "$root" \
  --prefix "$scratch/install" xgboost
bindings="$scratch/install/lib/xgboost/bindings"
# Confirm the requested native provider is the one that was actually built:
# on the vendored path the bundled archives carry real objects, on the
# system path they are empty placeholders (the real library is a system
# .so resolved through pkg-config).
for archive in libxgboost_vendored.a libdmlc_vendored.a; do
  test -s "$bindings/$archive"
  members=$(ar -t "$bindings/$archive" | grep -c . || true)
  if [[ "$mode" == vendored ]]; then
    if [[ "$members" -eq 0 ]]; then
      echo "Expected vendored fallback, but $archive has no objects" >&2
      exit 1
    fi
  else
    if [[ "$members" -ne 0 ]]; then
      echo "Expected system provider, but $archive is a non-empty vendored archive" >&2
      exit 1
    fi
  fi
done
# foreign_archives records the vendored archives and their dll companions in
# the link/dll-load closure in BOTH modes, so on the system path four inert
# placeholders are installed by design (they cannot be dropped without
# breaking the recorded references — see src/bindings/dune). Assert they are
# genuinely inert there, and real on the vendored path: the stub dll export
# XGBoost symbols only when the fallback is active.
stublibs="$scratch/install/lib/stublibs"
test -e "$stublibs/dllxgboost_vendored.so"
test -e "$stublibs/dlldmlc_vendored.so"
case "$(uname -s)" in
  Darwin) xgb_syms=$(nm -gU "$stublibs/dllxgboost_vendored.so" 2>/dev/null || true) ;;
  *)      xgb_syms=$(nm -D --defined-only "$stublibs/dllxgboost_vendored.so" 2>/dev/null || true) ;;
esac
# Match in-shell rather than through a pipe: piping the (large) symbol dump
# into `grep -q` would SIGPIPE the writer and, under pipefail, misreport.
if [[ "$mode" == vendored ]]; then
  if [[ "$xgb_syms" != *XGBoosterCreate* ]]; then
    echo "Expected vendored dll to export XGBoost symbols" >&2
    exit 1
  fi
else
  if [[ "$xgb_syms" == *XGBoosterCreate* ]]; then
    echo "Expected system placeholder dll to export no XGBoost symbols" >&2
    exit 1
  fi
fi
# The installed link metadata must carry no build-tree path: neither a
# temporary source directory nor a dangling build-relative -L. dune locates
# the bundled archives (declared via foreign_archives) through its own
# per-package search path, so relocation is exercised by the consumer build
# and run below rather than by any recorded path.
for suffix in cma cmxa; do
  metadata=$(opam exec -- ocamlobjinfo "$bindings/xgboost_bindings.$suffix")
  if printf '%s\n' "$metadata" | grep '^Extra C' | grep -Fq "$root"; then
    echo "Installed link metadata retains the build source path" >&2
    exit 1
  fi
  if printf '%s\n' "$metadata" | grep '^Extra C' | grep -Fq -- '-Lsrc/bindings'; then
    echo "Installed link metadata retains a dangling relative -L" >&2
    exit 1
  fi
done

# Make the build directory disappear, as opam does after installing.
if [[ "$reuse_build" == 1 ]]; then
  mv -- "$root/_build" "$hidden_build"
else
  rm -rf -- "$root"
fi
mv "$scratch/install" "$scratch/relocated-install"
export OCAMLPATH="$scratch/relocated-install/lib${OCAMLPATH:+:$OCAMLPATH}"
resolved=$(opam exec -- ocamlfind query xgboost.bindings)
test "$resolved" = "$scratch/relocated-install/lib/xgboost/bindings"

cat > "$scratch/consumer/dune-project" <<'DUNE_PROJECT'
(lang dune 3.18)
(name installed_xgboost_probe)
DUNE_PROJECT
cat > "$scratch/consumer/dune" <<'DUNE_FILE'
(executable (name probe) (modes exe byte_complete byte) (libraries xgboost))
DUNE_FILE
opam exec -- dune build --root "$scratch/consumer" --profile release \
  probe.exe probe.bc.exe probe.bc
"$scratch/consumer/_build/default/probe.exe"
"$scratch/consumer/_build/default/probe.bc.exe"
# Plain bytecode dlopens the C stubs, which in turn need the bundled
# vendored dlls; point the runtime at the relocated stublibs dir. Use
# ocamlrun -I rather than CAML_LD_LIBRARY_PATH: on OCaml < 5.5 the ocaml
# opam package sets CAML_LD_LIBRARY_PATH with `=`, so `opam exec` would
# overwrite a value set here.
opam exec -- ocamlrun -I "$scratch/relocated-install/lib/stublibs" \
  "$scratch/consumer/_build/default/probe.bc"
if [[ "$mode" == vendored ]]; then
  for executable in probe.exe probe.bc.exe; do
    case "$(uname -s)" in
      Darwin) dependencies=$(otool -L "$scratch/consumer/_build/default/$executable") ;;
      Linux) dependencies=$(ldd "$scratch/consumer/_build/default/$executable") ;;
      *) continue ;;
    esac
    if printf '%s\n' "$dependencies" | grep -Eq 'libxgboost\.(so|dylib)'; then
      echo "Vendored consumer unexpectedly depends on shared libxgboost" >&2
      exit 1
    fi
  done
fi
echo "Installed consumer passed after build-directory removal and prefix relocation ($mode)"
