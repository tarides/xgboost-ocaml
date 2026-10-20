#!/usr/bin/env bash
set -euo pipefail

checkout=$(cd "${1:?Usage: check-installed-linking.sh CHECKOUT vendored|system}" && pwd)
mode=${2:?Select vendored or system}
case "$mode" in
  vendored|system) ;;
  *) echo "Unknown native provider: $mode" >&2; exit 2 ;;
esac

scratch=$(mktemp -d "${TMPDIR:-/tmp}/xgboost-installed-linking.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/source" "$scratch/empty-pkgconfig" "$scratch/consumer"
# Copy the package inputs, including uncommitted edits, without local build caches.
tar -C "$checkout" -cf - dune-project xgboost.opam LICENSE CHANGES.md README.md \
  config src vendor test | tar -x -C "$scratch/source"
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

opam exec -- dune build --root "$scratch/source" --only-packages xgboost \
  --profile release -j "${BUILD_JOBS:-2}" @install @runtest

opam exec -- dune install --root "$scratch/source" \
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
  if printf '%s\n' "$metadata" | grep '^Extra C' | grep -Fq "$scratch/source"; then
    echo "Installed link metadata retains a temporary source path" >&2
    exit 1
  fi
  if printf '%s\n' "$metadata" | grep '^Extra C' | grep -Fq -- '-Lsrc/bindings'; then
    echo "Installed link metadata retains a dangling relative -L" >&2
    exit 1
  fi
done

rm -rf -- "$scratch/source"
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
# vendored dlls; point the runtime at the relocated stublibs dir.
CAML_LD_LIBRARY_PATH="$scratch/relocated-install/lib/stublibs${CAML_LD_LIBRARY_PATH:+:$CAML_LD_LIBRARY_PATH}" \
  opam exec -- ocamlrun "$scratch/consumer/_build/default/probe.bc"
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
echo "Installed consumer passed after source removal and prefix relocation ($mode)"
