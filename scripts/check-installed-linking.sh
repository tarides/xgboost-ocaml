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
flags="$scratch/source/_build/default/config/c_library_flags.sexp"
if [[ "$mode" == vendored ]]; then
  grep -F -- '-lxgboost_vendored' "$flags"
  grep -F -- '-ldmlc_vendored' "$flags"
else
  if grep -Fq -- '-lxgboost_vendored' "$flags"; then
    echo "Expected system native provider, got vendored fallback" >&2
    exit 1
  fi
fi

opam exec -- dune install --root "$scratch/source" \
  --prefix "$scratch/install" xgboost
bindings="$scratch/install/lib/xgboost/bindings"
for archive in libxgboost_vendored.a libdmlc_vendored.a; do
  if [[ "$mode" == vendored ]]; then
    test -s "$bindings/$archive"
    ar -t "$bindings/$archive" > /dev/null
  elif [[ -e "$bindings/$archive" ]]; then
    echo "System provider unexpectedly installed $archive" >&2
    exit 1
  fi
done
for suffix in cma cmxa; do
  metadata=$(opam exec -- ocamlobjinfo "$bindings/xgboost_bindings.$suffix")
  printf '%s\n' "$metadata" | grep -F -- '-L$CAMLORIGIN'
  if printf '%s\n' "$metadata" | grep '^Extra C' | grep -Fq "$scratch/source"; then
    echo "Installed link metadata retains a temporary source path" >&2
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
(executable (name probe) (modes exe byte_complete) (libraries xgboost))
DUNE_FILE
opam exec -- dune build --root "$scratch/consumer" --profile release \
  probe.exe probe.bc.exe
"$scratch/consumer/_build/default/probe.exe"
"$scratch/consumer/_build/default/probe.bc.exe"
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
