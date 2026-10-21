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
# config/provider records which xgboost_bindings library was enabled.
provider_vendored=$(cat "$root/_build/default/config/provider/vendored")
expected=$([[ "$mode" == vendored ]] && echo true || echo false)
if [[ "$provider_vendored" != "$expected" ]]; then
  echo "Expected $mode native provider, config/provider/vendored says $provider_vendored" >&2
  exit 1
fi

opam exec -- dune install --root "$root" \
  --prefix "$scratch/install" xgboost
bindings="$scratch/install/lib/xgboost/bindings"
# dune installs foreign archives next to the library and their dll
# companions in lib/stublibs.
stublibs="$scratch/install/lib/stublibs"
for name in xgboost_vendored dmlc_vendored; do
  if [[ "$mode" == vendored ]]; then
    test -s "$bindings/lib$name.a"
    ar -t "$bindings/lib$name.a" > /dev/null
    test -s "$stublibs/dll$name.so"
  elif [[ -e "$bindings/lib$name.a" || -e "$stublibs/dll$name.so" ]]; then
    echo "System provider unexpectedly installed vendored $name files" >&2
    exit 1
  fi
done
# Plain bytecode loads three dlls in order: dmlc, xgboost, then the stubs.
# The stubs dll does not declare the xgboost dll as a dependency, so every
# XGBoost symbol it uses must come from dllxgboost_vendored.so; and dmlc
# must be loaded once, from that dll, so dlldmlc_vendored.so stays empty.
if [[ "$mode" == vendored ]]; then
  case "$(uname -s)" in
    Darwin)
      defined_syms() { nm -gU "$1" | awk '{print $NF}'; }
      undefined_syms() { nm -gu "$1" | awk '{print $NF}'; } ;;
    *)
      defined_syms() { nm -D --defined-only "$1" | awk '{print $NF}'; }
      undefined_syms() { nm -D --undefined-only "$1" | awk '{print $NF}'; } ;;
  esac
  stub_refs=$(undefined_syms "$stublibs/dllxgboost_bindings_stubs.so" | grep 'XGB' | sort -u)
  if [[ -z "$stub_refs" ]]; then
    echo "Expected the stubs dll to reference XGBoost symbols" >&2
    exit 1
  fi
  missing=$(comm -23 <(printf '%s\n' "$stub_refs") \
    <(defined_syms "$stublibs/dllxgboost_vendored.so" | sort -u))
  if [[ -n "$missing" ]]; then
    printf 'XGBoost symbols the stubs dll needs but dllxgboost_vendored.so lacks:\n%s\n' "$missing" >&2
    exit 1
  fi
  if defined_syms "$stublibs/dlldmlc_vendored.so" | grep -q 'dmlc'; then
    echo "dlldmlc_vendored.so carries a second copy of dmlc" >&2
    exit 1
  fi
fi
for suffix in cma cmxa; do
  metadata=$(opam exec -- ocamlobjinfo "$bindings/xgboost_bindings.$suffix")
  if printf '%s\n' "$metadata" | grep '^Extra C' | grep -Fq -- '-Lsrc/bindings'; then
    echo "Installed link metadata retains a build-relative -L" >&2
    exit 1
  fi
  if printf '%s\n' "$metadata" | grep '^Extra C' | grep -Fq "$root"; then
    echo "Installed link metadata retains the build source path" >&2
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
(executable (name probe) (modes exe byte byte_complete) (libraries xgboost))
DUNE_FILE
opam exec -- dune build --root "$scratch/consumer" --profile release \
  probe.exe probe.bc probe.bc.exe
"$scratch/consumer/_build/default/probe.exe"
relocated_stublibs="$scratch/relocated-install/lib/stublibs"
opam exec -- ocamlrun -I "$relocated_stublibs" \
  "$scratch/consumer/_build/default/probe.bc"
# The same, finding the dlls through CAML_LD_LIBRARY_PATH as in an opam
# switch. It is set inside `opam exec`: on OCaml < 5.5 the ocaml package's
# setenv replaces a value set outside it.
with_stublibs() {
  opam exec -- sh -c 'CAML_LD_LIBRARY_PATH="$1${CAML_LD_LIBRARY_PATH:+:$CAML_LD_LIBRARY_PATH}"; export CAML_LD_LIBRARY_PATH; shift; exec "$@"' \
    sh "$relocated_stublibs" "$@"
}
with_stublibs ocamlrun "$scratch/consumer/_build/default/probe.bc"
# The toplevel loads the dlls itself, through findlib.
cat > "$scratch/consumer/toplevel.ml" <<'TOPLEVEL'
#use "topfind";;
#require "xgboost";;
let () =
  let major, minor, patch = Xgboost.version () in
  Printf.printf "Toplevel loaded xgboost %d.%d.%d\n" major minor patch
TOPLEVEL
with_stublibs ocaml "$scratch/consumer/toplevel.ml"
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
echo "Installed consumer passed after build-directory removal and prefix relocation ($mode)"
