#!/bin/sh
# Install xgboost from a checkout the way opam-repository CI does, then
# build and run an installed consumer. Normally run inside one of
# opam-repository CI's ocaml/opam images, as the image's opam user, with
# the checkout mounted read-only at /src:
#
#   docker run --rm -v "$PWD:/src:ro" ocaml/opam:alpine-3.24-ocaml-5.5 \
#     sh /src/scripts/check-distro-install.sh vendored
#
# It also runs on a host with an initialised opam switch (Distros' macOS
# and FreeBSD jobs): set SRC to the checkout.
#
# The argument is the native provider this distro must end up with
# (system or vendored); the script fails if opam took the other path.
#
# LOWER_BOUNDS=1 makes opam's solver pick the oldest versions the opam
# file allows for every dependency, as opam-repository CI's lower-bounds
# job does.
#
# opam installs the depexts it declares (libxgboost-dev where the distro
# packages it, cmake, a C++ compiler, OpenMP), so each distro takes the
# path it would take in opam-repository CI: the system libxgboost or the
# vendored build. The consumer runs as native, plain bytecode (dlls found
# through the switch's own stublibs), byte_complete and in the toplevel.

set -eu

expected=${1:?Usage: check-distro-install.sh system|vendored}
case "$expected" in
  system|vendored) ;;
  *) echo "Unknown native provider: $expected" >&2; exit 2 ;;
esac

src=${SRC:-/src}

export OPAMYES=1 OPAMCONFIRMLEVEL=unsafe-yes OMP_NUM_THREADS=1

if [ "${LOWER_BOUNDS:-0}" = 1 ]; then
  # As opam-repository CI's lower-bounds job (ocurrent/opam-repo-ci,
  # opam-ci-check/lib/opam_build.ml): the 0install solver with criteria
  # that prefer the oldest versions. The default mccs solver times out
  # on them, even given 20 minutes.
  opam option solver=builtin-0install
  criteria='+removed,+count[version-lag,solution]'
  export OPAMCRITERIA="$criteria" OPAMFIXUPCRITERIA="$criteria" \
    OPAMUPGRADECRITERIA="$criteria"
fi

if [ -r /etc/os-release ]; then
  . /etc/os-release
else
  ID=$(uname -s | tr '[:upper:]' '[:lower:]') VERSION_ID=$(uname -r)
fi
distro="$ID ${VERSION_ID:-rolling} $(uname -m)"
variant=
if [ "${LOWER_BOUNDS:-0}" = 1 ]; then variant=", lower bounds"; fi
echo "== $distro, OCaml $(opam exec -- ocamlc -version)$variant"

# Arch is rolling: the mirrors drop old package versions, but the image's
# pacman database is a snapshot, so opam's depext step (pacman -Su) can ask
# for a pkgconf/cmake build the mirror no longer serves (404). Refresh the
# database (and keyring) with a full sync+upgrade first, so depexts resolve
# to the versions the mirror actually has.
if [ "$ID" = arch ]; then
  sudo pacman -Sy --noconfirm archlinux-keyring || true
  sudo pacman -Syu --noconfirm
fi

# An ocaml/opam image's opam-repository is a snapshot; bring it up to
# date. (A host switch uses opam.ocaml.org, already current.)
if [ -d "$HOME/opam-repository/.git" ]; then
  git -C "$HOME/opam-repository" pull -q origin master
  opam update -q
fi

cp -R "$src" "$HOME/xgboost"
rm -rf "$HOME/xgboost/_build"
cd "$HOME/xgboost"

# Install it as opam-repository CI does: as a released package, from a
# source tarball, not as a pin. A pin is a development package, for which
# opam runs `dune subst` ({dev} in xgboost.opam); that needs git version
# information, which opam strips when copying a path pin, and dune < 3.20
# (what the lower-bounds job picks) then fails. The tarball carries the
# files as they are, uncommitted changes included.
version=0.0.0+ci
tarball="$HOME/xgboost-$version.tar.gz"
tar --exclude=./.git --exclude=./_build -czf "$tarball" .
sha256() {
  if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum > /dev/null; then shasum -a 256 "$1" | cut -d' ' -f1
  else sha256 -q "$1"   # FreeBSD
  fi
}
repo="$HOME/ci-opam-repository"
mkdir -p "$repo/packages/xgboost/xgboost.$version"
printf 'opam-version: "2.0"\n' > "$repo/repo"
{
  cat xgboost.opam
  printf 'url {\n  src: "file://%s"\n  checksum: "sha256=%s"\n}\n' \
    "$tarball" "$(sha256 "$tarball")"
} > "$repo/packages/xgboost/xgboost.$version/opam"
opam repository add ci "$repo" --this-switch --rank=1

if [ "${LOWER_BOUNDS:-0}" = 1 ]; then
  # Like opam-repository CI, build without tests: test-only dependencies
  # aren't part of what the lower-bounds job checks.
  opam install "xgboost.$version"
else
  opam install --with-test "xgboost.$version"
fi
if [ "${LOWER_BOUNDS:-0}" = 1 ]; then
  echo "== dependency versions picked:"
  opam list --installed --columns=name,version \
    dune dune-configurator ctypes ctypes-foreign integers
fi

lib=$(opam var lib)
if [ -s "$lib/xgboost/bindings/libxgboost_vendored.a" ]; then
  provider=vendored
else
  provider=system
fi
echo "== installed with the $provider provider"
# A broken conf-libxgboost filter, a failed depext or a distro dropping
# xgboost.pc would otherwise vendor silently and still pass.
if [ "$provider" != "$expected" ]; then
  echo "== expected the $expected provider on $distro" >&2
  exit 1
fi

consumer="$HOME/consumer"
mkdir -p "$consumer"
cp scripts/installed_linking_probe.ml "$consumer/probe.ml"
printf '(lang dune 3.18)\n' > "$consumer/dune-project"
printf '(executable (name probe) (modes exe byte byte_complete) (libraries xgboost))\n' \
  > "$consumer/dune"
opam exec -- dune build --root "$consumer" probe.exe probe.bc probe.bc.exe
"$consumer/_build/default/probe.exe"
opam exec -- ocamlrun "$consumer/_build/default/probe.bc"
"$consumer/_build/default/probe.bc.exe"

# The toplevel loads xgboost through topfind, from ocamlfind, which
# xgboost doesn't depend on: with lower bounds nothing pulls it in. Name
# xgboost in the same request: the lower-bounds criteria favour
# removals, and opam otherwise removed xgboost and its dependencies
# ("1 to install | 15 to remove") to install ocamlfind.
opam install ocamlfind "xgboost.$version"
# On OCaml 4.x, #thread must load the systhreads stubs before
# ctypes-foreign's (see check-installed-linking.sh). Harmless on 5.x.
cat > "$consumer/toplevel.ml" <<'TOPLEVEL'
#use "topfind";;
#thread;;
#require "xgboost";;
let () =
  let major, minor, patch = Xgboost.version () in
  Printf.printf "Toplevel loaded xgboost %d.%d.%d\n" major minor patch
TOPLEVEL
opam exec -- ocaml "$consumer/toplevel.ml"

echo "== $distro$variant: installed consumer passed ($provider)"
