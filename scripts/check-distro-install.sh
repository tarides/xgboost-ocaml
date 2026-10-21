#!/bin/sh
# Install xgboost from a checkout the way opam-repository CI does, inside
# one of its ocaml/opam images, then build and run an installed consumer.
#
# Run as the image's opam user, with the checkout mounted read-only at
# /src:
#
#   docker run --rm -v "$PWD:/src:ro" ocaml/opam:alpine-3.24-ocaml-5.5 \
#     sh /src/scripts/check-distro-install.sh
#
# opam installs the depexts it declares (libxgboost-dev where the distro
# packages it, cmake, a C++ compiler, OpenMP), so each distro takes the
# path it would take in opam-repository CI: the system libxgboost or the
# vendored build. The consumer runs as native, plain bytecode (dlls found
# through the switch's own stublibs), byte_complete and in the toplevel.

set -eu

export OPAMYES=1 OPAMCONFIRMLEVEL=unsafe-yes OMP_NUM_THREADS=1

. /etc/os-release
distro="$ID ${VERSION_ID:-rolling}"
echo "== $distro, $(opam exec -- ocamlc -version)"

# Arch is rolling: the mirrors drop old package versions, but the image's
# pacman database is a snapshot, so opam's depext step (pacman -Su) can ask
# for a pkgconf/cmake build the mirror no longer serves (404). Refresh the
# database (and keyring) with a full sync+upgrade first, so depexts resolve
# to the versions the mirror actually has.
if [ "$ID" = arch ]; then
  sudo pacman -Sy --noconfirm archlinux-keyring || true
  sudo pacman -Syu --noconfirm
fi

# The image's opam-repository is a snapshot; bring it up to date.
git -C "$HOME/opam-repository" pull -q origin master
opam update -q

cp -R /src "$HOME/xgboost"
rm -rf "$HOME/xgboost/_build"
cd "$HOME/xgboost"
# --kind=path: pin the files as they are, not a git revision, so a copy
# of a git worktree (whose .git points outside the container) or of a
# checkout with uncommitted changes works too.
opam pin add -n --kind=path xgboost .
opam install --with-test xgboost

lib=$(opam var lib)
if [ -s "$lib/xgboost/bindings/libxgboost_vendored.a" ]; then
  provider=vendored
else
  provider=system
fi
echo "== installed with the $provider provider"

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

cat > "$consumer/toplevel.ml" <<'TOPLEVEL'
#use "topfind";;
#require "xgboost";;
let () =
  let major, minor, patch = Xgboost.version () in
  Printf.printf "Toplevel loaded xgboost %d.%d.%d\n" major minor patch
TOPLEVEL
opam exec -- ocaml "$consumer/toplevel.ml"

echo "== $distro: installed consumer passed ($provider)"
