(* Hybrid libxgboost discovery.
   =============================

   The binding needs libxgboost >= 3.0 (it binds the 2.0/3.0 C API). We
   pick, at configure time, between two ways of satisfying that:

   - Fast path: a system libxgboost >= 3.0 discoverable through
     pkg-config. Its presence is arranged by opam [depexts] on distros
     that package it (Debian/Ubuntu [libxgboost-dev], Homebrew
     [xgboost]); a developer may also have installed it by hand.

   - Fallback: build the vendored XGBoost sources under [vendor/xgboost]
     with CMake (static libs) and link those. This runs only when no
     usable system library is available, and needs no network — the
     sources are committed — so it also covers the opam-repo-ci sandbox
     and distros without a package (Fedora, Arch, ...).

   The choice is made by detect.ml (config/provider/). It selects one of
   two [xgboost_bindings] libraries with mutually exclusive [enabled_if]
   fields: src/bindings/system (pkg-config flags) or src/bindings/vendored
   ([(foreign_archives ...)]). Nothing vendored is built or installed on
   the system path, so there are no placeholders.

   This program is the vendored build, run only on the vendored path.
   Outputs (declared targets of the rule in config/dune):
   - c_flags.sexp / c_library_flags.sexp — the include paths and runtime
     dependencies (C++, OpenMP) for src/bindings/vendored's ctypes stanza.
   - libxgboost_vendored.a / libdmlc_vendored.a — the CMake-built static
     archives, and dllxgboost_vendored.so / dlldmlc_vendored.so — shared
     companions for plain bytecode. *)

module C = Configurator.V1

let () =
  C.main ~name:"xgboost" (fun c ->
      let cflags, libs = Xgboost_config.Common.vendored_conf c in
      C.Flags.write_sexp "c_flags.sexp" (Xgboost_config.Common.warn_flags @ cflags);
      C.Flags.write_sexp "c_library_flags.sexp" libs)
