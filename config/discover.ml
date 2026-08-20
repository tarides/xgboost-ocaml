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

   Either way we emit [c_flags.sexp] / [c_library_flags.sexp], consumed
   by the ctypes stanza in src/bindings/dune via [(:include ...)]. *)

module C = Configurator.V1

let min_major = 3

(* ctypes' generated stubs declare XGBoost's borrowed-output pointers as
   non-const [T**] against the header's [const T**]; silence the benign
   cast warnings at the C compiler. *)
let warn_flags = [ "-Wno-incompatible-pointer-types"; "-Wno-discarded-qualifiers" ]

let parse_major v =
  (* "3.0.4" -> Some 3 *)
  match String.split_on_char '.' (String.trim v) with
  | major :: _ -> int_of_string_opt major
  | [] -> None

let system_conf c =
  match C.Pkg_config.get c with
  | None -> None
  | Some pc -> (
      match C.Pkg_config.query pc ~package:"xgboost" with
      | None -> None
      | Some conf -> (
          (* pkg-config version constraints are unreliable across .pc
             files; gate explicitly on --modversion. *)
          let r = C.Process.run c "pkg-config" [ "--modversion"; "xgboost" ] in
          match if r.exit_code = 0 then parse_major r.stdout else None with
          | Some major when major >= min_major -> Some conf
          | _ -> None))

let is_macos c =
  match C.ocaml_config_var c "system" with
  | Some ("macosx" | "macos") -> true
  | _ -> false

(* OpenMP and C++ runtime link flags, per platform. On macOS, Homebrew's
   libomp is keg-only, so add its lib dir explicitly. *)
let cxx_omp_libs c =
  if is_macos c then
    let brew_libomp =
      let r = C.Process.run c "brew" [ "--prefix"; "libomp" ] in
      if r.exit_code = 0 then [ "-L" ^ Filename.concat (String.trim r.stdout) "lib" ]
      else []
    in
    brew_libomp @ [ "-lc++"; "-lomp" ]
  else [ "-lstdc++"; "-lgomp" ]

(* Parallelism for the vendored CMake build. Deliberately NOT `nproc`,
   which honours OMP_NUM_THREADS (pinned to 1 in our test/CI env for
   deterministic libxgboost results) and would force a single-threaded,
   many-minutes build. [recommended_domain_count] reflects the available
   CPUs (respecting cgroup/affinity limits) and ignores OMP_NUM_THREADS. *)
let build_jobs () = max 1 (Domain.recommended_domain_count ())

let die_on_error label (r : C.Process.result) =
  if r.exit_code <> 0 then
    C.die "%s failed (exit %d)\nstdout:\n%s\nstderr:\n%s" label r.exit_code
      r.stdout r.stderr

(* Build vendored XGBoost as static libraries and return (cflags, libs).
   [cwd] is this rule's build directory (config/); the vendored sources
   were copied to ../vendor/xgboost via the (source_tree ...) dep. Paths
   are made absolute so they survive the later link step. *)
let vendored_conf c =
  let cwd = Sys.getcwd () in
  let abs p = if Filename.is_relative p then Filename.concat cwd p else p in
  let src = abs (Filename.concat Filename.parent_dir_name "vendor/xgboost") in
  let build = abs "vendor-build" in
  let cmake = Option.value (C.which c "cmake") ~default:"cmake" in
  die_on_error "cmake configure"
    (C.Process.run c cmake
       [ "-S"; src; "-B"; build;
         "-DCMAKE_BUILD_TYPE=Release";
         "-DBUILD_STATIC_LIB=ON";
         "-DUSE_OPENMP=ON";
         "-DUSE_CUDA=OFF";
         "-DBUILD_WITH_SHARED_NCCL=OFF" ]);
  die_on_error "cmake build"
    (C.Process.run c cmake
       [ "--build"; build; "--config"; "Release"; "-j"; string_of_int (build_jobs ()) ]);
  (* Locate the produced static archives; layout varies — XGBoost sets
     LIBRARY_OUTPUT_DIRECTORY to <src>/lib, while dmlc lands under the
     build dir — so search both roots. *)
  let find name =
    let r = C.Process.run c "find" [ build; src; "-name"; name; "-type"; "f" ] in
    match
      if r.exit_code = 0 then
        List.filter (fun s -> s <> "") (String.split_on_char '\n' (String.trim r.stdout))
      else []
    with
    | path :: _ -> path
    | [] -> C.die "vendored build did not produce %s under %s" name build
  in
  let xgb = find "libxgboost.a" in
  let dmlc = find "libdmlc.a" in
  let cflags =
    [ "-I"; Filename.concat src "include";
      "-I"; Filename.concat src "dmlc-core/include" ]
  in
  let libs = [ xgb; dmlc ] @ cxx_omp_libs c @ [ "-lm"; "-lpthread"; "-ldl" ] in
  (cflags, libs)

let () =
  C.main ~name:"xgboost" (fun c ->
      let cflags, libs =
        match system_conf c with
        | Some conf -> (conf.C.Pkg_config.cflags, conf.C.Pkg_config.libs)
        | None -> vendored_conf c
      in
      C.Flags.write_sexp "c_flags.sexp" (warn_flags @ cflags);
      C.Flags.write_sexp "c_library_flags.sexp" libs)
