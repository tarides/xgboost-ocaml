(* Shared helpers for config/provider/detect.ml and config/discover.ml. *)

module C = Configurator.V1

let min_major = 3

(* Declared targets of the vendored rule in config/dune: persistent copies
   of the CMake-built static archives, plus shared companions built from
   them. src/bindings/vendored declares them as [(foreign_archives ...)],
   so dune links, installs and locates them itself. *)
let out_xgb = "libxgboost_vendored.a"
let out_dmlc = "libdmlc_vendored.a"
let out_xgb_dll = "dllxgboost_vendored.so"
let out_dmlc_dll = "dlldmlc_vendored.so"

(* ctypes' generated stubs declare XGBoost's borrowed-output pointers as
   non-const [T**] against the header's [const T**]; silence the benign
   cast warnings at the C compiler. *)
let warn_flags = [ "-Wno-incompatible-pointer-types"; "-Wno-discarded-qualifiers" ]

let parse_major v =
  (* "3.0.4" -> Some 3 *)
  match String.split_on_char '.' (String.trim v) with
  | major :: _ -> int_of_string_opt major
  | [] -> None

let copy_file ~src ~dst =
  let ic = open_in_bin src and oc = open_out_bin dst in
  let len = 65536 in
  let buf = Bytes.create len in
  let rec loop () =
    let n = input ic buf 0 len in
    if n > 0 then (output oc buf 0 n; loop ())
  in
  loop ();
  close_in ic;
  close_out oc

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

(* backtrace/backtrace_symbols — used by XGBoost's bundled dmlc logging —
   are in libc on glibc and in libSystem on macOS, but on the BSDs they
   live in a separate libexecinfo that must be linked explicitly. *)
let is_bsd c =
  let r = C.Process.run c "uname" [ "-s" ] in
  r.exit_code = 0
  && List.mem
       (String.lowercase_ascii (String.trim r.stdout))
       [ "freebsd"; "openbsd"; "netbsd"; "dragonfly" ]

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
   CPUs (respecting cgroup/affinity limits) and ignores OMP_NUM_THREADS.
   Capped at [max_build_jobs]: CI sandboxes (opam-repo CI) report the
   host's CPU count (80+) but not its memory budget, and that many
   parallel cc1plus processes get OOM-killed. *)
let max_build_jobs = 6

let build_jobs () =
  max 1 (min max_build_jobs (Domain.recommended_domain_count ()))

let die_on_error label (r : C.Process.result) =
  if r.exit_code <> 0 then
    C.die "%s failed (exit %d)\nstdout:\n%s\nstderr:\n%s" label r.exit_code
      r.stdout r.stderr

(* The C/C++ compiler driver used to link the shared companions. *)
let cc c = Option.value (C.ocaml_config_var c "c_compiler") ~default:"cc"

(* Link [archives] whole into a self-contained shared object [out], pulling
   in [libs] (C++/OpenMP runtime, etc.), for the plain-bytecode load path.
   GNU ld uses --whole-archive; the macOS linker uses -force_load per
   archive and an @rpath install name. *)
let build_dll c ~out ~archives ~libs =
  let args =
    if is_macos c then
      ("-dynamiclib" :: "-o" :: out
       :: List.map (fun a -> "-Wl,-force_load," ^ a) archives)
      @ [ "-install_name"; "@rpath/" ^ out ] @ libs
    else
      ("-shared" :: "-o" :: out :: "-Wl,--whole-archive" :: archives)
      @ ("-Wl,--no-whole-archive" :: libs)
  in
  die_on_error ("link " ^ out) (C.Process.run c (cc c) args)

(* Build vendored XGBoost as static libraries, copy them to the declared
   archive targets, build their shared companions, and return (cflags,
   libs). [cwd] is this rule's build directory (config/); the vendored sources were copied to
   ../vendor/xgboost via the (source_tree ...) dep. Absolute paths are
   needed for CMake inputs and header discovery, not installed linking. *)
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
  (* Copy into the declared, persistent targets and link against those. *)
  copy_file ~src:(find "libxgboost.a") ~dst:out_xgb;
  copy_file ~src:(find "libdmlc.a") ~dst:out_dmlc;
  let cflags =
    [ "-I"; Filename.concat src "include";
      "-I"; Filename.concat src "dmlc-core/include" ]
  in
  (* The archives themselves come in through [(foreign_archives ...)];
     only their runtime dependencies are named here. *)
  let libs =
    cxx_omp_libs c @ [ "-lm"; "-lpthread"; "-ldl" ]
    @ (if is_bsd c then [ "-lexecinfo" ] else [])
  in
  (* xgboost's dll also carries dmlc, so it satisfies the stubs' references
     whatever the load order. *)
  build_dll c ~out:out_xgb_dll ~archives:[ out_xgb; out_dmlc ] ~libs;
  build_dll c ~out:out_dmlc_dll ~archives:[ out_dmlc ] ~libs;
  (cflags, libs)
