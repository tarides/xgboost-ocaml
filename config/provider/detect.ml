(* Native provider decision, run in config/provider/.

   Writes [vendored] and [system] ("true"/"false"), read by the
   [enabled_if] fields that pick one of the two [xgboost_bindings]
   libraries (src/bindings/{system,vendored}) and gate the vendored CMake
   build. They must live in a different directory from those stanzas:
   dune reads them while loading rules there.

   On the system path it also writes the pkg-config flags for
   src/bindings/system; on the vendored path those files are empty lists. *)

module C = Configurator.V1

let write_bool file b =
  let oc = open_out_bin file in
  output_string oc (string_of_bool b);
  close_out oc

let () =
  C.main ~name:"xgboost-detect" (fun c ->
      let cflags, libs, system =
        match Xgboost_config.Common.system_conf c with
        | Some conf -> (conf.C.Pkg_config.cflags, conf.C.Pkg_config.libs, true)
        | None -> ([], [], false)
      in
      write_bool "system" system;
      write_bool "vendored" (not system);
      C.Flags.write_sexp "c_flags.sexp" (Xgboost_config.Common.warn_flags @ cflags);
      C.Flags.write_sexp "c_library_flags.sexp" libs)
