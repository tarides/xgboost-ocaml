module C = Configurator.V1

let default_cflags = []
let default_libs = [ "-lxgboost" ]

let () =
  C.main ~name:"xgboost" (fun c ->
      let conf =
        match C.Pkg_config.get c with
        | None -> { C.Pkg_config.libs = default_libs; cflags = default_cflags }
        | Some pc -> (
            match C.Pkg_config.query pc ~package:"xgboost" with
            | Some conf -> conf
            | None ->
                { C.Pkg_config.libs = default_libs; cflags = default_cflags })
      in
      C.Flags.write_sexp "c_flags.sexp" conf.cflags;
      C.Flags.write_sexp "c_library_flags.sexp" conf.libs)
