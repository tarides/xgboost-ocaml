(* Type-level bindings for libxgboost.
 *
 * The XGBoost C API is built almost entirely on `void*` opaque handles
 * (DMatrixHandle, BoosterHandle, DataIterHandle) and integer/float
 * scalars. There are no struct or enum declarations from the header that
 * we need to materialise on the OCaml side. This functor stays empty;
 * the plugin still requires it to produce the generated bindings
 * scaffold. Concrete types are introduced in [Function_description]. *)

module Types (F : Ctypes.TYPE) = struct
  let _ = (module F : Ctypes.TYPE)
end
