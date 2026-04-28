(* Typed errors crossing the FFI boundary.
 *
 * Every XGBoost C call returns int (0=ok, -1=err); the binding raises
 * [Xgboost_error] with a captured copy of [XGBGetLastError ()]. Other
 * variants describe binding-side preconditions (Bigarray shape, etc.). *)

type t =
  | Xgb_error of string
  | Invalid_argument of string
  | Shape_mismatch of { expected : int * int; got : int * int }

exception Xgboost_error of t

let pp ppf = function
  | Xgb_error s -> Format.fprintf ppf "Xgboost: %s" s
  | Invalid_argument s -> Format.fprintf ppf "Invalid argument: %s" s
  | Shape_mismatch { expected = (er, ec); got = (gr, gc) } ->
      Format.fprintf ppf "Shape mismatch: expected %dx%d, got %dx%d" er ec gr gc

let to_string e = Format.asprintf "%a" pp e

let () =
  Printexc.register_printer (function
    | Xgboost_error e -> Some (Format.asprintf "Xgboost.Xgboost_error(%a)" pp e)
    | _ -> None)
