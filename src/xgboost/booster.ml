(* Booster — high-level OCaml interface to libxgboost's gradient-boosting
 * model.
 *
 * Lifetime rules (see architecture plan):
 *   - The C-side XGBoost booster does not own its training DMatrices, only
 *     a snapshot of their pointers. We must keep the OCaml DMatrix.t live
 *     for as long as XGBoost might re-touch it.
 *   - [cache_pin] permanently holds DMatrices passed at create time.
 *   - [iter_pin] holds the [dtrain] argument to [update_one_iter] for the
 *     duration of that single call (released on return). *)

module F = Internal.F
open Ctypes

type t = {
  handle : Internal.booster_handle;
  cache_pin : Dmatrix.t list;
  mutable iter_pin : Dmatrix.t option;
  mutable n_features : int option;
  freed : bool ref;
}

let make ~handle ~cache_pin =
  let freed = ref false in
  let t =
    { handle; cache_pin; iter_pin = None; n_features = None; freed }
  in
  Gc.finalise_last
    (fun () ->
      if not !freed then begin
        freed := true;
        ignore (F.xgbooster_free handle)
      end)
    t;
  t

let check_live t =
  if !(t.freed) then
    raise (Error.Xgboost_error (Error.Invalid_argument "Booster.t is freed"))

let create ?(cache = []) () =
  List.iter Dmatrix.check_live cache;
  let n = List.length cache in
  let h_out = Internal.alloc_booster_handle () in
  let dmats = CArray.make F.dmatrix_handle (max n 1) in
  List.iteri
    (fun i d -> CArray.set dmats i (Dmatrix.handle d))
    cache;
  Internal.xgb_check
    (F.xgbooster_create
       (CArray.start dmats) (Internal.ulong n) h_out);
  make ~handle:!@h_out ~cache_pin:cache

let set_param t k v =
  check_live t;
  Internal.xgb_check (F.xgbooster_set_param t.handle k v)

let set_params t ps =
  check_live t;
  List.iter
    (fun (k, v) -> Internal.xgb_check (F.xgbooster_set_param t.handle k v))
    ps

let update_one_iter t ~iter ~dtrain =
  check_live t;
  Dmatrix.check_live dtrain;
  t.iter_pin <- Some dtrain;
  Fun.protect
    ~finally:(fun () -> t.iter_pin <- None)
    (fun () ->
      Internal.xgb_check
        (F.xgbooster_update_one_iter t.handle iter (Dmatrix.handle dtrain)))

let eval_one_iter t ~iter ~evals =
  check_live t;
  let n = List.length evals in
  let dmats = CArray.make F.dmatrix_handle (max n 1) in
  let names = CArray.make string (max n 1) in
  List.iteri
    (fun i (name, d) ->
      Dmatrix.check_live d;
      CArray.set dmats i (Dmatrix.handle d);
      CArray.set names i name)
    evals;
  let out = allocate string "" in
  Internal.xgb_check
    (F.xgbooster_eval_one_iter t.handle iter
       (CArray.start dmats) (CArray.start names) (Internal.ulong n) out);
  !@out

let num_features t =
  check_live t;
  match t.n_features with
  | Some n -> n
  | None ->
      let n =
        Internal.with_out_ulong (F.xgbooster_get_num_feature t.handle)
      in
      t.n_features <- Some n;
      n

let boosted_rounds t =
  check_live t;
  Internal.with_out_int (F.xgbooster_boosted_rounds t.handle)

let reset t =
  check_live t;
  Internal.xgb_check (F.xgbooster_reset t.handle);
  t.n_features <- None

let predict ?(ntree_limit = 0) ?(training = false) t dmat =
  check_live t;
  Dmatrix.check_live dmat;
  let out_len = allocate uint64_t Unsigned.UInt64.zero in
  let out_result = allocate (ptr float) (from_voidp float null) in
  Internal.xgb_check
    (F.xgbooster_predict t.handle (Dmatrix.handle dmat) 0
       (Internal.uintv ntree_limit) (if training then 1 else 0)
       out_len out_result);
  let len = Internal.ulong_to_int !@out_len in
  let src = !@out_result in
  Internal.copy_borrowed_float32 ~len ~src

(* In-place predict from a Bigarray.Array2. Avoids allocating a transient
   DMatrix per call — useful for tight inference loops. The input is
   passed to libxgboost as a JSON __array_interface__ pointing into the
   Bigarray's memory; we keep the Bigarray reachable across the call. *)
let predict_dense ?(ntree_limit = 0) ?(training = false)
    ?(missing = Float.nan) t m =
  check_live t;
  let values_json = Array_interface.dense_array2 m in
  (* JSON config: type=0 (normal predict), iteration_end=ntree_limit
     (0 means all), strict_shape=false (we compute total length from
     out_shape). NaN literal is accepted by RapidJSON. *)
  let missing_str =
    if Float.is_nan missing then "NaN"
    else if missing = Float.infinity then "Infinity"
    else if missing = Float.neg_infinity then "-Infinity"
    else string_of_float missing
  in
  let config_json =
    Printf.sprintf
      {|{"type":0,"training":%b,"iteration_begin":0,"iteration_end":%d,"strict_shape":false,"missing":%s}|}
      training ntree_limit missing_str
  in
  let out_shape =
    allocate (ptr uint64_t) (from_voidp uint64_t null)
  in
  let out_dim = allocate uint64_t Unsigned.UInt64.zero in
  let out_result = allocate (ptr float) (from_voidp float null) in
  Internal.xgb_check
    (F.xgbooster_predict_from_dense t.handle values_json config_json
       null out_shape out_dim out_result);
  ignore (Sys.opaque_identity m);
  let dim = Internal.ulong_to_int !@out_dim in
  let shape_ptr = !@out_shape in
  let total = ref 1 in
  for i = 0 to dim - 1 do
    total := !total * Internal.ulong_to_int !@(shape_ptr +@ i)
  done;
  let len = !total in
  let src = !@out_result in
  Internal.copy_borrowed_float32 ~len ~src

let save_model t ~path =
  check_live t;
  Internal.xgb_check (F.xgbooster_save_model t.handle path)

let load_model t ~path =
  check_live t;
  Internal.xgb_check (F.xgbooster_load_model t.handle path);
  t.n_features <- None

let save_model_buffer ?(format = "{\"format\": \"json\"}") t =
  check_live t;
  let blen = allocate uint64_t Unsigned.UInt64.zero in
  let bptr = allocate (ptr char) (from_voidp char null) in
  Internal.xgb_check
    (F.xgbooster_save_model_to_buffer t.handle format blen bptr);
  let len = Internal.ulong_to_int !@blen in
  let src = !@bptr in
  Internal.copy_borrowed_bytes ~len ~src

let load_model_buffer t buf =
  check_live t;
  let len = Bytes.length buf in
  (* Pin the bytes by copying to a CArray for the duration of the call. *)
  let carr = CArray.make char len in
  for i = 0 to len - 1 do
    CArray.set carr i (Bytes.unsafe_get buf i)
  done;
  Internal.xgb_check
    (F.xgbooster_load_model_from_buffer t.handle
       (to_voidp (CArray.start carr)) (Internal.ulong len));
  t.n_features <- None

let save_json_config t =
  check_live t;
  let blen = allocate uint64_t Unsigned.UInt64.zero in
  let bptr = allocate (ptr char) (from_voidp char null) in
  Internal.xgb_check
    (F.xgbooster_save_json_config t.handle blen bptr);
  let len = Internal.ulong_to_int !@blen in
  let src = !@bptr in
  Bytes.to_string (Internal.copy_borrowed_bytes ~len ~src)

let load_json_config t json =
  check_live t;
  Internal.xgb_check (F.xgbooster_load_json_config t.handle json)

let free t =
  if not !(t.freed) then begin
    t.freed := true;
    Internal.xgb_check (F.xgbooster_free t.handle)
  end

let with_ ?cache f =
  let t = create ?cache () in
  Fun.protect ~finally:(fun () -> free t) (fun () -> f t)
