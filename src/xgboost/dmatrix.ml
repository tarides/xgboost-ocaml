(* DMatrix — high-level OCaml interface to libxgboost's dense and sparse
 * input matrices.
 *
 * Lifetime model:
 *   - [Gc.finalise_last] frees the C handle when the OCaml record becomes
 *     unreachable.
 *   - The mutable [freed] flag and explicit [free] make double-free safe.
 *   - The [pin] field holds back-references to source Bigarrays (for
 *     of_bigarray2 / of_csr) so the GC keeps them alive while the DMatrix
 *     is live. XGBoost copies dense input on construction, but pinning is
 *     a defensive zero-cost safety net.
 *)

module F = Internal.F
open Ctypes

type t = {
  handle : Internal.dmatrix_handle;
  rows : int;
  cols : int;
  pin : Obj.t list;
  freed : bool ref;
}

let make ~handle ~rows ~cols ~pin =
  let freed = ref false in
  let t = { handle; rows; cols; pin; freed } in
  Gc.finalise_last
    (fun () ->
      if not !freed then begin
        freed := true;
        ignore (F.xgdmatrix_free handle)
      end)
    t;
  t

(* Same as [make] but invokes [extra_free ()] right after the C handle
   is freed. Used by the streaming iterator path to also free the proxy
   DMatrix that backs the iteration. *)
let make_with_extra_free ~handle ~rows ~cols ~pin ~extra_free =
  let freed = ref false in
  let t = { handle; rows; cols; pin; freed } in
  Gc.finalise_last
    (fun () ->
      if not !freed then begin
        freed := true;
        ignore (F.xgdmatrix_free handle);
        extra_free ()
      end)
    t;
  t

let rows t = t.rows
let cols t = t.cols
let handle t = t.handle

let check_live t =
  if !(t.freed) then
    raise (Error.Xgboost_error (Error.Invalid_argument "DMatrix.t is freed"))

let of_bigarray2 ?(missing = Float.nan) m =
  let rows = Bigarray.Array2.dim1 m in
  let cols = Bigarray.Array2.dim2 m in
  (* Modern API path: pass the Bigarray buffer via a JSON
     __array_interface__ string. Skips the per-row conversion that the
     legacy XGDMatrixCreateFromMat does internally; the buffer is
     pinned in the resulting DMatrix.t. *)
  let json_data = Array_interface.dense_array2 m in
  let missing_str =
    if Float.is_nan missing then "NaN" else string_of_float missing
  in
  let config = Printf.sprintf {|{"missing":%s}|} missing_str in
  let h_out = Internal.alloc_dmatrix_handle () in
  Internal.xgb_check
    (F.xgdmatrix_create_from_dense json_data config h_out);
  make ~handle:!@h_out ~rows ~cols ~pin:[ Obj.repr m ]

let of_csr ~indptr ~indices ~data ~n_cols =
  let missing = Float.nan in
  let n_rows = Bigarray.Array1.dim indptr - 1 in
  let nnz = Bigarray.Array1.dim data in
  if Bigarray.Array1.dim indices <> nnz then
    raise
      (Error.Xgboost_error
         (Error.Invalid_argument
            (Printf.sprintf
               "of_csr: indices length %d != data length %d"
               (Bigarray.Array1.dim indices) nnz)));
  if n_rows < 0 then
    raise
      (Error.Xgboost_error
         (Error.Invalid_argument
            "of_csr: indptr must have length n_rows+1 (>=1)"));

  (* Modern API path: pass the Bigarray buffers via JSON
     __array_interface__ strings, no per-element copy. The buffers are
     pinned in the resulting DMatrix.t. *)
  let json_indptr = Array_interface.dense_array1_int32 indptr in
  let json_indices = Array_interface.dense_array1_int32 indices in
  let json_data = Array_interface.dense_array1_float32 data in
  let missing_str =
    if Float.is_nan missing then "NaN" else string_of_float missing
  in
  let config = Printf.sprintf {|{"missing":%s}|} missing_str in
  let h_out = Internal.alloc_dmatrix_handle () in
  Internal.xgb_check
    (F.xgdmatrix_create_from_csr
       json_indptr json_indices json_data
       (Internal.ulong n_cols) config h_out);
  make ~handle:!@h_out ~rows:n_rows ~cols:n_cols
    ~pin:[ Obj.repr indptr; Obj.repr indices; Obj.repr data ]

let set_label t labels =
  check_live t;
  let n = Bigarray.Array1.dim labels in
  if n <> t.rows then
    raise
      (Error.Xgboost_error
         (Error.Shape_mismatch
            { expected = (t.rows, 0); got = (n, 0) }));
  Internal.xgb_check
    (F.xgdmatrix_set_float_info t.handle "label"
       (bigarray_start array1 labels) (Internal.ulong n))

let set_weight t weights =
  check_live t;
  let n = Bigarray.Array1.dim weights in
  if n <> t.rows then
    raise
      (Error.Xgboost_error
         (Error.Shape_mismatch
            { expected = (t.rows, 0); got = (n, 0) }));
  Internal.xgb_check
    (F.xgdmatrix_set_float_info t.handle "weight"
       (bigarray_start array1 weights) (Internal.ulong n))

let num_non_missing t =
  check_live t;
  Internal.with_out_ulong (F.xgdmatrix_num_non_missing t.handle)

let slice t idxset =
  check_live t;
  let n = Bigarray.Array1.dim idxset in
  let parent_rows = t.rows in
  for i = 0 to n - 1 do
    let r = Int32.to_int idxset.{i} in
    if r < 0 || r >= parent_rows then
      raise
        (Error.Xgboost_error
           (Error.Invalid_argument
              (Printf.sprintf
                 "DMatrix.slice: index %d at position %d out of range \
                  [0,%d)" r i parent_rows)))
  done;
  (* libxgboost expects [const int*]; our Bigarray is int32 with c_layout,
     which matches int on every platform we target. Pass via the Bigarray's
     start pointer (no copy). *)
  let h_out = Internal.alloc_dmatrix_handle () in
  let idx_ptr =
    coerce (ptr int32_t) (ptr int) (bigarray_start array1 idxset)
  in
  Internal.xgb_check
    (F.xgdmatrix_slice_dmatrix t.handle idx_ptr (Internal.ulong n) h_out);
  (* Pin the parent DMatrix and the index buffer for the child's lifetime.
     The parent's own [pin] field is captured transitively by pinning the
     parent record. *)
  make ~handle:!@h_out ~rows:n ~cols:t.cols
    ~pin:[ Obj.repr t; Obj.repr idxset ]

let free t =
  if not !(t.freed) then begin
    t.freed := true;
    Internal.xgb_check (F.xgdmatrix_free t.handle)
  end

let with_ make_t f =
  let t = make_t () in
  Fun.protect ~finally:(fun () -> free t) (fun () -> f t)
