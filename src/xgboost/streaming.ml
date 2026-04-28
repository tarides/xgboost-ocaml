(* Streaming-iterator construction of DMatrices.
 *
 * The modern callback API in libxgboost (XGProxyDMatrixCreate +
 * XGDMatrixCreateFromCallback) lets the caller feed batches of data on
 * demand, optionally backed by an on-disk cache for datasets larger
 * than RAM. The two callbacks (next, reset) take the iterator handle
 * (a void pointer) but we ignore it and rely on closure capture instead.
 *
 * These functions are bound via [Foreign.foreign] (libffi runtime FFI)
 * rather than the dune ctypes plugin's static stubs because:
 *   - they're called at most once per DMatrix construction (rare
 *     compared to the train / predict hot paths), so libffi's per-call
 *     overhead is below noise;
 *   - libffi's [Foreign.funptr] gives a clean way to pass OCaml
 *     closures as C function pointers without writing a C shim; the
 *     plugin's [static_funptr] would require additional plumbing for
 *     the same result.
 *
 * Lifetime invariants:
 *   - The callback closures (next_fn, reset_fn) and the proxy DMatrix
 *     handle are pinned in the resulting DMatrix.t for its lifetime.
 *     XGBoost may invoke the callbacks again during prediction in
 *     external-memory mode (cache page-in).
 *   - The CURRENT batch's source Bigarrays are pinned in a mutable ref
 *     held by the closure environment, so they survive across
 *     [XGProxyDMatrixSetData*]'s borrowed-pointer use until the next
 *     [next ()] call drops them.
 *)

open Ctypes

(* --- Foreign bindings ------------------------------------------------ *)

let dmatrix_handle = ptr void

let ulong = Unsigned.UInt64.of_int

let xgproxy_dmatrix_create =
  Foreign.foreign "XGProxyDMatrixCreate"
    (ptr dmatrix_handle @-> returning int)

let xgproxy_dmatrix_set_data_dense =
  Foreign.foreign "XGProxyDMatrixSetDataDense"
    (dmatrix_handle @-> string @-> returning int)

let xgproxy_dmatrix_set_data_csr =
  Foreign.foreign "XGProxyDMatrixSetDataCSR"
    (dmatrix_handle @-> string @-> string @-> string
    @-> returning int)

(* Modern (2.1+) replacement for XGDMatrixSetDenseInfo: takes the
   field name and a JSON __array_interface__ describing the buffer. *)
let xgdmatrix_set_info_from_interface =
  Foreign.foreign "XGDMatrixSetInfoFromInterface"
    (dmatrix_handle @-> string @-> string @-> returning int)

let next_funptr_t = Foreign.funptr (ptr void @-> returning int)
let reset_funptr_t = Foreign.funptr (ptr void @-> returning void)

let xgdmatrix_create_from_callback =
  Foreign.foreign "XGDMatrixCreateFromCallback"
    (ptr void                (* user iter handle (we pass NULL) *)
    @-> dmatrix_handle       (* proxy *)
    @-> reset_funptr_t       (* reset callback *)
    @-> next_funptr_t        (* next callback *)
    @-> string               (* config json *)
    @-> ptr dmatrix_handle   (* out *)
    @-> returning int)

(* --- Public API ------------------------------------------------------ *)

type batch =
  | Batch_dense of
      (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t
  | Batch_csr of {
      indptr :
        (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t;
      indices :
        (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t;
      data :
        (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t;
      n_cols : int;
    }

(* Optional per-batch labels. If supplied, set on the proxy after each
   data setter call. *)
type labelled_batch = {
  data : batch;
  labels :
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
    option;
}

let xgb_check rc =
  if rc <> 0 then
    raise
      (Error.Xgboost_error
         (Error.Xgb_error
            (Xgboost_bindings.C.Functions.xgb_get_last_error ())))

(* Set labels on the proxy by handing libxgboost a JSON
   __array_interface__ describing the Bigarray buffer (no copy). *)
let set_proxy_labels proxy labels =
  let json = Array_interface.dense_array1_float32 labels in
  xgb_check
    (xgdmatrix_set_info_from_interface proxy "label" json)

let feed_batch proxy lb =
  match lb.data with
  | Batch_dense m ->
      let json = Array_interface.dense_array2 m in
      xgb_check (xgproxy_dmatrix_set_data_dense proxy json)
  | Batch_csr { indptr; indices; data; n_cols = _ } ->
      let json_indptr = Array_interface.dense_array1_int32 indptr in
      let json_indices = Array_interface.dense_array1_int32 indices in
      let json_data = Array_interface.dense_array1_float32 data in
      xgb_check
        (xgproxy_dmatrix_set_data_csr proxy
           json_indptr json_indices json_data)

(* Pin the source Bigarrays of [lb] so the GC keeps them alive across the
   subsequent libxgboost call(s). Returned as Obj.t list to be stashed
   in a closure-captured ref. *)
let pin_batch lb =
  let data_pins =
    match lb.data with
    | Batch_dense m -> [ Obj.repr m ]
    | Batch_csr { indptr; indices; data; _ } ->
        [ Obj.repr indptr; Obj.repr indices; Obj.repr data ]
  in
  match lb.labels with
  | None -> data_pins
  | Some l -> Obj.repr l :: data_pins

let of_iterator ?(cache_prefix = "") ?(missing = Float.nan)
    ~next ~reset () =
  (* Create the proxy DMatrix. *)
  let proxy_out = allocate dmatrix_handle null in
  xgb_check (xgproxy_dmatrix_create proxy_out);
  let proxy = !@proxy_out in

  (* Mutable pin for the current batch's source data. Cleared on
     [reset] and replaced on each [next]. *)
  let current_pin : Obj.t list ref = ref [] in
  (* Track the proxy lifetime so we can free it if construction fails. *)
  let freed_proxy = ref false in
  let free_proxy () =
    if not !freed_proxy then begin
      freed_proxy := true;
      ignore
        (Xgboost_bindings.C.Functions.xgdmatrix_free proxy)
    end
  in

  let next_cb _iter_handle : int =
    current_pin := [];
    match
      try next ()
      with e ->
        Printf.eprintf "xgboost: iterator next raised: %s\n%!"
          (Printexc.to_string e);
        None
    with
    | None -> 0
    | Some lb ->
        feed_batch proxy lb;
        (match lb.labels with
         | None -> ()
         | Some l -> set_proxy_labels proxy l);
        current_pin := pin_batch lb;
        1
  in
  let reset_cb _iter_handle : unit =
    current_pin := [];
    try reset ()
    with e ->
      Printf.eprintf "xgboost: iterator reset raised: %s\n%!"
        (Printexc.to_string e)
  in

  let missing_str =
    if Float.is_nan missing then "NaN"
    else string_of_float missing
  in
  (* libxgboost requires "cache_prefix" to be present even in in-memory
     mode; passing "" disables external-memory caching. *)
  let config_json =
    Printf.sprintf
      {|{"missing":%s,"cache_prefix":"%s"}|}
      missing_str cache_prefix
  in

  let out = allocate dmatrix_handle null in
  let rc =
    xgdmatrix_create_from_callback null proxy reset_cb next_cb config_json
      out
  in
  if rc <> 0 then begin
    let msg =
      Xgboost_bindings.C.Functions.xgb_get_last_error ()
    in
    free_proxy ();
    raise (Error.Xgboost_error (Error.Xgb_error msg))
  end;

  let handle = !@out in
  (* Query rows/cols on the assembled DMatrix. *)
  let module B = Xgboost_bindings.C.Functions in
  let rows =
    let p = allocate uint64_t Unsigned.UInt64.zero in
    xgb_check (B.xgdmatrix_num_row handle p);
    Unsigned.UInt64.to_int !@p
  in
  let cols =
    let p = allocate uint64_t Unsigned.UInt64.zero in
    xgb_check (B.xgdmatrix_num_col handle p);
    Unsigned.UInt64.to_int !@p
  in
  (* Pin everything XGBoost may still touch:
     - the callback closures (libffi-allocated trampolines depend on
       these OCaml values being alive)
     - the proxy DMatrix (for external memory mode)
     - the current batch (after [next] returned 0 we set current_pin :=
       []; but if we are in external memory mode this ref will hold the
       latest batch fed in via subsequent next() calls)
     - the proxy free thunk so that finalisation walks it as well. *)
  let pin =
    [
      Obj.repr next_cb;
      Obj.repr reset_cb;
      Obj.repr proxy;
      Obj.repr current_pin;
      Obj.repr free_proxy;
    ]
  in
  let dmatrix =
    Dmatrix.make_with_extra_free ~handle ~rows ~cols ~pin
      ~extra_free:free_proxy
  in
  dmatrix
