(* Layer-C internals: shared helpers around the ctypes-generated Functions
 * module. Not part of the public API. *)

module F = Xgboost_bindings.C.Functions
open Ctypes

(* The C ABI uses [void*] for both DMatrixHandle and BoosterHandle. We
 * keep them as a single OCaml type (unit ptr) and rely on phantom type
 * tags in the higher layers to keep them apart. *)
type dmatrix_handle = unit ptr
type booster_handle = unit ptr

let ulong = Unsigned.UInt64.of_int
let ulong_to_int = Unsigned.UInt64.to_int
let usize = Unsigned.Size_t.of_int
let uintv = Unsigned.UInt.of_int

let xgb_check rc =
  if rc <> 0 then
    raise (Error.Xgboost_error (Error.Xgb_error (F.xgb_get_last_error ())))

let alloc_dmatrix_handle () = allocate F.dmatrix_handle null
let alloc_booster_handle () = allocate F.booster_handle null

(* Read an out parameter [bst_ulong*]. *)
let with_out_ulong f =
  let p = allocate uint64_t Unsigned.UInt64.zero in
  xgb_check (f p);
  ulong_to_int !@p

(* Read an out parameter [int*]. *)
let with_out_int f =
  let p = allocate int 0 in
  xgb_check (f p);
  !@p

(* For functions that emit a borrowed [const float**] of length [out_len]:
 * copy [len] floats into a fresh OCaml-owned Bigarray.Array1 of float32.
 * The borrowed pointer is invalidated by the next call on the same
 * booster, so we copy eagerly.
 *
 * Implementation: wrap the borrowed pointer as a temporary Bigarray view
 * and call [Bigarray.Array1.blit], which is implemented as memcpy when
 * the source and destination kind/layout match. Replaces an earlier
 * element-by-element loop that was costing ~4ns/elem (W3 was 5pt over
 * the +30% Phase-2 gate; the loop was the smoking gun, see G3 + W3
 * decomposition in BENCH.md). *)
let copy_borrowed_float32 ~len ~src =
  let dst =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout len
  in
  if len > 0 then begin
    let view =
      bigarray_of_ptr array1 len Bigarray.Float32 src
    in
    Bigarray.Array1.blit view dst
  end;
  dst

(* Copy [len] bytes from a borrowed [const char*] into fresh OCaml [bytes].
 * Used only for save_model_buffer and save_json_config (infrequent
 * per-model calls), so the element loop is fine. *)
let copy_borrowed_bytes ~len ~src =
  let buf = Bytes.create len in
  for i = 0 to len - 1 do
    Bytes.unsafe_set buf i !@(src +@ i)
  done;
  buf
