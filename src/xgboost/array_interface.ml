(* Build numpy __array_interface__ JSON strings for Bigarrays.
 *
 * libxgboost 2.0+ takes raw data through this interface (see e.g.
 * XGBoosterPredictFromDense, XGProxyDMatrixSetDataDense, the modern
 * XGDMatrixCreateFromCSR). The format is documented at
 *   https://numpy.org/doc/stable/reference/arrays.interface.html
 * and consists of:
 *   { "data": [<address>, false],
 *     "shape": [...],
 *     "strides": null,
 *     "typestr": "<f4" | "<i4" | ...,
 *     "version": 3 }
 *
 * The [false] in "data" is the read-only flag (false = writable, but
 * libxgboost doesn't actually write to passed-in arrays).
 *
 * The caller is responsible for keeping the Bigarray reachable for the
 * lifetime of any C call that holds the JSON-encoded address. *)

open Ctypes

let typestr_float32 = "<f4"
let typestr_int32 = "<i4"
let typestr_uint64 = "<u8"

(* Convert a ctypes pointer address to a JSON integer literal. We can't
   use [%d because the address may overflow OCaml's int range on 32-bit;
   but on 64-bit a [nativeint] fits in a JSON number (libxgboost parses
   it with a 64-bit integer). *)
let address_of_ptr p =
  Nativeint.to_string (raw_address_of_ptr (to_voidp p))

let dense_array2 (m : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t) =
  let rows = Bigarray.Array2.dim1 m in
  let cols = Bigarray.Array2.dim2 m in
  let addr = address_of_ptr (bigarray_start array2 m) in
  Printf.sprintf
    {|{"data":[%s,false],"shape":[%d,%d],"strides":null,"typestr":"%s","version":3}|}
    addr rows cols typestr_float32

let dense_array1_float32 (a : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t) =
  let n = Bigarray.Array1.dim a in
  let addr = address_of_ptr (bigarray_start array1 a) in
  Printf.sprintf
    {|{"data":[%s,false],"shape":[%d],"strides":null,"typestr":"%s","version":3}|}
    addr n typestr_float32

let dense_array1_int32 (a : (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t) =
  let n = Bigarray.Array1.dim a in
  let addr = address_of_ptr (bigarray_start array1 a) in
  Printf.sprintf
    {|{"data":[%s,false],"shape":[%d],"strides":null,"typestr":"%s","version":3}|}
    addr n typestr_int32
