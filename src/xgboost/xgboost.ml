module Error = Error

module DMatrix = struct
  include Dmatrix

  type batch = Streaming.batch =
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

  type labelled_batch = Streaming.labelled_batch = {
    data : batch;
    labels :
      (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
      option;
  }

  let of_iterator = Streaming.of_iterator
end

module Booster = Booster

exception Xgboost_error = Error.Xgboost_error

let version () =
  let module F = Internal.F in
  let major = Ctypes.allocate Ctypes.int 0 in
  let minor = Ctypes.allocate Ctypes.int 0 in
  let patch = Ctypes.allocate Ctypes.int 0 in
  F.xgboost_version major minor patch;
  Ctypes.( !@major, !@minor, !@patch )

let last_error () =
  let module F = Internal.F in
  F.xgb_get_last_error ()
