module Error = Error
module DMatrix = Dmatrix
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
