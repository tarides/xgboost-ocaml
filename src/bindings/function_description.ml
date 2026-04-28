(* Function-level bindings for libxgboost (Phase 1 happy-path subset).
 *
 * Every C call returns [int] (0=ok, -1=err). The error message is
 * recovered via XGBGetLastError on the caller side. NULL out-pointers
 * are handled by the layer-C wrapper, not here.
 *
 * Handles (DMatrixHandle, BoosterHandle) are typedef'd to [void*] in
 * c_api.h. We expose them as [unit ptr] here; the high-level OCaml API
 * wraps them in record types with phantom-distinguished tags.
 *)

open Ctypes

module Functions (F : Ctypes.FOREIGN) = struct
  open F

  (* bst_ulong = uint64_t in c_api.h *)
  let bst_ulong = uint64_t

  (* Both DMatrixHandle and BoosterHandle are [void*]. We use [ptr void] for
     both; the high-level API recovers safety via phantom typing. *)
  let dmatrix_handle = ptr void
  let booster_handle = ptr void

  (* ----- meta ----- *)

  let xgboost_version =
    foreign "XGBoostVersion"
      (ptr int @-> ptr int @-> ptr int @-> returning void)

  let xgb_get_last_error =
    foreign "XGBGetLastError" (void @-> returning string)

  let xgb_set_global_config =
    foreign "XGBSetGlobalConfig" (string @-> returning int)

  (* ----- DMatrix ----- *)

  let xgdmatrix_create_from_mat =
    foreign "XGDMatrixCreateFromMat"
      (ptr float @-> bst_ulong @-> bst_ulong @-> float
      @-> ptr dmatrix_handle @-> returning int)

  let xgdmatrix_create_from_csr_ex =
    foreign "XGDMatrixCreateFromCSREx"
      (ptr size_t @-> ptr uint @-> ptr float
      @-> size_t @-> size_t @-> size_t
      @-> ptr dmatrix_handle @-> returning int)

  let xgdmatrix_set_float_info =
    foreign "XGDMatrixSetFloatInfo"
      (dmatrix_handle @-> string @-> ptr float @-> bst_ulong
      @-> returning int)

  let xgdmatrix_set_uint_info =
    foreign "XGDMatrixSetUIntInfo"
      (dmatrix_handle @-> string @-> ptr uint @-> bst_ulong
      @-> returning int)

  let xgdmatrix_get_float_info =
    foreign "XGDMatrixGetFloatInfo"
      (dmatrix_handle @-> string
      @-> ptr bst_ulong @-> ptr (ptr float) @-> returning int)

  let xgdmatrix_get_uint_info =
    foreign "XGDMatrixGetUIntInfo"
      (dmatrix_handle @-> string
      @-> ptr bst_ulong @-> ptr (ptr uint) @-> returning int)

  let xgdmatrix_num_row =
    foreign "XGDMatrixNumRow"
      (dmatrix_handle @-> ptr bst_ulong @-> returning int)

  let xgdmatrix_num_col =
    foreign "XGDMatrixNumCol"
      (dmatrix_handle @-> ptr bst_ulong @-> returning int)

  let xgdmatrix_num_non_missing =
    foreign "XGDMatrixNumNonMissing"
      (dmatrix_handle @-> ptr bst_ulong @-> returning int)

  let xgdmatrix_save_binary =
    foreign "XGDMatrixSaveBinary"
      (dmatrix_handle @-> string @-> int @-> returning int)

  let xgdmatrix_free =
    foreign "XGDMatrixFree" (dmatrix_handle @-> returning int)

  (* ----- Booster ----- *)

  let xgbooster_create =
    foreign "XGBoosterCreate"
      (ptr dmatrix_handle @-> bst_ulong @-> ptr booster_handle
      @-> returning int)

  let xgbooster_free =
    foreign "XGBoosterFree" (booster_handle @-> returning int)

  let xgbooster_reset =
    foreign "XGBoosterReset" (booster_handle @-> returning int)

  let xgbooster_set_param =
    foreign "XGBoosterSetParam"
      (booster_handle @-> string @-> string @-> returning int)

  let xgbooster_get_num_feature =
    foreign "XGBoosterGetNumFeature"
      (booster_handle @-> ptr bst_ulong @-> returning int)

  let xgbooster_boosted_rounds =
    foreign "XGBoosterBoostedRounds"
      (booster_handle @-> ptr int @-> returning int)

  let xgbooster_update_one_iter =
    foreign "XGBoosterUpdateOneIter"
      (booster_handle @-> int @-> dmatrix_handle @-> returning int)

  (* Modern (>=2.0) custom-objective training: grad and hess as JSON
     __array_interface__ strings instead of raw float* (the deprecated
     XGBoosterBoostOneIter took those directly). *)
  let xgbooster_train_one_iter =
    foreign "XGBoosterTrainOneIter"
      (booster_handle @-> dmatrix_handle @-> int @-> string @-> string
      @-> returning int)

  let xgbooster_eval_one_iter =
    foreign "XGBoosterEvalOneIter"
      (booster_handle @-> int
      @-> ptr dmatrix_handle @-> ptr string @-> bst_ulong
      @-> ptr string @-> returning int)

  let xgbooster_predict =
    foreign "XGBoosterPredict"
      (booster_handle @-> dmatrix_handle @-> int @-> uint @-> int
      @-> ptr bst_ulong @-> ptr (ptr float) @-> returning int)

  (* Modern in-place predict path. [values] is a JSON-encoded
     __array_interface__ object describing the input pointer and shape.
     [m] may be NULL or a proxy DMatrix carrying meta info. The output
     shape is written through [out_shape] (a borrowed const bst_ulong*
     of length out_dim) and the predictions through [out_result]. *)
  let xgbooster_predict_from_dense =
    foreign "XGBoosterPredictFromDense"
      (booster_handle @-> string @-> string @-> dmatrix_handle
      @-> ptr (ptr bst_ulong) @-> ptr bst_ulong @-> ptr (ptr float)
      @-> returning int)

  (* ----- Persistence ----- *)

  let xgbooster_save_model =
    foreign "XGBoosterSaveModel"
      (booster_handle @-> string @-> returning int)

  let xgbooster_load_model =
    foreign "XGBoosterLoadModel"
      (booster_handle @-> string @-> returning int)

  let xgbooster_save_model_to_buffer =
    foreign "XGBoosterSaveModelToBuffer"
      (booster_handle @-> string
      @-> ptr bst_ulong @-> ptr (ptr char) @-> returning int)

  let xgbooster_load_model_from_buffer =
    foreign "XGBoosterLoadModelFromBuffer"
      (booster_handle @-> ptr void @-> bst_ulong @-> returning int)

  let xgbooster_save_json_config =
    foreign "XGBoosterSaveJsonConfig"
      (booster_handle @-> ptr bst_ulong @-> ptr (ptr char) @-> returning int)

  let xgbooster_load_json_config =
    foreign "XGBoosterLoadJsonConfig"
      (booster_handle @-> string @-> returning int)

  (* Per-feature importance scores. [config] is a JSON string with at
     least {"importance_type": "weight"|"gain"|"cover"|...}. The output
     is two borrowed arrays: [features] (n_features strings) and
     [scores] (a tensor of shape [out_shape] whose product equals the
     score length). For non-multiclass models out_dim=1 and the shape
     is just [n_features]. *)
  let xgbooster_feature_score =
    foreign "XGBoosterFeatureScore"
      (booster_handle @-> string
      @-> ptr bst_ulong                  (* out_n_features *)
      @-> ptr (ptr (ptr char))           (* out_features *)
      @-> ptr bst_ulong                  (* out_dim *)
      @-> ptr (ptr bst_ulong)            (* out_shape *)
      @-> ptr (ptr float)                (* out_scores *)
      @-> returning int)
end
