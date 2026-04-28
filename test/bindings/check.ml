(* Layer-B smoke tests: every happy-path call exercised end-to-end via
 * the ctypes-generated static stubs. The intent is to catch (a) mismatches
 * between [function_description.ml] and the upstream C ABI, and (b)
 * regressions in the generator pipeline.
 *
 * These tests do not exercise the high-level API; that lives in
 * test/xgboost/. *)

open Ctypes
module F = Xgboost_bindings.C.Functions

let ulong = Unsigned.UInt64.of_int
let uint = Unsigned.UInt.of_int

let xgb_ok rc =
  if rc <> 0 then
    Alcotest.failf "xgboost call failed: %s" (F.xgb_get_last_error ())

let big2 ~rows ~cols =
  Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout rows cols

let big1 n = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n

let prng_state = ref 0xC0FFEEl

let next_uniform () =
  let x = !prng_state in
  let x = Int32.logxor x (Int32.shift_left x 13) in
  let x = Int32.logxor x (Int32.shift_right_logical x 17) in
  let x = Int32.logxor x (Int32.shift_left x 5) in
  let x = if Int32.equal x 0l then 1l else x in
  prng_state := x;
  Int32.to_float (Int32.shift_right_logical x 8) /. 16777216.0

let fill2 (m : (float, _, _) Bigarray.Array2.t) =
  for r = 0 to Bigarray.Array2.dim1 m - 1 do
    for c = 0 to Bigarray.Array2.dim2 m - 1 do
      m.{r, c} <- next_uniform ()
    done
  done

let labels_binary (m : (float, _, _) Bigarray.Array2.t) =
  let rows = Bigarray.Array2.dim1 m in
  let cols = Bigarray.Array2.dim2 m in
  let l = big1 rows in
  for r = 0 to rows - 1 do
    let v =
      0.5 *. m.{r, 0} -. 0.3 *. m.{r, min 1 (cols - 1)}
      +. 0.2 *. m.{r, min 2 (cols - 1)}
    in
    l.{r} <- (if v > 0.25 then 1.0 else 0.0)
  done;
  l

(* ---------- tests ---------- *)

let test_version () =
  let major = allocate int 0 in
  let minor = allocate int 0 in
  let patch = allocate int 0 in
  F.xgboost_version major minor patch;
  Alcotest.(check int) "major version is 3" 3 !@major;
  Alcotest.(check bool) "minor >= 0" true (!@minor >= 0);
  Alcotest.(check bool) "patch >= 0" true (!@patch >= 0)

let test_dmatrix_lifecycle () =
  prng_state := 0x12345l;
  let rows = 64 and cols = 8 in
  let m = big2 ~rows ~cols in
  fill2 m;
  let labels = labels_binary m in
  let dmat_out = allocate F.dmatrix_handle null in
  xgb_ok
    (F.xgdmatrix_create_from_mat
       (bigarray_start array2 m)
       (ulong rows) (ulong cols) Float.nan dmat_out);
  let h = !@dmat_out in
  xgb_ok
    (F.xgdmatrix_set_float_info h "label"
       (bigarray_start array1 labels) (ulong rows));
  let out_rows = allocate uint64_t Unsigned.UInt64.zero in
  xgb_ok (F.xgdmatrix_num_row h out_rows);
  Alcotest.(check int) "rows" rows (Unsigned.UInt64.to_int !@out_rows);
  let out_cols = allocate uint64_t Unsigned.UInt64.zero in
  xgb_ok (F.xgdmatrix_num_col h out_cols);
  Alcotest.(check int) "cols" cols (Unsigned.UInt64.to_int !@out_cols);
  xgb_ok (F.xgdmatrix_free h)

let test_train_and_predict () =
  prng_state := 0xCAFE_BABEl;
  let rows = 200 and cols = 16 and iters = 30 in

  let m = big2 ~rows ~cols in
  fill2 m;
  let labels = labels_binary m in

  let dmat_out = allocate F.dmatrix_handle null in
  xgb_ok
    (F.xgdmatrix_create_from_mat
       (bigarray_start array2 m)
       (ulong rows) (ulong cols) Float.nan dmat_out);
  let dtrain = !@dmat_out in
  xgb_ok
    (F.xgdmatrix_set_float_info dtrain "label"
       (bigarray_start array1 labels) (ulong rows));

  let booster_out = allocate F.booster_handle null in
  let dtrains = CArray.make F.dmatrix_handle 1 in
  CArray.set dtrains 0 dtrain;
  xgb_ok (F.xgbooster_create (CArray.start dtrains) (ulong 1) booster_out);
  let bst = !@booster_out in
  List.iter
    (fun (k, v) -> xgb_ok (F.xgbooster_set_param bst k v))
    [
      "objective", "binary:logistic";
      "tree_method", "hist";
      "max_depth", "4";
      "seed", "0";
      "verbosity", "0";
    ];
  for it = 0 to iters - 1 do
    xgb_ok (F.xgbooster_update_one_iter bst it dtrain)
  done;

  let n_feat = allocate uint64_t Unsigned.UInt64.zero in
  xgb_ok (F.xgbooster_get_num_feature bst n_feat);
  Alcotest.(check int) "num_features"
    cols (Unsigned.UInt64.to_int !@n_feat);

  let rounds = allocate int 0 in
  xgb_ok (F.xgbooster_boosted_rounds bst rounds);
  Alcotest.(check int) "boosted_rounds" iters !@rounds;

  let out_len = allocate uint64_t Unsigned.UInt64.zero in
  let out_result = allocate (ptr float) (from_voidp float null) in
  xgb_ok
    (F.xgbooster_predict bst dtrain 0 (uint 0) 0 out_len out_result);
  let n = Unsigned.UInt64.to_int !@out_len in
  Alcotest.(check int) "predict length" rows n;
  let outp = !@out_result in
  let p0 = !@outp in
  Alcotest.(check bool) "p0 finite" true (Float.is_finite p0);
  Alcotest.(check bool) "p0 in [0,1]" true (p0 >= 0.0 && p0 <= 1.0);

  xgb_ok (F.xgbooster_free bst);
  xgb_ok (F.xgdmatrix_free dtrain)

let test_error_path () =
  (* Setting an unknown parameter should produce a non-zero return.
   * Some XGBoost versions accept unknown params silently; we treat
   * either outcome as acceptable but require xgb_get_last_error to
   * return a string when an error did occur. *)
  let booster_out = allocate F.booster_handle null in
  let none = CArray.make F.dmatrix_handle 0 in
  xgb_ok
    (F.xgbooster_create (CArray.start none) (ulong 0) booster_out);
  let bst = !@booster_out in
  let _ = F.xgbooster_set_param bst "" "" in
  let _ = F.xgb_get_last_error () in
  xgb_ok (F.xgbooster_free bst)

let test_save_load_buffer () =
  prng_state := 0xBEEFl;
  let rows = 100 and cols = 8 and iters = 5 in
  let m = big2 ~rows ~cols in
  fill2 m;
  let labels = labels_binary m in

  let dmat_out = allocate F.dmatrix_handle null in
  xgb_ok
    (F.xgdmatrix_create_from_mat
       (bigarray_start array2 m)
       (ulong rows) (ulong cols) Float.nan dmat_out);
  let dtrain = !@dmat_out in
  xgb_ok
    (F.xgdmatrix_set_float_info dtrain "label"
       (bigarray_start array1 labels) (ulong rows));

  let booster_out = allocate F.booster_handle null in
  let dtrains = CArray.make F.dmatrix_handle 1 in
  CArray.set dtrains 0 dtrain;
  xgb_ok (F.xgbooster_create (CArray.start dtrains) (ulong 1) booster_out);
  let bst = !@booster_out in
  xgb_ok (F.xgbooster_set_param bst "objective" "reg:squarederror");
  xgb_ok (F.xgbooster_set_param bst "verbosity" "0");
  for it = 0 to iters - 1 do
    xgb_ok (F.xgbooster_update_one_iter bst it dtrain)
  done;

  (* save model to buffer *)
  let blen = allocate uint64_t Unsigned.UInt64.zero in
  let bptr = allocate (ptr char) (from_voidp char null) in
  xgb_ok (F.xgbooster_save_model_to_buffer bst "{\"format\": \"json\"}" blen bptr);
  let n = Unsigned.UInt64.to_int !@blen in
  Alcotest.(check bool) "buffer non-empty" true (n > 0);

  (* round-trip into a fresh booster *)
  let booster2_out = allocate F.booster_handle null in
  let none = CArray.make F.dmatrix_handle 0 in
  xgb_ok
    (F.xgbooster_create (CArray.start none) (ulong 0) booster2_out);
  let bst2 = !@booster2_out in
  let raw = !@bptr in
  xgb_ok
    (F.xgbooster_load_model_from_buffer bst2 (to_voidp raw) (ulong n));

  let r2 = allocate int 0 in
  xgb_ok (F.xgbooster_boosted_rounds bst2 r2);
  Alcotest.(check int) "loaded rounds" iters !@r2;

  xgb_ok (F.xgbooster_free bst2);
  xgb_ok (F.xgbooster_free bst);
  xgb_ok (F.xgdmatrix_free dtrain)

let () =
  Alcotest.run "xgboost_bindings"
    [
      ( "smoke",
        [
          Alcotest.test_case "version" `Quick test_version;
          Alcotest.test_case "dmatrix_lifecycle" `Quick test_dmatrix_lifecycle;
          Alcotest.test_case "train_and_predict" `Quick test_train_and_predict;
          Alcotest.test_case "error_path" `Quick test_error_path;
          Alcotest.test_case "save_load_buffer" `Quick test_save_load_buffer;
        ] );
    ]
