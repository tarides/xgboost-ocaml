(* High-level (layer C) alcotest unit tests.
 *
 * These tests treat Xgboost as a black box: only the public API
 * (Xgboost.DMatrix, Xgboost.Booster, Xgboost.Xgboost_error) is touched.
 * They cover the happy path, the scoped combinators, save/load round
 * trips and the typed error path. *)

open Bigarray

(* ---------- helpers ---------- *)

let big2 ~rows ~cols = Array2.create float32 c_layout rows cols
let big1 n = Array1.create float32 c_layout n

let prng_state = ref 0xC0FFEEl

let next_uniform () =
  let x = !prng_state in
  let x = Int32.logxor x (Int32.shift_left x 13) in
  let x = Int32.logxor x (Int32.shift_right_logical x 17) in
  let x = Int32.logxor x (Int32.shift_left x 5) in
  let x = if Int32.equal x 0l then 1l else x in
  prng_state := x;
  Int32.to_float (Int32.shift_right_logical x 8) /. 16777216.0

let fill2 (m : (float, _, _) Array2.t) =
  for r = 0 to Array2.dim1 m - 1 do
    for c = 0 to Array2.dim2 m - 1 do
      m.{r, c} <- next_uniform ()
    done
  done

let labels_binary (m : (float, _, _) Array2.t) =
  let rows = Array2.dim1 m in
  let cols = Array2.dim2 m in
  let l = big1 rows in
  for r = 0 to rows - 1 do
    let v =
      0.5 *. m.{r, 0}
      -. 0.3 *. m.{r, min 1 (cols - 1)}
      +. 0.2 *. m.{r, min 2 (cols - 1)}
      +. 0.1 *. m.{r, min 3 (cols - 1)}
    in
    l.{r} <- (if v > 0.25 then 1.0 else 0.0)
  done;
  l

let train_simple ?(seed = 0xDEADBEEFl) ?(rows = 200) ?(cols = 12)
    ?(iters = 30) () =
  prng_state := seed;
  let m = big2 ~rows ~cols in
  fill2 m;
  let labels = labels_binary m in
  let dtrain = Xgboost.DMatrix.of_bigarray2 m in
  Xgboost.DMatrix.set_label dtrain labels;
  let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
  Xgboost.Booster.set_params bst
    [
      "objective", "binary:logistic";
      "tree_method", "hist";
      "max_depth", "4";
      "seed", "0";
      "verbosity", "0";
    ];
  for it = 0 to iters - 1 do
    Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
  done;
  (m, labels, dtrain, bst)

(* ---------- tests ---------- *)

let test_version () =
  let major, _, _ = Xgboost.version () in
  Alcotest.(check int) "major version" 3 major

let test_train_predict () =
  let m, _, dtrain, bst = train_simple () in
  Alcotest.(check int) "num_features"
    (Array2.dim2 m) (Xgboost.Booster.num_features bst);
  Alcotest.(check int) "boosted_rounds" 30
    (Xgboost.Booster.boosted_rounds bst);
  let preds = Xgboost.Booster.predict bst dtrain in
  Alcotest.(check int) "predict length"
    (Array2.dim1 m) (Array1.dim preds);
  let p0 = preds.{0} in
  Alcotest.(check bool) "p0 finite" true (Float.is_finite p0);
  Alcotest.(check bool) "p0 in [0,1]" true (p0 >= 0.0 && p0 <= 1.0)

let test_dmatrix_shape_mismatch () =
  prng_state := 0xCAFEl;
  let m = big2 ~rows:50 ~cols:6 in
  fill2 m;
  let dtrain = Xgboost.DMatrix.of_bigarray2 m in
  let labels = big1 49 (* wrong size *) in
  match Xgboost.DMatrix.set_label dtrain labels with
  | () -> Alcotest.fail "expected Shape_mismatch"
  | exception Xgboost.Xgboost_error (Xgboost.Error.Shape_mismatch _) -> ()

let test_dmatrix_with_ () =
  let observed = ref None in
  Xgboost.DMatrix.with_
    (fun () ->
      let m = big2 ~rows:10 ~cols:4 in
      Xgboost.DMatrix.of_bigarray2 m)
    (fun d -> observed := Some (Xgboost.DMatrix.rows d));
  Alcotest.(check (option int)) "rows observed inside scope"
    (Some 10) !observed

let test_save_load_buffer () =
  let _, _, dtrain, bst = train_simple ~seed:0xBEEF11l () in
  let preds_before = Xgboost.Booster.predict bst dtrain in
  let buf = Xgboost.Booster.save_model_buffer bst in
  Alcotest.(check bool) "buffer non-empty" true (Bytes.length buf > 0);
  Xgboost.Booster.with_ (fun bst2 ->
      Xgboost.Booster.load_model_buffer bst2 buf;
      Alcotest.(check int) "loaded rounds" 30
        (Xgboost.Booster.boosted_rounds bst2);
      let preds_after = Xgboost.Booster.predict bst2 dtrain in
      Alcotest.(check int) "predict shapes match"
        (Array1.dim preds_before) (Array1.dim preds_after);
      let max_diff = ref 0.0 in
      for i = 0 to Array1.dim preds_before - 1 do
        let d = Float.abs (preds_before.{i} -. preds_after.{i}) in
        if d > !max_diff then max_diff := d
      done;
      Alcotest.(check (float 1e-6))
        "predictions byte-identical after round-trip" 0.0 !max_diff)

let test_save_load_path () =
  let _, _, dtrain, bst = train_simple ~seed:0xBEEF22l () in
  let path = Filename.temp_file "xgb" ".json" in
  let preds_before = Xgboost.Booster.predict bst dtrain in
  Xgboost.Booster.save_model bst ~path;
  Xgboost.Booster.with_ (fun bst2 ->
      Xgboost.Booster.load_model bst2 ~path;
      let preds_after = Xgboost.Booster.predict bst2 dtrain in
      let max_diff = ref 0.0 in
      for i = 0 to Array1.dim preds_before - 1 do
        let d = Float.abs (preds_before.{i} -. preds_after.{i}) in
        if d > !max_diff then max_diff := d
      done;
      Alcotest.(check (float 1e-6))
        "predictions equal after path round-trip" 0.0 !max_diff);
  Sys.remove path

let test_json_config_round_trip () =
  let _, _, _, bst = train_simple ~seed:0x1234l () in
  let cfg = Xgboost.Booster.save_json_config bst in
  Alcotest.(check bool) "config non-empty" true (String.length cfg > 0);
  Xgboost.Booster.load_json_config bst cfg;
  let cfg2 = Xgboost.Booster.save_json_config bst in
  Alcotest.(check string) "json config round-trips identically" cfg cfg2

let test_csr () =
  prng_state := 0xC5_3F_00l;
  let n_rows = 50 in
  let n_cols = 8 in
  let density = 0.4 in
  let cap = n_rows * n_cols in
  let indptr = Array1.create int32 c_layout (n_rows + 1) in
  let indices_buf = Array1.create int32 c_layout cap in
  let data_buf = big1 cap in
  let nnz = ref 0 in
  indptr.{0} <- 0l;
  for r = 0 to n_rows - 1 do
    for c = 0 to n_cols - 1 do
      if next_uniform () < density then begin
        indices_buf.{!nnz} <- Int32.of_int c;
        data_buf.{!nnz} <- next_uniform ();
        incr nnz
      end
    done;
    indptr.{r + 1} <- Int32.of_int !nnz
  done;
  let nnz = !nnz in
  let indices = Array1.sub indices_buf 0 nnz in
  let data = Array1.sub data_buf 0 nnz in
  let d = Xgboost.DMatrix.of_csr ~indptr ~indices ~data ~n_cols in
  Alcotest.(check int) "csr rows" n_rows (Xgboost.DMatrix.rows d);
  Alcotest.(check int) "csr cols" n_cols (Xgboost.DMatrix.cols d)

let test_double_free_safe () =
  let m = big2 ~rows:5 ~cols:3 in
  let d = Xgboost.DMatrix.of_bigarray2 m in
  Xgboost.DMatrix.free d;
  Xgboost.DMatrix.free d;  (* must not crash or raise *)
  let bst = Xgboost.Booster.create () in
  Xgboost.Booster.free bst;
  Xgboost.Booster.free bst

let () =
  Alcotest.run "xgboost"
    [
      ( "smoke",
        [
          Alcotest.test_case "version" `Quick test_version;
          Alcotest.test_case "train_predict" `Quick test_train_predict;
          Alcotest.test_case "dmatrix_shape_mismatch" `Quick
            test_dmatrix_shape_mismatch;
          Alcotest.test_case "dmatrix_with_" `Quick test_dmatrix_with_;
          Alcotest.test_case "save_load_buffer" `Quick test_save_load_buffer;
          Alcotest.test_case "save_load_path" `Quick test_save_load_path;
          Alcotest.test_case "json_config_round_trip" `Quick
            test_json_config_round_trip;
          Alcotest.test_case "csr" `Quick test_csr;
          Alcotest.test_case "double_free_safe" `Quick
            test_double_free_safe;
        ] );
    ]
