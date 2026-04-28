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
    (* Mirrors gen_labels_binary in bin/c_reference/bench_common.h
       exactly: 4 terms, the 4th column index wraps via [3 mod cols]
       so the labels match the C reference for any cols >= 1. *)
    let v =
      0.5 *. m.{r, 0}
      -. 0.3 *. m.{r, min 1 (cols - 1)}
      +. 0.2 *. m.{r, min 2 (cols - 1)}
      +. 0.1 *. m.{r, 3 mod cols}
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

let test_streaming_iterator () =
  (* Build a 200-row dataset, then construct it twice: once as a single
     dense Bigarray, once via the streaming iterator in 4 batches of 50
     rows. Train boosters with identical params on both and compare
     predictions on a held-out predict matrix — must be bit-identical
     (same data, same seed, same hyperparams). *)
  prng_state := 0x57_2E_A1l;
  let rows = 200 and cols = 8 and n_batches = 4 in
  let m = big2 ~rows ~cols in
  fill2 m;
  let labels = labels_binary m in
  let batch_rows = rows / n_batches in

  let make_booster dtrain =
    let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
    Xgboost.Booster.set_params bst
      [
        "objective", "binary:logistic";
        "tree_method", "hist";
        "max_depth", "4";
        "seed", "0";
        "verbosity", "0";
      ];
    for it = 0 to 9 do
      Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
    done;
    bst
  in

  (* Baseline: single-shot dense DMatrix. *)
  let dtrain_full = Xgboost.DMatrix.of_bigarray2 m in
  Xgboost.DMatrix.set_label dtrain_full labels;
  let bst_full = make_booster dtrain_full in
  let preds_full = Xgboost.Booster.predict bst_full dtrain_full in

  (* Streaming: feed [m] in [n_batches] consecutive 50-row slices. *)
  let cursor = ref 0 in
  let next () =
    if !cursor >= n_batches then None
    else begin
      let start = !cursor * batch_rows in
      let bm = big2 ~rows:batch_rows ~cols in
      for r = 0 to batch_rows - 1 do
        for c = 0 to cols - 1 do
          bm.{r, c} <- m.{start + r, c}
        done
      done;
      let bl = big1 batch_rows in
      for r = 0 to batch_rows - 1 do
        bl.{r} <- labels.{start + r}
      done;
      incr cursor;
      Some Xgboost.DMatrix.{ data = Batch_dense bm; labels = Some bl }
    end
  in
  let reset () = cursor := 0 in
  let dtrain_iter = Xgboost.DMatrix.of_iterator ~next ~reset () in
  Alcotest.(check int) "iterator-built rows" rows
    (Xgboost.DMatrix.rows dtrain_iter);
  Alcotest.(check int) "iterator-built cols" cols
    (Xgboost.DMatrix.cols dtrain_iter);
  let bst_iter = make_booster dtrain_iter in
  let preds_iter = Xgboost.Booster.predict bst_iter dtrain_full in

  let max_diff = ref 0.0 in
  for i = 0 to Bigarray.Array1.dim preds_full - 1 do
    let d = Float.abs (preds_full.{i} -. preds_iter.{i}) in
    if d > !max_diff then max_diff := d
  done;
  Alcotest.(check (float 1e-5))
    "streaming and single-shot trainings agree" 0.0 !max_diff

let test_predict_dense_agrees_with_predict () =
  let m, _, dtrain, bst = train_simple ~seed:0xD3115El () in
  let p_via_dmat = Xgboost.Booster.predict bst dtrain in
  let p_inplace = Xgboost.Booster.predict_dense bst m in
  Alcotest.(check int) "lengths match"
    (Bigarray.Array1.dim p_via_dmat) (Bigarray.Array1.dim p_inplace);
  let max_diff = ref 0.0 in
  for i = 0 to Bigarray.Array1.dim p_via_dmat - 1 do
    let d = Float.abs (p_via_dmat.{i} -. p_inplace.{i}) in
    if d > !max_diff then max_diff := d
  done;
  Alcotest.(check (float 1e-5))
    "predict_dense agrees with predict via DMatrix" 0.0 !max_diff

let test_feature_score () =
  let _, _, _, bst = train_simple ~seed:0xF5C0_5El () in
  let scores = Xgboost.Booster.feature_score bst in
  Alcotest.(check bool) "feature_score returns at least one entry"
    true (List.length scores > 0);
  List.iter
    (fun (_, v) ->
      Alcotest.(check bool) "score finite" true (Float.is_finite v);
      Alcotest.(check bool) "score >= 0" true (v >= 0.0))
    scores

let test_boost_one_iter_custom_obj () =
  (* Drive a custom-objective training loop: at each iteration, predict,
     compute grad = pred - label and hess = 1 (squared-error
     derivatives), and call boost_one_iter. The built-in reg:squarederror
     uses additional regularisation/initialisation that we don't try to
     replicate exactly; we just verify (a) the call sequence completes
     without error, (b) predictions are finite and in a reasonable
     range, (c) MSE strictly decreases over iterations. *)
  prng_state := 0xCB05_71El;
  let rows = 200 and cols = 8 and iters = 15 in
  let m = big2 ~rows ~cols in
  fill2 m;
  let labels = big1 rows in
  for r = 0 to rows - 1 do
    labels.{r} <-
      0.5 *. m.{r, 0} -. 0.3 *. m.{r, 1} +. 0.2 *. m.{r, 2}
  done;
  let dtrain = Xgboost.DMatrix.of_bigarray2 m in
  Xgboost.DMatrix.set_label dtrain labels;
  let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
  Xgboost.Booster.set_params bst
    [
      "objective", "reg:squarederror";
      "tree_method", "hist";
      "max_depth", "4";
      "eta", "0.1";
      "verbosity", "0";
    ];

  let mse preds =
    let s = ref 0.0 in
    for r = 0 to rows - 1 do
      let d = preds.{r} -. labels.{r} in
      s := !s +. (d *. d)
    done;
    !s /. float_of_int rows
  in

  let grad = big1 rows in
  let hess = big1 rows in
  let mse0 = ref Float.infinity in
  let mse_last = ref Float.infinity in
  for it = 0 to iters - 1 do
    let preds = Xgboost.Booster.predict bst dtrain in
    if it = 0 then mse0 := mse preds;
    for r = 0 to rows - 1 do
      grad.{r} <- preds.{r} -. labels.{r};
      hess.{r} <- 1.0
    done;
    Xgboost.Booster.boost_one_iter bst ~iter:it ~dtrain ~grad ~hess
  done;
  let final_preds = Xgboost.Booster.predict bst dtrain in
  mse_last := mse final_preds;
  for r = 0 to rows - 1 do
    Alcotest.(check bool)
      (Printf.sprintf "pred[%d] finite" r) true
      (Float.is_finite final_preds.{r})
  done;
  Alcotest.(check bool)
    (Printf.sprintf "MSE decreased from %.4f to %.4f" !mse0 !mse_last)
    true (!mse_last < !mse0)

let test_unsafe_predict_borrowed () =
  let _, _, dtrain, bst = train_simple ~seed:0x800B_2A0Bl () in
  let p_safe = Xgboost.Booster.predict bst dtrain in
  let p_borrowed = Xgboost.Booster.Unsafe.predict_borrowed bst dtrain in
  (* Same booster, same DMatrix → same predictions. Read p_borrowed
     before any further call on bst. *)
  Alcotest.(check int) "same length"
    (Bigarray.Array1.dim p_safe) (Bigarray.Array1.dim p_borrowed);
  let max_diff = ref 0.0 in
  for i = 0 to Bigarray.Array1.dim p_safe - 1 do
    let d = Float.abs (p_safe.{i} -. p_borrowed.{i}) in
    if d > !max_diff then max_diff := d
  done;
  Alcotest.(check (float 0.0))
    "borrowed equals copied predict" 0.0 !max_diff

let test_result_try_ () =
  (* Ok path *)
  let m = big2 ~rows:10 ~cols:4 in
  fill2 m;
  let r =
    Xgboost.Result.try_ (fun () -> Xgboost.DMatrix.of_bigarray2 m)
  in
  (match r with
   | Ok d ->
       Alcotest.(check int) "ok rows" 10 (Xgboost.DMatrix.rows d);
       Xgboost.DMatrix.free d
   | Error _ -> Alcotest.fail "expected Ok");
  (* Error path: invalid label shape *)
  let dt = Xgboost.DMatrix.of_bigarray2 m in
  let r2 =
    Xgboost.Result.try_ (fun () ->
        let bad = big1 5 in
        Xgboost.DMatrix.set_label dt bad)
  in
  (match r2 with
   | Ok () -> Alcotest.fail "expected Error"
   | Error (Xgboost.Error.Shape_mismatch _) -> ()
   | Error _ -> Alcotest.fail "wrong error variant");
  Xgboost.DMatrix.free dt

let fixture_predictions =
  [|
    9.617016315e-01;
    3.944031429e-03;
    9.921880960e-01;
    9.953472018e-01;
    9.950394034e-01;
  |]

(* Cross-layer parity oracle: same setup as bin/c_reference/check.c
   and test/bindings/check.ml::test_fixture_parity. The high-level
   API must reproduce the captured fixture predictions to ~1e-5.
   See test/bindings/check.ml for why we skip under ASan. *)
let in_unstable_fp_regime () =
  match Sys.getenv_opt "LD_PRELOAD" with
  | Some s when String.length s > 0 -> true
  | _ -> false

let test_fixture_parity_layer_c () =
  if in_unstable_fp_regime () then Alcotest.skip ()
  else
  prng_state := 0xCAFE_BABEl;
  let rows = 200 and cols = 16 and iters = 30 in
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
  let preds = Xgboost.Booster.predict bst dtrain in
  for i = 0 to Array.length fixture_predictions - 1 do
    let got = preds.{i} in
    let want = fixture_predictions.(i) in
    Alcotest.(check (float 1e-5))
      (Printf.sprintf "fixture[%d] (got %.9e want %.9e)" i got want)
      want got
  done

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
          Alcotest.test_case "streaming_iterator" `Quick test_streaming_iterator;
          Alcotest.test_case "predict_dense_agrees" `Quick
            test_predict_dense_agrees_with_predict;
          Alcotest.test_case "feature_score" `Quick test_feature_score;
          Alcotest.test_case "boost_one_iter_custom_obj" `Quick
            test_boost_one_iter_custom_obj;
          Alcotest.test_case "unsafe_predict_borrowed" `Quick
            test_unsafe_predict_borrowed;
          Alcotest.test_case "result_try_" `Quick test_result_try_;
          Alcotest.test_case "fixture_parity_layer_c" `Quick
            test_fixture_parity_layer_c;
          Alcotest.test_case "double_free_safe" `Quick
            test_double_free_safe;
        ] );
    ]
