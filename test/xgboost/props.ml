(* High-level (layer C) property-based tests.
 *
 * These exercise invariants of the XGBoost binding that are quantified
 * over inputs (matrix shape, parameter combinations, etc.). qcheck
 * generates the inputs; we run a relatively small count (100 by default)
 * to keep wall-clock acceptable while still catching regressions.
 *
 * Properties:
 *   prop_predict_shape       — predict length == rows of input
 *   prop_model_buffer_rt     — save+load yields identical predictions
 *   prop_json_config_rt      — save_json_config; load → save returns same
 *   prop_dmatrix_lifetime    — many constructions+frees leave heap clean
 *   prop_double_free_safe    — explicit free + GC finaliser does not crash
 *)

open Bigarray

(* ---------- helpers ---------- *)

let big2 ~rows ~cols = Array2.create float32 c_layout rows cols
let big1 n = Array1.create float32 c_layout n

let fill_rand st (m : (float, _, _) Array2.t) =
  for r = 0 to Array2.dim1 m - 1 do
    for c = 0 to Array2.dim2 m - 1 do
      m.{r, c} <- QCheck.Gen.generate1 ~rand:st (QCheck.Gen.float_range 0.0 1.0)
    done
  done

let labels_binary (m : (float, _, _) Array2.t) =
  let rows = Array2.dim1 m in
  let l = big1 rows in
  for r = 0 to rows - 1 do
    l.{r} <- (if m.{r, 0} > 0.5 then 1.0 else 0.0)
  done;
  l

let train_small ~rng ~rows ~cols ~iters =
  let m = big2 ~rows ~cols in
  fill_rand rng m;
  let labels = labels_binary m in
  let dtrain = Xgboost.DMatrix.of_bigarray2 m in
  Xgboost.DMatrix.set_label dtrain labels;
  let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
  Xgboost.Booster.set_params bst
    [
      "objective", "binary:logistic";
      "tree_method", "hist";
      "max_depth", "3";
      "seed", "0";
      "verbosity", "0";
    ];
  for it = 0 to iters - 1 do
    Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
  done;
  (m, dtrain, bst)

(* ---------- properties ---------- *)

let arb_size = QCheck.pair (QCheck.int_range 8 80) (QCheck.int_range 4 16)

let prop_predict_shape =
  QCheck.Test.make ~name:"predict shape == rows" ~count:50 arb_size
    (fun (rows, cols) ->
      let st = Random.State.make [| rows; cols; 42 |] in
      let _, dtrain, bst = train_small ~rng:st ~rows ~cols ~iters:5 in
      let p = Xgboost.Booster.predict bst dtrain in
      Array1.dim p = rows)

let max_abs_diff a b =
  let n = Array1.dim a in
  let d = ref 0.0 in
  for i = 0 to n - 1 do
    let v = Float.abs (a.{i} -. b.{i}) in
    if v > !d then d := v
  done;
  !d

let prop_model_buffer_rt =
  QCheck.Test.make ~name:"model buffer round-trip preserves predictions"
    ~count:30 arb_size (fun (rows, cols) ->
      let st = Random.State.make [| rows; cols; 7 |] in
      let _, dtrain, bst = train_small ~rng:st ~rows ~cols ~iters:5 in
      let preds_before = Xgboost.Booster.predict bst dtrain in
      let buf = Xgboost.Booster.save_model_buffer bst in
      Xgboost.Booster.with_ (fun bst2 ->
          Xgboost.Booster.load_model_buffer bst2 buf;
          let preds_after = Xgboost.Booster.predict bst2 dtrain in
          let diff = max_abs_diff preds_before preds_after in
          diff = 0.0))

let prop_json_config_rt =
  QCheck.Test.make ~name:"JSON config round-trip is identity" ~count:30
    arb_size (fun (rows, cols) ->
      let st = Random.State.make [| rows; cols; 99 |] in
      let _, _, bst = train_small ~rng:st ~rows ~cols ~iters:3 in
      let cfg = Xgboost.Booster.save_json_config bst in
      Xgboost.Booster.load_json_config bst cfg;
      let cfg2 = Xgboost.Booster.save_json_config bst in
      String.equal cfg cfg2)

let prop_dmatrix_lifetime =
  QCheck.Test.make ~name:"DMatrix lifetime: 200 alloc/free leaves heap clean"
    ~count:5
    QCheck.unit
    (fun () ->
      let before = (Gc.stat ()).live_words in
      for _ = 1 to 200 do
        let m = big2 ~rows:50 ~cols:8 in
        let d = Xgboost.DMatrix.of_bigarray2 m in
        Xgboost.DMatrix.free d;
        ignore (Sys.opaque_identity m)
      done;
      Gc.full_major ();
      let after = (Gc.stat ()).live_words in
      (* Allow a bit of growth from internal logger / lazy init paths. *)
      after - before < 100_000)

let prop_double_free_safe =
  QCheck.Test.make ~name:"double-free is safe via explicit + GC" ~count:5
    QCheck.unit
    (fun () ->
      for _ = 1 to 50 do
        let m = big2 ~rows:20 ~cols:5 in
        let d = Xgboost.DMatrix.of_bigarray2 m in
        Xgboost.DMatrix.free d;
        Xgboost.DMatrix.free d;
        ignore (Sys.opaque_identity m)
      done;
      Gc.full_major ();
      true)

let () =
  let tests =
    [
      prop_predict_shape;
      prop_model_buffer_rt;
      prop_json_config_rt;
      prop_dmatrix_lifetime;
      prop_double_free_safe;
    ]
    |> List.map QCheck_alcotest.to_alcotest
  in
  Alcotest.run "xgboost-props" [ "props", tests ]
