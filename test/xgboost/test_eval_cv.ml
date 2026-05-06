(* Tests for the Eval and Cv modules. *)

open Bigarray

let big1 n = Array1.create float32 c_layout n
let big2 ~rows ~cols = Array2.create float32 c_layout rows cols
let big1_int n = Array1.create int c_layout n

let prng_state = ref 0xC0FFEEl

let next_uniform () =
  let x = !prng_state in
  let x = Int32.logxor x (Int32.shift_left x 13) in
  let x = Int32.logxor x (Int32.shift_right_logical x 17) in
  let x = Int32.logxor x (Int32.shift_left x 5) in
  let x = if Int32.equal x 0l then 1l else x in
  prng_state := x;
  Int32.to_float (Int32.shift_right_logical x 8) /. 16777216.0

let fill2 m =
  for r = 0 to Array2.dim1 m - 1 do
    for c = 0 to Array2.dim2 m - 1 do
      m.{r, c} <- next_uniform ()
    done
  done

let labels_binary m =
  let rows = Array2.dim1 m in
  let cols = Array2.dim2 m in
  let l = big1 rows in
  for r = 0 to rows - 1 do
    let v =
      0.5 *. m.{r, 0}
      -. 0.3 *. m.{r, min 1 (cols - 1)}
      +. 0.2 *. m.{r, min 2 (cols - 1)}
      +. 0.1 *. m.{r, 3 mod cols}
    in
    l.{r} <- (if v > 0.25 then 1.0 else 0.0)
  done;
  l

let load_array1 xs =
  let n = Array.length xs in
  let a = big1 n in
  Array.iteri (fun i v -> a.{i} <- v) xs;
  a

(* ---------- Eval.parse / get ---------- *)

let test_parse_basic () =
  let s = "[12]\ttest-auc:0.998612\ttrain-logloss:0.05" in
  let parsed = Xgboost.Eval.parse s in
  Alcotest.(check int) "two metrics" 2 (List.length parsed);
  Alcotest.(check string)
    "first metric name" "test-auc" (fst (List.nth parsed 0));
  Alcotest.(check (float 1e-9))
    "first metric value" 0.998612 (snd (List.nth parsed 0));
  Alcotest.(check string)
    "second metric name" "train-logloss" (fst (List.nth parsed 1));
  Alcotest.(check (float 1e-9))
    "second metric value" 0.05 (snd (List.nth parsed 1))

let test_parse_trailing_newline () =
  let parsed = Xgboost.Eval.parse "[0]\teval-auc:0.5\n" in
  Alcotest.(check (float 1e-9))
    "trimmed value" 0.5
    (List.assoc "eval-auc" parsed)

let test_get () =
  let s = "[3]\ttrain-error:0.0123\ttest-error:0.0456" in
  Alcotest.(check (float 1e-9))
    "train-error" 0.0123
    (Xgboost.Eval.get ~metric:"train-error" s);
  Alcotest.check_raises "missing metric raises"
    (Xgboost.Xgboost_error
       (Xgboost.Error.Invalid_argument
          "Eval.get: metric \"missing\" not found in \
           \"[3]\\ttrain-error:0.0123\\ttest-error:0.0456\""))
    (fun () -> ignore (Xgboost.Eval.get ~metric:"missing" s))

(* ---------- Eval.auc ---------- *)

(* Hand-computed fixture: predictions [0.9; 0.8; 0.4; 0.3],
   labels [1; 0; 1; 0]. Sorted ROC: (0,0)→(0,0.5)→(0.5,0.5)→(0.5,1)
   →(1,1). Trapezoidal AUC = 0.75. *)
let test_auc_fixture () =
  let preds = load_array1 [| 0.9; 0.8; 0.4; 0.3 |] in
  let labels = load_array1 [| 1.0; 0.0; 1.0; 0.0 |] in
  let auc = Xgboost.Eval.auc ~predictions:preds ~labels in
  Alcotest.(check (float 1e-9)) "fixture AUC" 0.75 auc

let test_auc_perfect () =
  let preds = load_array1 [| 0.99; 0.98; 0.05; 0.01 |] in
  let labels = load_array1 [| 1.0; 1.0; 0.0; 0.0 |] in
  Alcotest.(check (float 1e-9)) "perfect ranking"
    1.0
    (Xgboost.Eval.auc ~predictions:preds ~labels)

let test_auc_inverted () =
  let preds = load_array1 [| 0.99; 0.98; 0.05; 0.01 |] in
  let labels = load_array1 [| 0.0; 0.0; 1.0; 1.0 |] in
  Alcotest.(check (float 1e-9)) "inverted ranking"
    0.0
    (Xgboost.Eval.auc ~predictions:preds ~labels)

let test_auc_random_close_to_half () =
  (* 1000 random predictions vs random labels — AUC should sit near
     0.5. Use the LCG PRNG with a fixed seed for reproducibility. *)
  prng_state := 0xDEADBEEFl;
  let n = 1000 in
  let preds = big1 n in
  let labels = big1 n in
  for i = 0 to n - 1 do
    preds.{i} <- next_uniform ();
    labels.{i} <- (if next_uniform () < 0.5 then 0.0 else 1.0)
  done;
  let auc = Xgboost.Eval.auc ~predictions:preds ~labels in
  if auc < 0.45 || auc > 0.55 then
    Alcotest.failf "random AUC %f outside [0.45,0.55]" auc

let test_auc_ties () =
  (* All predictions equal → AUC must be 0.5 by tie-handling. *)
  let preds = load_array1 [| 0.5; 0.5; 0.5; 0.5 |] in
  let labels = load_array1 [| 1.0; 0.0; 1.0; 0.0 |] in
  Alcotest.(check (float 1e-9)) "ties yield 0.5"
    0.5
    (Xgboost.Eval.auc ~predictions:preds ~labels)

(* ---------- Eval.roc ---------- *)

let test_roc_endpoints () =
  let preds = load_array1 [| 0.9; 0.8; 0.4; 0.3 |] in
  let labels = load_array1 [| 1.0; 0.0; 1.0; 0.0 |] in
  let pts = Xgboost.Eval.roc ~predictions:preds ~labels in
  let first = List.hd pts in
  let last = List.nth pts (List.length pts - 1) in
  Alcotest.(check (pair (float 1e-9) (float 1e-9))) "starts at origin"
    (0.0, 0.0) first;
  Alcotest.(check (pair (float 1e-9) (float 1e-9))) "ends at (1,1)"
    (1.0, 1.0) last

(* ---------- DMatrix.slice ---------- *)

let train_simple_booster ~rows ~cols ~iters seed =
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

let test_slice_round_trip () =
  let _m, _labels, dtrain, bst =
    train_simple_booster ~rows:200 ~cols:8 ~iters:10 0xCAFEl
  in
  (* Slice rows 5,7,9,11,13 and verify dimensions. *)
  let idx = Array1.of_array int32 c_layout [| 5l; 7l; 9l; 11l; 13l |] in
  let child = Xgboost.DMatrix.slice dtrain idx in
  Alcotest.(check int) "child rows" 5 (Xgboost.DMatrix.rows child);
  Alcotest.(check int) "child cols" 8 (Xgboost.DMatrix.cols child);
  let preds_full = Xgboost.Booster.predict bst dtrain in
  let preds_child = Xgboost.Booster.predict bst child in
  for i = 0 to 4 do
    let parent_row = Int32.to_int idx.{i} in
    Alcotest.(check (float 1e-5))
      (Printf.sprintf "child[%d] = parent[%d]" i parent_row)
      preds_full.{parent_row} preds_child.{i}
  done;
  Xgboost.DMatrix.free child;
  Xgboost.DMatrix.free dtrain;
  Xgboost.Booster.free bst

let test_slice_bad_index_rejected () =
  let m = big2 ~rows:10 ~cols:3 in
  fill2 m;
  let d = Xgboost.DMatrix.of_bigarray2 m in
  let bad = Array1.of_array int32 c_layout [| 0l; 100l |] in
  Alcotest.check_raises "out-of-range index"
    (Xgboost.Xgboost_error
       (Xgboost.Error.Invalid_argument
          "DMatrix.slice: index 100 at position 1 out of range [0,10)"))
    (fun () -> ignore (Xgboost.DMatrix.slice d bad));
  Xgboost.DMatrix.free d

(* ---------- Cv.k_fold ---------- *)

let make_create_booster () =
  fun ~dtrain ->
    let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
    Xgboost.Booster.set_params bst
      [
        "objective", "binary:logistic";
        "tree_method", "hist";
        "max_depth", "4";
        "seed", "0";
        "verbosity", "0";
      ];
    bst

let test_k_fold_dmatrix () =
  let m, labels, dtrain, _bst =
    train_simple_booster ~rows:200 ~cols:8 ~iters:0 0xBEEFl
  in
  let results =
    Xgboost.Cv.k_fold ~k:5
      ~create_booster:(make_create_booster ())
      ~features:dtrain ~labels ~iters_per_fold:5 ()
  in
  Alcotest.(check int) "5 folds" 5 (List.length results);
  let mean, _std = Xgboost.Cv.summarise results ~metric:`Test_auc in
  if mean < 0.7 then
    Alcotest.failf
      "synthetic test AUC %f is below 0.7 (signal expected to be \
       learnable)" mean;
  List.iter
    (fun r -> Xgboost.Booster.free r.Xgboost.Cv.booster)
    results;
  Xgboost.DMatrix.free dtrain;
  ignore m

let test_k_fold_array2 () =
  let m, labels, dtrain, _bst =
    train_simple_booster ~rows:200 ~cols:8 ~iters:0 0xBEEFl
  in
  Xgboost.DMatrix.free dtrain;
  let results =
    Xgboost.Cv.k_fold_array2 ~k:5
      ~create_booster:(make_create_booster ())
      ~features:m ~labels ~iters_per_fold:5 ()
  in
  Alcotest.(check int) "5 folds (array2 path)" 5 (List.length results);
  let mean, _std = Xgboost.Cv.summarise results ~metric:`Test_auc in
  if mean < 0.7 then
    Alcotest.failf
      "synthetic test AUC %f is below 0.7 (array2 path)" mean;
  List.iter
    (fun r -> Xgboost.Booster.free r.Xgboost.Cv.booster)
    results

let test_k_fold_deterministic () =
  let m, labels, dtrain, _bst =
    train_simple_booster ~rows:200 ~cols:8 ~iters:0 0x1234l
  in
  let run () =
    let results =
      Xgboost.Cv.k_fold ~k:5
        ~create_booster:(make_create_booster ())
        ~features:dtrain ~labels ~iters_per_fold:3 ~seed:42 ()
    in
    let xs =
      List.map (fun r -> r.Xgboost.Cv.test_auc) results
    in
    List.iter
      (fun r -> Xgboost.Booster.free r.Xgboost.Cv.booster)
      results;
    xs
  in
  let r1 = run () in
  let r2 = run () in
  List.iter2
    (fun a b ->
      Alcotest.(check (float 1e-6))
        (Printf.sprintf "deterministic test_auc %f vs %f" a b)
        a b)
    r1 r2;
  Xgboost.DMatrix.free dtrain;
  ignore m

let test_fold_indices_partition () =
  (* Without groups: the 5 fold index arrays must be disjoint and
     together cover [0, n). *)
  let n = 100 in
  let folds = Xgboost.Cv.fold_indices ~n ~k:5 ~seed:7 () in
  Alcotest.(check int) "k folds" 5 (Array.length folds);
  let seen = Array.make n false in
  Array.iter
    (fun f ->
      Array.iter
        (fun i ->
          if seen.(i) then
            Alcotest.failf "row %d appears in two folds" i;
          seen.(i) <- true)
        f)
    folds;
  for i = 0 to n - 1 do
    if not seen.(i) then
      Alcotest.failf "row %d missing from all folds" i
  done

let test_fold_indices_group_coherent () =
  (* With group_ids: no group's rows are split across folds. *)
  let n = 120 in
  let groups = big1_int n in
  for i = 0 to n - 1 do
    groups.{i} <- i / 4 (* 30 groups of 4 rows *)
  done;
  let folds =
    Xgboost.Cv.fold_indices ~n ~k:5 ~group_ids:groups ~seed:13 ()
  in
  let group_to_fold = Hashtbl.create 32 in
  Array.iteri
    (fun fold_id rows ->
      Array.iter
        (fun i ->
          let g = groups.{i} in
          match Hashtbl.find_opt group_to_fold g with
          | None -> Hashtbl.add group_to_fold g fold_id
          | Some prev when prev <> fold_id ->
              Alcotest.failf
                "group %d split across folds %d and %d" g prev fold_id
          | Some _ -> ())
        rows)
    folds;
  (* And the partition is total. *)
  let seen = Array.make n false in
  Array.iter
    (fun f -> Array.iter (fun i -> seen.(i) <- true) f)
    folds;
  for i = 0 to n - 1 do
    if not seen.(i) then
      Alcotest.failf "row %d missing from all folds (group split test)" i
  done

let test_fold_indices_deterministic () =
  let r1 = Xgboost.Cv.fold_indices ~n:50 ~k:5 ~seed:99 () in
  let r2 = Xgboost.Cv.fold_indices ~n:50 ~k:5 ~seed:99 () in
  Array.iter2
    (fun a b ->
      Alcotest.(check (array int))
        "same seed reproduces same partition"
        (Array.to_list a |> List.sort compare |> Array.of_list)
        (Array.to_list b |> List.sort compare |> Array.of_list))
    r1 r2

let test_summarise_basic () =
  let bst = Xgboost.Booster.create () in
  let r =
    [
      { Xgboost.Cv.fold = 0; train_auc = 0.9; test_auc = 0.85; booster = bst };
      { fold = 1; train_auc = 0.91; test_auc = 0.86; booster = bst };
      { fold = 2; train_auc = 0.89; test_auc = 0.84; booster = bst };
    ]
  in
  let mean, std = Xgboost.Cv.summarise r ~metric:`Test_auc in
  Alcotest.(check (float 1e-9)) "mean" 0.85 mean;
  (* Population std: sqrt(((0.85-0.85)^2 + (0.86-0.85)^2 + (0.84-0.85)^2)/3)
     = sqrt(0.0002/3) ≈ 0.008164965809 *)
  Alcotest.(check (float 1e-6)) "std" (sqrt (0.0002 /. 3.0)) std;
  Xgboost.Booster.free bst

(* ---------- exposed group-coherence test via make_folds-equivalent ---------- *)
(* We can verify group coherence directly by replicating the helper
   logic: run k_fold with group_ids and then for each fold inspect the
   set of rows it actually saw via the booster's training behavior is
   too indirect. For now the previous test only checks that the call
   doesn't crash. A stronger structural test would require Cv to
   expose the row-index arrays — out of scope for this PR. *)

let () =
  Alcotest.run "eval_cv"
    [
      ( "eval_parse",
        [
          Alcotest.test_case "parse_basic" `Quick test_parse_basic;
          Alcotest.test_case "parse_trailing_newline" `Quick
            test_parse_trailing_newline;
          Alcotest.test_case "get" `Quick test_get;
        ] );
      ( "eval_auc",
        [
          Alcotest.test_case "auc_fixture" `Quick test_auc_fixture;
          Alcotest.test_case "auc_perfect" `Quick test_auc_perfect;
          Alcotest.test_case "auc_inverted" `Quick test_auc_inverted;
          Alcotest.test_case "auc_random" `Quick
            test_auc_random_close_to_half;
          Alcotest.test_case "auc_ties" `Quick test_auc_ties;
          Alcotest.test_case "roc_endpoints" `Quick test_roc_endpoints;
        ] );
      ( "dmatrix_slice",
        [
          Alcotest.test_case "round_trip" `Quick test_slice_round_trip;
          Alcotest.test_case "bad_index" `Quick
            test_slice_bad_index_rejected;
        ] );
      ( "cv",
        [
          Alcotest.test_case "k_fold_dmatrix" `Quick test_k_fold_dmatrix;
          Alcotest.test_case "k_fold_array2" `Quick test_k_fold_array2;
          Alcotest.test_case "k_fold_deterministic" `Quick
            test_k_fold_deterministic;
          Alcotest.test_case "fold_indices_partition" `Quick
            test_fold_indices_partition;
          Alcotest.test_case "fold_indices_group_coherent" `Quick
            test_fold_indices_group_coherent;
          Alcotest.test_case "fold_indices_deterministic" `Quick
            test_fold_indices_deterministic;
          Alcotest.test_case "summarise" `Quick test_summarise_basic;
        ] );
    ]
