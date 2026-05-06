(* Cv — k-fold cross validation built on top of Dmatrix.slice and
 * Eval.auc. Supports group-coherent splitting via an optional
 * [group_ids] array (rows sharing a group are kept in the same
 * fold). *)

open Bigarray

type fold_result = {
  fold : int;
  train_auc : float;
  test_auc : float;
  booster : Booster.t;
}

let raise_invalid msg =
  raise (Error.Xgboost_error (Error.Invalid_argument msg))

(* Fisher-Yates shuffle on an int array, in-place, with a per-call PRNG. *)
let shuffle_in_place ~seed arr =
  let st = Random.State.make [| seed |] in
  let n = Array.length arr in
  for i = n - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let tmp = arr.(i) in
    arr.(i) <- arr.(j);
    arr.(j) <- tmp
  done

(* Shape: result.(f) is the int array of test-row indices in fold f. *)
let make_folds ~n ~k ~group_ids ~seed =
  if k < 2 then raise_invalid "Cv.k_fold: k must be >= 2";
  if n < k then
    raise_invalid
      (Printf.sprintf "Cv.k_fold: not enough rows (%d) for %d folds" n k);
  match group_ids with
  | None ->
      let perm = Array.init n (fun i -> i) in
      shuffle_in_place ~seed perm;
      let folds = Array.make k [||] in
      let base = n / k and extra = n mod k in
      let cursor = ref 0 in
      for f = 0 to k - 1 do
        let len = base + (if f < extra then 1 else 0) in
        folds.(f) <- Array.sub perm !cursor len;
        cursor := !cursor + len
      done;
      folds
  | Some gids ->
      if Array1.dim gids <> n then
        raise
          (Error.Xgboost_error
             (Error.Shape_mismatch
                { expected = (n, 0); got = (Array1.dim gids, 0) }));
      (* Bucket rows by group id, then shuffle the *list of groups*
         and assign each group greedily to the smallest current fold
         (by row count). Guarantees no group spans folds. *)
      let buckets : (int, int list) Hashtbl.t = Hashtbl.create 64 in
      for i = 0 to n - 1 do
        let g = gids.{i} in
        let prev = try Hashtbl.find buckets g with Not_found -> [] in
        Hashtbl.replace buckets g (i :: prev)
      done;
      let groups =
        Hashtbl.fold (fun _ rows acc -> Array.of_list rows :: acc) buckets []
        |> Array.of_list
      in
      let idx = Array.init (Array.length groups) (fun i -> i) in
      shuffle_in_place ~seed idx;
      let groups = Array.map (fun i -> groups.(i)) idx in
      let fold_rows = Array.make k [] in
      let fold_size = Array.make k 0 in
      Array.iter
        (fun rows ->
          let smallest = ref 0 in
          for f = 1 to k - 1 do
            if fold_size.(f) < fold_size.(!smallest) then smallest := f
          done;
          let f = !smallest in
          fold_rows.(f) <- rows :: fold_rows.(f);
          fold_size.(f) <- fold_size.(f) + Array.length rows)
        groups;
      Array.init k (fun f ->
          List.fold_left
            (fun acc a -> Array.append a acc)
            [||] fold_rows.(f))

(* Convert an int array of row indices to a fresh int32 Bigarray for
   passing into Dmatrix.slice. *)
let to_int32_array1 arr =
  let n = Array.length arr in
  let out = Array1.create int32 c_layout n in
  for i = 0 to n - 1 do
    out.{i} <- Int32.of_int arr.(i)
  done;
  out

(* Slice a labels Array1 by a row-index int array. *)
let slice_labels labels rows =
  let n = Array.length rows in
  let out = Array1.create float32 c_layout n in
  for i = 0 to n - 1 do
    out.{i} <- labels.{rows.(i)}
  done;
  out

(* Build the train index set as the concatenation of all folds except
   [test_fold]. *)
let train_indices folds ~test_fold =
  let parts =
    Array.to_list folds
    |> List.mapi (fun f rows -> (f, rows))
    |> List.filter_map (fun (f, rows) ->
           if f = test_fold then None else Some rows)
  in
  Array.concat parts

let run_fold ~create_booster ~slice_to_dmatrix ~labels ~train_rows
    ~test_rows ~iters_per_fold ~fold =
  let dtrain = slice_to_dmatrix train_rows in
  let dtest = slice_to_dmatrix test_rows in
  Fun.protect
    ~finally:(fun () ->
      Dmatrix.free dtest;
      Dmatrix.free dtrain)
    (fun () ->
      let train_labels = slice_labels labels train_rows in
      let test_labels = slice_labels labels test_rows in
      Dmatrix.set_label dtrain train_labels;
      Dmatrix.set_label dtest test_labels;
      let booster = create_booster ~dtrain in
      for it = 0 to iters_per_fold - 1 do
        Booster.update_one_iter booster ~iter:it ~dtrain
      done;
      let train_pred = Booster.predict booster dtrain in
      let test_pred = Booster.predict booster dtest in
      let train_auc =
        Eval.auc ~predictions:train_pred ~labels:train_labels
      in
      let test_auc =
        Eval.auc ~predictions:test_pred ~labels:test_labels
      in
      { fold; train_auc; test_auc; booster })

let default_seed = 0xC0FFEE

(* Public version: returns the int array array of test-row indices per
   fold. Useful for inspection or for building custom CV loops on top
   of the same split. *)
let fold_indices ~n ~k ?group_ids ?(seed = default_seed) () =
  make_folds ~n ~k ~group_ids ~seed

let k_fold ~k ~create_booster ~features ~labels ?group_ids
    ?(seed = default_seed) ~iters_per_fold () =
  let n = Dmatrix.rows features in
  if Array1.dim labels <> n then
    raise
      (Error.Xgboost_error
         (Error.Shape_mismatch
            { expected = (n, 0); got = (Array1.dim labels, 0) }));
  let folds = make_folds ~n ~k ~group_ids ~seed in
  let slice_to_dmatrix rows =
    Dmatrix.slice features (to_int32_array1 rows)
  in
  List.init (Array.length folds) (fun f ->
      let test_rows = folds.(f) in
      let train_rows = train_indices folds ~test_fold:f in
      run_fold ~create_booster ~slice_to_dmatrix ~labels ~train_rows
        ~test_rows ~iters_per_fold ~fold:f)

(* Build a fresh dense Array2 from a parent Array2 and a row-index
   array. The new buffer is allocated row-by-row; downstream
   Dmatrix.of_bigarray2 may copy or pin it depending on libxgboost's
   internal path. *)
let select_rows_array2 src rows =
  let n = Array.length rows in
  let cols = Array2.dim2 src in
  let out = Array2.create float32 c_layout n cols in
  for i = 0 to n - 1 do
    let r = rows.(i) in
    for c = 0 to cols - 1 do
      out.{i, c} <- src.{r, c}
    done
  done;
  out

let k_fold_array2 ~k ~create_booster ~features ~labels ?group_ids
    ?(seed = default_seed) ~iters_per_fold () =
  let n = Array2.dim1 features in
  if Array1.dim labels <> n then
    raise
      (Error.Xgboost_error
         (Error.Shape_mismatch
            { expected = (n, 0); got = (Array1.dim labels, 0) }));
  let folds = make_folds ~n ~k ~group_ids ~seed in
  let slice_to_dmatrix rows =
    let m = select_rows_array2 features rows in
    Dmatrix.of_bigarray2 m
  in
  List.init (Array.length folds) (fun f ->
      let test_rows = folds.(f) in
      let train_rows = train_indices folds ~test_fold:f in
      run_fold ~create_booster ~slice_to_dmatrix ~labels ~train_rows
        ~test_rows ~iters_per_fold ~fold:f)

let summarise results ~metric =
  let xs =
    List.map
      (fun r ->
        match metric with
        | `Train_auc -> r.train_auc
        | `Test_auc -> r.test_auc)
      results
  in
  let n = List.length xs in
  if n = 0 then raise_invalid "Cv.summarise: empty results";
  let mean = List.fold_left ( +. ) 0.0 xs /. float_of_int n in
  let variance =
    List.fold_left
      (fun acc x -> acc +. ((x -. mean) ** 2.0))
      0.0 xs
    /. float_of_int n
  in
  (mean, sqrt variance)
