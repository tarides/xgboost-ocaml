(* Layer-C benchmark harness — mirrors bin/c_reference/perf.c and
 * bench/bindings/perf.ml, but using the high-level Xgboost API.
 *
 * Pair with the lower layers to compute layer-C overhead.
 *)

open Bigarray

(* ---------- options ---------- *)

type opts = {
  mutable workload : string;
  mutable rows : int;
  mutable cols : int;
  mutable iters : int;
  mutable density : float;
  mutable repeat : int;
  mutable warmup : int;
  mutable seed : int32;
  mutable verbose : bool;
}

let default_opts () =
  {
    workload = "";
    rows = 0;
    cols = 0;
    iters = 0;
    density = 0.05;
    repeat = 5;
    warmup = 1;
    seed = 0xC0FFEEl;
    verbose = false;
  }

let parse_args () =
  let o = default_opts () in
  let speclist =
    [
      "--workload", Arg.String (fun s -> o.workload <- s), "";
      "--rows", Arg.Int (fun n -> o.rows <- n), "";
      "--cols", Arg.Int (fun n -> o.cols <- n), "";
      "--iters", Arg.Int (fun n -> o.iters <- n), "";
      "--density", Arg.Float (fun f -> o.density <- f), "";
      "--repeat", Arg.Int (fun n -> o.repeat <- n), "";
      "--warmup", Arg.Int (fun n -> o.warmup <- n), "";
      "--seed", Arg.Int (fun n -> o.seed <- Int32.of_int n), "";
      "--verbose", Arg.Unit (fun () -> o.verbose <- true), "";
    ]
  in
  Arg.parse speclist
    (fun s -> raise (Arg.Bad ("unexpected positional " ^ s)))
    "perf --workload {W1|W2|W3|W4|W5|W6} ...";
  if o.workload = "" then (
    prerr_endline "--workload required";
    exit 2);
  o

let apply_defaults o =
  let setz r v = if r = 0 then v else r in
  match o.workload with
  | "W1" ->
      o.rows <- setz o.rows 1000;
      o.cols <- setz o.cols 50;
      o.iters <- setz o.iters 100
  | "W2" | "W3" | "W4" ->
      o.rows <- setz o.rows 1_000_000;
      o.cols <- setz o.cols 100;
      o.iters <- setz o.iters 100
  | "W5" | "W6" ->
      o.rows <- setz o.rows 100_000;
      o.cols <- setz o.cols 100
  | _ -> ()

(* ---------- timing ---------- *)

let now_ns () = Mtime_clock.now () |> Mtime.to_uint64_ns
let elapsed_ns since = Int64.sub (now_ns ()) since
let to_ms_f ns = Int64.to_float ns /. 1e6

(* ---------- helpers ---------- *)

let prng_state = ref 0xC0FFEEl

let next_uniform () =
  let x = !prng_state in
  let x = Int32.logxor x (Int32.shift_left x 13) in
  let x = Int32.logxor x (Int32.shift_right_logical x 17) in
  let x = Int32.logxor x (Int32.shift_left x 5) in
  let x = if Int32.equal x 0l then 1l else x in
  prng_state := x;
  Int32.to_float (Int32.shift_right_logical x 8) /. 16777216.0

let big2 ~rows ~cols = Array2.create float32 c_layout rows cols
let big1 n = Array1.create float32 c_layout n

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
      +. 0.1 *. m.{r, min 3 (cols - 1)}
    in
    l.{r} <- (if v > 0.25 then 1.0 else 0.0)
  done;
  l

let labels_reg m =
  let rows = Array2.dim1 m in
  let cols = Array2.dim2 m in
  let l = big1 rows in
  for r = 0 to rows - 1 do
    l.{r} <-
      0.5 *. m.{r, 0}
      -. 0.3 *. m.{r, min 1 (cols - 1)}
      +. 0.2 *. m.{r, min 2 (cols - 1)}
      +. 0.1 *. m.{r, min 3 (cols - 1)}
  done;
  l

let train_model_binary ~rows ~cols ~iters seed =
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
      "max_depth", "6";
      "verbosity", "0";
    ];
  for it = 0 to iters - 1 do
    Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
  done;
  (m, labels, dtrain, bst)

(* ---------- workloads ---------- *)

let run_W1 o =
  prng_state := o.seed;
  let m = big2 ~rows:o.rows ~cols:o.cols in
  fill2 m;
  let labels = labels_reg m in
  let dtrain = Xgboost.DMatrix.of_bigarray2 m in
  Xgboost.DMatrix.set_label dtrain labels;
  let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
  Xgboost.Booster.set_params bst
    [ "objective", "reg:squarederror"; "verbosity", "0" ];
  let t0 = now_ns () in
  for it = 0 to o.iters - 1 do
    Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
  done;
  let elapsed = elapsed_ns t0 in
  Xgboost.Booster.free bst;
  Xgboost.DMatrix.free dtrain;
  ignore (Sys.opaque_identity (m, labels));
  elapsed

let run_W2 o =
  prng_state := o.seed;
  let m = big2 ~rows:o.rows ~cols:o.cols in
  fill2 m;
  let labels = labels_binary m in
  let dtrain = Xgboost.DMatrix.of_bigarray2 m in
  Xgboost.DMatrix.set_label dtrain labels;
  let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
  Xgboost.Booster.set_params bst
    [
      "objective", "binary:logistic";
      "tree_method", "hist";
      "max_depth", "6";
      "verbosity", "0";
    ];
  let t0 = now_ns () in
  for it = 0 to o.iters - 1 do
    Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
  done;
  let elapsed = elapsed_ns t0 in
  Xgboost.Booster.free bst;
  Xgboost.DMatrix.free dtrain;
  ignore (Sys.opaque_identity (m, labels));
  elapsed

let run_W3 o =
  let m, labels, dtrain, bst =
    train_model_binary ~rows:o.rows ~cols:o.cols ~iters:o.iters o.seed
  in
  prng_state := Int32.logxor o.seed 0xA5A5A5A5l;
  let pm = big2 ~rows:o.rows ~cols:o.cols in
  fill2 pm;
  let dpred = Xgboost.DMatrix.of_bigarray2 pm in

  let t0 = now_ns () in
  let preds = Xgboost.Booster.predict bst dpred in
  let n = Array1.dim preds in
  let sink = ref 0.0 in
  for i = 0 to n - 1 do
    sink := !sink +. preds.{i}
  done;
  let elapsed = elapsed_ns t0 in
  ignore (Sys.opaque_identity !sink);

  Xgboost.DMatrix.free dpred;
  Xgboost.DMatrix.free dtrain;
  Xgboost.Booster.free bst;
  ignore (Sys.opaque_identity (m, labels, pm));
  elapsed

let run_W4 o =
  let m, labels, dtrain, bst =
    train_model_binary ~rows:o.rows ~cols:o.cols ~iters:o.iters o.seed
  in
  let n_pred = min o.rows 100_000 in
  prng_state := Int32.logxor o.seed 0xA5A5A5A5l;
  let prow_buf = big2 ~rows:1 ~cols:o.cols in

  (* warmup predict outside the timed region *)
  for c = 0 to o.cols - 1 do prow_buf.{0, c} <- next_uniform () done;
  (let d1 = Xgboost.DMatrix.of_bigarray2 prow_buf in
   ignore (Xgboost.Booster.predict bst d1);
   Xgboost.DMatrix.free d1);

  let sink = ref 0.0 in
  let t0 = now_ns () in
  for _ = 0 to n_pred - 1 do
    for c = 0 to o.cols - 1 do prow_buf.{0, c} <- next_uniform () done;
    let d1 = Xgboost.DMatrix.of_bigarray2 prow_buf in
    let p = Xgboost.Booster.predict bst d1 in
    sink := !sink +. p.{0};
    Xgboost.DMatrix.free d1
  done;
  let elapsed = elapsed_ns t0 in
  ignore (Sys.opaque_identity !sink);

  Xgboost.DMatrix.free dtrain;
  Xgboost.Booster.free bst;
  ignore (Sys.opaque_identity (m, labels, prow_buf));
  elapsed

let run_W5 o =
  prng_state := o.seed;
  let m = big2 ~rows:o.rows ~cols:o.cols in
  fill2 m;
  let t0 = now_ns () in
  let d = Xgboost.DMatrix.of_bigarray2 m in
  let elapsed = elapsed_ns t0 in
  Xgboost.DMatrix.free d;
  ignore (Sys.opaque_identity m);
  elapsed

let run_W6 o =
  prng_state := o.seed;
  let cap = o.rows * o.cols in
  let indptr = Array1.create int32 c_layout (o.rows + 1) in
  let indices_buf = Array1.create int32 c_layout cap in
  let data_buf = big1 cap in
  indptr.{0} <- 0l;
  let nnz = ref 0 in
  for r = 0 to o.rows - 1 do
    for c = 0 to o.cols - 1 do
      if next_uniform () < o.density then begin
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

  let t0 = now_ns () in
  let d =
    Xgboost.DMatrix.of_csr ~indptr ~indices ~data ~n_cols:o.cols
  in
  let elapsed = elapsed_ns t0 in
  Xgboost.DMatrix.free d;
  ignore (Sys.opaque_identity (indptr, indices_buf, data_buf));
  elapsed

let workloads =
  [
    "W1", run_W1;
    "W2", run_W2;
    "W3", run_W3;
    "W4", run_W4;
    "W5", run_W5;
    "W6", run_W6;
  ]

(* ---------- main ---------- *)

let () =
  let o = parse_args () in
  apply_defaults o;
  let fn =
    try List.assoc o.workload workloads
    with Not_found ->
      Printf.eprintf "unknown workload: %s\n" o.workload;
      exit 2
  in
  if o.verbose then begin
    let mj, mn, pt = Xgboost.version () in
    Printf.eprintf "# libxgboost %d.%d.%d\n" mj mn pt
  end;
  for i = 0 to o.warmup - 1 do
    let ns = fn o in
    if o.verbose then
      Printf.eprintf "# warmup %d: %.3f ms (discarded)\n" i (to_ms_f ns)
  done;
  let samples = Array.make o.repeat 0L in
  for i = 0 to o.repeat - 1 do
    let ns = fn o in
    samples.(i) <- ns;
    if o.verbose then
      Printf.eprintf "# run %d: %.3f ms\n" i (to_ms_f ns)
  done;
  let mn = Array.fold_left (fun a b -> if Int64.compare b a < 0 then b else a) samples.(0) samples in
  let mx = Array.fold_left (fun a b -> if Int64.compare b a > 0 then b else a) samples.(0) samples in
  let sum = Array.fold_left Int64.add 0L samples in
  let mean = Int64.div sum (Int64.of_int (Array.length samples)) in
  if Sys.getenv_opt "NO_HEADER" <> Some "1" then
    print_endline "workload,rows,cols,iters,density,repeat,min_ms,mean_ms,max_ms";
  Printf.printf "%s,%d,%d,%d,%.4f,%d,%.3f,%.3f,%.3f\n"
    o.workload o.rows o.cols o.iters o.density o.repeat
    (to_ms_f mn) (to_ms_f mean) (to_ms_f mx)
