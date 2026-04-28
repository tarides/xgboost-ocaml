(* Layer-B benchmark harness — mirrors bin/c_reference/perf.c exactly.
 *
 * Same workloads, same default sizes, same CSV output. The only thing
 * that varies is the language layer doing the FFI calls. Pair this binary
 * with bin/c_reference/perf to compute OCaml/C ratios.
 *
 * Usage:
 *   dune exec bench/bindings/perf.exe -- --workload W1 --repeat 5
 *)

open Ctypes
module F = Xgboost_bindings.C.Functions

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
  let usage =
    "perf --workload {W1|W2|W3|W4|W5|W6} [--rows N] [--cols N] [--iters N] \
     [--density F] [--repeat K] [--warmup K] [--seed N] [--verbose]"
  in
  let speclist =
    [
      ("--workload", Arg.String (fun s -> o.workload <- s), "");
      ("--rows", Arg.Int (fun n -> o.rows <- n), "");
      ("--cols", Arg.Int (fun n -> o.cols <- n), "");
      ("--iters", Arg.Int (fun n -> o.iters <- n), "");
      ("--density", Arg.Float (fun f -> o.density <- f), "");
      ("--repeat", Arg.Int (fun n -> o.repeat <- n), "");
      ("--warmup", Arg.Int (fun n -> o.warmup <- n), "");
      ("--seed", Arg.Int (fun n -> o.seed <- Int32.of_int n), "");
      ("--verbose", Arg.Unit (fun () -> o.verbose <- true), "");
    ]
    |> List.map (fun (k, sp, _) -> (k, sp, ""))
  in
  Arg.parse speclist (fun s -> raise (Arg.Bad ("unexpected positional " ^ s))) usage;
  if o.workload = "" then (
    prerr_endline "--workload is required";
    prerr_endline usage;
    exit 2);
  o

let apply_defaults o =
  let set_if_zero r v = if !r = 0 then r := v in
  let set_if_zero_int r v = if r = 0 then v else r in
  match o.workload with
  | "W1" ->
      o.rows <- set_if_zero_int o.rows 1000;
      o.cols <- set_if_zero_int o.cols 50;
      o.iters <- set_if_zero_int o.iters 100;
      ignore set_if_zero
  | "W2" | "W3" | "W4" ->
      o.rows <- set_if_zero_int o.rows 1_000_000;
      o.cols <- set_if_zero_int o.cols 100;
      o.iters <- set_if_zero_int o.iters 100
  | "W5" | "W6" ->
      o.rows <- set_if_zero_int o.rows 100_000;
      o.cols <- set_if_zero_int o.cols 100
  | "G3" -> o.iters <- set_if_zero_int o.iters 10_000_000
  | _ -> ()

(* ---------- timing ---------- *)

let now_ns () = Mtime_clock.now () |> Mtime.to_uint64_ns

let elapsed_ns since =
  let now = now_ns () in
  Int64.sub now since

let to_ms_f ns = Int64.to_float ns /. 1e6

(* ---------- helpers ---------- *)

let ulong = Unsigned.UInt64.of_int
let uint_v = Unsigned.UInt.of_int

let xgb_ok rc =
  if rc <> 0 then
    Printf.ksprintf failwith "xgboost call failed: %s"
      (F.xgb_get_last_error ())

(* Deterministic xorshift32 PRNG mirroring bench_common.h. *)
let prng_state = ref 0xC0FFEEl

let next_uniform () =
  let x = !prng_state in
  let x = Int32.logxor x (Int32.shift_left x 13) in
  let x = Int32.logxor x (Int32.shift_right_logical x 17) in
  let x = Int32.logxor x (Int32.shift_left x 5) in
  let x = if Int32.equal x 0l then 1l else x in
  prng_state := x;
  Int32.to_float (Int32.shift_right_logical x 8) /. 16777216.0

let big2 ~rows ~cols =
  Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout rows cols

let big1 n = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n

let fill2 (m : (float, _, _) Bigarray.Array2.t) =
  for r = 0 to Bigarray.Array2.dim1 m - 1 do
    for c = 0 to Bigarray.Array2.dim2 m - 1 do
      m.{r, c} <- next_uniform ()
    done
  done

let gen_labels_binary (m : (float, _, _) Bigarray.Array2.t) =
  let rows = Bigarray.Array2.dim1 m in
  let cols = Bigarray.Array2.dim2 m in
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

let gen_labels_reg (m : (float, _, _) Bigarray.Array2.t) =
  let rows = Bigarray.Array2.dim1 m in
  let cols = Bigarray.Array2.dim2 m in
  let l = big1 rows in
  for r = 0 to rows - 1 do
    l.{r} <-
      0.5 *. m.{r, 0}
      -. 0.3 *. m.{r, min 1 (cols - 1)}
      +. 0.2 *. m.{r, min 2 (cols - 1)}
      +. 0.1 *. m.{r, min 3 (cols - 1)}
  done;
  l

let dmatrix_create_dense m =
  let rows = Bigarray.Array2.dim1 m in
  let cols = Bigarray.Array2.dim2 m in
  let dmat_out = allocate F.dmatrix_handle null in
  xgb_ok
    (F.xgdmatrix_create_from_mat
       (bigarray_start array2 m)
       (ulong rows) (ulong cols) Float.nan dmat_out);
  !@dmat_out

let booster_create_with_dmat dmat =
  let booster_out = allocate F.booster_handle null in
  let arr = CArray.make F.dmatrix_handle 1 in
  CArray.set arr 0 dmat;
  xgb_ok
    (F.xgbooster_create (CArray.start arr) (ulong 1) booster_out);
  !@booster_out

let set_params bst ps =
  List.iter (fun (k, v) -> xgb_ok (F.xgbooster_set_param bst k v)) ps

(* Trains a binary:logistic hist booster on a fresh dense dataset. Returns
 * (booster, dtrain) — caller frees both. *)
let train_model rows cols iters seed =
  prng_state := seed;
  let m = big2 ~rows ~cols in
  fill2 m;
  let labels = gen_labels_binary m in
  let dtrain = dmatrix_create_dense m in
  xgb_ok
    (F.xgdmatrix_set_float_info dtrain "label"
       (bigarray_start array1 labels) (ulong rows));
  let bst = booster_create_with_dmat dtrain in
  set_params bst
    [
      "objective", "binary:logistic";
      "tree_method", "hist";
      "max_depth", "6";
      "verbosity", "0";
    ];
  for it = 0 to iters - 1 do
    xgb_ok (F.xgbooster_update_one_iter bst it dtrain)
  done;
  (bst, dtrain, m, labels)

(* ---------- workloads ---------- *)

let run_W1 o =
  prng_state := o.seed;
  let m = big2 ~rows:o.rows ~cols:o.cols in
  fill2 m;
  let labels = gen_labels_reg m in
  let dtrain = dmatrix_create_dense m in
  xgb_ok
    (F.xgdmatrix_set_float_info dtrain "label"
       (bigarray_start array1 labels) (ulong o.rows));
  let bst = booster_create_with_dmat dtrain in
  set_params bst [ "objective", "reg:squarederror"; "verbosity", "0" ];

  let t0 = now_ns () in
  for it = 0 to o.iters - 1 do
    xgb_ok (F.xgbooster_update_one_iter bst it dtrain)
  done;
  let elapsed = elapsed_ns t0 in

  xgb_ok (F.xgbooster_free bst);
  xgb_ok (F.xgdmatrix_free dtrain);
  (* keep [m] and [labels] alive across the timed region *)
  ignore (Sys.opaque_identity (m, labels));
  elapsed

let run_W2 o =
  prng_state := o.seed;
  let m = big2 ~rows:o.rows ~cols:o.cols in
  fill2 m;
  let labels = gen_labels_binary m in
  let dtrain = dmatrix_create_dense m in
  xgb_ok
    (F.xgdmatrix_set_float_info dtrain "label"
       (bigarray_start array1 labels) (ulong o.rows));
  let bst = booster_create_with_dmat dtrain in
  set_params bst
    [
      "objective", "binary:logistic";
      "tree_method", "hist";
      "max_depth", "6";
      "verbosity", "0";
    ];

  let t0 = now_ns () in
  for it = 0 to o.iters - 1 do
    xgb_ok (F.xgbooster_update_one_iter bst it dtrain)
  done;
  let elapsed = elapsed_ns t0 in

  xgb_ok (F.xgbooster_free bst);
  xgb_ok (F.xgdmatrix_free dtrain);
  ignore (Sys.opaque_identity (m, labels));
  elapsed

let run_W3 o =
  let bst, dtrain, m, labels = train_model o.rows o.cols o.iters o.seed in

  prng_state := Int32.logxor o.seed 0xA5A5A5A5l;
  let pm = big2 ~rows:o.rows ~cols:o.cols in
  fill2 pm;
  let dpred = dmatrix_create_dense pm in

  let out_len = allocate uint64_t Unsigned.UInt64.zero in
  let out_result = allocate (ptr float) (from_voidp float null) in
  let t0 = now_ns () in
  xgb_ok
    (F.xgbooster_predict bst dpred 0 (uint_v 0) 0 out_len out_result);
  (* touch the borrowed output to defeat any lazy materialisation *)
  let n = Unsigned.UInt64.to_int !@out_len in
  let outp = !@out_result in
  let sink = ref 0.0 in
  for i = 0 to n - 1 do
    sink := !sink +. !@(outp +@ i)
  done;
  let elapsed = elapsed_ns t0 in
  ignore (Sys.opaque_identity !sink);

  xgb_ok (F.xgdmatrix_free dpred);
  xgb_ok (F.xgdmatrix_free dtrain);
  xgb_ok (F.xgbooster_free bst);
  ignore (Sys.opaque_identity (m, labels, pm));
  elapsed

let run_W4 o =
  let bst, dtrain, m, labels = train_model o.rows o.cols o.iters o.seed in

  let n_pred = min o.rows 100_000 in
  prng_state := Int32.logxor o.seed 0xA5A5A5A5l;
  let prow = big1 o.cols in

  (* warm: run one predict outside the timed loop *)
  for c = 0 to o.cols - 1 do prow.{c} <- next_uniform () done;
  let _ =
    let dh = allocate F.dmatrix_handle null in
    xgb_ok
      (F.xgdmatrix_create_from_mat
         (bigarray_start array1 prow)
         (ulong 1) (ulong o.cols) Float.nan dh);
    let d = !@dh in
    let ol = allocate uint64_t Unsigned.UInt64.zero in
    let op = allocate (ptr float) (from_voidp float null) in
    xgb_ok (F.xgbooster_predict bst d 0 (uint_v 0) 0 ol op);
    xgb_ok (F.xgdmatrix_free d)
  in

  let sink = ref 0.0 in
  let t0 = now_ns () in
  for _ = 0 to n_pred - 1 do
    for c = 0 to o.cols - 1 do prow.{c} <- next_uniform () done;
    let dh = allocate F.dmatrix_handle null in
    xgb_ok
      (F.xgdmatrix_create_from_mat
         (bigarray_start array1 prow)
         (ulong 1) (ulong o.cols) Float.nan dh);
    let d = !@dh in
    let ol = allocate uint64_t Unsigned.UInt64.zero in
    let op = allocate (ptr float) (from_voidp float null) in
    xgb_ok (F.xgbooster_predict bst d 0 (uint_v 0) 0 ol op);
    let outp = !@op in
    sink := !sink +. !@outp;
    xgb_ok (F.xgdmatrix_free d)
  done;
  let elapsed = elapsed_ns t0 in
  ignore (Sys.opaque_identity !sink);

  xgb_ok (F.xgdmatrix_free dtrain);
  xgb_ok (F.xgbooster_free bst);
  ignore (Sys.opaque_identity (m, labels, prow));
  elapsed

(* Build a JSON __array_interface__ string for a Bigarray buffer.
   Mirrors src/xgboost/array_interface.ml; duplicated here so this
   bench harness doesn't take a dependency on the high-level library. *)
let array_interface_2d_f32 m =
  let rows = Bigarray.Array2.dim1 m in
  let cols = Bigarray.Array2.dim2 m in
  let addr =
    Nativeint.to_string
      (raw_address_of_ptr (to_voidp (bigarray_start array2 m)))
  in
  Printf.sprintf
    {|{"data":[%s,false],"shape":[%d,%d],"strides":null,"typestr":"<f4","version":3}|}
    addr rows cols

let array_interface_1d typestr ptr_addr len =
  Printf.sprintf
    {|{"data":[%s,false],"shape":[%d],"strides":null,"typestr":"%s","version":3}|}
    ptr_addr len typestr

let run_W5 o =
  prng_state := o.seed;
  let m = big2 ~rows:o.rows ~cols:o.cols in
  fill2 m;
  let json = array_interface_2d_f32 m in
  let config = {|{"missing":NaN}|} in
  let dmat_out = allocate F.dmatrix_handle null in
  let t0 = now_ns () in
  xgb_ok (F.xgdmatrix_create_from_dense json config dmat_out);
  let elapsed = elapsed_ns t0 in
  xgb_ok (F.xgdmatrix_free !@dmat_out);
  ignore (Sys.opaque_identity m);
  elapsed

let run_W6 o =
  prng_state := o.seed;
  (* Generate CSR triplet into Bigarrays so we can pass their address
     directly via __array_interface__ — no per-element copy. *)
  let cap = o.rows * o.cols in
  let indptr =
    Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout (o.rows + 1)
  in
  let indices_buf =
    Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout cap
  in
  let values_buf = big1 cap in
  indptr.{0} <- 0L;
  let nnz = ref 0 in
  for r = 0 to o.rows - 1 do
    for c = 0 to o.cols - 1 do
      if next_uniform () < o.density then begin
        indices_buf.{!nnz} <- Int32.of_int c;
        values_buf.{!nnz} <- next_uniform ();
        incr nnz
      end
    done;
    indptr.{r + 1} <- Int64.of_int !nnz
  done;
  let nnz = !nnz in
  let indices = Bigarray.Array1.sub indices_buf 0 nnz in
  let values = Bigarray.Array1.sub values_buf 0 nnz in

  let addr_of_ba1 ba =
    Nativeint.to_string
      (raw_address_of_ptr (to_voidp (bigarray_start array1 ba)))
  in
  let json_indptr =
    array_interface_1d "<i8" (addr_of_ba1 indptr) (o.rows + 1)
  in
  let json_indices =
    array_interface_1d "<i4" (addr_of_ba1 indices) nnz
  in
  let json_values =
    array_interface_1d "<f4" (addr_of_ba1 values) nnz
  in
  let config = {|{"missing":NaN}|} in
  let dmat_out = allocate F.dmatrix_handle null in
  let t0 = now_ns () in
  xgb_ok
    (F.xgdmatrix_create_from_csr
       json_indptr json_indices json_values (ulong o.cols) config dmat_out);
  let elapsed = elapsed_ns t0 in
  xgb_ok (F.xgdmatrix_free !@dmat_out);
  ignore (Sys.opaque_identity (indptr, indices_buf, values_buf));
  elapsed

(* ---------- G3: FFI roundtrip microbench ----------
 * Calls XGBoostVersion --iters times in a tight loop. Pair with the C
 * reference's G3 to read off per-call FFI overhead from layer B. *)
let run_G3 o =
  let n = if o.iters > 0 then o.iters else 10_000_000 in
  let major = allocate int 0 in
  let minor = allocate int 0 in
  let patch = allocate int 0 in
  let sink = ref 0 in
  let t0 = now_ns () in
  for _ = 1 to n do
    F.xgboost_version major minor patch;
    sink := !@major
  done;
  let elapsed = elapsed_ns t0 in
  ignore (Sys.opaque_identity !sink);
  elapsed

let workloads =
  [
    "W1", run_W1;
    "W2", run_W2;
    "W3", run_W3;
    "W4", run_W4;
    "W5", run_W5;
    "W6", run_W6;
    "G3", run_G3;
  ]

(* ---------- stats ---------- *)

let stats samples =
  let n = Array.length samples in
  if n = 0 then (0L, 0L, 0L)
  else
    let mn = ref samples.(0) in
    let mx = ref samples.(0) in
    let sum = ref 0L in
    Array.iter
      (fun v ->
        if Int64.compare v !mn < 0 then mn := v;
        if Int64.compare v !mx > 0 then mx := v;
        sum := Int64.add !sum v)
      samples;
    (!mn, Int64.div !sum (Int64.of_int n), !mx)

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

  (* warmup: force libxgboost lazy init, plus discarded runs *)
  (let major = allocate int 0 in
   let minor = allocate int 0 in
   let patch = allocate int 0 in
   F.xgboost_version major minor patch;
   if o.verbose then
     Printf.eprintf "# libxgboost %d.%d.%d\n" !@major !@minor !@patch);

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
  let mn, mean, mx = stats samples in

  let no_hdr =
    match Sys.getenv_opt "NO_HEADER" with Some "1" -> true | _ -> false
  in
  if not no_hdr then
    print_endline "workload,rows,cols,iters,density,repeat,min_ms,mean_ms,max_ms";
  Printf.printf "%s,%d,%d,%d,%.4f,%d,%.3f,%.3f,%.3f\n"
    o.workload o.rows o.cols o.iters o.density o.repeat
    (to_ms_f mn) (to_ms_f mean) (to_ms_f mx)
