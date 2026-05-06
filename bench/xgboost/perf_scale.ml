(* Scale-validation harness — closes issue #1.
 *
 * Validates that xgboost-ocaml survives at the data sizes downstream
 * blockchain pipelines need (50M+ rows × ~36 cols). Three scales:
 *
 *   1M   × 36, ~50% sparsity   — DMatrix.of_csr (in-memory)
 *   10M  × 36, ~50% sparsity   — DMatrix.of_csr (in-memory)
 *   50M  × 36, ~50% sparsity   — DMatrix.of_iterator + cache_prefix
 *                                (libxgboost external-memory mode)
 *
 * For each scale, reports:
 *   - DMatrix construction wall-time and peak RSS
 *   - One round of update_one_iter (tree_method=hist, max_depth=6,
 *     binary:logistic)
 *   - predict_dense on a 100k held-out matrix
 *   - At 50M only: cache-prefix directory size on disk
 *
 * This is NOT a perf gate. It's a feasibility probe. A "scaling
 * breaks at N" outcome is a valid result and must be documented in
 * BENCH.md alongside successful runs.
 *)

open Bigarray

(* ---------- options ---------- *)

type scale = One_m | Ten_m | Fifty_m

let scale_to_string = function
  | One_m -> "1M"
  | Ten_m -> "10M"
  | Fifty_m -> "50M"

let scale_of_string = function
  | "1M" -> One_m
  | "10M" -> Ten_m
  | "50M" -> Fifty_m
  | s ->
      Printf.eprintf "unknown scale %s (expected 1M|10M|50M)\n" s;
      exit 2

let rows_of = function
  | One_m -> 1_000_000
  | Ten_m -> 10_000_000
  | Fifty_m -> 50_000_000

let cols_default = 36
let density_default = 0.5
let iters_default = 1
let holdout_rows = 100_000

type opts = {
  mutable scales : scale list;
  mutable cols : int;
  mutable density : float;
  mutable cache_prefix : string;
  mutable batch_rows : int;
  mutable seed : int32;
  mutable verbose : bool;
}

let default_opts () =
  {
    scales = [];
    cols = cols_default;
    density = density_default;
    cache_prefix = "";
    batch_rows = 1_000_000;
    seed = 0xC0FFEEl;
    verbose = false;
  }

let parse_args () =
  let o = default_opts () in
  let single_scale = ref None in
  let speclist =
    [
      "--scale", Arg.String (fun s -> single_scale := Some (scale_of_string s)),
        " run a single scale (1M|10M|50M)";
      "--all", Arg.Unit (fun () -> o.scales <- [ One_m; Ten_m; Fifty_m ]),
        " run all three scales";
      "--cols", Arg.Int (fun n -> o.cols <- n), " columns (default 36)";
      "--density", Arg.Float (fun f -> o.density <- f),
        " sparsity (default 0.5)";
      "--cache-prefix", Arg.String (fun s -> o.cache_prefix <- s),
        " cache prefix dir for 50M external memory";
      "--batch-rows", Arg.Int (fun n -> o.batch_rows <- n),
        " streaming batch size (default 1M)";
      "--seed", Arg.Int (fun n -> o.seed <- Int32.of_int n), " PRNG seed";
      "--verbose", Arg.Unit (fun () -> o.verbose <- true), " stderr progress";
    ]
  in
  Arg.parse speclist
    (fun s -> raise (Arg.Bad ("unexpected positional " ^ s)))
    "perf_scale [--scale 1M|10M|50M | --all] [--cache-prefix DIR] ...";
  (match !single_scale, o.scales with
   | Some s, [] -> o.scales <- [ s ]
   | _ -> ());
  if o.scales = [] then begin
    prerr_endline "either --scale or --all is required";
    exit 2
  end;
  o

(* ---------- timing + RSS ---------- *)

let now_ns () = Mtime_clock.now () |> Mtime.to_uint64_ns
let elapsed_ns since = Int64.sub (now_ns ()) since
let to_ms_f ns = Int64.to_float ns /. 1e6

(* Read VmHWM (high-water-mark RSS, in kB) from /proc/self/status.
   Returns bytes. Linux-only; on other platforms returns 0. *)
let peak_rss_bytes () =
  try
    let ic = open_in "/proc/self/status" in
    let r = ref 0 in
    (try
       while true do
         let line = input_line ic in
         match String.split_on_char ':' line with
         | [ "VmHWM"; rest ] ->
             let s = String.trim rest in
             let n = String.length s in
             let kb_str =
               if n > 3 && String.sub s (n - 3) 3 = " kB"
               then String.sub s 0 (n - 3)
               else s
             in
             r := int_of_string (String.trim kb_str) * 1024
         | _ -> ()
       done
     with End_of_file -> ());
    close_in ic;
    !r
  with _ -> 0

let bytes_to_gib b = float_of_int b /. (1024.0 *. 1024.0 *. 1024.0)

(* Recursively compute the size of [dir] (bytes). Returns 0 if dir
   doesn't exist. *)
let rec dir_size_bytes dir =
  if not (Sys.file_exists dir) then 0
  else if not (Sys.is_directory dir) then begin
    try (Unix.stat dir).Unix.st_size with _ -> 0
  end
  else begin
    let acc = ref 0 in
    Array.iter
      (fun name ->
        let path = Filename.concat dir name in
        try
          let st = Unix.lstat path in
          match st.Unix.st_kind with
          | Unix.S_REG -> acc := !acc + st.Unix.st_size
          | Unix.S_DIR -> acc := !acc + dir_size_bytes path
          | _ -> ()
        with _ -> ())
      (Sys.readdir dir);
    !acc
  end

(* ---------- synthetic data ---------- *)

(* xorshift32, copied from perf.ml so the harness is self-contained.
   The state is per-call closure-captured so concurrent generators
   don't interfere if anyone parallelises this later. *)
let make_prng seed =
  let st = ref seed in
  fun () ->
    let x = !st in
    let x = Int32.logxor x (Int32.shift_left x 13) in
    let x = Int32.logxor x (Int32.shift_right_logical x 17) in
    let x = Int32.logxor x (Int32.shift_left x 5) in
    let x = if Int32.equal x 0l then 1l else x in
    st := x;
    Int32.to_float (Int32.shift_right_logical x 8) /. 16777216.0

(* Generate a CSR batch of [rows × cols] with the given density.
   Returns (indptr, indices, data) as int32/float32 Bigarrays sized
   to the actual nnz. Labels follow the same recipe as perf.ml's
   labels_binary so the synthetic signal is learnable. *)
let gen_csr_batch ~rows ~cols ~density rng =
  (* Worst case: every cell present. We allocate the worst-case
     scratch buffer, then copy out the exact-nnz prefix. For 1M rows
     × 36 cols × 50% density that's 18M nnz × 4 bytes = 72 MB scratch
     beyond the final buffer. Acceptable for 1M / 10M; for 50M we'd
     blow up here, which is precisely why 50M uses smaller batches. *)
  let cap = rows * cols in
  let indptr = Array1.create int32 c_layout (rows + 1) in
  let indices_buf = Array1.create int32 c_layout cap in
  let data_buf = Array1.create float32 c_layout cap in
  let labels = Array1.create float32 c_layout rows in
  indptr.{0} <- 0l;
  let nnz = ref 0 in
  for r = 0 to rows - 1 do
    (* Sample features once per row so the label can use them. *)
    let feature_cache = Array.make cols 0.0 in
    let feature_present = Array.make cols false in
    for c = 0 to cols - 1 do
      if rng () < density then begin
        let v = rng () in
        feature_cache.(c) <- v;
        feature_present.(c) <- true;
        indices_buf.{!nnz} <- Int32.of_int c;
        data_buf.{!nnz} <- v;
        incr nnz
      end
    done;
    indptr.{r + 1} <- Int32.of_int !nnz;
    (* Same recipe as perf.ml.labels_binary: linear combination of
       the first 4 columns, then threshold. Missing columns
       contribute 0. *)
    let f i =
      let c = i mod cols in
      if feature_present.(c) then feature_cache.(c) else 0.0
    in
    let v =
      0.5 *. f 0 -. 0.3 *. f 1 +. 0.2 *. f 2 +. 0.1 *. f 3
    in
    labels.{r} <- (if v > 0.25 then 1.0 else 0.0)
  done;
  let n = !nnz in
  let indices = Array1.sub indices_buf 0 n in
  let data = Array1.sub data_buf 0 n in
  (indptr, indices, data, labels)

let gen_dense_batch ~rows ~cols rng =
  let m = Array2.create float32 c_layout rows cols in
  let labels = Array1.create float32 c_layout rows in
  for r = 0 to rows - 1 do
    for c = 0 to cols - 1 do
      m.{r, c} <- rng ()
    done;
    let v =
      0.5 *. m.{r, 0}
      -. 0.3 *. m.{r, min 1 (cols - 1)}
      +. 0.2 *. m.{r, min 2 (cols - 1)}
      +. 0.1 *. m.{r, 3 mod cols}
    in
    labels.{r} <- (if v > 0.25 then 1.0 else 0.0)
  done;
  (m, labels)

(* ---------- per-scale runners ---------- *)

(* Common params for a tree-method=hist, depth-6 binary classifier. *)
let booster_params =
  [
    "objective", "binary:logistic";
    "tree_method", "hist";
    "max_depth", "6";
    "verbosity", "0";
  ]

type bench_result = {
  scale : scale;
  rows : int;
  cols : int;
  construct_ms : float;
  rss_after_construct_gib : float;
  train_one_round_ms : float;
  predict_dense_ms : float;
  cache_dir_bytes : int;
}

let report_csv r =
  let cache_gib =
    if r.cache_dir_bytes = 0 then "n/a"
    else Printf.sprintf "%.3f" (bytes_to_gib r.cache_dir_bytes)
  in
  Printf.printf "%s,%d,%d,%.1f,%.3f,%.1f,%.1f,%s\n"
    (scale_to_string r.scale)
    r.rows r.cols
    r.construct_ms r.rss_after_construct_gib
    r.train_one_round_ms r.predict_dense_ms
    cache_gib;
  ()

let log_verbose verbose fmt =
  if verbose then Printf.eprintf (fmt ^^ "\n%!")
  else Printf.ifprintf stderr fmt

(* In-memory CSR path. Used for 1M and 10M. *)
let run_in_memory ~scale ~(o : opts) =
  let rows = rows_of scale in
  let cols = o.cols in
  log_verbose o.verbose "[%s] generating CSR (%d rows × %d cols, density %.2f)"
    (scale_to_string scale) rows cols o.density;
  let rng = make_prng o.seed in
  let indptr, indices, data, labels =
    gen_csr_batch ~rows ~cols ~density:o.density rng
  in
  let nnz = Bigarray.Array1.dim data in
  log_verbose o.verbose "[%s] generated %d nnz (%.2f%% density)"
    (scale_to_string scale) nnz
    (100. *. float_of_int nnz /. float_of_int (rows * cols));

  log_verbose o.verbose "[%s] constructing DMatrix" (scale_to_string scale);
  let t0 = now_ns () in
  let dtrain =
    Xgboost.DMatrix.of_csr ~indptr ~indices ~data ~n_cols:cols
  in
  let construct_ns = elapsed_ns t0 in
  let rss = peak_rss_bytes () in
  Xgboost.DMatrix.set_label dtrain labels;
  log_verbose o.verbose "[%s] DMatrix built in %.1f ms; RSS %.2f GiB"
    (scale_to_string scale)
    (to_ms_f construct_ns)
    (bytes_to_gib rss);

  log_verbose o.verbose "[%s] training one round" (scale_to_string scale);
  let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
  Xgboost.Booster.set_params bst booster_params;
  let t1 = now_ns () in
  Xgboost.Booster.update_one_iter bst ~iter:0 ~dtrain;
  let train_ns = elapsed_ns t1 in
  log_verbose o.verbose "[%s] one round: %.1f ms"
    (scale_to_string scale) (to_ms_f train_ns);

  log_verbose o.verbose "[%s] generating 100k held-out and predicting"
    (scale_to_string scale);
  let pred_rng = make_prng (Int32.logxor o.seed 0xA5A5A5A5l) in
  let pm, _ = gen_dense_batch ~rows:holdout_rows ~cols pred_rng in
  let t2 = now_ns () in
  let preds = Xgboost.Booster.predict_dense bst pm in
  let predict_ns = elapsed_ns t2 in
  let _sink =
    let s = ref 0.0 in
    for i = 0 to Array1.dim preds - 1 do s := !s +. preds.{i} done;
    !s
  in
  ignore (Sys.opaque_identity _sink);
  log_verbose o.verbose "[%s] predict_dense 100k: %.1f ms"
    (scale_to_string scale) (to_ms_f predict_ns);

  Xgboost.Booster.free bst;
  Xgboost.DMatrix.free dtrain;
  ignore (Sys.opaque_identity (indptr, indices, data, labels, pm));
  {
    scale;
    rows;
    cols;
    construct_ms = to_ms_f construct_ns;
    rss_after_construct_gib = bytes_to_gib rss;
    train_one_round_ms = to_ms_f train_ns;
    predict_dense_ms = to_ms_f predict_ns;
    cache_dir_bytes = 0;
  }

(* External-memory path: 50M rows fed via streaming iterator with
   cache_prefix, so libxgboost spills batches to disk. *)
let run_external_memory ~(o : opts) =
  let scale = Fifty_m in
  let rows = rows_of scale in
  let cols = o.cols in
  let cache_prefix =
    if o.cache_prefix <> "" then o.cache_prefix
    else
      let dir =
        Filename.concat (Filename.get_temp_dir_name ())
          (Printf.sprintf "xgb-cache-%d" (Unix.getpid ()))
      in
      Unix.mkdir dir 0o755;
      Filename.concat dir "ext"
  in
  let cache_dir = Filename.dirname cache_prefix in
  if not (Sys.file_exists cache_dir) then Unix.mkdir cache_dir 0o755;

  log_verbose o.verbose
    "[50M] streaming %d rows × %d cols in batches of %d to cache prefix %s"
    rows cols o.batch_rows cache_prefix;

  let n_batches =
    (rows + o.batch_rows - 1) / o.batch_rows
  in
  let consumed = ref 0 in
  let rng = make_prng o.seed in
  let next () =
    if !consumed >= rows then None
    else begin
      let take = min o.batch_rows (rows - !consumed) in
      let m, labels = gen_dense_batch ~rows:take ~cols rng in
      consumed := !consumed + take;
      log_verbose o.verbose "[50M] emitted batch %d/%d (%d rows so far)"
        (!consumed / o.batch_rows) n_batches !consumed;
      Some
        Xgboost.DMatrix.{
          data = Batch_dense m;
          labels = Some labels;
        }
    end
  in
  let reset () =
    consumed := 0;
    (* re-seed so the second pass over the iterator (libxgboost may
       call reset+next during prediction) regenerates the same data. *)
    let new_rng = make_prng o.seed in
    let _ = new_rng in
    ()
  in
  log_verbose o.verbose "[50M] constructing DMatrix via of_iterator";
  let t0 = now_ns () in
  let dtrain =
    Xgboost.DMatrix.of_iterator ~cache_prefix ~next ~reset ()
  in
  let construct_ns = elapsed_ns t0 in
  let rss = peak_rss_bytes () in
  log_verbose o.verbose "[50M] DMatrix built in %.1f ms; RSS %.2f GiB"
    (to_ms_f construct_ns) (bytes_to_gib rss);

  log_verbose o.verbose "[50M] training one round";
  let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
  Xgboost.Booster.set_params bst booster_params;
  let t1 = now_ns () in
  Xgboost.Booster.update_one_iter bst ~iter:0 ~dtrain;
  let train_ns = elapsed_ns t1 in
  log_verbose o.verbose "[50M] one round: %.1f ms" (to_ms_f train_ns);

  log_verbose o.verbose "[50M] generating 100k held-out and predicting";
  let pred_rng = make_prng (Int32.logxor o.seed 0xA5A5A5A5l) in
  let pm, _ = gen_dense_batch ~rows:holdout_rows ~cols pred_rng in
  let t2 = now_ns () in
  let preds = Xgboost.Booster.predict_dense bst pm in
  let predict_ns = elapsed_ns t2 in
  let _sink =
    let s = ref 0.0 in
    for i = 0 to Array1.dim preds - 1 do s := !s +. preds.{i} done;
    !s
  in
  ignore (Sys.opaque_identity _sink);

  let cache_bytes = dir_size_bytes cache_dir in
  log_verbose o.verbose "[50M] predict_dense 100k: %.1f ms; cache dir %.2f GiB"
    (to_ms_f predict_ns) (bytes_to_gib cache_bytes);

  Xgboost.Booster.free bst;
  Xgboost.DMatrix.free dtrain;
  ignore (Sys.opaque_identity pm);
  {
    scale;
    rows;
    cols;
    construct_ms = to_ms_f construct_ns;
    rss_after_construct_gib = bytes_to_gib rss;
    train_one_round_ms = to_ms_f train_ns;
    predict_dense_ms = to_ms_f predict_ns;
    cache_dir_bytes = cache_bytes;
  }

(* ---------- main ---------- *)

let run_one ~o scale =
  match scale with
  | One_m | Ten_m -> run_in_memory ~scale ~o
  | Fifty_m -> run_external_memory ~o

let () =
  let o = parse_args () in
  let _ = iters_default in
  if Sys.getenv_opt "NO_HEADER" <> Some "1" then
    print_endline
      "scale,rows,cols,construct_ms,rss_construct_gib,train_one_round_ms,predict_dense_ms,cache_dir_gib";
  List.iter
    (fun s ->
      let r = run_one ~o s in
      report_csv r)
    o.scales
