(* Eval — pure-OCaml helpers for evaluating predictions.
 *
 * No FFI; operates on the strings emitted by Booster.eval_one_iter
 * and on Bigarray buffers of predictions / labels. *)

open Bigarray

let raise_invalid msg =
  raise (Error.Xgboost_error (Error.Invalid_argument msg))

(* ----- parse + get ----- *)

(* libxgboost emits eval lines of the form
     "[<iter>]\t<dataset>-<metric>:<value>\t..."
   sometimes with a trailing newline. We tokenise on TAB, drop the
   leading [<iter>] header, then split each remaining token on the
   *last* colon (metric names may legally contain colons in some
   custom configurations, so right-splitting is the safer choice
   even though stock libxgboost names never do). *)
let split_on_tab s =
  String.split_on_char '\t' s
  |> List.map (fun t ->
         (* Drop a single trailing '\n' if present. *)
         let n = String.length t in
         if n > 0 && t.[n - 1] = '\n' then String.sub t 0 (n - 1) else t)
  |> List.filter (fun t -> t <> "")

let split_metric token =
  match String.rindex_opt token ':' with
  | None ->
      raise_invalid
        (Printf.sprintf "Eval.parse: malformed token %S (no ':')" token)
  | Some i ->
      let name = String.sub token 0 i in
      let value = String.sub token (i + 1) (String.length token - i - 1) in
      let v =
        try float_of_string value
        with _ ->
          raise_invalid
            (Printf.sprintf
               "Eval.parse: token %S has non-numeric value %S" token value)
      in
      (name, v)

let parse s =
  match split_on_tab s with
  | [] -> []
  | header :: rest ->
      (* The first token is "[<iter>]"; we ignore it. If it doesn't look
         like a header, treat it as a metric token instead — robust to
         callers that pass already-trimmed input. *)
      let tokens =
        if String.length header > 0 && header.[0] = '[' then rest
        else header :: rest
      in
      List.map split_metric tokens

let get ~metric s =
  match List.assoc_opt metric (parse s) with
  | Some v -> v
  | None ->
      raise_invalid
        (Printf.sprintf "Eval.get: metric %S not found in %S" metric s)

(* ----- AUC + ROC ----- *)

(* Convert a float32 Bigarray.Array1 to a float array for sorting.
   AUC needs O(n log n) sort, and the cost of copying once into a
   regular OCaml array is dwarfed by the sort itself. Returns the
   array of (prediction, label) pairs. *)
let pairs ~predictions ~labels =
  let n = Array1.dim predictions in
  if Array1.dim labels <> n then
    raise
      (Error.Xgboost_error
         (Error.Shape_mismatch
            { expected = (n, 0); got = (Array1.dim labels, 0) }));
  let arr = Array.make n (0.0, 0.0) in
  for i = 0 to n - 1 do
    arr.(i) <- (predictions.{i}, labels.{i})
  done;
  arr

(* Sweep sorted (score desc, label) pairs accumulating TP/FP counts and
   trapezoidal AUC. Ties on score are flushed as a single ROC step (TPR
   and FPR move simultaneously) rather than emitting an arbitrary order
   between tied points; this is the standard AUC definition. *)
let auc_and_roc ~want_roc ~predictions ~labels =
  let arr = pairs ~predictions ~labels in
  Array.sort
    (fun (p1, _) (p2, _) -> Float.compare p2 p1)
    arr;
  let n = Array.length arr in
  let p_total = ref 0.0 and n_total = ref 0.0 in
  for i = 0 to n - 1 do
    let _, l = arr.(i) in
    if l > 0.5 then p_total := !p_total +. 1.0
    else n_total := !n_total +. 1.0
  done;
  if !p_total = 0.0 || !n_total = 0.0 then
    raise_invalid
      "Eval.auc/roc: labels must contain both positive and negative \
       examples";
  let tp = ref 0.0 and fp = ref 0.0 in
  let prev_score = ref Float.infinity in
  let prev_tp = ref 0.0 and prev_fp = ref 0.0 in
  let auc = ref 0.0 in
  let roc = ref [] in
  if want_roc then roc := [ (0.0, 0.0) ];
  let flush () =
    if !tp <> !prev_tp || !fp <> !prev_fp then begin
      let dfpr = (!fp -. !prev_fp) /. !n_total in
      let avg_tpr = (!tp +. !prev_tp) /. (2.0 *. !p_total) in
      auc := !auc +. (dfpr *. avg_tpr);
      if want_roc then
        roc :=
          (!fp /. !n_total, !tp /. !p_total) :: !roc;
      prev_tp := !tp;
      prev_fp := !fp
    end
  in
  for i = 0 to n - 1 do
    let s, l = arr.(i) in
    if s <> !prev_score then begin
      flush ();
      prev_score := s
    end;
    if l > 0.5 then tp := !tp +. 1.0 else fp := !fp +. 1.0
  done;
  flush ();
  let roc =
    if want_roc then List.rev !roc else []
  in
  (!auc, roc)

let auc ~predictions ~labels =
  let v, _ = auc_and_roc ~want_roc:false ~predictions ~labels in
  v

let roc ~predictions ~labels =
  let _, r = auc_and_roc ~want_roc:true ~predictions ~labels in
  r
