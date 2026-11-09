(* Regression tests: the binding must keep alive everything a C call
 * uses until that call (and any copy of a result it borrows) is done:
 * Bigarrays handed to libxgboost by address, through a JSON
 * __array_interface__ string, and the Booster.t / DMatrix.t records whose
 * finalisers free the libxgboost objects behind their raw handles.
 *
 * Booster and DMatrix: functions passed only the raw handle to C, so a
 * booster or DMatrix the caller no longer referenced could be finalised,
 * and freed, right before libxgboost used it (a segfault in
 * InplacePredictImpl, found by the tests below). Bigarrays:
 * On OCaml < 4.13, [Booster.predict_dense] and [Booster.boost_one_iter]
 * did not: their [ignore (Sys.opaque_identity x)] after the call was
 * compiled away, so a GC triggered by the allocations just before the
 * call could finalise the input, and libxgboost read freed memory (see
 * Internal.keep_alive). That only failed when a GC happened to land in
 * that window, so the property tests caught it now and then.
 *
 * This test makes it deterministic and independent of what the allocator
 * does with freed memory:
 * - the input is only reachable through the argument of the call under
 *   test, and a finaliser poisons its data (fills it with 1e30) as soon
 *   as the GC finds it unreachable;
 * - while the call runs, Gc.Memprof forces a full major GC (which runs
 *   finalisers) at every allocation.
 * If the binding lets the input die before the C call, libxgboost reads
 * poisoned values and the result differs from the reference.
 *
 * Gc.Memprof isn't implemented on OCaml 5.0 to 5.2 (start fails); the
 * test skips itself there. *)

open Bigarray

let rows = 64
let cols = 8
let poison = 1e30

(* Deterministic data, same xorshift as check.ml. *)
let data seed n =
  let x = ref seed in
  Array.init n (fun _ ->
      let v = !x in
      let v = Int32.logxor v (Int32.shift_left v 13) in
      let v = Int32.logxor v (Int32.shift_right_logical v 17) in
      let v = Int32.logxor v (Int32.shift_left v 5) in
      let v = if Int32.equal v 0l then 1l else v in
      x := v;
      Int32.to_float (Int32.shift_right_logical v 8) /. 16777216.0)

let matrix values =
  let m = Array2.create float32 c_layout rows cols in
  Array.iteri (fun i v -> m.{i / cols, i mod cols} <- v) values;
  m

let vector values =
  let a = Array1.create float32 c_layout (Array.length values) in
  Array.iteri (fun i v -> a.{i} <- v) values;
  a

(* A fresh copy of [values] that poisons itself once unreachable. *)
let poisoned_when_dead_matrix values =
  let m = matrix values in
  Gc.finalise (fun m -> Array2.fill m poison) m;
  m

let poisoned_when_dead_vector values =
  let a = vector values in
  Gc.finalise (fun a -> Array1.fill a poison) a;
  a

(* Run [f] while forcing a full major GC at every allocation. *)
let with_gc_at_every_allocation f =
  let force _ =
    Gc.full_major ();
    None
  in
  let tracker =
    { Gc.Memprof.null_tracker with alloc_minor = force; alloc_major = force }
  in
  (* [start] returns unit before OCaml 5.2 and a profile from 5.2 on; it
     fails on 5.0 to 5.2, where memprof isn't implemented. *)
  (match Gc.Memprof.start ~sampling_rate:1.0 tracker with
   | _ -> ()
   | exception (Failure _ | Invalid_argument _) ->
       Alcotest.skip ());
  Fun.protect ~finally:Gc.Memprof.stop f

let train_values = data 0xC0FFEEl (rows * cols)
let test_values = data 0xBADC0DEl (rows * cols)

let labels =
  vector
    (Array.init rows (fun r ->
         if train_values.(r * cols) > 0.5 then 1.0 else 0.0))

let new_booster () =
  let dtrain = Xgboost.DMatrix.of_bigarray2 (matrix train_values) in
  Xgboost.DMatrix.set_label dtrain labels;
  let bst = Xgboost.Booster.create ~cache:[ dtrain ] () in
  Xgboost.Booster.set_params bst
    [
      "objective", "binary:logistic";
      "tree_method", "hist";
      "max_depth", "3";
      "seed", "0";
      "nthread", "1";
      "verbosity", "0";
    ];
  (dtrain, bst)

(* NaN (e.g. from a model boosted on poison) counts as infinitely far. *)
(* Hand [x] to a call as its only reference: [take cell] empties the cell. *)
let take cell =
  match !cell with
  | Some x ->
      cell := None;
      x
  | None -> assert false

let trained_booster () =
  let dtrain, bst = new_booster () in
  for it = 0 to 4 do
    Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
  done;
  bst

let max_abs_diff a b =
  let d = ref 0.0 in
  for i = 0 to Array1.dim a - 1 do
    let x = Float.abs (a.{i} -. b.{i}) in
    d := Float.max !d (if Float.is_nan x then Float.infinity else x)
  done;
  !d

let test_predict_dense () =
  let dtrain, bst = new_booster () in
  for it = 0 to 4 do
    Xgboost.Booster.update_one_iter bst ~iter:it ~dtrain
  done;
  let expected = Xgboost.Booster.predict_dense bst (matrix test_values) in
  (* The poison must change the predictions, or the test proves nothing. *)
  let all_poison = Xgboost.Booster.predict_dense bst
      (matrix (Array.make (rows * cols) poison)) in
  Alcotest.(check bool) "poisoned input changes predictions" true
    (max_abs_diff expected all_poison > 1e-3);
  let got =
    with_gc_at_every_allocation (fun () ->
        Xgboost.Booster.predict_dense bst
          (poisoned_when_dead_matrix test_values))
  in
  (* A real use of [bst] after the call, so this case only tests the
     input Bigarray (booster_only_the_call_keeps tests the booster). *)
  Xgboost.Booster.free bst;
  Alcotest.(check (float 1e-6)) "predictions on an input only the call keeps"
    0.0 (max_abs_diff expected got)

let test_booster_only_the_call_keeps () =
  let expected =
    Xgboost.Booster.predict_dense (trained_booster ()) (matrix test_values)
  in
  let cell = ref (Some (trained_booster ())) in
  let got =
    with_gc_at_every_allocation (fun () ->
        Xgboost.Booster.predict_dense (take cell) (matrix test_values))
  in
  Alcotest.(check (float 1e-6)) "predict_dense on a booster only the call keeps"
    0.0 (max_abs_diff expected got)

let test_dmatrix_only_the_call_keeps () =
  let bst = trained_booster () in
  let expected = Xgboost.Booster.predict_dense bst (matrix test_values) in
  let cell = ref (Some (Xgboost.DMatrix.of_bigarray2 (matrix test_values))) in
  let got =
    with_gc_at_every_allocation (fun () ->
        Xgboost.Booster.predict bst (take cell))
  in
  Alcotest.(check (float 1e-6)) "predict on a DMatrix only the call keeps"
    0.0 (max_abs_diff expected got)

let test_boost_one_iter () =
  (* Two identical boosters, one boosted with grad/hess kept alive by the
     test, the other with copies only the call keeps. *)
  let grad = Array.init rows (fun r -> labels.{r} -. 0.5) in
  let hess = Array.make rows 0.25 in
  let boost ?(wrap = fun f -> f ()) mk =
    let dtrain, bst = new_booster () in
    wrap (fun () ->
        Xgboost.Booster.boost_one_iter bst ~iter:0 ~dtrain ~grad:(mk grad)
          ~hess:(mk hess));
    Xgboost.Booster.predict_dense bst (matrix test_values)
  in
  let expected = boost vector in
  let all_poison = boost (fun v -> vector (Array.map (fun _ -> poison) v)) in
  Alcotest.(check bool) "poisoned grad/hess change the model" true
    (max_abs_diff expected all_poison > 1e-3);
  let got = boost ~wrap:with_gc_at_every_allocation poisoned_when_dead_vector in
  Alcotest.(check (float 1e-6)) "model boosted with grad/hess only the call keeps"
    0.0 (max_abs_diff expected got)

let () =
  Alcotest.run "keep_alive"
    [
      ( "keep_alive",
        [
          Alcotest.test_case "predict_dense" `Quick test_predict_dense;
          Alcotest.test_case "boost_one_iter" `Quick test_boost_one_iter;
          Alcotest.test_case "booster_only_the_call_keeps" `Quick
            test_booster_only_the_call_keeps;
          Alcotest.test_case "dmatrix_only_the_call_keeps" `Quick
            test_dmatrix_only_the_call_keeps;
        ] );
    ]
