open Bigarray

let () =
  let features =
    Array2.of_array float32 c_layout
      [| [| 0.; Float.nan |]; [| 1.; 2. |]; [| 2.; 1. |]; [| 3.; 3. |] |]
  in
  Xgboost.DMatrix.with_
    (fun () -> Xgboost.DMatrix.of_bigarray2 features)
    (fun matrix ->
      Xgboost.DMatrix.set_label matrix
        (Array1.of_array float32 c_layout [| 0.; 1.; 4.; 9. |]);
      Xgboost.Booster.with_ ~cache:[ matrix ] (fun model ->
          Xgboost.Booster.set_params model
            [
              ("objective", "reg:squarederror");
              ("tree_method", "hist");
              ("device", "cpu");
              ("nthread", "1");
              ("max_depth", "2");
              ("seed", "17");
            ];
          for iter = 0 to 3 do
            Xgboost.Booster.update_one_iter model ~iter ~dtrain:matrix
          done;
          let before = Xgboost.Booster.predict_dense model features in
          let bytes = Xgboost.Booster.save_model_buffer model in
          Xgboost.Booster.with_ (fun restored ->
              Xgboost.Booster.load_model_buffer restored bytes;
              Xgboost.Booster.set_param restored "nthread" "1";
              let after = Xgboost.Booster.predict_dense restored features in
              if Array1.dim before <> 4 || Array1.dim after <> 4 then
                failwith "Unexpected prediction shape";
              for row = 0 to 3 do
                if
                  (not (Float.is_finite before.{row}))
                  || before.{row} <> after.{row}
                then failwith "Model restore changed predictions"
              done;
              print_endline
                "Installed XGBoost linking, training, prediction, and restore \
                 passed")))
