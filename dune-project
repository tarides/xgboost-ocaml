(lang dune 3.18)

(using ctypes 0.3)

(generate_opam_files true)

(name xgboost)

(source (github tarides/xgboost-ocaml))
(authors "Cuihtlauac ALVARADO <cuihtlauac@tarides.com>")
(maintainers "Cuihtlauac ALVARADO <cuihtlauac@tarides.com>")
(license MIT)

(package
 (name xgboost)
 (synopsis "OCaml bindings for the XGBoost gradient boosting library")
 (description
   "Native OCaml bindings to libxgboost (>=3.0). Three-layer FFI:\
   \nthe upstream stable C ABI, statically generated ctypes stubs,\
   \nand a high-level GC-aware OCaml API. Supports dense and sparse\
   \nDMatrix construction (Bigarray, zero-copy), training, prediction,\
   \nJSON config, and external-memory streaming via a callback iterator.")
 (tags ("xgboost" "machine-learning" "gradient-boosting" "bindings" "ctypes"))
 (depends
  (ocaml (>= 5.1))
  (dune-configurator :build)
  (ctypes (>= 0.20))
  (ctypes-foreign (>= 0.20))
  (alcotest :with-test)
  (qcheck-core (and :with-test (>= 0.20)))
  (qcheck-alcotest (and :with-test (>= 0.20)))))

(package
 (name xgboost-bench)
 (synopsis "Benchmarks for xgboost-ocaml against the C reference and Python xgboost")
 (description
   "Benchmark harness using bechamel; compares the high-level API and\
   \nlow-level bindings against a pure-C reference and (optionally) the\
   \nupstream Python xgboost package on a fixed grid of workloads.")
 (tags ("xgboost" "benchmark" "bechamel"))
 (depends
  (ocaml (>= 5.1))
  xgboost
  (mtime (>= 2.0))
  (bechamel (>= 0.5))
  (bechamel-perf (>= 0.5))
  (bechamel-notty (>= 0.5))))
