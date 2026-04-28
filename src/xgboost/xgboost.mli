(** OCaml bindings for the XGBoost gradient boosting library.

    The public API is divided into two submodules:
    - {!DMatrix} for input matrices (dense or sparse, owned by libxgboost
      but pinned to OCaml-side data).
    - {!Booster} for trained models.

    Errors crossing the FFI boundary are raised as {!Xgboost_error}. The
    payload carries either an upstream message (from
    [XGBGetLastError]) or a binding-side precondition violation. *)

module Error : sig
  type t =
    | Xgb_error of string
    | Invalid_argument of string
    | Shape_mismatch of { expected : int * int; got : int * int }

  val to_string : t -> string
  val pp : Format.formatter -> t -> unit
end

exception Xgboost_error of Error.t

(** Result-returning wrappers for callers who prefer [result] over
    raised exceptions. The hot-path API raises by default. *)
module Result : sig
  val try_ : (unit -> 'a) -> ('a, Error.t) result
end

module DMatrix : sig
  (** Input matrices for libxgboost.

      A [DMatrix.t] owns a libxgboost handle and pins back-references to
      the source Bigarrays (or other OCaml data) for the lifetime of the
      handle. Freeing happens deterministically via {!free} or {!with_},
      and as a safety net via {!Gc.finalise_last}. Double-free is safe. *)

  type t

  val rows : t -> int
  val cols : t -> int

  (** [of_bigarray2 ?missing m] constructs a dense DMatrix from a 2D
      Float32 Bigarray (c_layout, row-major). The Bigarray is pinned for
      the lifetime of the result; mutating it after construction has
      undefined effects (libxgboost typically copies on construction).
      [missing] is the value treated as missing (default: [Float.nan]). *)
  val of_bigarray2 :
    ?missing:float ->
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t ->
    t

  (** [of_csr ~indptr ~indices ~data ~n_cols] constructs a sparse
      DMatrix in CSR form. [indptr.(i)] is the start of row [i] in
      [indices] / [data]; [Array1.dim indptr - 1] is the number of
      rows. All three Bigarrays are pinned for the lifetime of the
      result. The buffers are passed to libxgboost via JSON
      __array_interface__ — no element-wise copy. *)
  val of_csr :
    indptr:(int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    indices:(int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    data:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    n_cols:int ->
    t

  (** A single batch of data fed by an iterator into the streaming
      construction path. *)
  type batch =
    | Batch_dense of
        (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t
    | Batch_csr of {
        indptr :
          (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t;
        indices :
          (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t;
        data :
          (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t;
        n_cols : int;
      }

  type labelled_batch = {
    data : batch;
    labels :
      (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
      option;
  }

  (** [of_iterator ?cache_prefix ?missing ~next ~reset ()] builds a
      DMatrix by repeatedly calling [next ()] until it returns [None].
      Each batch's source Bigarrays are pinned for the duration of the
      libxgboost call that consumes them. With a non-empty
      [cache_prefix], libxgboost spills batches to disk and the
      iterator may be re-invoked during prediction (external memory
      mode); the iterator state must therefore remain meaningful for
      the lifetime of the returned DMatrix. *)
  val of_iterator :
    ?cache_prefix:string ->
    ?missing:float ->
    next:(unit -> labelled_batch option) ->
    reset:(unit -> unit) ->
    unit ->
    t

  (** Set the [label] field used for supervised training. The Bigarray
      length must equal [rows t]. *)
  val set_label :
    t -> (float, Bigarray.float32_elt, _) Bigarray.Array1.t -> unit

  (** Set the per-row [weight] field. *)
  val set_weight :
    t -> (float, Bigarray.float32_elt, _) Bigarray.Array1.t -> unit

  (** Number of non-missing entries in the matrix. *)
  val num_non_missing : t -> int

  (** Explicitly free the underlying handle. Idempotent. *)
  val free : t -> unit

  (** [with_ create f] constructs a DMatrix with [create ()] and runs
      [f] on it; the DMatrix is freed at scope exit even on exception. *)
  val with_ : (unit -> t) -> (t -> 'a) -> 'a
end

module Booster : sig
  (** Trained gradient boosting model.

      A [Booster.t] does not own its training DMatrices at the C level,
      but the OCaml wrapper pins them via [cache] (permanent) and
      [iter_pin] (held only during a single [update_one_iter] call). *)

  type t

  (** [create ?cache ()] creates a fresh booster. The [cache] DMatrices
      are passed to [XGBoosterCreate] (used internally by libxgboost as
      training/eval references) and pinned for the booster's lifetime. *)
  val create : ?cache:DMatrix.t list -> unit -> t

  val set_param : t -> string -> string -> unit
  val set_params : t -> (string * string) list -> unit

  (** Run one round of training. The training DMatrix is pinned only for
      the duration of this call. *)
  val update_one_iter : t -> iter:int -> dtrain:DMatrix.t -> unit

  (** Custom-objective training: instead of letting libxgboost compute
      gradients from the configured objective, the caller supplies the
      first and second derivatives of their loss directly. [grad] and
      [hess] must have the same length, normally [rows dtrain * n_classes].
      The Bigarrays are pinned for the duration of the call only. *)
  val boost_one_iter :
    t ->
    iter:int ->
    dtrain:DMatrix.t ->
    grad:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    hess:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    unit

  (** Evaluate the model against a list of (name, dmat) pairs and return
      the upstream-formatted evaluation string. *)
  val eval_one_iter :
    t -> iter:int -> evals:(string * DMatrix.t) list -> string

  (** Reset accumulated training state. *)
  val reset : t -> unit

  (** Number of features the model expects (cached after first call). *)
  val num_features : t -> int

  (** Number of boosted rounds. *)
  val boosted_rounds : t -> int

  (** [predict ?ntree_limit ?training t dmat] returns a fresh OCaml-owned
      Bigarray of float32 predictions of length [rows dmat *
      output_dim]. The borrowed XGBoost buffer is copied eagerly so the
      result is safe to keep across subsequent calls. *)
  val predict :
    ?ntree_limit:int ->
    ?training:bool ->
    t ->
    DMatrix.t ->
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

  (** [predict_dense ?missing t m] runs prediction directly against the
      Bigarray [m] without allocating a DMatrix. Useful for tight
      inference loops where DMatrix construction overhead would
      dominate. The input Bigarray is pinned for the duration of the
      call only. *)
  val predict_dense :
    ?ntree_limit:int ->
    ?training:bool ->
    ?missing:float ->
    t ->
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t ->
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

  val save_model : t -> path:string -> unit
  val load_model : t -> path:string -> unit

  (** [save_model_buffer ?format t] returns the serialised model bytes.
      [format] is a JSON config string; the default selects "json". *)
  val save_model_buffer : ?format:string -> t -> bytes
  val load_model_buffer : t -> bytes -> unit

  val save_json_config : t -> string
  val load_json_config : t -> string -> unit

  (** Per-feature importance scores. [importance_type] is one of
      ["weight"] (default — split count), ["gain"], ["cover"],
      ["total_gain"], ["total_cover"]. For multiclass models the score
      returned per feature is the sum across classes. *)
  val feature_score :
    ?importance_type:string -> t -> (string * float) list

  val free : t -> unit

  val with_ : ?cache:DMatrix.t list -> (t -> 'a) -> 'a

  (** Expert-only no-copy variants. The returned Bigarrays wrap
      libxgboost's internal output buffer directly and are invalidated
      by the next call on the same booster. Callers MUST consume (or
      copy) the result before any subsequent call on this booster. *)
  module Unsafe : sig
    val predict_borrowed :
      ?ntree_limit:int ->
      ?training:bool ->
      t ->
      DMatrix.t ->
      (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  end
end

(** [version ()] returns the (major, minor, patch) of the loaded
    libxgboost. *)
val version : unit -> int * int * int

(** [last_error ()] returns the libxgboost thread-local error string.
    Mainly useful for low-level debugging; high-level callers receive the
    string already wrapped in [Xgboost_error]. *)
val last_error : unit -> string
