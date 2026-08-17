# task024 — sparse-basis / tiny-tail dead route (task024, bo_ng_cdx2_worker2, 2026-07-14)

Live bar used: **56**, sha256 `9090d7fc66fcf617eb83597d5b2020a5bfb9affae3720e3e34eb0391a8abca35`
from `python tools/ng_current_bar.py --task 24 --json` / team manifest.

Attempted route: keep Vamshi's one-node free-output `Einsum` exactly but convert
the shared `basis[4,10]` initializer to an ONNX sparse initializer. Intended cost:
`basis` 16 stored nonzero values + dense `mix[4,4]` 16 = 32, with identical math.
Prototype: `../prototypes/build_sparse_basis_probe.py` ->
`../prototypes/cand_sparse_basis.onnx`.

Exact failed gates / checks:
- `score_network_v2(Team_Best_Onnx/task024.onnx)` = `(0, 0, 56)`.
- `score_network_v2(../prototypes/cand_sparse_basis.onnx)` = `None`.
- ONNX `check_model(..., full_check=True)` and strict shape inference fail:
  `(op_type:Einsum): [ShapeInferenceError] Rank of input 3 (0) does not match the equation indices (2).`
- `python tools/ng_gate.py visible --task 24 --cand ../prototypes/cand_sparse_basis.onnx`
  fails before comparison:
  `Invalid model. Node input 'safe_name_1' is not a graph input, initializer, or output of a previous node.`

Why dead: sparse initializers are not a legal drop-in for `Einsum` in the scorer
path. Shape inference cannot supply the sparse operand rank, and the sanitizer
does not preserve the sparse initializer reference, so ORT sees a dangling input.
Adding a dense `value_info`/Constant sparse output would re-materialize the
`basis` tensor and exceed the 56-byte bar.

Tiny Equal-tail / Where-as-output note: a materialized color-index, row-vector,
or column-vector decision core is already at least one charged 30-byte BOOL/U8
vector per row/col feature, and the necessary row-fill + col-2 + active/sentinel
features cross the 56-byte live bar before the terminal writer. The current
single free-output `Einsum` has zero charged intermediates, so a tail rewrite must
avoid *all* pre-output planes/vectors to be competitive.

One remaining hypothesis: find a one-node `Einsum` factorization that replaces the
dense `basis[4,10]` color selector with free input-channel axes while still keeping
all terms in one contraction. The obvious polynomial formulation needs either a
dense channel-mode factor or multiple full-output terms, so it was not built.
