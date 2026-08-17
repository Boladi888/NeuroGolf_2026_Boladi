task043 hard record - sparse Einsum selectors

Live bar:
- cost 84
- sha256 1e0d69485acbed45aafb3f94ab63b182f635896031144d8fb7fa9865c6c8ad01
- source: `python tools/ng_current_bar.py --task 43 --json` (manifest authoritative)

Attempted route:
- Replace dense one-hot selector initializers `top_row[30]` and `right_col[30]` in the live one-node Einsum with sparse initializers.
- Prototype: `The_400_Tasks/Task_by_Task/task043/prototypes/task043_sparse_selectors_named_probe.onnx`
- Prototype sha256: d658c0725cbeb93ebd82960cc4313c2bf394520e70c9edb534f7cdb8519122e5

Exact failed gate:
- `python -c "from tools._score_v2 import score_network_v2; ..."` returned `score None None`.
- `onnx.checker.check_model(full_check=True)` failed with:
  `[ShapeInferenceError] Rank of input 4 (0) does not match the equation indices (1).`

Why this route is dead:
- ONNX sparse initializers bind, but strict shape inference treats the sparse Einsum operand as rank 0. This matches traps section 8e: sparse op inputs are not a valid scorer route here.
- No visible/truth/profile gates were run because the model is scorer-invalid.

One remaining hypothesis:
- Only dense selector factorization remains plausible; do not retry sparse selector operands for Einsum unless the scorer's sparse shape-inference behavior changes.
