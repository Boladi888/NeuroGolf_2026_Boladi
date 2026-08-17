task043 hard record - shared selector with live coefficients

Live bar:
- cost 84
- sha256 1e0d69485acbed45aafb3f94ab63b182f635896031144d8fb7fa9865c6c8ad01
- source: manifest via `python tools/ng_current_bar.py --task 43 --json`

Attempted route:
- Cost-54 one-node `Einsum` prototype reusing live `C[2,10]` and `M[2,2]`, but replacing the two live 30-element selectors (`top_row`, `right_col`) with one shared 30-element selector `S = delta0 + delta9`.
- Prototype: `The_400_Tasks/Task_by_Task/task043/prototypes/task043_shared_selector_alpha_1p0.onnx`
- Static score: `(0, 0, 54)`, total cost 54.
- Prototype sha256: `a8db7d86f8b543d89ba61dfced00cca033ebe347919316ad3d6ead4eacb3df56`

Exact failed gate:
- `python tools/ng_gate.py visible --task 43 --cand The_400_Tasks/Task_by_Task/task043/prototypes/task043_shared_selector_alpha_1p0.onnx`
- Result: `[visible] 0/266 (0.00%)  FAIL`

Why this route is dead:
- Sharing the selector while keeping the live coefficient tables loses the separate top-row/right-column reads. The live sign table does not survive that collapse; visible is a certain hidden zero.
- Per gate protocol, no truth, A/B, battery, profile, or full gate was run after the visible-zero.

One remaining hypothesis:
- A fresh 54-byte shared-selector sign table may still exist if `S[30]`, `C[2,10]`, and `M[2,2]` are solved jointly. The attempted random/LP searches were not conclusive before timeout; do not retry the live-coefficient version.
