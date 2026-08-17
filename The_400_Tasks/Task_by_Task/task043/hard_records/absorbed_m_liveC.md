task043 hard record - absorbed M with live C

Live bar:
- cost 84
- sha256 1e0d69485acbed45aafb3f94ab63b182f635896031144d8fb7fa9865c6c8ad01
- source: `python tools/ng_current_bar.py --task 43 --json` (manifest authoritative)

Attempted route:
- Delete the live `M[2,2]` initializer and replace the final contraction with
  `bqrw,lq,barx,ka,x,bdyw,kd,y,kc->bcrw`, reusing live `C[2,10]` for output coefficients.
- Prototype: `The_400_Tasks/Task_by_Task/task043/prototypes/task043_absorb_m_liveC_cost80_probe.onnx`
- Static score: `(0, 0, 80)`, total cost 80.
- Prototype sha256: 72e6d89d2c4f5d9e15b86a6dcb310b130c4c869d37f913ba178db840a3640ef5

Exact failed gate:
- `python tools/ng_gate.py visible --task 43 --cand The_400_Tasks/Task_by_Task/task043/prototypes/task043_absorb_m_liveC_cost80_probe.onnx`
- Result: `[visible] 0/266 (0.00%)  FAIL`

Why this route is dead:
- Removing `M` makes the signed classifier leak positive wrong channels: live `C` without `M` makes background and mixed marker cases multi-hot/incorrect. A visible-zero is a certain hidden zero, so no truth/ab/profile gates were run.

One remaining hypothesis:
- A custom sub-4-param replacement for `M` would need to stay inside the final Einsum without a charged constructed tensor. The observed live `M` has a repeated first row, but no valid 3-param ONNX factorization was found without adding memory that exceeds the saved param.
