task043 hard record - drop right-column selector / no-axis row sum

Live bar:
- cost 58
- sha256 e770aa014557dd2e05c610a51aa46a97f6aa42e52348a2f39bd4dd338ab08d03
- source: `python tools/ng_current_bar.py --task 43 --json` (manifest authoritative, owner Vamshi)

Attempted route:
- Start from the current team-best one-node Einsum:
  `ndhw,nehx,nfyw,qd,se,x,sf,pc,pqs->nchw`
- Delete the remaining charged selector `X[30]` and let the right-column read sum over the whole row:
  `ndhw,nehx,nfyw,qd,se,sf,pc,pqs->nchw`
- This keeps the current `A[2,10]` and `D[2,2,2]`, removes `X`, and targets the hoped-for
  cost-28 floor (`20 + 8` params, memory 0).
- Builder: `The_400_Tasks/Task_by_Task/task043/prototypes/build_drop_right_selector_probe.py`
- Prototype: `The_400_Tasks/Task_by_Task/task043/prototypes/task043_drop_right_selector_cost28_probe.onnx`
- Static score: `(0, 0, 28)`, total cost 28.
- Prototype sha256: d45f23b2f1d77311c52379724b1da9c202331d58d80cfaf0d11448b40054cfd0

Exact failed gate:
- `python tools/ng_gate.py visible --task 43 --cand The_400_Tasks/Task_by_Task/task043/prototypes/task043_drop_right_selector_cost28_probe.onnx`
- Result: `[visible] 0/266 (0.00%)  FAIL`

Why this route is dead:
- The current 58-byte graph already uses a whole-column sum for top-row markers, but still needs
  the one-hot `X=e9` read to isolate right-column markers. Replacing that read with a whole-row sum
  loses the special handling for row 0, where top-row column markers contaminate the row detector.
- Per the visible-fail axiom, a visible-zero representation is a certain hidden zero; no truth,
  A/B, battery, profile, or safety gates were run.

One remaining hypothesis:
- A genuinely different output basis may still avoid an explicit length-30 right-column selector,
  but it cannot be the direct no-axis row-sum deletion of Vamshi's 58-byte Einsum.
