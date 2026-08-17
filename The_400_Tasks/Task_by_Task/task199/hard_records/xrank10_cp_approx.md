# xrank10_cp_approx

- Live bar: 570, parent sha `5baef59b7af166cb2d617e5362dd5708bfd26119aea945033481d6dda6e3b263` from `python tools/ng_current_bar.py --task 199 --json` / team manifest.
- Attempted route: replace Sho's `x=11` color/channel CP factor with an arbitrary ALS rank-10 factorization of the full `G[k,p,r]` tensor; prototype `The_400_Tasks/Task_by_Task/task199/prototypes/task199_xrank10.onnx`, sha `27c496848383e99db39ad4648d53ed8774ea50a386673fea261db1cf86baa8aa`.
- Cost/profile before failure: static=547, profile=547.
- Failed gate: `python tools/ng_gate.py visible --task 199 --cand The_400_Tasks/Task_by_Task/task199/prototypes/task199_xrank10.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the float64 rank-10 fit rounds to a numerically different FP32 Einsum; the coefficient error is amplified by the polynomial row/column factors and destroys all visible outputs. Do not retry arbitrary numeric CP compression of the full color tensor.
- One remaining hypothesis: only exact structural/generator-invariant x-column removal is viable here; further rank cuts need symbolic proof or a clean generator-invariant branch, not fitted approximate factors.
