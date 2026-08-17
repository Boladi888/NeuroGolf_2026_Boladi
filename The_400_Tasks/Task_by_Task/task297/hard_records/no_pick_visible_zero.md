# task297 no-pick visible-zero route

- Bar used: task-local staged-pending cost 130 (`task297_factor_de_cost130`, sha256 9afc560beacedef1c6f957432bbf34f0742861dc49267d81183bf480f1102a8f); authoritative workbook+manifest live remains 136, sha256 fefcaa50289d057f74b9cfcefa9188d236be69cb956ceed87c16d2a41f7d9d22.
- Attempted route: remove the 2-param `pick_second` selector from the current factorized single-Einsum graph, letting direct basis reads sum both rows (`task297_no_pick.onnx`, static cost 128).
- Failed gate: `python tools/ng_gate.py visible --task 297 --cand The_400_Tasks/Task_by_Task/task297/prototypes/task297_no_pick.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: the selector is load-bearing; dropping it shifts color/row coordinate reads and produces visible-zero, which is certain hidden zero. Do not run truth/AB/profile or build unadjusted no-pick variants.
- Remaining hypothesis: only a true algebraic rewrite that keeps exact second-row selection without a standalone 2-vector, or a different sub-130 single-op factorization of `coord_basis`/`color_basis`, can improve over the cost-130 staged candidate.
