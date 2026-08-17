# task115 4-term painter dead route

- Live/team-best bar: **389**, SHA-256 `e959b06e6782df52dbb55e802bb0613d0825fb4f71bad262712975dcf8022357`, source `tools/ng_current_bar.py --task 115 --json` / team manifest updated 2026-07-02T02:35:15Z.
- Attempted route: reduce the live terminal separable painter from 5 terms to 4 terms to save `term_slot_h[1,4]` + `ra/rb/ca/cb[1,2]` and the small term-weight memory.
- Prototype/check: `prototypes/bo_ng_cdx_worker2_4term_rank/rank_check_4term_painter.py`.
- Exact failed gate: pre-build algebraic feasibility check, `python The_400_Tasks/Task_by_Task/task115/prototypes/bo_ng_cdx_worker2_4term_rank/rank_check_4term_painter.py`.
- Result: row-output mode alone needs four active signed-distance components in the live painter family: slot constant `(1-s^2)`, row penalty `-r^2`, column penalty `-c^2`, and slot-linear column term `2s*c`. The column-output mode needs the symmetric slot-linear row term `2s*r`. A dual-orientation renderer therefore needs the three shared components plus separate row/column linear components; the first feasible term count in this family is 5, matching LIVE.
- Why dead: any 4-term graph that keeps the live cost-equivalent terminal-Einsum structure must reserve different row/column linear terms, leaving too few active components for at least one orientation. No ONNX was emitted; running visible/truth on a known-infeasible skeleton would be noise.
- Remaining hypothesis: only a different non-polynomial renderer or a different selector/orientation formulation outside the live painter family could beat 389; no task-local evidence currently points to one.
