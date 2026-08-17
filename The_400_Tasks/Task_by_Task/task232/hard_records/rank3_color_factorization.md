# task232 hard record - rank3 color factorization

- Live/team-best bar: cost 205, profile_cost 205, sha256 `f54c7425eeb6a1da3977db56be71af2cadeafdc729064e47ce402f3cc5cf80e5`.
- Bar source: `NG_OnnxlyFans_TaskStatus.xlsx` row task232 and `Team_Submission/Backend/manifest.csv` row task232, both updated `2026-07-02T02:35:15Z`.
- Attempted route: reduce the 205-param one-Einsum color basis below the bar by replacing the shared `R[10,4]` color basis with rank-3 output/input color bases. Concrete probe: `The_400_Tasks/Task_by_Task/task232/prototypes/probe_svd_color_rank3_cost204.onnx`.
- Exact failed gate: `ng_gate visible --task 232 --cand probe_svd_color_rank3_cost204.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the 204-param SVD color compression loses the background-left-of-seed margin; a separate fixed rank-3 hand color basis with current `Q/S` was also LP-infeasible over all generator row states.
- One remaining hypothesis: only revisit color rank if a genuinely different threshold formulation supplies separate background, gray, and seed-hue degrees without using the current `R/Q/M/S` decomposition.
