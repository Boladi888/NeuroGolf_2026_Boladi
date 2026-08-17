# task232 hard record - two-term factorization

- Live/team-best bar: cost 205, profile_cost 205, sha256 `f54c7425eeb6a1da3977db56be71af2cadeafdc729064e47ce402f3cc5cf80e5`.
- Bar source: `NG_OnnxlyFans_TaskStatus.xlsx` row task232 and `Team_Submission/Backend/manifest.csv` row task232, both updated `2026-07-02T02:35:15Z`.
- Attempted route: flatten the live kernel `F[o,c,x,j]`, keep the best rank-2 SVD approximation, and project it back through the same `R[10,4]` and `Q[30,3]` bases. Concrete probe: `The_400_Tasks/Task_by_Task/task232/prototypes/probe_t2_svd_termrank_cost180.onnx`.
- Exact failed gate: `ng_gate visible --task 232 --cand probe_t2_svd_termrank_cost180.onnx` -> `[visible] 11/266 (4.14%)  FAIL`.
- Why dead: the third term separates the constant/background, right-of-seed ramp, and parity/same-color effects; rank-2 drops the background-left-of-seed positive margin and fails visible immediately.
- One remaining hypothesis: a win below 205 would need a non-SVD threshold formulation, not another approximation of the current three-term kernel.
