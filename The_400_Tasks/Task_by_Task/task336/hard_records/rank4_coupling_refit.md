# rank4_coupling_refit

- Live/team-best bar: cost 400, static/profile 400 (`181` memory + `219` params), sha256 `418eee6e3107fe8c2c28ab6ecf0464fc47e7b07f007446ba5d25c1d83fcabc89`, source `Team_Submission/Backend/manifest.csv`, owner Vamshi, updated `2026-07-02T02:35:15Z`.
- Attempted route: keep the full five-row quadratic feature bank including `z2coef`, but refit the final CP coupling from rank 5 to rank 4 by replacing `U[5,5]`, `Vfac[5,5]`, and `Q[2,5,2]` with `U[5,4]`, `Vfac[5,4]`, and `Q[2,4,2]`.
- Builder: `The_400_Tasks/Task_by_Task/task336/prototypes/build_rank4_coupling_fit.py`.
- Probe artifact: `The_400_Tasks/Task_by_Task/task336/prototypes/rank4_coupling_fit.onnx`.
- Probe sha256: `0995e4596b266ef5986f69e2dbe5f535b375c2dc33a5db41cbc7e28d1271c0b5`.
- Static score: `(0, 181, 205)`, cost `386`, strict-below 400 before gate.
- Failed gate: `python tools/ng_gate.py visible --task 336 --cand The_400_Tasks/Task_by_Task/task336/prototypes/rank4_coupling_fit.onnx` -> `[visible] 0/31 (0.00%)  FAIL`.
- Why dead: the live final tensor needs separate rank capacity for horizontal background, vertical background, and gray passthrough. The best capped rank-4 fit kept the large background slices nearly exact but left a gray-passthrough residual; thresholded visible outputs all diverged. Algebraic check also shows the relevant effective matrices have combined row/column span rank 5, so a true four-wide feature-basis refit cannot preserve the live render exactly.
- One remaining hypothesis: only a non-rank-reduction micro-golf that preserves the five-row feature bank and rank-5 coupling could beat 400; no known candidate route remains below the live bar.
