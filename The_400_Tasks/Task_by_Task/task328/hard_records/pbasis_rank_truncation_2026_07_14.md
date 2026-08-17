# task328 hard record: P-basis rank truncation

- Live bar used: manifest/team-best cost 456, sha256 `6a7846c6e08678ba5abdf4044d4a906ca43c2c2ddce695bca66e3694574e137c` (Aditya, updated 2026-07-04T05:54:11Z).
- Attempted route: factor the live terminal-Einsum `P[2,3,30]` power basis into `PfA[2,3,k] * PfB[k,30]` inside the output Einsum (`prototypes/build_live456_pfactor.py`).
- Cost math: rank-4 scores `(macs=0, memory=28, params=392)`, cost 420; rank-3 scores `(0, 28, params=356)`, cost 384. Both are strict static wins if correct.
- Exact failed gate: `python tools/ng_gate.py visible --task 328 --cand The_400_Tasks/Task_by_Task/task328/prototypes/task328_live456_pfactor_rank4_balanced.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: the cost-saving factors require truncating `P.reshape(6,30)` below rank 5; the active 18 columns have six nonzero singular values `[5.825e23, 8.061e14, 7.744e13, 9.507e6, 3.606e-1, 3.091e-2]`. Rank-4 approximation destroys the logit signs on every visible case. Per visible-fail axiom, do not run truth/AB/profile/safety or rank-3/rank-4 rescalings.
- One remaining hypothesis: a different exact analytic parameterization that removes the board-size `Pow(Vbases,H)` family entirely might beat 456, but not by low-rank/truncated reparameterization of the current `P` basis.
