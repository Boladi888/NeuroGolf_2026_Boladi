# task328 hard record: color-coupling rank truncation

- Live bar used: manifest/team-best cost 456, sha256 `6a7846c6e08678ba5abdf4044d4a906ca43c2c2ddce695bca66e3694574e137c` (Aditya, updated 2026-07-04T05:54:11Z).
- Attempted route: replace the live terminal-Einsum color coupling `C[10,10]` with `CfA[10,k] * CfB[k,10]` inside the same output Einsum (`prototypes/build_live456_cfactor.py`).
- Cost math: rank-4 scores `(macs=0, memory=28, params=408)`, cost 436, a nominal 20-byte win over live.
- Exact failed gate: `python tools/ng_gate.py visible --task 328 --cand The_400_Tasks/Task_by_Task/task328/prototypes/task328_live456_cfactor_rank4.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: `C` has rank 9 with singular values `[3.1623, 1, 1, 1, 1, 1, 1, 1, 1, ~0]`. Any cost-saving rank (`k <= 4`) discards exact color identity directions and destroys visible outputs. `k=5` only ties dense params (`20*k = 100`) and is still approximate; exact `k=9` is 180 params and loses.
- One remaining hypothesis: any sub-456 color route must avoid the dense `C` coupling by changing the analytic output formula, not by low-rank truncating the current color map.
