# task050 - no-active-bias direct155 route is dead

- Live bar at probe time: manifest/team-best cost 202, sha256 `893bdc8f733c8aa5ce9a03acb31f7dce7a41cc51a6ff1ad551043739b18d6b20`; stricter task-local direct-Einsum bar was 182 from `hard_records/direct_einsum_init_shaves.md`.
- Attempted route: remove the active-count feature and active^2 negative baseline from the direct output Einsum, leaving only marker count/sum/sumsq; expected params were `G[30,3] 90 + A[3,3,3] 27 + S[2,3,3] 18 + P[10,2] 20 = 155`.
- Failed gate: `python tools/ng_gate.py visible --task 50 --cand The_400_Tasks/Task_by_Task/task050/prototypes/task050_direct155_no_active_bias_probe.onnx` -> `[visible] 0/271 (0.00%)  FAIL`.
- Why dead: the sign-coded one-Einsum output needs a negative baseline on active rows/columns so ordinary background cells stay positive in channel 0; marker-only rows/columns evaluate to zero when no aligned markers exist.
- One remaining hypothesis: keep the active baseline but encode position degree 2 by reusing a `[1,x]` basis twice; this became `task050_direct176_g2_product.onnx`.
