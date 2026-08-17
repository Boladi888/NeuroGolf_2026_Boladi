# rank2_spatial_einsum

Live bar: cost 129, sha256 `0a68d150f2eb578e63cabb7e9e741bfdbf68ae40b15cbc3a94d244d15d58f842`, source `python tools/ng_current_bar.py --task 60 --json` (manifest, owner Vamshi, updated 2026-07-12T15:15:54Z).

Attempted route: keep the current one-node free-output `Einsum` family but lower the spatial rank from 3 to 2, with zeros outside the active width. Two probes were fitted from the decoded row rule: shared channel embedding (`R=2,T=3`, target cost 96) and separate input/output channel embeddings (`R=2,T=3`, target cost 126). Builder/probes: `The_400_Tasks/Task_by_Task/task060/prototypes/fit_task060_lowrank.py`, `task060_lowrank_r2_t3_shared.onnx`, `task060_lowrank_r2_t3_sep.onnx`.

Exact failed gates:
- Shared fit: optimizer best `ok=8936/9020`, `worst_margin=-0.957319`; `python tools/ng_gate.py visible --task 60 --cand The_400_Tasks/Task_by_Task/task060/prototypes/task060_lowrank_r2_t3_shared.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Separate fit: optimizer best `ok=8936/9020`, `worst_margin=-0.195255`; `python tools/ng_gate.py visible --task 60 --cand The_400_Tasks/Task_by_Task/task060/prototypes/task060_lowrank_r2_t3_sep.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.

Why dead: rank-2 spatial factors cannot separate the required left fill, gray divider, right fill, and blank-row background behavior inside the current final-output contraction. Both under-129 variants visibly fail, so no truth/profile/AB gates were run.

One remaining hypothesis: a different one-node final op family, such as a legal sparse/dictionary operator that keeps zero charged intermediates while supporting cross-channel gray routing, would be needed; rank-2 compression of Vamshi's `Einsum` scaffold should not be retried.
