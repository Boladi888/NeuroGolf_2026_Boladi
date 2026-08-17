# task301 CP-rank5 coefficient factorization

- Live/team-best bar: cost 430, SHA-256 `b01f14e6667d853308d75e797b8f3b702bb204524a57468e06e03aa72537555c`, source `python tools/ng_current_bar.py --task 301 --json` / team manifest.
- Effective local staged rung to beat: cost/profile cost 428, SHA-256 `cf6a3738253302d9118cb8959c7b2fedfab9d703c541f2d5b465ba2d0e2a11d4`, path `The_400_Tasks/Task_by_Task/task301/staged_candidates/task301_seek64bool_cost428/task301.onnx`.
- Attempted route: replace dense `u[4,4,4]` with CP factors `u_a[4,5]`, `u_b[4,5]`, `u_c[5,4]` inside the final free-output `Einsum`, saving 4 params while keeping the same memory profile.
- Prototype: `The_400_Tasks/Task_by_Task/task301/prototypes/task301_cp_rank5_u.onnx`, SHA-256 `a8abfe22c8e3f81bdb3ff75d926af99eec245d6003eb8d5d18856ac1d236c553`, score `(macs=0, memory=216, params=208)`, cost 424.
- Exact failed gate: `python tools/ng_gate.py visible --task 301 --cand The_400_Tasks/Task_by_Task/task301/prototypes/task301_cp_rank5_u.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Support refit check: a bounded squared-hinge refit over the full `(W=3..9, H=W..min(W+3,12), L=0..W)` support with the same rank-5 parameterization still left negative margins (`min=-2.6603778611697635`; F16-quantized `min=-443.85608244327886`), so this is not just a poor SVD-style initialization.
- Why dead: rank 5 is a 4-byte nominal saving but does not preserve visible behavior; F16 quantization makes the already-negative support margins much worse. No truth/AB gates were run after the visible-zero.
- One remaining hypothesis: an exact rank-6-or-higher factor is likely needed, but rank 6 costs 72 params for the factors versus 64 for dense `u`, so it cannot beat the 428 rung through this factorization shape.
