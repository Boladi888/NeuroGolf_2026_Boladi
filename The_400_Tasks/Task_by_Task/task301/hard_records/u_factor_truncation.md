# task301 u_factor_truncation hard record

Live bar:
- Authoritative live/team bar: cost 430, sha256 `b01f14e6667d853308d75e797b8f3b702bb204524a57468e06e03aa72537555c`, source `python tools/ng_current_bar.py --task 301 --json` / manifest.
- Effective staged rung to beat: cost/profile cost 428, sha256 `cf6a3738253302d9118cb8959c7b2fedfab9d703c541f2d5b465ba2d0e2a11d4`, path `staged_candidates/task301_seek64bool_cost428/task301.onnx`.

Attempted route:
- Replaced dense `u` `[4,4,4]` with SVD factors `u_m[f,g,s]` and `u_c[s,k]` inside the final free-output Einsum.
- Rank prices: rank1 cost 384, rank2 cost 404, rank3 cost 424, rank4 cost 444. Rank4 is not useful because factor overhead exceeds dense `u`.

Exact failed gate:
- `python tools/ng_gate.py visible --task 301 --cand The_400_Tasks/Task_by_Task/task301/prototypes/task301_u_rank1.onnx` -> `[visible] 0/266 (0.00%)  FAIL`
- `python tools/ng_gate.py visible --task 301 --cand The_400_Tasks/Task_by_Task/task301/prototypes/task301_u_rank2.onnx` -> `[visible] 0/266 (0.00%)  FAIL`
- `python tools/ng_gate.py visible --task 301 --cand The_400_Tasks/Task_by_Task/task301/prototypes/task301_u_rank3.onnx` -> `[visible] 0/266 (0.00%)  FAIL`

Why dead:
- `u` is full rank in all mode flattenings. Truncating the small singular direction is enough to make every visible case fail, so approximate low-rank coefficient compression is not viable.
- No truth/AB gates were run on these visible-zero probes.

One remaining hypothesis:
- Exact structural savings would need to remove or re-express `v`/`basis` spatial terms, not approximate `u`.
