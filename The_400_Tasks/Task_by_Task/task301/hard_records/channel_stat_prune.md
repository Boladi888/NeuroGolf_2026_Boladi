# task301 channel_stat_prune hard record

Live bar:
- Authoritative live/team bar: cost 430, sha256 `b01f14e6667d853308d75e797b8f3b702bb204524a57468e06e03aa72537555c`, source `python tools/ng_current_bar.py --task 301 --json` / manifest.
- Effective staged rung to beat: cost/profile cost 428, sha256 `cf6a3738253302d9118cb8959c7b2fedfab9d703c541f2d5b465ba2d0e2a11d4`, path `staged_candidates/task301_seek64bool_cost428/task301.onnx`.

Attempted route:
- Refitted the free-output Einsum with smaller channel feature sets derived from current `chan_feat = [nz, t_nz, t_sq, black]`.
- LP separability over generator support `W=3..9, H=W..min(W+3,12)` had zero positive margin for every one-stat and two-stat removal.
- LSQ probe costs: drop `nz` 392, drop `t_sq` 372, drop `t` 392, drop `black` 382, two-feature probes 316-356.

Exact failed gate:
- `python tools/ng_gate.py visible --task 301 --cand The_400_Tasks/Task_by_Task/task301/prototypes/task301_drop_nz_t_t2_black_lstsq.onnx` -> `[visible] 0/266 (0.00%)  FAIL`
- `python tools/ng_gate.py visible --task 301 --cand The_400_Tasks/Task_by_Task/task301/prototypes/task301_drop_t2_nz_t_black_lstsq.onnx` -> `[visible] 0/266 (0.00%)  FAIL`
- `python tools/ng_gate.py visible --task 301 --cand The_400_Tasks/Task_by_Task/task301/prototypes/task301_drop_t_nz_t2_black_lstsq.onnx` -> `[visible] 0/266 (0.00%)  FAIL`
- `python tools/ng_gate.py visible --task 301 --cand The_400_Tasks/Task_by_Task/task301/prototypes/task301_drop_black_nz_t_t2_lstsq.onnx` -> `[visible] 0/266 (0.00%)  FAIL`

Why dead:
- `nz`, `t_nz`, `t_sq`, and `black` are all load-bearing in this fixed basis/Einsum formulation. Removing any one stat destroys visible correctness before truth/AB.
- No full/truth/AB gates were run on these visible-zero probes.

One remaining hypothesis:
- A different final basis, not just a smaller channel-stat vector under the current `basis/v/u` polynomial form, would be needed to approach the prompt's ~300 floor.
