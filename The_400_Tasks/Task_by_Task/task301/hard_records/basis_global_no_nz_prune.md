# task301 basis/global/no-nz prune checks

- Live/team-best bar: cost 430, SHA-256 `b01f14e6667d853308d75e797b8f3b702bb204524a57468e06e03aa72537555c`, source `python tools/ng_current_bar.py --task 301 --json` / team manifest.
- Effective local staged rung to beat: cost/profile cost 428, SHA-256 `cf6a3738253302d9118cb8959c7b2fedfab9d703c541f2d5b465ba2d0e2a11d4`, path `The_400_Tasks/Task_by_Task/task301/staged_candidates/task301_seek64bool_cost428/task301.onnx`.
- Attempted route A: drop one current global feature from `[1, W, H, H*W]` or one spatial term from `[1, col, col^2 + 24*row^2, row]` and refit dense `u`.
- Exact failed check: support LP over the full generator support returned zero margin for every single drop: `drop one`, `drop W`, `drop H`, `drop HW`, and spatial `k=0..3` all reported `margin=-0.0`; no ONNX was emitted.
- Attempted route B: remove the `nz` selector by using raw `t/count` powers plus the existing `black` selector, including expanded cheap global sets such as `[1,W,H,H*W,d]`, `[1,W,H,H*W,d,d^2]`, and `[1,W,H,H*W,W^2,c0]`.
- Exact failed checks:
  - `python The_400_Tasks/Task_by_Task/task301/prototypes/build_task301_feature_search.py` -> `NOPE raw_t_t2_black`, `NOPE raw_t_t2_t3_black`, `NOPE count_count2_black`, and `NOPE count_count2_count3_black`, all with no positive margin.
  - Expanded squared-hinge support fits still left hundreds of wrong-side rows; examples: `t_t2_black + [1,W,H,H*W,d]` had `bad=646`, `min=-1.4098940769515975`; `t_t2_t3_black + [1,W,H,H*W,d,d2]` had `bad=676`, `min=-1.4931982297975352`; `count_count2_count3_black + [1,W,H,H*W,W2,c0]` had `bad=689`, `min=-1.3010185278283273`.
- Why dead: the current `nz`, all four global features, and all four spatial terms are load-bearing in this polynomial/free-output basis. Removing `nz` pollutes the black channel, and the tested cheap extra globals do not restore separability.
- One remaining hypothesis: a different spatial renderer, not a pruned/refit version of the current polynomial basis, would be required to move below the 428-byte staged rung.
