# positive_score_output dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (`python tools/ng_current_bar.py --task 355 --json`, authoritative source `manifest`).
- Attempted route: delete `score_max = ReduceMax(scores)` and `best_oh = Equal(scores, score_max)`, then feed U8 `scores` directly to the final `Pad`; static estimate 406. The bool `Greater(scores, 0)` sibling is decode-equivalent and was not gated after this fail.
- Failed gate: `visible` on `task355_bo_ng_cdx_worker2_scores_pad.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: non-winning rectangles often have positive scores; NeuroGolf output thresholding would mark multiple color channels, so the max-selection tail is load-bearing.
- Remaining hypothesis: only a cheaper exact argmax/equal tail than `ReduceMax+Equal` can reopen this, not positive thresholding.
