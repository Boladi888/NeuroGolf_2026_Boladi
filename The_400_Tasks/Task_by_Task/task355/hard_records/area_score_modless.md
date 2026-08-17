# area_score_modless dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (`python tools/ng_current_bar.py --task 355 --json`, authoritative source `manifest`).
- Attempted route: keep the exact row/column decode, but replace `Mod -> Div -> Greater -> Sub -> Where` scoring with direct U8 `area = height * width; holes = area - cnt; scores = valid ? holes : 0`. Static estimate was about 387 if exact, because it deletes three U8/BOOL `[1,10,1,1]` tensors.
- Candidate: `The_400_Tasks/Task_by_Task/task355/prototypes/task355_bo_ng_worker1_area_score.onnx` built by `build_bo_ng_worker1_area_score.py`.
- Failed gate: `visible` -> `[visible] 264/267 (98.88%)  FAIL`.
- Why dead: the direct area-minus-count score is exact for the clean ARC-GEN 2x2 unique-color generator, but visible officials include repeated base colors / non-2x2 row layouts. For those, the decoded `height * width` is a union/bounding estimate; `area - cnt` underflows or overcounts, while the live modulo/height ladder intentionally preserves the visible quirks.
- Related pre-gate probe: `probe_no_row_score_rules.py` tested rowless residual variants; best rule was also `264/267`, with the same repeated-color ambiguity.
- Remaining hypothesis: a cost-neutral safety rewrite can replace `Div(cnt,width) -> Greater(height,q)` with `Greater(Mul(height,width), cnt)` to avoid dummy divide-by-zero, but a staged win still needs a separate strict byte shave below 417 or a cheaper exact modulo-with-positive test.
