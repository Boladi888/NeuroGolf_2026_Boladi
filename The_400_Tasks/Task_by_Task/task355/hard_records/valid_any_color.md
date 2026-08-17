# valid_any_color dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (`python tools/ng_current_bar.py --task 355 --json`, authoritative source `manifest`).
- Attempted route: replace the base-color validity test `cnt_u8 > 18` with `cnt_u8 > 0`, deleting the `u8eighteen` initializer for a static cost of 416.
- Candidate: `The_400_Tasks/Task_by_Task/task355/prototypes/task355_bo_ng_worker1_valid_any_color.onnx`, built by `build_bo_ng_worker1_tiny_probes.py`.
- Failed gate: `visible` -> `[visible] 130/267 (48.69%)  FAIL`.
- Why dead: the marker color must stay invalid; admitting every nonzero color pollutes the row/column split decode and can let marker-color geometry influence scores.
- Remaining hypothesis: a win needs a cheaper marker-color exclusion than the current `cnt > 18` scalar threshold, not a weaker validity threshold.
