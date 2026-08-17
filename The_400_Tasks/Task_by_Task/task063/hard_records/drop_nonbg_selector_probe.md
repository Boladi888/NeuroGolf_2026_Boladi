# drop_nonbg_selector_probe

- live bar: cost 90, SHA `df2edc223f90406977cf5aab0f116dd2f08e474f82ff22d67ed53dbb024c1754` from `python tools/ng_current_bar.py --task 63 --json` / manifest.
- attempted route: delete the 2-param `nonbg_selector[2]` from the one-node `Einsum`, letting the row/column count legs sum both signed channel modes instead of explicitly selecting the `(ch2 - ch8)^2` non-background mode.
- candidate: `The_400_Tasks/Task_by_Task/task063/prototypes/task063_drop_nonbg_selector_probe.onnx`, SHA `77fd1e1d23cde3be87a03ac4f905b4b4d3bfc5db7fe896c6a71d4648f55d9799`, static cost 88 (`memory=0`, `params=88`).
- exact failed gate: `python tools/ng_gate.py visible --task 63 --cand The_400_Tasks/Task_by_Task/task063/prototypes/task063_drop_nonbg_selector_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- why dead: the removed selector is load-bearing; the row/column count branches must isolate the non-background red/cyan mode before squaring. Summing both signed modes mixes background/fill channel algebra into the count polynomial and breaks every visible pair.
- one remaining hypothesis: any sub-90 route must avoid the 2-param selector by a different exact non-background count representation, or attack the larger `axis_spatial_basis[2,30]`; do not re-run this selector-drop family.
