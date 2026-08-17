# task245 area-derived width route

- Live bar: manifest cost 940, sha256 `1a7e75f1764b6283cf5a4cb8030bdfd272ffc43bad23737d94f493c73eab3ca5`; local staged rung to beat: 939 (`task245_expdecodeavoidance_cost939`).
- Attempted route: `prototypes/build_task245_area_width.py` / `task245_area_width_probe.onnx`; deleted the separate `col_dim_code_* -> w_f` weighted-width chain and derived `w_f = area_h / (h_f - 0.5) + 1`.
- Failed gate: `python tools/ng_gate.py visible --task 245 --cand The_400_Tasks/Task_by_Task/task245/prototypes/task245_area_width_probe.onnx` -> `[visible] 216/267 (80.90%)  FAIL`.
- Why dead: the derived width under-pads several W=8..10 examples; first miss was a 9x9 official where ORT produced a 29-wide output and the decoded grid lost the rightmost background column.
- Remaining hypothesis: an integer-floor `H` then `area/H + epsilon` derivation might be exact, but it needs extra Cast/Div/Add tensors and likely erases the small width-chain saving; do not retune this family without a lower-cost integer divide construction.
