# task369 - centered affine one-code state core is dead

- Live bar: cost **1111**, sha256 `3583d772e665e5eb9bfad145de708f8e7f916af80642d92ef96da6e1cc3d8d85`, source `python tools/ng_current_bar.py --task 369 --json` / manifest.
- Attempted route: collapse the hidden state to one U8 feature with a nonzero hidden zero point, then use the final `ConvInteger` `(h - x_zero_point)` term as an implicit bias to remove the explicit complement/state channels.
- Probe: `The_400_Tasks/Task_by_Task/task369/prototypes/worker3_centered_state_core_probe.py`.
- Result artifact: `The_400_Tasks/Task_by_Task/task369/prototypes/worker3_centered_state_core_probe.json`.
- Failed gate: pre-build LP separability screen -> `any_all_colors_feasible=False`.
- Exact failure: centered candidates with no feature conflicts (`centered_80_m16_h128`, `centered_64_m8_h128`) were infeasible for colors `1`, `2`, `3`, and `5` under HiGHS; other screened affine codes had direct feature-label conflicts before LP.
- Why dead: a single centered affine U8 state code cannot feed one output-exempt `ConvInteger` head that separates all four required labels, so this family cannot produce a visible-clean below-1111 ONNX without adding more hidden channels and losing the memory saving.
- One remaining hypothesis: a broader symbolic/MILP search over non-affine scalar codes could be run if this task is revisited, but prior K=2 and signed-hidden probes already point to the three hidden U8 planes being load-bearing; the cleaner remaining lever is a legal sub-byte/BOOL spatial feature producer.
