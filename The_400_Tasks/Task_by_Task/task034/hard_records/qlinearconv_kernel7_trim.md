# task034 hard record: qlinearconv_kernel7_trim

- live bar: 1334 by `python tools/ng_current_bar.py --task 34 --json`, sha256 `642a13a7dd051aff43afb6a4d6b9217d378d42f747d099d80ed8b15c30e3a268`; `_score_v2` on `Team_Best_Onnx/task034.onnx` = memory 1129 + params 205.
- attempted route: trim current-best diagonal spread kernels `v_w14`/`v_w15` from `[1,1,8,8]` to `[1,1,7,7]` and reduce end pads to preserve `[1,1,9,9]` outputs; probe `prototypes/task034_kernel7_probe.onnx`.
- exact failed gate: `python tools/ng_gate.py visible --task 34 --cand The_400_Tasks/Task_by_Task/task034/prototypes/task034_kernel7_probe.onnx` -> `[visible] 117/267 (43.82%)  FAIL`.
- why dead: despite a real static save (`_score_v2` 1304; `ng_graph_cost_diff` delta -30 params from `v_w14`/`v_w15`), official visible examples require the full 8-cell diagonal reach. Do not build aligned 7x7 variants in this representation family after this visible miss.
- one remaining hypothesis: a distinct rewrite would need to avoid dense diagonal kernels entirely, e.g. factor the ray write into a final-output contraction without materializing extra 9x9 planes; not a kernel-trim variant.
