live bar: cost 96, sha256 c63f5f1ce665a4b91771fcc5c4400f791f5a64c68684f9f1760fecd6fb7b7060.

Attempted route: remove Pad from the old GridSample crop and declare `[1,10,9,4]` as graph `output` (`prototypes/task211_gridcrop72_output9x4_probe.onnx`, static cost 80 with stale pad init still counted).

Failed gate: `python tools/ng_gate.py visible --task 211 --cand The_400_Tasks/Task_by_Task/task211/prototypes/task211_gridcrop72_output9x4_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.

Why dead: NeuroGolf visible/official comparison expects the padded `[1,10,30,30]` one-hot tensor from `convert_to_numpy(output)`; the logical 9x4 crop output is shape-wrong even though it represents the active grid.

Remaining hypothesis: only full-extent free-output formulations are viable; keep reducing the terminal Einsum basis/factor count instead of emitting a smaller output tensor.
