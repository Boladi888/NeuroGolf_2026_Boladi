# task400 dead route - positive-step 180-source slice without reversal

- **Live bar:** 1084, SHA-256 `49f88a9d910b85ab712b0e76cf65e3b834b7606ea8c02f56f325a6dc6365fac4`, owner Aditya. Source: `python tools/ng_current_bar.py --task 400 --json`; authoritative source `manifest`.
- **Attempted route:** remove the `steps_180_i32` Slice initializer by using a positive-step source window. The ramp `0.844 - 0.04*i` yields `19 - y0` / `19 - x0`, so the graph can Slice forward and price at 1082 (`memory 1032 + params 50`).
- **Exact failed gate:** `python tools/ng_gate.py visible --task 400 --cand The_400_Tasks/Task_by_Task/task400/prototypes/task400_worker2_positive_step_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- **Why dead:** the `steps=[-1,-1]` reversal is semantically load-bearing. A forward source window reads the 180-degree counterpart in source orientation, not in the hole orientation, so every visible pair fails despite the 2-byte param saving.
- **One remaining hypothesis:** only a different final-output primitive that writes the reversed 5x5 source patch directly into the free graph output, without materializing `selected_5x5` or carrying a `steps` initializer, could break 1084. No known ONNX op does this under the current scorer.
