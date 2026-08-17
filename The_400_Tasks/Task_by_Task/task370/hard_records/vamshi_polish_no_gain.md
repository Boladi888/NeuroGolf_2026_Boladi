# task370 hard record — mechanical polish on Vamshi 3607

- Live bar: cost `3607`, profile `3607`, sha256 `c5a03117601f462483d58aa02b9006f792e53b9c77502fd501481d1ad84d89c0`, source `python tools/ng_current_bar.py --task 370 --json` (`authoritative_source=manifest`, owner Vamshi, updated `2026-07-12T15:15:54Z`).
- Attempted route: run the standard `tools.polish_candidate.polish()` pipeline on the 3607 Team_Best ONNX with task-local work dir `The_400_Tasks/Task_by_Task/task370/prototypes/polish_tmp`.
- Artifact: `The_400_Tasks/Task_by_Task/task370/prototypes/task370_vamshi_polished_probe.onnx`.
- Exact result: original `{'macs': 0, 'memory': 3108, 'params': 499, 'cost': 3607}`; final `{'macs': 0, 'memory': 3108, 'params': 499, 'cost': 3607}`. Steps: `onnxsim` no-savings delta `0`; `safe_opt`, `const_to_init`, `clip_mul_to_min`, `scalar_broadcast_ones`, `slice_axis_collapse` all no-op/no-savings.
- Why dead: the current Vamshi graph is already mechanically minimized for the standard safe polish passes; this route cannot create a sub-3607 candidate.
- One remaining hypothesis: a win must be a semantic/structural rewrite, not generic cleanup.
