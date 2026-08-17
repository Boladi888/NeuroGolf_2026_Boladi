# task030 rank-1 left-vector trim dead route

- Live bar used: 1054 profile cost from `python tools/ng_current_bar.py --task 30 --json`; live SHA-256 `88cbb94de8c671d33eef5aefee30ea2224755632b487d15dc5f68528598835e8`.
- Attempted route: make the color-2/4 left vector rank-1 and remove the `left1_2d` Unsqueeze plus `unsqueeze_axis1` initializer.
- Probe artifact: `The_400_Tasks/Task_by_Task/task030/prototypes/worker3_trim_lefts_unsqueeze.onnx`.
- Failed gate: `python tools/ng_gate.py visible --task 30 --cand ...` -> `[visible] ERROR: Fail: [ONNXRuntimeError] : 1 : FAIL : Node (safe_name_39) Op (Add) [ShapeInferenceError] Incompatible dimensions`; `_score_v2` -> `None None`.
- Why dead: the position add needs `lefts_i8` shaped `[3,1]` so it can broadcast against the nine local offsets to `[3,9]`; rank `[3]` and `[9]` are incompatible.
- Remaining hypothesis: only a different position-index construction can remove the singleton axis; moving the Unsqueeze later gives back the same 1-byte tensor and does not beat the bar.
