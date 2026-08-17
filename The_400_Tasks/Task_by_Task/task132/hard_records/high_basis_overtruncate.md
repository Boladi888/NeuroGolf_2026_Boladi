# task132 high-basis overtruncation dead route

- Live bar: 516, SHA256 `0c6b6d8bbe081c188c1cb4b6f37cfca44e825a3e05fcad1b9be451feeca1b774` from `python tools/ng_current_bar.py --task 132 --json`.
- Attempted route: shrink `rxH` to only two real high buckets for rows/cols 0..11 and zero rows/cols >=12 (`task132_v22_h2_zero_tail.onnx`, static cost 384).
- Failed gate: `python tools/ng_gate.py visible --task 132 --cand The_400_Tasks/Task_by_Task/task132/prototypes/task132_v22_h2_zero_tail.onnx` -> `[visible] 131/267 (49.06%)  FAIL`.
- Why dead: coordinates in the 12..14 high bucket are still visible/load-bearing; only rows/cols 15..29 can be zero-tailed.
- One remaining hypothesis: keep three real high buckets but search for a different low-comparison decomposition.
