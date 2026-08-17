# task132 drop_M dead route

- Live bar: 516, SHA256 `0c6b6d8bbe081c188c1cb4b6f37cfca44e825a3e05fcad1b9be451feeca1b774` from `python tools/ng_current_bar.py --task 132 --json`.
- Attempted route: remove `M[10,10]` color map from Sho's one-node Einsum parent and emit the contracted color axis directly (`task132_v21_drop_M.onnx`, static cost 416).
- Failed gate: `python tools/ng_gate.py visible --task 132 --cand The_400_Tasks/Task_by_Task/task132/prototypes/task132_v21_drop_M.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: the color map is not just a rename; it carries the background suppression / channel semantics needed by the decoded output.
- One remaining hypothesis: replace `M` with a genuinely different sign-preserving low-rank channel formulation; do not retry direct removal.
