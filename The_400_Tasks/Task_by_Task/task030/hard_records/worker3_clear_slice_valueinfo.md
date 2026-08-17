# task030 clear-slice-value-info dead route

- Live bar used: 1054 profile cost from `python tools/ng_current_bar.py --task 30 --json`; live SHA-256 `88cbb94de8c671d33eef5aefee30ea2224755632b487d15dc5f68528598835e8`.
- Attempted route: remove the manual `patch_f`/`patch_u8` value_info entries from the dynamic 3x3 Slice in the team-best graph.
- Probe artifact: `The_400_Tasks/Task_by_Task/task030/prototypes/worker3_probe_clear_slice_valueinfo.onnx`.
- Gates: `python tools/ng_gate.py visible --task 30 --cand ...` -> `[visible] 266/266 (100.00%)  PASS`; `python -c "from tools._score_v2 import score_network_v2; ..."` -> `None None`; `python tools/agent_helpers/profile_real_cost.py --task 30 --cand ... --arcgen 266` -> `FAIL: score_network returned None on real-input trace`.
- Why dead: strict shape inference can infer most downstream tensors but not the dynamic Slice outputs; the scorer requires every non-output node output to appear in the inferred tensor map.
- Remaining hypothesis: a real win has to remove or replace the dynamic Slice itself, not hide its declarations.
