# CP9 final-channel map factor route - visible fail (2026-07-12)

- Live bar: `2028`, SHA-256 `da5d9b3cc90ccbc82e3147aef22c969b5f29b6ad7d8b344c5a42274a8ab39596`, from `python tools/ng_current_bar.py --task 187 --json` (`manifest`, reconciled with workbook and `Team_Best_Onnx/task187.onnx`).
- Attempted route: replace dense final `v_w1[2,10,10]` (200 params) with a 9-term CP/sign factor inside the same terminal `Einsum` (`sm,sk,sx`, 198 params), candidate `The_400_Tasks/Task_by_Task/task187/prototypes/task187_cp9_final_map_probe.onnx`.
- Exact failed gate: `python tools/ng_gate.py visible --task 187 --cand The_400_Tasks/Task_by_Task/task187/prototypes/task187_cp9_final_map_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: exact CP rank of the original final map is at least 10 from the channel-input unfolding, so a rank-9 shave must rely on omitted/extra logits being semantically irrelevant. They are not; visible failure is total.
- Remaining hypothesis: a different final-output algebra that eliminates the dense channel map entirely could still save bytes, but rank-9/less CP-style factoring of `v_w1` is ruled out.
