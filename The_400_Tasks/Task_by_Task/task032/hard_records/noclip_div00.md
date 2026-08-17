# task032 noclip_div00 hard record

- Live bar used: cost 403, sha256 `e3ca4a62f12fe88ff9852a6159c24108f8438d030eff62ecf6a74267146754ce` from `python tools/ng_current_bar.py --task 32 --json` / team manifest.
- Attempted route: delete `Clip(max(k,1))` in Sho's 403 graph and feed `Div(v_t3, v_t4)` directly, accepting `UINT8 Div(0,0)` on empty columns if ORT made it harmless.
- Candidate: `The_400_Tasks/Task_by_Task/task032/prototypes/task032_noclip_div00.onnx`; builder `build_task032_noclip_div00.py`.
- Failed gate: `python tools/ng_gate.py visible --task 32 --cand The_400_Tasks/Task_by_Task/task032/prototypes/task032_noclip_div00.onnx` exited code 1 with no stdout/stderr; direct ORT `sess.run` and `score_network_v2` also exited code 1 with no stdout/stderr.
- Why dead: empty columns are legal in ARC-GEN (`active=[]`); the denominator guard is load-bearing. Removing it triggers ORT/runtime failure before a visible PASS can exist.
- One remaining hypothesis: avoid the guard only if a different packed encoding maps empty columns to a nonzero denominator without changing the color quotient; the simple Conv-bias variant is algebraically invalid because the bias changes `packed / count`.
