# crop16_raw dead route

- Live/team bar used: cost 278, profile 278, sha256 `cc1f7c0947f4951eb199dfa07fa2c52d82c7b81da34717aa98a7f736a3a19f06` (manifest/workbook, Vamshi, 2026-07-02T02:35:15Z).
- Attempted route: emit the free-output polynomial directly as a raw `[1,10,16,16]` canvas, deleting the 30-row `xfeat` tail and the active-mask input operand.
- Exact failed gate: `python tools/ng_gate.py visible --task 305 --cand The_400_Tasks/Task_by_Task/task305/prototypes/probe_crop16_raw.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: official verifier encodes expected outputs as `[1,10,30,30]`; a raw 16x16 ONNX output is shape-wrong. A Pad sibling would make the 16x16 polynomial a charged F32 intermediate, which is far above the 278 bar.
- One remaining hypothesis: only revisit crop if a future representation writes the final 30x30 output directly while using 16x16 row/col factors without materializing a cropped canvas.
