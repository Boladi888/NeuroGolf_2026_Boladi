live bar: 2360, sha f12c26da73d8ad2a657625cc50a07fa1faac4f95694b6195ffb763355439e9cf

attempted route: replace the per-side extent chain `r0 + r0*r1 + r0*r1*r2` with cheaper extent reducers.

failed gates:
- `build_task365_col_axis_reducesum_probe.py` -> ONNX shape inference rejected `ReduceSum` on `tensor(uint8)`.
- `python tools/ng_gate.py visible --task 365 --cand The_400_Tasks/Task_by_Task/task365/prototypes/task365_col_axis_sum3_probe.onnx` -> `[visible] 227/266 (85.34%)  FAIL`.

why dead: UINT8 ReduceSum is not scorer-legal, and plain `r0+r1+r2` overcounts when samples after the first zero hit another box beyond a gap. The multiplicative prefix chain is doing required stop-after-gap work.

one remaining hypothesis: a cheaper exact prefix-length primitive would need a legal UINT8 min/prefix operator or a different extent representation, not simple summation.
