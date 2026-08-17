# output_bool_tail dead route

- Live/team bar used: cost 278, profile 278, sha256 `cc1f7c0947f4951eb199dfa07fa2c52d82c7b81da34717aa98a7f736a3a19f06` (manifest/workbook, Vamshi, 2026-07-02T02:35:15Z).
- Attempted route: apply the BOOL-output lever to the current free-output polynomial.
- Exact failed gates:
  - Metadata-only BOOL: `python tools/ng_gate.py visible --task 305 --cand The_400_Tasks/Task_by_Task/task305/prototypes/probe_declared_bool.onnx` -> ORT load error `Type (tensor(bool)) of output arg (output) of node (output) does not match expected type (tensor(float))`.
  - Valid BOOL tail: `python tools/ng_gate.py visible --task 305 --cand The_400_Tasks/Task_by_Task/task305/prototypes/probe_bool_tail_greater.onnx` -> `[visible] 266/266 (100.00%)  PASS`; `score_network_v2` -> `(0, 36064, 215) 36279`; `profile_real_cost.py --arcgen 266` -> `static=36279 profile=36279 (memory 36064 + params 215; 266 real inputs)`.
- Why dead: the Vamshi graph already makes the F32 polynomial the graph output, so output memory is exempt. A real BOOL final op requires `Greater(poly, 0)`, which charges the full `[1,10,30,30]` F32 polynomial as an intermediate.
- One remaining hypothesis: only revisit if the polynomial sign test itself can be expressed as the graph-output BOOL op without a charged F32 parent; direct metadata changes and tail-thresholding are dead.
