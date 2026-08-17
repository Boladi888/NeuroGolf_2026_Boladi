# no_height_rem_gate dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (OnnxlyFans workbook + manifest).
- Attempted route: delete row/height log decode and score from width plus `cnt % width` only.
- Failed gates:
  - `task355_bo_ng_worker1_rem_nonzero_gate.onnx`, static estimate 374 -> `[visible] 229/267 (85.77%)  FAIL`.
  - `task355_bo_ng_worker1_no_height_all_valid.onnx`, static estimate 354 -> `[visible] 86/267 (32.21%)  FAIL`.
- Why dead: full-row marker counts have `cnt % width == 0` but should score `width`; no-marker rectangles also have `cnt % width == 0` but should score zero, so the row/height comparison is load-bearing.
- Remaining hypothesis: derive the `q < height` predicate from existing count/width/side tensors cheaper than the current row log path; simple max-q heuristics missed at least 54/267 visible+arcgen cases.
