# sparse_einsum_initializers dead route

- Live/team bar used: cost 278, profile 278, sha256 `cc1f7c0947f4951eb199dfa07fa2c52d82c7b81da34717aa98a7f736a3a19f06` (`python tools/ng_current_bar.py --task 305 --json`, manifest authority, Vamshi, 2026-07-02T02:35:15Z).
- Attempted route: convert zero-heavy `Einsum` initializers (`rmap`, `cmap`, then small-zero siblings) to sparse initializers so params count stored NNZ instead of dense element count.
- Exact failed gates:
  - Naive sparse names: `python tools/ng_gate.py visible --task 305 --cand The_400_Tasks/Task_by_Task/task305/prototypes/probe_sparse_rmap_cmap.onnx` -> ORT load error `Node input 'safe_name_22' is not a graph input, initializer, or output of a previous node`.
  - Pre-sanitized sparse names: `python tools/ng_gate.py visible --task 305 --cand The_400_Tasks/Task_by_Task/task305/prototypes/probe_sparse_presafe_rmap_cmap.onnx` -> `[visible] 266/266 (100.00%)  PASS`; `score_network_v2` -> `None`; `profile_real_cost.py --arcgen 266` -> `ShapeInferenceError: Rank of input 3 (0) does not match the equation indices (2)`.
  - Coordinate-index sparse names: `python tools/ng_gate.py visible --task 305 --cand The_400_Tasks/Task_by_Task/task305/prototypes/probe_sparse_presafe_rmap_cmap_idx2d.onnx` -> `[visible] 266/266 (100.00%)  PASS`; `score_network_v2` -> `None`; `profile_real_cost.py --arcgen 266` -> same `Einsum` rank-0 shape-inference error.
- Why dead: ORT can run a pre-sanitized sparse initializer, but the official scorer calls `onnx.checker.check_model(full_check=True)` inside `calculate_memory`; ONNX shape inference treats the sparse `Einsum` operand as rank 0, so the candidate scores `None`.
- One remaining hypothesis: sparse may still be usable only for ops whose ONNX shape inference consumes sparse initializer dims correctly; for task305 final `Einsum`, do not retry sparse operands.
