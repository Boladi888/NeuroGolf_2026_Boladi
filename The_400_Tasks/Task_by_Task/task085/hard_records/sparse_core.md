# sparse_core route

- Live/team-best bar used: cost 136, sha256 c2d0c80df4932d3b3c782185af15b9d2749d2f6caee9c1fbfe9a5e74895873b7, source workbook + team manifest refreshed 2026-07-03.
- Attempted route: preserve Vamshi's one-Einsum `task085_period2_colorpoly_redfirst_score20` graph and replace dense `channel_core` [2,4,4] (32 params, 9 nonzero values) with an ONNX sparse initializer.
- Exact failed gate: `python The_400_Tasks/Task_by_Task/task085/prototypes/build_task085_sparse_core_probe.py` failed at `onnx.checker.check_model(..., full_check=True)` with `ShapeInferenceError: Rank of input 6 (0) does not match the equation indices (3)`.
- Why dead: strict ONNX shape inference does not expose the sparse initializer's `[2,4,4]` rank to `Einsum`, so `score_network_v2` would reject before cost/profile gates. Adding dense value_info would charge the recovered 32-float tensor as memory and loses the 136 bar.
- One remaining hypothesis: a non-sparse CP/Tucker factorization of `channel_core` with no intermediate tensors, but mode-2 unfolding rank is 4, so a rank-3 CP save is impossible.
