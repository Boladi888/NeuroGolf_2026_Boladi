live_bar: cost 1161, sha256 ca47e6f02338f613e92d1a6e9ab87ea4cfffca5192ac7360a322e462aa41d65d, source `python tools/ng_current_bar.py --task 281 --json` manifest/Sho 2026-07-09T01:09:17Z

attempted_route: Rewrote the Sho 1161 team-best graph's 13 zero-heavy dense initializer tables as SparseTensorProto entries, preserving topology and tensor values. Builder/probe: `The_400_Tasks/Task_by_Task/task281/prototypes/build_task281_sparse_inits_probe.py` -> `task281_sparse_inits_probe.onnx`.

failed_gate: `onnx.checker.check_model(full_check=True)`, `onnx.shape_inference.infer_shapes(strict_mode=True)`, and `tools._score_v2.score_network_v2` all failed before visible. Exact error family: `ShapeInferenceError` on `Einsum` nodes, e.g. `Rank of input 2 (0) does not match the equation indices (2)`; scorer returned `None`.

why_dead: Official strict shape inference does not treat sparse initializers as ranked dense `Einsum` operands, so the tempting dense-param 537 -> sparse-nnz 157 rewrite is unscoreable. This matches the sparse-op-input trap in `core.md`/`traps.md`.

remaining_hypothesis: A legal param shave must stay dense or move structure into the final free `Einsum`; sparse initializer layout should not be retried for task281 unless the scorer/shape-inference contract changes.
