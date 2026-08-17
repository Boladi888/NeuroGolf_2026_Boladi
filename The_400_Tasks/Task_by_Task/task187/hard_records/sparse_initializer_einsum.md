# Sparse initializer Einsum route - scorer rejection (2026-07-12)

- Live bar: `2028`, SHA-256 `da5d9b3cc90ccbc82e3147aef22c969b5f29b6ad7d8b344c5a42274a8ab39596`, from `python tools/ng_current_bar.py --task 187 --json` (`manifest`, reconciled with workbook and `Team_Best_Onnx/task187.onnx`).
- Attempted route: replace dense `Einsum` initializer banks with `graph.sparse_initializer` to exploit the sparse logical tensors; builder `The_400_Tasks/Task_by_Task/task187/prototypes/build_sparse_init_probes.py`.
- Exact failed gate: local scorer/checker rejection before visible. `score_network_v2` returned `None` for `task187_sparse_vw1_probe.onnx` and `task187_sparse_all_probe.onnx`; `onnx.checker.check_model(..., full_check=True)` raised `ShapeInferenceError` because the sparse `Einsum` input rank was inferred as `0` instead of the equation rank.
- Why dead: adding value_info to rescue sparse initializer shapes would charge the dense F32 tensor memory, wiping out the params saving (`v_w1` alone would add an 800-byte memory tensor; `v_w2` would add 7200 bytes). Sparse initializer inputs are therefore not a valid sub-2028 path here.
- Remaining hypothesis: only a dense/factored operand that stays inside the terminal output `Einsum` can be safe; sparse ONNX storage is scorer-incompatible for this model family.
