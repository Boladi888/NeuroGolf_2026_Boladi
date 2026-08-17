# task315 sparse-B terminal Einsum probe

- Live bar used: 146, SHA-256 `013dbc694b331f0991b4e4f9a637a01c318e96a0f89289052e15bf526280ebb7`, source `python tools/ng_current_bar.py --task 315 --json` manifest result, owner Vamshi.
- Attempted route: keep the current one-node terminal `Einsum`, but replace dense `B[3,30]` with a sparse initializer storing only nonzero entries.
- Candidate: `The_400_Tasks/Task_by_Task/task315/prototypes/task315_sparse_b_probe.onnx`.
- Exact failed gates: visible and static score/checker. `ng_gate.py visible` failed with `Invalid model. Node input 'safe_name_6' is not a graph input, initializer, or output of a previous node.` `score_network_v2` returned `None None`; `onnx.checker.check_model(full_check=True)` failed with `[ShapeInferenceError] Rank of input 5 (0) does not match the equation indices (2)`.
- Why dead: ONNX shape inference does not expose sparse initializer rank to `Einsum`, and the NeuroGolf sanitizer does not preserve the sparse initializer name, matching the project sparse-initializer trap. Adding dense `value_info` for `B` would charge the dense tensor and exceed the 146 bar.
- Remaining hypothesis: only a smaller dense single-`Einsum` factorization can beat 146; sparse initializer compression is not a viable route.
