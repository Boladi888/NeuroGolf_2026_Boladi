# task117 hard record — sparse `v_w2` Einsum initializer is DEAD

- **Live bar:** 2380, source `python tools/ng_current_bar.py --task 117 --json` / authoritative manifest, owner Sho, sha `79dcba6fb80f7b66ba3f4ed64b9c7d3e4e5358a02972745772b1c6fdc3a0f879`, updated 2026-07-09T01:09:17Z.
- **Attempted route:** keep Sho's one-node free-output `Einsum`, but store dominant sparse operand `v_w2[6,15,15]` as `SparseTensorProto`. Dense params 1350; nnz 78; hypothetical saving -1272 if the scorer accepted it.
- **Prototype:** `The_400_Tasks/Task_by_Task/task117/prototypes/build_task117_sparse_vw2_probe.py`
- **Probe artifact:** `The_400_Tasks/Task_by_Task/task117/prototypes/task117_sparse_vw2_probe.onnx`, sha `feeeef888e81ab435e4786ed67beddde18dfd4180834d1cbc907727ecc395b48`.
- **Exact failed gate:** static score gate. `score_network_v2(...) -> None`. Direct checker detail: `onnx.checker.check_model(full_check=True)` and `onnx.shape_inference.infer_shapes(strict_mode=True)` both fail with `ShapeInferenceError`: `Rank of input 45 (0) does not match the equation indices (3)`.
- **Why dead:** sparse initializers are not visible to strict shape inference as rank-3 Einsum operands under the official scorer path. Raw storage sparsity cannot be converted into credited params for this graph.
- **Remaining hypothesis:** only dense equation/shape factorization can reduce `v_w2`; sparse storage, dtype narrowing, and Constant-node generation are not viable for this operand.
