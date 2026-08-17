# task350 sparse Einsum initializers dead route

- Live bar: 533, SHA-256 `4303e186062a423a999d6323dcc90aa58de1249380ab8f1db3fff82b7795babc` (`python tools/ng_current_bar.py --task 350 --json`, team manifest / Team_Best).
- Attempted route: convert the 507-param one-node free-output `Einsum` factor banks to ONNX sparse initializers; dense NNZ is 175, so this would have beaten the prompt's 239 target if accepted.
- Exact failed gate: static build/check/scorer before visible. `onnx.checker.check_model(full_check=True)` failed with `ShapeInferenceError (op_type:Einsum): Rank of input 0 (0) does not match the equation indices (2)`, and `tools._score_v2.score_network_v2(...)` returned `None`.
- Why dead: ONNX strict shape inference does not expose sparse initializer rank to `Einsum`; this matches the project sparse-initializer trap. It cannot reach visible/truth gates.
- One remaining hypothesis: dense factor-rank reduction inside the terminal `Einsum`; sparse initializer storage is not viable for this graph.
