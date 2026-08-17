# task128 sparse_initializer_zero_storage

- Live bar: cost 104, SHA256 `ccdb0c525b1f0ba110466e0c36a9f4eedb963c098794a264c725a71595205798`, source `python tools/ng_current_bar.py --task 128 --json` manifest authority.
- Attempted route: exact clone of the live one-node free-output `Einsum`, replacing zero-heavy dense initializers with `SparseTensorProto` sparse initializers.
- Builder: `The_400_Tasks/Task_by_Task/task128/prototypes/build_task128_sparse_inits_probe.py`
- Probe: `The_400_Tasks/Task_by_Task/task128/prototypes/task128_sparse_inits_probe.onnx`, nominal stored values 85 vs dense 104, SHA256 `2b7a768df4f35fcd58a81501235e5da0481a9b949e91413d7cc1408896b92127`.
- Exact failed scorer/gate:
  - `score_network_v2(...)` -> `None`
  - `python tools/ng_gate.py visible --task 128 --cand The_400_Tasks/Task_by_Task/task128/prototypes/task128_sparse_inits_probe.onnx` -> `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_0' is not a graph input, initializer, or output of a previous node.`
- Why dead: the official sanitizer renames node inputs but not `graph.sparse_initializer` value names, leaving the `Einsum` operands dangling. This matches traps/core sparse-initializer doctrine; sparse zero-storage is unstageable even though the nominal NNZ count is lower.
- One remaining hypothesis: none inside sparse storage unless the official sanitizer/scorer changes. A real sub-104 candidate needs a non-sparse algebraic formulation with fewer dense operand elements.
