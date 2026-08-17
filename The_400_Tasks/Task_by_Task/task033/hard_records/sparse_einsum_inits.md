route: sparse_einsum_inits
live_bar: 260
live_sha256: b6093795a15eb6fbd91b33f42692e47f6693195f16f0abc4a8ec43cd22689a35
attempt: Converted Sho's one-node free-output Einsum initializers from dense tensors to sparse initializers after zeroing |x|<1e-9; dense params 260 would become 106 stored sparse values if accepted.
failed_gate: `python tools/ng_gate.py visible --task 33 --cand The_400_Tasks/Task_by_Task/task033/prototypes/task033_sparse_einsum_inits.onnx`
exact_result: `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_0' is not a graph input, initializer, or output of a previous node.`
why_dead: Official scorer/sanitizer does not preserve sparse initializer names for node inputs; `tools._score_v2.score_network_v2` returns `None`, so the candidate is invalid before correctness.
remaining_hypothesis: Look for dense algebraic factorization of Sho's `v_w5`/polynomial operands; do not retry ONNX sparse initializers for task033 Einsum operands.
