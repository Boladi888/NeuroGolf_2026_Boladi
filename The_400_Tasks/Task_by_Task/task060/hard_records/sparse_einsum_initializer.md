# sparse_einsum_initializer

Live bar: cost 129, sha256 `0a68d150f2eb578e63cabb7e9e741bfdbf68ae40b15cbc3a94d244d15d58f842`, source `python tools/ng_current_bar.py --task 60 --json` (manifest, owner Vamshi, updated 2026-07-12T15:15:54Z).

Attempted route: keep Vamshi's exact one-node `Einsum` graph and replace the zero-heavy dense `width_basis[3,30]` initializer with an ONNX `sparse_initializer`. Dense params would drop from 90 width entries to 16 stored width values, implying a theoretical cost near 55 if the official scorer accepted sparse `Einsum` operands. Builder/probes: `The_400_Tasks/Task_by_Task/task060/prototypes/build_task060_sparse_width.py`; candidates `task060_sparse_width*.onnx`.

Exact failed gates:
- `python The_400_Tasks/Task_by_Task/task060/prototypes/build_task060_sparse_width.py` -> `full_check=FAIL InferenceError: [ShapeInferenceError] Inference error(s): (op_type:Einsum): [ShapeInferenceError] Rank of input 1 (0) does not match the equation indices (2).`
- `python tools/spread_score_isolated.py The_400_Tasks/Task_by_Task/task060/prototypes/task060_sparse_width.onnx` -> `{"error": "score_network_v2 returned None"}`
- `python tools/agent_helpers/profile_real_cost.py --task 60 --cand The_400_Tasks/Task_by_Task/task060/prototypes/task060_sparse_width.onnx --arcgen 266` -> `Invalid model. Node input 'safe_name_2' is not a graph input, initializer, or output of a previous node.`
- `python tools/ng_gate.py visible --task 60 --cand The_400_Tasks/Task_by_Task/task060/prototypes/task060_sparse_width.onnx` -> `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_2' is not a graph input, initializer, or output of a previous node.`

Why dead: ONNX shape inference does not expose sparse initializer dense rank to `Einsum`, and NeuroGolf's `sanitize_model` renames dense initializers/node inputs but not `graph.sparse_initializer`, leaving sparse operands dangling in the official scorer path. Pre-naming sparse values to the predicted `safe_name_2` also scored `None`; the raw graph is topologically invalid before sanitize and still rejected by the scorer.

One remaining hypothesis: only revisit sparse initializers if the official sanitizer is updated to rename `graph.sparse_initializer` and ONNX/ORT shape inference accepts sparse tensor operands for `Einsum`; otherwise sparse params are not a legal way to beat the 129 free-output graph.
