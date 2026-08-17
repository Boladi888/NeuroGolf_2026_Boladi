# task031 sparse Einsum initializers

- Live bar used: 400, sha256 `963fb663b0272cda192e00965a1681bfe006b13d5da05c178d6c3dd6dc9d8904`, source `python tools/ng_current_bar.py --task 31 --json` / OnnxlyFans manifest.
- Attempted route: replace zero-heavy dense initializers in the one-node 400-byte `Einsum` with ONNX sparse initializers to make params count stored nonzeros.
- Candidate: `The_400_Tasks/Task_by_Task/task031/prototypes/worker3_sparse_einsum/task031_worker3_sparse_einsum.onnx`, sha256 `fededb7b63e6a68462f4fea40d308a85fcffae1b3de044849e62b77abd3f72aa`.
- Exact failed gate: `python tools/ng_gate.py visible --task 31 --cand The_400_Tasks/Task_by_Task/task031/prototypes/worker3_sparse_einsum/task031_worker3_sparse_einsum.onnx` -> `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_1' is not a graph input, initializer, or output of a previous node.`
- Supporting failure: `onnx.checker.check_model(..., full_check=True)` -> `ShapeInferenceError ... Einsum ... Rank of input 0 (0) does not match the equation indices (1).`; `score_network_v2` returned `None`.
- Why dead: the official scorer/sanitizer path does not preserve sparse initializer bindings for this `Einsum`, and ONNX shape inference does not supply the expected dense rank to the node. This is invalid before correctness.
- One remaining hypothesis: sparse `Constant` nodes might load, but their outputs would become charged intermediates and likely erase the params win; only revisit if scorer semantics for sparse constants are separately proven on a one-node output path.
