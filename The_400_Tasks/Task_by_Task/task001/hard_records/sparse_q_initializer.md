# Sparse Q Initializer Probe

- live bar: 118 bytes, manifest/team-best SHA-256 `5121f6312e9e3bd6ee13b5b4a91bdd98dd1ec249358a67bf65a63fea07ff3f7e`
- attempted route: convert the current `Q[30,3]` initializer, whose rows 9..29 are all zero, to an ONNX sparse initializer. Theoretical charged params would be `Q nnz 27 + K 18 + s 10 = 55`, below the 118-byte bar.
- prototype: `The_400_Tasks/Task_by_Task/task001/prototypes/probe_sparse_q.py`
- probe ONNX: `The_400_Tasks/Task_by_Task/task001/prototypes/task001_sparse_q.onnx`
- failed gate: `python tools/ng_gate.py visible --task 1 --cand The_400_Tasks/Task_by_Task/task001/prototypes/task001_sparse_q.onnx --no-watch-queue`
- exact result: `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_2' is not a graph input, initializer, or output of a previous node.`
- static validation result: `onnx.checker.check_model(full_check=True)` raises `InferenceError: [ShapeInferenceError] Inference error(s): (op_type:Einsum): [ShapeInferenceError] Rank of input 2 (0) does not match the equation indices (2).`
- scorer result: `_score_v2.score_network_v2(...) -> None`
- why dead: sparse initializers are not visible to strict shape inference / sanitizer as valid dense `Einsum` operands. Adding dense `value_info` would charge the full `Q[30,3]` tensor as memory and lose the intended 55-byte floor.
- one remaining hypothesis: find a non-sparse way to make the 21 zero rows structural, e.g. a different free-output writer whose active bound is intrinsic and does not require a charged crop/pad.
