live bar: 1454 bytes, Team_Best_Onnx/task222.onnx sha256 796a53ac92654a261b32e46945196f32ba83d8e5a352768abb0f98762add76c2.

route attempted: convert the dominant dense selector c_sh F32 [2,30,16] (960 params, 32 nonzeros) to an ONNX sparse_initializer so the one-node fused Einsum would price by nnz.

exact failed gate: `onnx.checker.check_model(..., full_check=True)` on `prototypes/task222_sparse_csh_probe.onnx` failed with `[ShapeInferenceError] Rank of input 2 (0) does not match the equation indices (3)`; `score_network_v2` returned `None`. Adding tensor value_info in `prototypes/task222_sparse_csh_valueinfo_probe.onnx` failed with `[TypeInferenceError] type case mismatch. existing=tensor_type inferred=sparse_tensor_type`; score remained `None`.

why dead: scorer/strict shape inference treats sparse Einsum operands as rank-0/sparse-typed, matching the global sparse-op-input trap. A dense value_info repair is type-incompatible here and would be dense-charged even if accepted.

one remaining hypothesis: only a scorer/ORT change that accepts sparse tensor operands in Einsum shape inference would reopen this; do not retry as a task-local construction.
