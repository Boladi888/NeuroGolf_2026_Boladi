# task099 hard record: sparse S Einsum operand

- Live bar used: 504 bytes, SHA-256 `6c419ff35ddc8651209c1ac5f32b0f2aecce69501cd274348e10c97a469bcfed`, from `python tools/ng_current_bar.py --task 99 --json` authoritative manifest source.
- Attempted route: convert the current one-node parent initializer `S` from dense `[5,30]` F32 to a `SparseTensorProto` because only 50/150 entries are nonzero, which would price at 404 if accepted.
- Probe builder: `The_400_Tasks/Task_by_Task/task099/prototypes/build_sparse_s_einsum.py`.
- Exact failed gate: ONNX full-check during build failed before visible/score: `ShapeInferenceError (op_type:Einsum): Rank of input 4 (0) does not match the equation indices (2)`.
- Why dead: strict shape inference does not type sparse `Einsum` operands as the declared dense rank, matching the project-wide sparse-initializer trap. No scoreable ONNX artifact was emitted.
- One remaining hypothesis: only revisit sparse storage if the scorer/sanitizer changes to preserve sparse initializer rank for `Einsum`; otherwise reduce dense params by changing the factorization, not by sparse encoding.
