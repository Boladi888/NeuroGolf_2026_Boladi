# task297 sparse-Einsum initializer route

- Bar used: task-local staged-pending cost 128 (`task297_shifted_basis_no_pick_cost128`, sha256 ba8146c6fea0bd3f347e296fd37a5a25667565d75e5ac1807606e19a262aa6c6); authoritative manifest live remains 136, sha256 fefcaa50289d057f74b9cfcefa9188d236be69cb956ceed87c16d2a41f7d9d22.
- Attempted route: convert all zero-heavy dense initializers in the one-node `Einsum` graph to ONNX sparse initializers (`task297_sparse_inits_probe.onnx`, sha256 dcb62b717b686bbc4d05ae8af6a58ae63c55717d9ad80f59abf0f99047f7f1cf).
- Failed gate: `onnx.checker.check_model(..., full_check=True)` -> `ShapeInferenceError: Rank of input 0 (0) does not match the equation indices (2)`; `score_network_v2` -> `None`.
- Why dead: strict shape inference does not expose sparse initializer shapes to `Einsum`, so the scorer rejects before any correctness gate. Do not run visible/truth/AB on sparse-Einsum variants.
- Remaining hypothesis: only dense-parameter factorization of `coord_basis`/`color_basis`/small affine tensors, or a different free-output one-node formulation, can beat the current cost128 rung without adding memory.
