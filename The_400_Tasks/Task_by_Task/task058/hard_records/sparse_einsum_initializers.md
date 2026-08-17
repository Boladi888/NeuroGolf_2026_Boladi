# task058 sparse Einsum initializers dead route

Live bar: cost/profile 128, sha256 `795ce7543fc7358b026acfaf9e6db16a21813c32d9f43c5dd9b7b1afd232eff9`.

Attempted route: convert the sparse-valued `channel_sign` initializer (2/10 nonzero) and `coeff` initializer (5/9 nonzero) in the current analytic Einsum graph into ONNX sparse initializers to save up to 12 params.

Failed gate: builder `onnx.checker.check_model(full_check=True)` before scoring/visible. Exact error: `ShapeInferenceError ... (op_type:Einsum, node name: output): Rank of input 6 (0) does not match the equation indices (2).`

Why dead: ONNX shape inference treats sparse initializer operands to `Einsum` as rank-0/unknown, so `score_network_v2` cannot be reached. This matches the sparse-initializer trap in the library.

One remaining hypothesis: a non-sparse algebraic rewrite of the analytic formula could remove the explicit 10-value `channel_sign` or the 3-value `phase`, but it must keep a single final output op; splitting into plane intermediates will exceed the 128-byte bar.
