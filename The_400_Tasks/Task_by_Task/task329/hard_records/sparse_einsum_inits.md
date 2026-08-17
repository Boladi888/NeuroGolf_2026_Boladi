# sparse_einsum_inits

- live bar: 358, sha `429513d344edac3c875299e2455f1ba5661f0fc31802c2bae1d0e473047206f2`.
- attempted route: convert mostly-zero `selector_coeff`, `spatial_basis`, `color_basis`, and `color_maps` initializers in the 308 factorized Einsum candidate to ONNX sparse initializers.
- failed gate: `onnx.checker.check_model(full_check=True)` / `onnx.shape_inference.infer_shapes(strict_mode=True)` before scoring.
- exact failure: `Einsum`: `Rank of input 2 (0) does not match the equation indices (3)`.
- why dead: ONNX shape inference does not expose sparse initializer rank to `Einsum` in this scorer path, so `score_network_v2` would reject it before any useful cost is credited.
- one remaining hypothesis: only revisit sparse params here if the scorer path gains sparse-initializer shape support for `Einsum`; dense factorization is the valid route now.
