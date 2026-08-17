# task301 sparse_initializer_coeffs

- Live/team-best bar: 430, SHA-256 `b01f14e6667d853308d75e797b8f3b702bb204524a57468e06e03aa72537555c` from workbook `NG_OnnxlyFans_TaskStatus.xlsx` and team manifest.
- Attempted route: replace dense F16 coefficient initializers `v` or `u+v` with ONNX sparse initializers in Aditya's 430-byte free-output/Einsum graph; artifacts `task301_sparse_v.onnx`, `task301_sparse_uv.onnx`, `task301_qactive_sparse_uv.onnx`.
- Failed gate: `score_network_v2` returned `None` for all three sparse-initializer variants.
- Why dead: scorer/checker/ORT path does not accept sparse initializers as ordinary op inputs; this matches traps/core sparse-initializer guidance.
- Remaining hypothesis: dense coefficient refactor is still viable; only algebraic feature deletion/factorization that keeps dense legal initializers should be retried.
