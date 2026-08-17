# sparse_pow_fwd dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (OnnxlyFans workbook + manifest).
- Attempted route: store `pow_fwd[30]` as a sparse initializer with only the first 20 nonzero values; estimated param saving 10 if accepted.
- Failed gate: checker/shape before visible on `task355_bo_ng_worker1_sparse_pow.onnx`; shape inference reports `Einsum` input rank 0 for sparse `pow_fwd`, and `visible` errors with a dangling sanitized input.
- Why dead: sparse initializers are not usable as `Einsum` operands under the official checker/sanitizer path.
- Remaining hypothesis: none for sparse `pow_fwd`; only a non-sparse generated/short position basis that costs less than 30 params could reopen the position-weight storage.
