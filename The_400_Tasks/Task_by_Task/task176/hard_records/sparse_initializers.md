# task176 sparse initializers dead route

- Live bar used: cost 140, sha256 `36dcb4901656db6303a277633faae8d4cfb63bbe189781b2f2d6851b4477a5ce` from `python tools/ng_current_bar.py --task 176 --json`.
- Attempted route: replace dense `row`, `col`, and/or `color` initializers in the one-node `Einsum` with ONNX sparse initializers so params count only nonzeros.
- Exact failed gate: scorer/checker before visible. `score_network_v2` returned `None` for `v_sparse_row.onnx`, `v_sparse_row_col.onnx`, and `v_sparse_all.onnx`; `onnx.checker.check_model(full_check=True)` reported `ShapeInferenceError`: sparse `Einsum` operands are treated as rank 0, e.g. `Rank of input 3 (0) does not match the equation indices (2)`.
- Why dead: official scorer path rejects sparse `Einsum` operands before correctness gates, so the apparent nonzero-count param save is not usable.
- One remaining hypothesis: sparse constants might still be useful only for an op with source-proven sparse support, but not for this terminal `Einsum` route.
