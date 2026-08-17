# task320 sparse initializer probe

Live bar used: 88 bytes, SHA256 `3b60b3cd73e4985df112d876a12b57ad7dbc9cd62e7ed87b71d9e2b22ee1c016`, source `python tools/ng_current_bar.py --task 320 --json` / manifest.

Attempted route: keep the current one-node free-output Einsum equation exactly, but store the sparse `basis`, `coord`, and/or `core` initializers as ONNX sparse initializers so params would count only nonzero values.

Failed gate: static score gate. `score_network_v2` returned `None` for all generated variants:
- `prototypes/task320_sparse_core.onnx`
- `prototypes/task320_sparse_basis.onnx`
- `prototypes/task320_sparse_coord.onnx`
- `prototypes/task320_sparse_basis_core.onnx`
- `prototypes/task320_sparse_all.onnx`

Why dead: sparse initializer inputs are not accepted by the scorer/ORT path for this Einsum graph, matching the project sparse-initializer trap. No visible/truth/profile gate is valid after a `None` score.

One remaining hypothesis: a true win would need a dense param-factorization of the live `basis(20)+coord(60)+core(8)` contract, most plausibly deleting a constant coordinate lane or replacing the dense 2x10 channel basis with a smaller analytic channel-code formulation without adding any charged intermediates.
