# task125 hard_record - sparse Einsum initializers are scorer-dead

- **Live bar:** 696, owner Sho, sha256 `23a5bffcbe112d13f3470a158ef56664d394c3997ad50ffc8035f947f12d8582`, source `python tools/ng_current_bar.py --task 125 --json` on 2026-07-13.
- **Attempted route:** convert the current one-node 696-param dense `Einsum` factor bank to `graph.sparse_initializer` entries, preserving names/equation and removing dense initializers. Probe builder: `prototypes/build_sparse_init_probe.py`; probe ONNX sha256 `e66b37953dc66f15a9ef735f6d66ba84e78e2114c2a35bf6636f1e96c7b09c97`.
- **Exact failed gate:** checker/scorer pre-gate. `onnx.checker.check_model(full_check=True)` fails with `[ShapeInferenceError] Rank of input 1 (0) does not match the equation indices (2).`; `score_network_v2(...)` returns `None`.
- **Why dead:** ONNX shape inference treats sparse initializer inputs to `Einsum` as rank-0 here, so the official scorer path rejects before visible/truth. This is not a candidate family; do not run visible/truth/ab on sparse `Einsum` variants.
- **One remaining hypothesis:** a lower dense one-node graph would need a different tensor factorization or a generator-specific profile/render formulation; sparse storage cannot provide the params discount.
