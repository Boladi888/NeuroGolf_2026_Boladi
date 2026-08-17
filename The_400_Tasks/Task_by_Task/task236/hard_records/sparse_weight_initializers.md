# task236 sparse weight initializer route

- Live/team-best bar used: cost 126, sha256 `b422d6b5c34a300040bd2268aef0d31544cd33ec7c4a233438e9c0f6b598d9d1`, source `python tools/ng_current_bar.py --task 236 --json` / team manifest.
- Attempted route: convert dense `w_pair` and/or `w_route` in the 126-byte parent graph to ONNX `sparse_initializer` tensors to reduce params while preserving the same `ConvTranspose -> Cast -> Shrink -> ConvInteger` graph.
- Artifacts: `prototypes/task236_sparse_wpair.onnx`, `prototypes/task236_sparse_wroute.onnx`, `prototypes/task236_sparse_both.onnx`, built by `prototypes/probe_sparse_weights.py`.
- Failed gate: `score_network_v2` returned `None` for all three sparse variants: `task236_sparse_wpair.onnx None None`, `task236_sparse_wroute.onnx None None`, `task236_sparse_both.onnx None None`. ORT load alone passed, but NeuroGolf scoring did not.
- Why dead: sparse initializers feeding Conv-family ops are not accepted by the official scorer path, matching `core.md` sparse-initializer warnings; no visible/truth gate should be spent on this family.
- One remaining hypothesis: only revisit if the official scorer/sanitizer changes sparse Conv-family support; otherwise params must be reduced by changing the legal dense graph, not by sparse storage.
