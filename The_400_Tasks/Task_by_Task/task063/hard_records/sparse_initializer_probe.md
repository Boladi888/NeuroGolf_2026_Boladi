# sparse_initializer_probe

- live bar: cost 108, SHA `63694aef1aa2deecabfdde154de4879279c6d05d6f6f1ad5d57a9fbaa511d4ae` from `python tools/ng_current_bar.py --task 63 --json` / manifest.
- attempted route: make the sparse `axis_spatial_basis` initializer feed the one-node `Einsum` so params count only 31 stored values instead of dense `[2,30] = 60`.
- exact failed gate: `python -c "from tools._score_v2 import score_network_v2; ... task063_sparse_probe.onnx ..."` returned `None`.
- why dead: raw ORT can load the unsanitized model, but NeuroGolf scorer sanitization renames node inputs without renaming the sparse initializer; sanitized ORT load fails with `Invalid model. Node input 'safe_name_3' is not a graph input, initializer, or output of a previous node.`
- one remaining hypothesis: a non-sparse lower route would need to delete/compress `axis_spatial_basis[2,30]` by mixing implicit all-column sums with the first-column selector inside the final `Einsum`.
