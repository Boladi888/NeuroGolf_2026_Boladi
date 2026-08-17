# task372 hard record: sparse Conv weight

- Live bar: 200 bytes, sha256 `b35beb50d83f1b9a90dbfed5a977cc6c9af9ac8d1b314310950baebbdc0da02e` from `python tools/ng_current_bar.py --task 372 --json`.
- Attempted route: replace dense Conv initializer `w[10,10,2,1]` with a sparse initializer containing the 162 nonzero values, sha256 `e5463953cc062f42f45e49dfc3894201eb3ffbdd8e90f4de092cc0cd28d1622a`.
- Exact failed gate: `score_network_v2` returned `None`; `python tools/agent_helpers/profile_real_cost.py --task 372 --cand The_400_Tasks/Task_by_Task/task372/prototypes/task372_probe_sparse_conv_weight.onnx --arcgen 266` failed with `ORT INVALID_ARGUMENT: Node input 'safe_name_0' is not a graph input, initializer, or output of a previous node.`
- Why dead: the scorer/sanitizer path does not preserve sparse initializer references as legal Conv inputs; this matches `core.md`/`traps.md` sparse-initializer warnings. Raw nonzero count cannot be credited for Conv weights.
- One remaining hypothesis: no sparse-Conv route remains unless the official scorer/sanitizer changes; dense element count is the charged floor for Conv weights.
