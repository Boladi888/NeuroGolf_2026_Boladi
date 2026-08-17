# task267 sparse_initializers dead route

- Live/team-best bar used: 196 bytes, `Team_Best_Onnx/task267.onnx`, sha256 `c930e33bcaa8c72eebc4265b0300d099f0d8ec65b946e2825e2b663863ce5dbd` from `python tools/ng_current_bar.py --task 267 --json`.
- Attempted route: convert the current-best `row` selector table, then selector tables, into ONNX sparse initializers to reduce params in the free-output Einsum graph.
- Exact failed gate: `python tools/ng_gate.py visible --task 267 --cand The_400_Tasks/Task_by_Task/task267/prototypes/task267_sparse_probe_row.onnx` -> `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_8' is not a graph input, initializer, or output of a previous node.`
- Why dead: scorer sanitization/ORT load does not preserve sparse initializer references for an `Einsum` input, so even the least-invasive row-only sparse probe is not a valid model.
- One remaining hypothesis: avoid sparse tensors and instead reduce dense selector params by algebraically refactoring the rank-2 `Einsum` factors (`row`, `a_any_fg`, `k_bg_sub_fg`) without materializing any full output term.
