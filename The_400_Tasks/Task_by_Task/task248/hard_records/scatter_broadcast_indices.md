# task248 scatter_broadcast_indices hard record

- Live bar: cost 151, SHA-256 `6a001ce01217f4e505777fb369eb36ec242ffdc413030285fdb28dc76479d892`, source `python tools/ng_current_bar.py --task 248 --json` / team manifest.
- Attempted route: delete the `Concat` that duplicates `path_cols` to `[1,2,9,1]`; cast only `[1,1,9,1]` path columns to INT32 and ask `ScatterElements` to broadcast those indices over the two-channel `[1,2,9,1]` updates tensor.
- Prototype: `The_400_Tasks/Task_by_Task/task248/prototypes/probe_broadcast_indices.onnx`, SHA-256 `5dce6943e790dac1ee0ce1d61364ded81d076c7829c65465d43dace690051c3d`.
- Exact failed gate: `_score_v2` returned `None`; `ng_gate visible` failed with `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running ScatterElements node. Name:'output' Status Message: Indices vs updates dimensions differs at position=1 1 vs 2`.
- Why dead: `ScatterElements` requires runtime-equal `indices` and `updates` dimensions; checker/strict shape inference/raw ORT load do not catch this, but execution and scorer do. The final two-channel index shape is therefore forced for the current overlay formulation.
- One remaining hypothesis: a strict sub-151 win must avoid `ScatterElements` overlay entirely or find a different one-op output formulation; shrinking/broadcasting its index/update operands is exhausted locally.
