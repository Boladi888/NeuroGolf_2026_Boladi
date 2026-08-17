# task248 sparse_updates hard record

- Live bar: cost 151, SHA-256 `6a001ce01217f4e505777fb369eb36ec242ffdc413030285fdb28dc76479d892`, source `python tools/ng_current_bar.py --task 248 --json` / team manifest.
- Attempted route: replace dense `updates` F32 `[1,2,9,1]` with a sparse initializer containing only the nine channel-1 ones, hoping to drop params from 29 to 20 while keeping the same ScatterElements overlay.
- Prototype: `The_400_Tasks/Task_by_Task/task248/prototypes/probe_sparse_updates.onnx`, SHA-256 `c352c0ed82344207812f1129423b396ddc773e59073eb81727ba260b6742459a`.
- Exact failed gate: `_score_v2` returned `None`; ONNX full-check reports `InferenceError: [ShapeInferenceError] (op_type:ScatterElements, node name: output): updates typestr: T, has unsupported type: sparse_tensor(float)`. `ng_gate visible` also fails after sanitizer with `Invalid model. Node input 'safe_name_10' is not a graph input, initializer, or output of a previous node.`
- Why dead: sparse initializers are not valid `ScatterElements` updates under the scorer path, and sanitizer does not preserve sparse initializer inputs.
- One remaining hypothesis: a strict sub-151 win likely needs avoiding the 18-element INT32 scatter index/update pair entirely, not sparse-encoding its zero half.
