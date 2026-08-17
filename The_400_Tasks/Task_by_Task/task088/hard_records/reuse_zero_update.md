live bar: manifest cost 901, sha256 c5dbb60f72207c6d5c260d0db689de2efee0a630c2a3ea7ee8b46d0955828feb (`ng_current_bar.py --task 88 --json`)

attempted route: reuse existing `zero_u8v` as the `ScatterElements` update tensor and delete `marker_update_i8` for a 1-param shave.

failed gate: `python tools/ng_gate.py visible --task 88 --cand The_400_Tasks/Task_by_Task/task088/prototypes/task088_reuse_zero_update.onnx` -> `ERROR: InvalidArgument ... ScatterElements ... Indices and updates must have the same rank`

why dead: ORT runtime requires the update tensor rank to match the index tensor rank; the existing `[1]` zero initializer cannot replace the `[1,1,1,1]` update without adding a Reshape that gives back the saved byte as charged memory.

remaining hypothesis: only same-rank initializer sharing or deleting a consumer can save the update param; rank-broadcast ScatterElements is not available.
