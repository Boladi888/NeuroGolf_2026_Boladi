# task263 color-index Equal-tail route

Live/team bar: 472 cost/profile cost, SHA-256 `8ddcdb4887dc6ad862198b673ec921675d830a239844bb40edf1b96a08ac7ea7`, source `Team_Submission/Backend/manifest.csv` via `python tools/ng_current_bar.py --task 263 --json`.

Attempted route: replace `color_vec_u8` + `zero_chan_u8` + U8 `Where` render with `ArgMax(color_vec_f)` to a scalar color index, `Where(bg_b, 0, color_idx)` for a 3x3 color-index stamp, `Equal(stamp, ch_ids)` for the small one-hot, and `Pad` BOOL to output.

Exact gates:
- `python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2b_color_index_tail.onnx` -> `[visible] 267/267 (100.00%)  PASS`.
- `python tools/agent_helpers/profile_real_cost.py --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2b_color_index_tail.onnx --arcgen 266` -> `task263 worker2b_color_index_tail.onnx: static=480 profile=480 (memory 353 + params 127; 267 real inputs)`.

Why dead: the scalar color-index path deletes the 10-byte `color_vec_u8` tensor but adds `ArgMax`'s 8-byte INT64 scalar, a 1-byte U8 scalar, and a 9-byte index stamp. Initializer swap `zero_chan_u8` -> `ch_ids_u8` is a param tie. `ng_graph_cost_diff` named net delta `+8`; official profile 480 loses to 472.

One remaining hypothesis: a cheaper color scalar would need to avoid `ArgMax`'s INT64 output or read the color without the 40-byte `color_vec_f` Slice; current ONNX options found so far make the one-hot vector cheaper than the scalar index.
