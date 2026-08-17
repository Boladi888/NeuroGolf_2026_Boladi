# task263 selected-ch0 Slice background route

Live/team bar: 472 cost/profile cost, SHA-256 `8ddcdb4887dc6ad862198b673ec921675d830a239844bb40edf1b96a08ac7ea7`, source `Team_Submission/Backend/manifest.csv` via `python tools/ng_current_bar.py --task 263 --json`.

Attempted route: replace selected-signature bit-unpack (`h_sel_f`, `v_sel_f`, `mask_sig_f`, `mask_sig_u16`, `bg_and_u16`, `bg_pow_u16`) with a dynamic `Slice` of the selected tile's channel-0 plane, then `Cast` to BOOL for `bg_b`.

Exact gates:
- `python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2b_ch0_slice_bg.onnx` -> `[visible] 267/267 (100.00%)  PASS`.
- `python tools/agent_helpers/profile_real_cost.py --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2b_ch0_slice_bg.onnx --arcgen 266` -> `task263 worker2b_ch0_slice_bg.onnx: static=501 profile=501 (memory 377 + params 124; 267 real inputs)`.

Why dead: deleting the 18-byte U16 bitwise plane and 9-param `bg_pow_u16` is outweighed by dynamic Slice start/ends plumbing plus the 36-byte F32 `bg_f` tile. `ng_graph_cost_diff` named net delta `+29` vs the live structure (`+32` memory, `-3` params under cost_predictor; official profile 501 vs 472).

One remaining hypothesis: only a background decoder that avoids both the 18-byte `bg_and_u16` plane and dynamic spatial Slice vectors could beat the current bit-unpack; no such ONNX primitive is identified.
