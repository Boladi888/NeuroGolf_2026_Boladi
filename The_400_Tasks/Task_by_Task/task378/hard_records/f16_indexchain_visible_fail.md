# task378 hard-record — F16 MaxPool-index chain visible-fails

- Live bar at session (2026-07-07): **2654**, sha
  `d30e61b3d2e68fd4f30df196ed557c28ca019b7754fc3d613aad0810eec10cfe`
  (`python tools/ng_current_bar.py --task 378 --json`, team manifest). Lower task-local
  reviewer-ready bar: **2614**, sha
  `ed6fef4fe2d3291f396572e15d94e658fb714d72c50ae77de8df24eab48b89f9`.
- Attempted route: cast `pool_indices4` INT64 to F16 before Squeeze/Div/Floor/Mod and keep
  only the score tie-break cast back to F32. Prototype:
  `prototypes/task378_f16_indexchain.onnx` from
  `prototypes/build_task378_f16_indexchain.py`.
- Failed gate: `python tools/ng_gate.py visible --task 378 --cand The_400_Tasks/Task_by_Task/task378/prototypes/task378_f16_indexchain.onnx`
  → `[visible] 124/267 (46.44%)  FAIL`.
- Why dead: MaxPool flattened indices include channel offsets up to 8999; F16 cannot preserve
  the low bits above 2048, so row/column extraction by Div/Mod changes visible outputs.
- One remaining hypothesis: scalarize exact F32 per-channel vectors after `payload_color` is
  known; do not cast flattened MaxPool indices to F16 before row/column extraction.
