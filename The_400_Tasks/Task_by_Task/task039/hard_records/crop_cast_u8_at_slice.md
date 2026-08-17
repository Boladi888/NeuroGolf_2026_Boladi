# task039 hard record - crop_cast_u8_at_slice

Live/team bar used: cost 441, profile cost 441, sha256
`dc9191feb18062f9d159affc29e09b0bd8b77a3259cefb129ada0876dd915e0b`,
from `python tools/ng_current_bar.py --task 39 --json` on 2026-07-06.

Attempted route: insert `Cast(crop_f -> UINT8)` immediately after the promoted
dynamic `Slice` and feed the U8 crop to the final `Pad`, with graph output dtype
UINT8. Prototype:
`The_400_Tasks/Task_by_Task/task039/prototypes/task039_cropcast_u8_probe.onnx`;
builder:
`The_400_Tasks/Task_by_Task/task039/prototypes/build_task039_cropcast_u8_probe.py`.

Exact gates:
- `python tools/ng_gate.py visible --task 39 --cand .../task039_cropcast_u8_probe.onnx`
  -> `[visible] 264/264 (100.00%)  PASS`
- `score_network_v2` -> `(0, 482, 49) 531`
- `python tools/agent_helpers/profile_real_cost.py --task 39 --cand .../task039_cropcast_u8_probe.onnx --arcgen 266`
  -> `static=531 profile=531 (memory 482 + params 49; 264 real inputs)`

Why dead: `Slice` inherits its data dtype from the F32 graph input, so
`source_crop` still emits `crop_f [1,10,3,3]` as F32 = 360 B. The added Cast then
creates `crop_u8 [1,10,3,3]` = 90 B, raising memory from 392 to 482. This does
not cast "at" the Slice producer; it double-charges after the F32-forcing op.
Because the profile cost is 90 bytes above the 441 live bar, no truth/AB/full
gates were run.

One remaining hypothesis: only a producer that directly emits a one-channel/U8
3x3 color-index crop from the free F32 input, or a final op that writes the
30x30 output without materializing the 10-channel F32 crop, can cross below this
bar. Post-Slice casts, QLinearConv after Slice, and output-dtype-only changes are
not candidates.
