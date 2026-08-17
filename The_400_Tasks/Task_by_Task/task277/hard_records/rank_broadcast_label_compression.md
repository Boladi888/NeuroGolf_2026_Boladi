# task277 — rank-broadcast label compression is dead

**live bar:** 2873 bytes, sha `8ef03725c8af8913224ba70d6a1b4df9120a2815f290f341b39de0365d053863`
(`ng_current_bar.py --task 277 --json`, manifest source, owner Sho, 2026-07-09T01:09:17Z).

**attempted route:** Replace Sho's 100-param `rank_raw_u8 [1,1,10,10]` per-cell component rank grid
with a broadcast row-rank initializer `[1,1,10,1]`, preserving the rest of the live graph. This would
only be valid if component identity never needs same-row tie-breaking after the 5-round max-rank flood.

**prototype:** `../prototypes/task277_rank_row_broadcast.onnx`, sha
`1d77c9641ee7708a2cf28a841e616f503a2acba7818dd7baddff670d0426c7ea`, built by
`../prototypes/build_task277_rank_row_broadcast_probe.py`.

**failed gate:** `python tools/ng_gate.py visible --task 277 --cand The_400_Tasks/Task_by_Task/task277/prototypes/task277_rank_row_broadcast.onnx`
→ `[visible] 206/266 (77.44%)  FAIL`.

**why dead:** The 4-connected component extraction needs a collision-free per-cell rank, not an
axis-broadcast rank. Same-bottom-row component collisions change which components are extracted as
`c1`/`c2`, and the recolor tail then paints the wrong component class. Per the visible-fail axiom, no
truth/profile/full gate was run. Do not try row/column/broadcast rank variants unless a new generator
proof identifies a strictly collision-free smaller label domain.

**one remaining hypothesis:** A genuinely different label primitive that preserves unique per-component
identity without a 100-element rank source and without materializing an extra 100-cell rank tensor.
Known `Range`/row+col generation variants appear to lose because they trade the 100-param initializer
for at least one extra charged 100-cell intermediate.
