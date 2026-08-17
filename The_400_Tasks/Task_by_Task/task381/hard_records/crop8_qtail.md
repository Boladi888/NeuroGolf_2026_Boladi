# crop8_qtail

- live bar: 1162, sha `8ded3453e806a64e32691d9de41bea2422d11c25054df8edc6ac528cca81377c` (`ng_current_bar`, manifest, 2026-07-09T01:09:17Z)
- attempted route: compute only original rows 1..8 and use final `QLinearConv` padding `[1,0,21,20]` to restore the 30x30 output.
- failed gate: `ng_gate visible --task 381 --cand task381_crop8_qtail.onnx`.
- exact result: `[visible] 0/265 (0.00%)  FAIL`; profile would have been `static=912 profile=912`.
- why dead: final `QLinearConv` padding emits all-zero outside the cropped rows, but rows 0 and 9 are active-grid background/red rows and must preserve channel 0/channel 2. The corpus has red in rows 0 and 9 on 166/265 bundled pairs.
- one remaining hypothesis: a crop8 detector could still be used only if a cheaper active-grid/background carrier restores rows 0 and 9, but a full 10x10 active carrier erases the saving.
