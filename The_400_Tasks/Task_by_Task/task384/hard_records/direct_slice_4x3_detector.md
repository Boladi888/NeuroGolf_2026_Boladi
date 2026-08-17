# direct_slice_4x3_detector

- Live/team-best bar: 495, SHA-256 `9dfe8199296cae425e71d69d161a347cb618832acee6faca0c396ef46a280291` (manifest + workbook, 2026-06-14T17:47:49Z).
- Attempted route: replace the accepted 7x7 foreground crop with a 4x3 top-left detector plus a direct 4x5 dynamic Slice from the original F32 input.
- Failed gate: `python tools/ng_gate.py visible --task 384 --cand The_400_Tasks/Task_by_Task/task384/prototypes/task384_worker3_direct_slice.onnx` -> `[visible] 230/266 (86.47%)  FAIL`.
- Why dead: the 4x3 detector is too narrow; top rows can have their first foreground outside cols 1..3 even though the bbox min-col appears lower, so y0 is sometimes detected too low.
- Remaining hypothesis: a correct direct-slice detector would need separate row/col stripes over the full orthogonal extent, but that reintroduces about the same F32/U8 detector cost as the 7x7 parent plus an extra direct compact Slice, so it is unlikely to beat 495.
