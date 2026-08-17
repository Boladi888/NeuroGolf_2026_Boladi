# task034 hard record: gathernd_contiguous_crop_price

- live bar: 1334 by `python tools/ng_current_bar.py --task 34 --json`, sha256 `642a13a7dd051aff43afb6a4d6b9217d378d42f747d099d80ed8b15c30e3a268`; `_score_v2` on `Team_Best_Onnx/task034.onnx` = memory 1129 + params 205.
- attempted route: replace the current first-node `Slice` over channels `{0,2}` and rows/cols `1:8` with a GatherND stencil read, then feed equivalent `[1,2,7,7]` data into the existing detector.
- exact failed gate: pre-build price gate. Current Slice reads `2*7*7 = 98` values with 15 slice-index params and 392B F32 + 98B U8 crop memory. Equivalent `GatherND(batch_dims=1)` point reads need at least `98*3 = 294` index params for `(channel,row,col)`, adding +279 params before any decision tail changes.
- why dead: this is a fixed contiguous crop, so techniques #145/#146 say Slice is the floor and GatherND is only for scattered sparse reads. Building the GatherND version cannot beat the 1334 live bar.
- one remaining hypothesis: use GatherND only if a different rule derivation reduces the read set far below the 98-value contiguous crop, not as a mechanical replacement for the current Slice.
