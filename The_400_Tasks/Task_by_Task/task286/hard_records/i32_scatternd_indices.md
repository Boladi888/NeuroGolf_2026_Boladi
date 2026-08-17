# task286 hard record - i32_scatternd_indices

- live bar: manifest/team-best 21558, sha `406cc1e321bd08ccce766912914f9e49b997ceb3ba68c1cc767d5fd71345ff89`.
- stricter local bar at attempt time: staged `task286_rank3tailshave_cost20646`.
- attempted route: convert dynamic channel-index path (`h_allowed`, `h_c0`, `h_c8`, therefore `h1051`) from INT64 to INT32 in `task286_rank3_tail_i32idx_probe.onnx`.
- exact failed gate: `_score_v2` returned `None`; visible/profile both failed ORT load with `InvalidGraph: Type 'tensor(int32)' of input parameter ... of operator (ScatterND) ... is invalid`.
- why dead: ORT 1.24.4 requires INT64 indices for `ScatterND`, so the apparent static `-32 B` dtype-narrow saving is unshippable; adding an INT64 cast before `ScatterND` gives back the saved bytes.
- one remaining hypothesis: only revisit if a future accepted runtime/opset supports INT32 `ScatterND` indices or if the scatter is replaced by a different final-output construction that avoids dense channel planes.
