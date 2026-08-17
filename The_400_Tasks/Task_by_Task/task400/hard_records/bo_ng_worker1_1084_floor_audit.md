# task400 hard record - bo_ng_worker1 1084 floor audit

- **Live bar:** 1084, SHA-256 `49f88a9d910b85ab712b0e76cf65e3b834b7606ea8c02f56f325a6dc6365fac4`, owner Aditya. Source: `python tools/ng_current_bar.py --task 400 --json` on 2026-07-03; authoritative source `manifest`.
- **Live graph audited:** `Team_Best_Onnx/task400.onnx`, graph `task400_moment_180_direct_start_vec_i32_v3d03`.
- **Charged structure:** one large intermediate, `selected_5x5` F32 `[1,10,5,5]` = 1000 B. Remaining cost is scalar/index runtime plumbing (32 B) plus params (52).

## Gates / proofs run

- Live visible: `[visible] 266/266 (100.00%)  PASS`.
- Live score: `(0, 1032, 52) 1084`.
- Live profile: `task400 task400.onnx: static=1084 profile=1084 (memory 1032 + params 52; 266 real inputs)`.
- F1 scalar-color-index probe path: `The_400_Tasks/Task_by_Task/task400/prototypes/task400_worker2_f1_colorindex.onnx`.
- F1 probe SHA-256: `84904cadf9cf04311411dddd19760da9273c739c294b5c8c2548f45256c65fe9`.
- F1 visible: `[visible] 266/266 (100.00%)  PASS`.
- F1 score: `(0, 5082, 73) 5155`.
- F1 profile: `task400 task400_worker2_f1_colorindex.onnx: static=5155 profile=5155 (memory 5082 + params 73; 266 real inputs)`.
- Cost diff vs live: memory `+4050`, params `+21`, net `+4071`. Named deltas: added `idx_canvas` F32 `[1,30,30]` +3600 B, added `oh5f` F32 `[1,10,5,5]` +1000 B, removed live `selected_5x5` -1000 B, added `oh5` BOOL `[1,10,5,5]` +250 B.

## Verdict

The F1 color-index route is dead for the current bar. It is visible-correct but costs 5155/profile 5155, which is +4071 worse than the manifest bar, so no truth/ab/full gates were run.

The current 1084 graph already sits on the straightforward valid-output floor for this representation. A data-dependent 5x5 one-hot patch read from the F32 input forces a counted F32 `[1,10,5,5]` Slice/Gather output before Pad; Slice/Gather/GatherND inherit input dtype, and Pad-as-output still charges its producer. The remaining micro-cost is already minimal: two F32-to-INT32 scalar starts, one INT32 start vector, one INT32 end vector, INT32 Slice axes/steps, and INT64 Pad pads/axes. Pad without axes was previously measured at 1086, so the current Pad-18 axes form is cheaper.

## Remaining hypothesis

Only a fused final-output primitive could break 1084: something that samples the dynamic 180-degree mirrored 5x5 source patch from the original F32 input and emits the full `[1,10,30,30]` output directly, with zero padding, without materializing either a 5x5 F32 patch or full row/column selector maps. Known ONNX candidates do not fit: two-axis Gather needs a huge intermediate, GatherND needs a large index map, Resize cannot flip and zero-pad this patch, and separable/Einsum selectors materialize row/column maps larger than the 1000-B patch.
