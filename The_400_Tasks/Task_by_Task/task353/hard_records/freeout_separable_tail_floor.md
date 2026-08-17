# task353 hard record: non-ScatterND separable/free-output tail loses to 238

- Live bar used: cost 238, SHA256 `df5d0635f518a326b888a7b6fa109fceefaa1309f96d384fde208a25d84c3a32`, source `python tools/ng_current_bar.py --task 353 --json` / team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
- Locked reviewer route: no `ScatterND`, no `GatherND`, no sparse initializers, no INT8/INT32 ScatterND-index trick; rebuild singleton old/new row/col vectors with #82 Equal-ramp, then make one #89 terminal free-output `Einsum`:
  `output = Einsum(input[nqhw], term_map[tkq], row_terms[th], col_terms[tw] -> nkhw)`.
- Semantics of the three terms: copy input channels 0 and 4; turn old green into background; turn new background into green and remove background.
- Prototype: `The_400_Tasks/Task_by_Task/task353/prototypes/task353_freeout_separable_tail_probe.onnx`, builder `build_task353_freeout_separable_tail_probe.py`.
- Pricing:
  - `score_network_v2` -> `(0, 1352, 380)` = cost 1732.
  - `profile_real_cost.py --task 353 --cand ... --arcgen 266` -> `static=1732 profile=1732 (memory 1352 + params 380; 271 real inputs)`.
  - `largest_intermediate.py --top 30` -> dominant tensors: `row_terms_f [3,30] F32 = 360 B`, `col_terms_f [3,30] F32 = 360 B`, four singleton F32 vectors `[1,30] = 120 B` each, and four bool Equal vectors `[1,30] = 30 B` each.
- Stop result: profile cost is not strictly below 238, so no visible/truth/ab/safety gate was run.
- Infeasibility certificate for this separable tail: without a scatter-style point write, the tail must localize two independent data-dependent cells, old `(r3,c3)` and new `(r3+sign(dr), c3+sign(dc))`. A separable final output needs at least four dynamic 1-D singleton factors: old row, old col, new row, new col. Because the terminal contraction reads the F32 graph input, those factors must be numeric F32 operands for the final `Einsum`; the #82 Equal-ramp implementation pays `4 * (30 B bool + 120 B F32) = 600 B`, already above the 238 bar before term stacking, channel routing, or scalar moment costs. Even a hypothetical direct legal F32 one-hot generator would pay `4 * 120 = 480 B`, still above 238. The measured exact skeleton confirms the tail lands at 1732, not near the prompt's ~114 estimate.
- One remaining hypothesis: any sub-238 route must avoid both `ScatterND`'s INT64 index tensor and the four exposed F32 singleton vectors, likely requiring an op that can combine dynamic point localization with passthrough inside a single output node; no such legal static ONNX primitive was found in the #67/#82/#89 family.
