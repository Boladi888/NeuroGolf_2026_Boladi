# task359 — compressed history (archived 2026-06-07)

Superseded attempt prose and superseded cost-analysis prose moved here from
`task359.md` to keep the live file under the 20 KB manual-content lint.
Current best, rule, floor proof, and don't-do log stay in `task359.md`.

## Superseded attempts (full prose)

2026-05-19 | `task359_boolmask` (BIG_RMAX_a): replace float full_frame_oh [1,10,30,30]=36000B + nz [1,1,30,30]=3600B with bool Or/And chain — cast one-hots to bool (300B each), And-select by pick_row orientation, Or-broadcast to [1,10,30,30] bool=9000B, And nz_bool=900B, Cast→float output; 266/266 verify, 100% adversarial, cost 45132→25694 (-19438B, +0.563 pts) | 25694 | staged `.hermes/board/staged_submits/task359_boolmask_t_FDB5A7B2_on_5806_87/`

2026-06-04 | `task359_confidence_stripes_v1`: full rewrite from live confidence rule. Keeps only the two `[1,10,*,*]` count tensors, casts dominant vectors to `UINT8`, builds skinny `[1,10,1,30]` / `[1,10,30,1]` stripe patterns, and makes the final full broadcast `And` the BOOL graph output. Gates: `ng_gate visible` 266/266; `ng_gate truth --full` 200000/200000; `ng_gate ab --full` 200266/200266 byte-equal to `submissions/submission.zip`. Cost 40808→4642 (-36166, raw ln gain +2.173733). | 4642 | `prototypes/build_task359_v1.py`, `prototypes/task359_v1.onnx`

2026-06-04 | `task359_confidence_stripes_v2`: v1 tail polish. Dominant label vectors gated only by active rows/cols; branch-specific all-channel active vectors override off-branch labels, deleting `keep_col`/`keep_row`. Gates: visible 266/266; truth --full 200000/200000; ab --full 200266/200266 byte-equal. Cost 4642→4582 (-60). | 4582 | `prototypes/build_task359_v2.py`, `prototypes/task359_v2.onnx`

2026-06-04 | `task359_direct_dominant_v3`: direct `Equal(counts,max)` dominant masks, deleting ArgMax/Cast/label-vector work. Invalid as written: padding lines have `counts == max == 0` for every color, so all colors become true in padding. Builder visible 0/266. Rejected. | 3971 invalid | `prototypes/build_task359_v3.py`, `prototypes/task359_v3.onnx`

2026-06-04 | `task359_direct_dominant_v5`: repaired direct dominant masks with active-line gating + `LessOrEqual(row_conf,col_conf)` complement. Gates: visible 266/266; truth --full 200000/200000; ab --full 200266/200266 byte-equal. Cost 4582→4571 (-11). Superseded by v6. | 4571 | `prototypes/build_task359_v5.py`, `prototypes/task359_v5.onnx`

2026-06-04 | `task359_direct_dominant_v6`: post-branch active gating. Gates `(raw_dominant_mask | branch_scalar)` by the active vector instead of building separate `pick & active` vectors. Gates: visible 266/266; truth --full 200000/200000; ab --full 200266/200266 byte-equal. Cost 4571→4511 (-60). | 4511 | `prototypes/build_task359_v6.py`, `prototypes/task359_v6.onnx`

2026-06-04 | `task359_v7_dynamic_crop`: dynamic active-grid crop, v6 core on the crop, Pad back as graph output. Visible 266/266 but `score_network_v2` returns `None` — the memory walker rejects inferred dynamic dims before profile-shape replacement. Rejected. | score None | `prototypes/build_task359_v7_dynamic_crop.py`

2026-06-04 | `task359_direct_dominant_v8`: v6 path with active masks from `Cast(col_max/row_max -> BOOL)` instead of `Greater(max, zero_f)`; counts are non-negative so this deletes the `zero_f` parameter. Gates: visible 266/266; truth --full 200000/200000; ab --full 200266/200266 byte-equal. Cost 4511→4510 (-1). Was LIVE before v14. | 4510 | `prototypes/build_task359_v8_cast_active.py`, `prototypes/task359_v8_cast_active.onnx`

2026-06-04 | `task359_conv5_svm_v9`: under-3k probe — single dense 5×5 Conv as graph output (memory 0, params 2510). 266/266 visible but visible-fit trap: truth --n 2000 548/2000, full generator 34666/200000. Unsafe. | 2510 unsafe | `prototypes/build_task359_v9_conv5_svm.py`

2026-06-04 | `task359_group2_conv7_v10`: `Conv(group=2)` 7×7 SVM, params 2460, 266/266 visible. truth --n 2000 765/2000, full generator 59321/200000. Unsafe. | 2460 unsafe | `prototypes/build_task359_v10_group2_conv7_svm.py`

2026-06-04 | `task359_group2_conv6x7_v12`: `Conv(group=2)` 6×7 SVM + folded bias, params 2110, 266/266 visible. truth --n 2000 731/2000, full generator 49516/200000. Unsafe. | 2110 unsafe | `prototypes/build_task359_v12_group2_conv6x7_svm.py`

2026-06-04 | `task359_depthwise_conv6x22_v13`: depthwise `Conv(group=10)` 6×22, params 1330, 266/266 visible. Best variant C=100 balanced: truth --n 2000 922/2000, 50k generator 18904/50000. Unsafe visible-fit. | 1330 unsafe | `prototypes/build_task359_v13_depthwise_conv6x22_svm.py`

## Superseded cost-analysis prose

Live model (cost 45132 = memory 45129 + params 3): dominated by full_frame_oh [1,10,30,30] float32 = 36000B (Where broadcasts row/col one-hots) and nz [1,1,30,30] float32 = 3600B (ReduceSum channels for padding mask).

Package before v1 remeasure: cost 40,808 (memory 40,805 + params 3), points 14.3834. Static graph only shows the F16 input cast `[1,10,30,30]` = 18,000 B; runtime/broadcast profiling accounts for the remaining cost.

The boolmask prototype (cost 25694 = memory 25690 + params 4) replaces both large float intermediates with bool equivalents 4× smaller: row_nz_bool/col_nz_bool [.,1] Greater→30B, nz_bool [1,1,30,30] And→900B (vs 3600B float), row/col_bool Cast→300B, row/col_selected And→300B, or_frame [1,10,30,30] Or→9000B (vs 36000B float), masked_bool And→9000B, output Cast→float (graph output, not counted). ORT does not support Where(bool,bool,bool) so Or-of-two-And-masked-branches is used.

v8 (cost 4,510 = memory 4,510 + params 0, points 16.5859) static value-info contributors: col_cnts/row_cnts F32 1,200 B each; col_dom_raw/row_dom_raw/col_pre/row_pre/x_pat/y_pat skinny BOOL stripes 300 B each; col_max/row_max F32 120 B each.

Local under-3k Conv probes (v9–v13) beat the scorer mechanically because the Conv result is the graph output (0 memory) but are visible-fit traps, NOT exact rules. Near misses below v13: depthwise 6×21 cost 1270 = 265/266; 6×20 cost 1210 = 265/266; 9×9 cost 820 = 265/266; group5 11×9 cost 1990 = 265/266; group2 5×8 cost 2010 = 264/266.
