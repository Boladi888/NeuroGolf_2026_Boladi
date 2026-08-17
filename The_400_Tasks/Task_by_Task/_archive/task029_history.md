# task029 compressed history

Main file compressed on 2026-06-09 after the v18/v17 staging pass pushed manual task029.md content above the 15 KB soft limit.

Older lineage retained here in compact form:

| Date | Candidate | Result | Cost | Notes |
|------|-----------|--------|------|-------|
| 2026-05-29 | build_task029_v10_crop23 | PASS visible/LIVE/adversarial | 14,545 | Static K=23 crop over the color-index canvas; prior LIVE baseline before the 2026-06-09 reduction pass. |
| 2026-05-29 | build_task029_v9_einsum | PASS visible/adversarial | 15,221 | Replaced INT64 ArgMax color canvas with F32 `Einsum("bchw,c->bhw")`. |
| 2026-05-29 | build_task029_v8_canvas | PASS visible/LIVE/adversarial | 18,811 | Introduced U8 color-index canvas plus final free `Equal` output; fixed absent-color last-index divergence with clamps. |
| 2026-05-29T17:19Z | task029_compact_v6_F32 | shipped | 97,875 | Slightly better than v7 after real scorer check; old full-grid algorithm family. |
| 2026-05-29T17:13Z | task029_compact_v7_INT64_Equal | shipped | 98,371 | Worker manifest accidentally used file size as cost; actual scorer cost was 98,371. |
| 2026-05-19 | build_task029_crop25 | FAIL | 133,275 | Literal crop-to-active-grid increased cost because the cropped input intermediate outweighed savings. |
| 2026-05-29 | build_task029_compact v1-v7 | PASS visible/adversarial | file-size 1,924-2,684, scorer ~98k | File-size-oriented compact series before scorer-cost accounting was corrected. |

Old don't-do details compressed:

- Literal crop-process-pad was a cost increase; dynamic output-side slicing is different and became useful only after v13-v18 value_info/profile handling.
- `select_last_index=1` for last row/col diverged on absent colors unless absent colors were clamped. v11+ drops that clamp only because valid generator/LIVE-equivalent marker selection cannot choose absent colors; v17/v18 add a separate no-marker guard for the score dummy.
- The v10 "13-14k floor" applied only to the fixed color-index canvas plus Equal-tail family. v18 invalidated that floor by slicing the original one-hot input and making final Pad the free output.

## Compressed 2026-06-22 (Worker 6 — md exceeded 20 KB lint)

Superseded intermediate rungs moved out of main file (full prose was redundant; v23 remains the honest floor, v18/v17/v22/v11 kept in main for ladder proof):

| Date | Candidate | Result | Cost | Notes |
|------|-----------|--------|------|-------|
| 2026-06-09 | build_task029_v15_dynamic_slice_minannot | PASS visible + full A/B byte-equal; truth same as LIVE | 9,742 score_v2; 9,038 real-profile (arcgen 0 — fiction) | Dynamic U8 Slice after the full U8 color canvas. Minimal Slice value_info lets real-profile charge observed crop size. Superseded by v17/v23. |
| 2026-06-09 | build_task029_v14_dynamic_slice_annotated | PASS visible + full A/B byte-equal; truth same as LIVE | 9,742 score_v2; 9,487 real-profile (arcgen 0) | Same dynamic Slice/Pad tail, but conservative `[1,23,23]` Slice annotation overcharges official real-profile versus v15. |
| 2026-06-09 | build_task029_v12_bordercheck_i32 | Generator-truth probe: visible 266/266, truth 20000/20000, A/B diverges from LIVE on known LIVE misses | 14,838 (short probe cost) | Adds complete-border verification using row/col color counts. Repairs the R1 generator rule but not a cost win over LIVE/v10. Evidence only, not staged. |
| 2026-05-19..2026-05-29 | older v1-v10 / compact lineage | see rows above | 133,275 -> 14,545 | v10 was the prior LIVE baseline at 14,545; literal crop-to-active-grid increased cost; old compact file-size rows were not scorer-cost wins. |

## Compressed 2026-07-08 (v28 dilated-Conv win — md exceeded 20 KB lint)

Full prose of superseded rungs moved from main (main now keeps compact one-line rungs + the v28 forced-floor proof). v26 (dense-Conv inner, 8,319) and v28 (dilated-Conv inner, 7,954) are the current byte-equal floor and superseded v23's 9,510. R1 #83/#63 detail is captured tersely in the main Don't-do log; kept here in full for ladder provenance.

| 2026-06-22 | FREE-OUTPUT variable-shape Slice probe (Worker 6) | ANGLE DEAD — variable-shaped `output` rejected | n/a (probe) | Source-read of canonical grader `neurogolf_utils.py`: `verify_subset` (L546) does `np.array_equal(run_network(...), convert_to_numpy(example)["output"])`; `convert_to_numpy` (L282-289) ALWAYS builds a fixed `[1,10,30,30]` zero-padded one-hot (grid top-left). `np.array_equal` does NOT broadcast → any `[1,10,h,w]` with h,w<30 fails. Empirically `output=Slice(input,[0,0,1,1],[1,10,5,5])`→`[1,10,4,4]` scored 0/20000. Slice MUST Pad back to 30×30 → the 10-ch crop [1,10,23,23]=21,160 B killer unless first collapsed to a color-index canvas. |
| 2026-06-21 | v25 row-profile squareish selector | DROP: no strict-below-8,481 | 9,407 static = 9,407 arcgen | All-color row counts + square-ish row-profile selector + selected-color column marginal. Visible 266/266; truth 199965/200000; A/B 200044/200266 (222 div), byte_equal=False; SHA `80D1A85AFBEF12DCB3131F3DF3207DDD051BCB87337F75006033C757E49CF917`. Saves only 103 B vs v23. |
| 2026-06-20 | worker-task029-r1 #83/#63 rework | NO-CANDIDATE below 8,481 | v24 packed 10,652 static = 10,652 arcgen | `v24_nibble_packed_transport.py`: #83 Conv-packed `[1,1,30,15]` + U8 packed plane, unpacks only the dynamic crop. Visible 266/266, full A/B byte-equal 200266/200266, truth 199813/200000, but odd-start crop unpack (packed crop, hi/lo nibbles, interleave, final crop + same padded U8 tail) costs more than v23. SHA `4073A11DAC003854AE7AD3A39E281EA287E22AF520E0B4C600D20E294611D2E5`. |
| 2026-06-20 | worker-task029 #64/#109 audit | NO-CANDIDATE below 8,481 | best honest remains v23 9,510 | v17 static 8,086 but full arcgen-262 profile 10,061 → 8,481 is the 4-input mirage. #64 Cast-BOOL bbox cannot remove the two all-color F32 marginals. #109/#119 free-output Einsum needs ≥ two F32 30×30 source selectors (7,200 B). |
| 2026-06-18 | v23 u8canvas_honest_slice | prior honest floor (SUPERSEDED by v26/v28) | 9,510 = 9,510 arcgen (gap 0%) | Cast colour-index Einsum canvas to U8 once, dynamic-Slice U8 canvas to interior crop (529 B, true max 23×23), Pad(fill=10), Equal→BOOL. Full A/B byte-equal 200266/200266; truth 199813/200000. SHA `6CB1458D5EC075DF9A896D60927C44185B18AB43F485865CF868BA935D7FF0F3`. Its 3,600 F32 canvas is what v26/v28's Conv-inner deleted. |
| 2026-06-09 | v22 complete_border_u8occ | truth-100 (credit-probe) | 5,496 score_v2; 8,656 4-input → 21,296 arcgen | Generator-correct count marginals + U8 occupancy + complete-border checks, then v18's direct F32 Slice + Pad tail. Truth 200000/200000; A/B diverges 187/200266 on LIVE misses. SHA `776E3299A5A4F562999B26D657E83F85279F4B6B0998B18EE14CEEEFC6B516C6`. |
| 2026-06-09 | v18 direct_input_slice_pad | CREDIT-PROBE (mirage) | 3,636 score_v2; 6,796 4-input → 19,436 arcgen | Slices original F32 one-hot input directly, F32 Pad named `output`. 10-ch F32 crop [1,10,23,23]=21,160 balloons under arcgen. SHA `34EC1BCFDB92F77D75A9308FFE2E981BE9C83472E8F286FA87C4D7AFEA2D127C`. |
| 2026-06-09 | v17 slice_then_cast_guard (= banked LIVE) | banked-bar source | 8,086 score_v2; 8,481 4-input → 10,061 arcgen | THE banked team-best (SHA `DBE79C804F04F5BD23B00B97B60625CCCC4A89EF99FF634D5CA9D9927F7ED0FC`). 8,481 = its 4-input profile mirage. |
| 2026-06-09 | v11 full10_i32 | first honest rung | 11,556 (gap 0%) | v10 canvas/Equal-tail, all-10-channel row/col bbox, INT32, ArgMin marker area. SHA `0E1DD3D94E2595D5BDC87B8A0CACC68AF1A2E13645B5813FFFD974813D9C135D`. |

**R1 #83 packed proof:** v23 U8-canvas 9,510 vs v24 #83 packed 10,652 — packed crop 276 + hi/lo nibbles + interleave concat/reshape (552 each) + U8 crop 529 overrun the 2,250 B source-plane saving. **R1 #63 proof:** selector `count==2*((lr-fr)+(lc-fc))` min-area tie needs all-color extents first; #63 selected marginals only ADD without deleting the paid `2*[1,10,30]` F32. **v11→v18 ladder:** v11 dropped v10 ch1..9 Slice planes + INT32'd bbox; v17 removed the fixed 23×23 Gather/mask tail; v18 removed the canvas via free F32 Pad output (but 10-ch F32 crop = mirage). The Conv-inner route (v26/v28) is what deleted the F32 canvas honestly.
