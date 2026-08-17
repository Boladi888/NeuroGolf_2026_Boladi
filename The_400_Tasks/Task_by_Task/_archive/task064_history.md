# task064 — history (archived 2026-06-22 by Worker 9; size-cap move per WORKER_PROTOCOL)

Superseded attempt rows + old cost-analysis prose. Current best, Don't-do log, and
floor proof live in `Task_by_Task/task064.md`. Renderer lineage: full-plane (v1 14706)
→ single-pack (v6 10860) → dual-pack micro-shrinks (v8 9819 / v11 9811 / v12 9751).

## Attempts (superseded)

### Pre-build novelty statement (2026-05-22)
Polish status: opset 10, already F16-output, prior where-fold tail staged at cost 131,805, and the 20260518 spread sweep reported 0 bytes saved from existing automatic patterns beyond advisory crop-to-active-grid. Hypothesis: the remaining cost is dominated by 8 F16 MaxPool directional expansions over binary marker/rectangle masks plus the full-crop F16 pipeline around two 3x3 Conv material detectors. My approach: first inspect live tensor shapes/values, then try opset 10→12 UINT8 MaxPool for binary expansion, larger directional kernels/fewer iters if equivalent, Rule 12 on Greater/Reduce chains where algebra permits, and if those stall, reformulate the marker-to-rectangle-line rule directly. Hidden-safety: task064 has 267 visible bundled pairs, so it is not a <10-visible-pair exception; preserve 267/267 visible/LIVE and run adversarial before staging.

2026-05-22 | **coloridx_exact_minmax** — opset 12 color-index rewrite: Conv(input,[0..9])→UINT8 label grid, exact valid mask from ReduceMax(input), same-color neighbor rectangle detection in [1,1,24,24], UINT8 MaxPool line expansion, and Min/Max fold for line intersections. | **NOT STAGED** — 267/267 official verify and cost target met, but adversarial-live only 572/5874 (9.7%); LIVE emits multi-channel `11` artifacts on legacy-encoded perturbations when padding channel 0 becomes the counted background, while this scalar-label rewrite emits a single channel. | 131805→39510 (−92295, +1.205 pts) | `task064/prototypes/task064_coloridx_t/task064_coloridx_exact_minmax.onnx`

2026-05-22 | **coloridx_nz_sentinel / minmarker** — more aggressive nonzero-color-domain variant: treat label 0 as invalid/padding, pad neighbor labels once, sentinel channel-0 decode, and keep UINT8 Min/Max line tail. | **REJECTED** — 267/267 official verify at ~28k, but adversarial-live fails badly (best full run 456/5874 = 7.8%; min-marker max-1 probe 136/1869 = 7.3%) for the same off-distribution multi-channel LIVE behavior and color-0 perturbations. | best 131805→28076 (−103729) | `task064/prototypes/task064_coloridx_t/`

2026-06-04 | **v1b runtime-Conv exact interval renderer** — full rewrite in `prototypes/build_task064_v1.py`: `TopK(count+0.01*color)` selects bg/box/dot colors; runtime-weight `Conv(input, dot_onehot)` directly selects a cropped 24x24 dot mask; box bbox comes from free-input `Einsum` marginals; dot row/column first/last indices drive four static interval masks; final `Where(paint30, dot_onehot, input)` overlays the dot color without rebuilding the output. | **STAGED-LOCAL BEST (superseded), NOT FLOOR** — visible 267/267; truth-full 199522/200000 (99.761%); A/B vs current `submission.zip` diverges 8/2267 (first_div=599), rule-rebuild candidate not byte-equal. | 39510→14706 (−24804) | `prototypes/build_task064_v1.py`

2026-06-04 | **v2 cropped ScatterND rows/cols** — render only top/left + 12 possible box rows/columns, then ScatterND strip updates back to 24x24. | **REJECTED (cost regression)** — visible 267/267, truth 199523/200000, score 15981. ScatterND canvases + 2D reshape + false-canvas params outweighed the saved planes. | 14707→15981 (+1274) | `prototypes/build_task064_v2_scatter12.py`

2026-06-04 | **v3 dynamic sparse-dot ScatterND** — runtime dot mask, dynamic `TopK` K=`ReduceSum(dot_score)`, point-level `ScatterND(reduction=max)`. | **REJECTED (score None)** — `(memory=None, params=673)`: shape inference emits symbolic K. Annotating K breaks real examples via ORT preallocation mismatch. | None | `prototypes/build_task064_v3_sparse_dynamic.py`

2026-06-04 | **v4 aligned sparse K=8** — zero unaligned dot pixels before `TopK`, render 8 aligned dots via row-slice `ScatterND(reduction=max)`. | **REJECTED (cost regression)** — visible 267/267; truth 199157/200000 (99.579%); score 17054. | 14707→17054 (+2347) | `prototypes/build_task064_v4_sparse_aligned_k8.py`

2026-06-08 | **v6 uint8 bit-packed interval renderer** — runtime-Conv geometry decoder, inclusive `GreaterOrEqual`/`LessOrEqual` tail, replaces 15 full 24x24 BOOL planes with 3-byte U8 row/col interval packs + bit tables + one packed unpack/threshold tail. | **STAGED (superseded)** — visible 267/267; truth 199522/200000; A/B byte-equal 200267/200267; score/profile 10860. | 14706→10860 (-3846), memory=10502, params=358 | `staged_submits/task064_bitpack_u8_cost10860/` / `prototypes/build_task064_v6_bitpack_u8.py`

2026-06-08 | **v8 route-B row/column-packed rays** — reads dot positions directly from the free one-hot into row/col 3-byte bitsets via `Einsum`, fills prefix/suffix ray intervals in packed U8 space, INT32 gather indices, unpacks once for the final `Where`. | **STAGED (superseded)** — visible 267/267; truth 199522/200000; A/B byte-equal 200267/200267; profile static=9819. | 10860→9819 (-1041), memory=8876, params=943 | `staged_submits/task064_rowpack_rays_cost9819/` / `prototypes/build_task064_v8_rowpack_rays.py`

2026-06-11 | **v11 micro einsum+gt table** — byte-equivalent v8 shrink: `Einsum('bchw->bc')` deletes axes init; direct `gt_table` deletes two BitwiseNot outputs. | **STAGED (superseded)** — visible 267/267; truth 199522/200000; A/B byte-equal; score/profile 9811; SHA256 `47A9532D71D62E54FF5A89E403A66F6CDCB3ECF1F34E6596FF3EB2C230B74DF3`. | 9819→9811 (-8), memory=8870, params=941 | `staged_submits/task064_micro_einsum_gt_cost9811/` / `prototypes/build_task064_v11_micro.py`

2026-06-19 | **free-output separable ray-Einsum (#109/#89) — falsified** — express painted region as row⊗col rank-1/rank-K terms folded into the free [1,10,30,30] output (target ~6500). numpy mirror + 30k corpus rank measurement. | **REJECTED — rule is NOT separable.** (1) 4 ray directions per-cell mutually exclusive (overlap>1 = 0/20000), so multi-term SUM == OR is valid. BUT (2) painted submask per side is multi-rank: up to 7-8 distinct rows/cols, ≤6 distinct column-intervals per side (per-row-varying `dot_first_col[h]`); ~22% of left cases have ≥2 intervals. Rank-1 outer product paints a rectangle not per-row rays; fixed-K direction-Einsum can't carry per-row interval variation; per-DOT-term Einsum is dynamic-K (v3 dead-end). Two orientations pack on perpendicular axes, can't OR in packed space → two 576B unpacks before OR are structural. | no candidate | `prototypes/separable_numpy_probe.py`

2026-06-08 | **v9 generator-exact color pick** — tests whether 2nd TopK color is a filled rectangle (`area==count`, height/width match), swaps to 3rd color when dots outnumber box. | **REJECTED (truth fix costs too much)** — visible 267/267; truth 200000/200000; score 10459. The cheap `area==count` test alone scored 10146 / 199997 but 3 random dot sets matched the product. | 9819→10459 (+640), memory=9515, params=944 | `prototypes/build_task064_v9_truth_colorpick.py`

2026-06-08 | **v10 directional U8 scan renderer** — four `MaxPool` prefix/suffix scans gated by box span + outside-box side tests, free final `Where`. | **REJECTED (large cost regression)** — visible 267/267; score 16285. Scan core is fine; the dynamic dot-color mask needed `dot_score30` 3600B + `dot_score24` 2304B before scans. | 9819→16285 (+6466), memory=16196, params=89 | `prototypes/build_task064_v10_directional_scan.py`

2026-05-19 | **where_fold_tail** — paint_mask binary {0,1}; replace Sub/Mul/Max keep+paint chain with Greater→Where; saves 23616B. | **STAGED (superseded)** — 267/267 verify, 5874/5874 adversarial, cost 155421→131805. | 155421→131805 | `.hermes/board/staged_submits/task064_wherefold_t_5ddee6cd_on_5806_87/`

## Cost analysis (historical profiles)

2026-06-04 v1b (`score=14706`, memory=14612, params=94):
- `dot_score` runtime-weight Conv `[1,1,24,24]` F32 = 2304 B = entry cost for dot-mask space.
- `paint30` BOOL pad = 900 B fixed output-overlay condition.
- 15 separate `[1,1,24,24]` BOOL planes (576 B each): 4 dot-bound comparisons, 8 side/window `And`s, 3 `Or`s = real sub-10k blocker.
- Box bbox: `Einsum(input, box_onehot, 'bchw,bc->bh/bw')` cut box mask to two 120 B vectors.

2026-06-08 v6 (`score=10860`, memory=10502, params=358):
- 15 full BOOL planes → U8 interval packs. Dominant tail = packed unpack/threshold chain: `h_bits_u8`, `v_bits_u8`, `v_bits_u8_t`, `v_bits_u8_hshape`, `paint_bits_u8`, `paint_bits_b`, `paint24` (7 × 576 B) + 900 B `paint30`.
- `dot_score` still largest single tensor (2304 B runtime Conv).

2026-06-08 v8 (`score=9819`, memory=8876, params=943):
- Box extraction 6 ops / 272 B (`box_row_counts`/`box_col_counts` 2×120 B + 4 scalar ArgMax edges 4×8 B); span gates 6 ops / 144 B.
- Dot-position extraction 936 B/orientation: packed `Einsum` (360 B) + Slice (288 B) + INT32 cast (288 B). The 2304 B dot_score plane is gone.
- H packed interval 921 B + unpack 1224 B (`h_pack4`/`h_bits_u8`/`h_flat_u8`); V symmetric.
- Shared tail 2052 B: `paint_bits_u8` 576 + `paint24` 576 + `paint30` 900, then free `Where`.

2026-06-08 post-v8 floor attacks:
- Dot-pack direct-24 collapse: `Einsum` preserves the unsummed axis label; a `[24,3]` table alone can't shrink the opposite axis. Selector-matrix saves two 360 B outputs but adds 1440 selector params. No win.
- Generator-exact color selection: strict rectangle test adds 640 B (10459). Hidden-safe but not the cost floor.
- Directional scan loses (no cheap starting dot mask): dynamic dot mask 5904 B before 4×576 B scans → 16285.

2026-06-11 v11 (`score=9811`, memory=8870, params=941):
- Byte-equivalent to v8, saves 8 B: `Einsum('bchw->bc')` deletes `axes_hw`; direct `gt_table_u8` deletes two `BitwiseNot` outputs. Dominant tensors unchanged.
- No-epsilon TopK probe scored 9761 but failed visible 266/267 + 9/5267 A/B divergence → `color_eps` ties are semantically charged, keep.

2026-06-21 v12 (`score=9751`, memory=8858, params=893):
- Direct `OneHot(axis=0)` removes color ramp + scalar cast/equal/cast; duplicate INT64 scalar inits shared; lt/gt tables 30x3→24x3 with two INT64 dummy clamps. Profile static-exact 9751.
- Dominant tensors unchanged: `paint30` 900 B, six 24x24 unpack/paint tensors 576 B each, two 30-axis dot-pack Einsums. Same #110 renderer family.

2026-06-22 W9 angle-B directional scan (`score=19889`, memory=18368, params=1521):
- Reads dots cheaply via 4-operand Einsum (no Conv dot mask), unpacks ONCE to [1,1,24,24], 4 full-axis MaxPool prefix/suffix fills clamped at scalar walls, free Where. Visible 267/267 (rule correct).
- Regresses because the fill happens in spatial 576B planes (4 scans + ~12 BOOL band/paint planes) instead of v8/v12's cheap 72B packed 3-byte space. Same failure mode as v10.
