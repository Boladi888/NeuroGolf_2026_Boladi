# task319 — archived history

Superseded prose moved out of `task319.md` (WORKER_PROTOCOL task-md-size rule). Current best, full Attempts table, Rule, and live Don't-do log stay in `task319.md`.

## 2026-05-27 RESUME EXAMINATION — Path A floor confirmation (SUPERSEDED by v35/v36)

**2026-06-20 superseded note:** the v34 floor below was formulation-relative. Later library technique #98 applies because the generator hard-bounds all active content to the top-left 19x19 region; a dilated Conv can combine channel colour projection with that fixed crop. v35 deletes the 30x30 colour-index carrier and stages at 13,681. **2026-06-24: v36 deletes the full UINT8 bbox mask via opset-20 ReduceMax(BOOL), staging at 12,714.**

After v34 SHIPPED at +15.27 LB (6240.91 → 6256.18), resumed to look for sub-16k savings. **CONFIRMED (then-relative): v34 polished 16,716 IS the structural floor for this rule.** Tried four user-suggested directions plus several others:

| Direction | Outcome |
|---|---|
| 1. Separable rank-1 Conv (techniques.md F1 Win 1) | **N/A**: our Conv is already 1×1 spatial (kernel `[1, 10, 1, 1]` — projecting across CHANNEL axis, not spatial). The channel dimension is not factorizable. The 30×30 output IS the floor because Conv on FLOAT input must produce FLOAT output at full spatial resolution. |
| 2. ConvTranspose `output_padding` to skip pre-pad | **REJECTED**: ORT's ConvTranspose with `output_padding` produces incorrect output (cells appear interleaved 1/0 pattern; bottom-right 6x6 of 16x16 has non-zero values where they should be zero). Confirms task319.md earlier note from v17 attempt. |
| 3. Bbox + count heuristic to skip QLinearConv pipeline | **REJECTED**: 64.6% synth pass rate vs ≥99.95% Path A gate. Heuristic gives many false positives in source identification. |
| 4. Sentinel-pad mask to skip canvas_bool And | **NET NEGATIVE**: pad adds 234 bytes (mask grows from `[1, 3, 19, 19]` to `[1, 3, 20, 20]` for both BOOL and UINT8 versions); saving canvas_bool And (300) yields +30 net worse. |
| 5. INT16 combined_idx (opset 14 Add INT16) | **NET ZERO**: GatherElements requires INT32/INT64 indices; INT16 → Cast → INT32 introduces same final intermediate. |

The (then) structural floor was attributed to these components:
- **Conv color-index Conv FLOAT [1, 1, 30, 30] = 3600** (techniques.md §23 projection-cache). *(v35 #98 dilated crop replaced this with [1,1,19,19]=1444.)*
- **mask intermediates 3 × 1083 = 3249** (Equal output BOOL, Cast UINT8 for ReduceMax bbox, Reshape BOOL for canvas GatherElements). *(v36 opset-20 ReduceMax(BOOL) deletes the UINT8 cast → 2 × 1083 now.)*
- **combined_idx + flat INT32 = 2400** (GatherElements needs INT32/INT64 indices; INT16 not accepted).
- **Pad + Resize + Transpose + QLinearConv at UPSCALE=16 = ~2200** (UPSCALE=16 needed for synth 99% gate: max picked offset is 6 in synth distribution).
- **Canvas pipeline (~2100)** + **output construction (~360)** + **bbox calc (~500 small ints)** + Conv weights (40 init).

**v33 (16,250) was unsafe**: its Conv weights [0..9] caused color-0 shape cells to be indistinguishable from padding cells (all-channels-zero). Visible has 0/267 color-0 shapes so it passed 267/267, but synth has 35.3% color-0 shapes and dropped to 80.6%. Fixed in v34 by shifting weights to [1..10] (padding → 0, color c → c+1). Also UPSCALE=14 missed picked offsets 5, 6 in synth (5.8% pairs); v34 uses UPSCALE=16 (offsets [0, 6] cover 99.93%).

**Pivot recommendation (2026-05-27, since reopened):** task319 was thought structurally complete at v34 polished. (Reopened: #98 → v35 13,681; opset-20 ReduceMax(BOOL) → v36 12,714.)

Top remaining intermediates in v18-era polished (historical):
- `mask_full_u8` (9000) — Cast(input, UINT8). Required for downstream Gather/Slice.
- `mask_full_3` (2700) — Gather(mask_full_u8, shape_indices, axis=1).
- `row_idx_b` (2280) — INT32 Expand'd index for GatherElements.
- `signed_corr` (1764) — Conv output [3, 3, 7, 7] FLOAT.
- 7 × 1200 = 8400 bytes from canvas_4d_f / two_canvas / signed_kernel / row_has_f / col_has_f / in_bbox_f / col_idx_b.

Failed v17 variant: ConvTranspose with `output_padding` to produce 16×16 directly (skips the upscale_10 + Pad chain) — ORT's ConvTranspose with output_padding produces incorrect output (cells appear interleaved 0/value), so we fall back to explicit Pad. Verified empirically on a 5×5 all-ones test.

Early (v1-era 635k) op dominators and headroom notes: the two `Conv` ops producing `[10, 10, 13, 13] F32` correlation tensors (~270k bytes each), plus the `Tile`-based 2x upscale producing `[10, 24, 24] F32`. Headroom then identified: drop canvas/upscale dims 12/24 → 10/20; cast intermediate canvases to UINT8/BOOL; pre-mask irrelevant channels; replace Tile+Reshape upscale with ConvTranspose. All of these were applied across v2–v35.

## 2026-05-20 OB_MID_SMALL worker notes (stale)

### 2026-05-20 - OB_MID_SMALL worker
- f16 conversion (robust + direct + opset bump): NO WIN
- Reason: small task (<20k cost) - Cast wrapper overhead (~1k bytes) exceeds f16 savings
- spread_scan (21 patterns): 0 wins
- Result: NO CANDIDATE
- Future: try ReduceMax(axis=[2])+ReduceMax(axis=[3]) for bbox; nonzero_shortcut pattern; or full from-scratch rebuild (multi-hour like task071/task049/task071).

### 2026-05-20 - OB_MID_SMALL worker (rebuild - SUCCESS)
- Technique: **opset11_bump_pad_uint8_output** (delta=-741 bytes)
- Verify: 267/267
- Adversarial: 6675/6675 (100%)
- Notes: Original opset 10 Pad uses pads-as-attribute and float-only types.
  Bumped opset 10 -> 11 to enable pads-as-input form, then changed Cast(BOOL->FLOAT)
  to Cast(BOOL->UINT8). Pad now operates on UINT8 (1 byte/elem instead of 4).
  Output type becomes UINT8. Safe under (out > 0.0) on uint8 tensors.

---

## 2026-07-15 cleanup compression source for task319.md

Full pre-compression task file archived from `Task_by_Task/task319.md` because repo_lint manual content exceeded 20 KB.

```markdown
# task319

<!-- AUTO:BEGIN -->
<!-- generated by tools/generate_task_mds.py; manual edits OUTSIDE this block -->

**STATUS:** LIVE cost **16,716** (live_costs.json, refreshed 2026-06-05).
(one row: `python tools/agent_helpers/show_task.py --task N` — NEVER Read task_state.csv whole)

**Linked research / prototypes** (auto-detected by filename):
- _none found_
<!-- AUTO:END -->

> **⚠ 2026-07-12 (bo_ng_cld_worker2, Claude lane) — LIVE BAR MOVED to 5833; WALL.**
> The AUTO STATUS above (16,716) and the v35/v36 floor notes below are STALE. Live team-best = **5833** (manifest, owner Vamshi, 2026-07-12, sha `5df0de2e…`), a newer **bit-pack** lineage (NOT in this repo's prototypes; the v34–v36 Conv color-index lineage here is 12,714–16,716 = 2–3× worse). Two passes returned a **priced WALL/DROP**: the prompt's floor_est ~1500 is fantasy — the source-shape selection needs an irreducible 2-D bitmap witness (all cheap scalar proxies falsified on 267 data), and the two `[10,30]` F32 carriers (2400 B) are near-forced. **Pass 2 built + validated (267/267) two from-scratch front-ends: single-carrier = 16,639 (column-derivation via unpack is refuted); lean two-carrier front-end = 5,268 for carriers+bboxes+aligned-bitmaps ALONE — already 90% of the live 5,833 total, leaving only 565 B for the witness+render that the live best spends ~2,000 B on.** A from-scratch rebuild cannot beat 5,833. Full evidence + floor proof: [`task319/hard_records/shape_only_rearrange_wall.md`](task319/hard_records/shape_only_rearrange_wall.md).

<!-- HIST:BEGIN -->
<!-- mined by tools/mine_task_history.py; rerun to refresh -->

**Git history** (commits touching files or subjects mentioning this task):
- `4390a932` — docs(task319): WORKER RETIRED, v34 = structural floor confirmed
- `0a28ff71` — +36.55 LB batch: task002/017/243/076/101 SHIPPED 5988.64 -> 6025.19
- `a81a00b0` — agent_helpers/ (9 scripts) + auto-graph-diag in all 401 task .md files
- `e987304c` — exemplars/02_bool_first/: 6 of 10 (task293, task021, task196, task273, task319, task299, task233, task368)
- `3a5ae306` — OB_MID_SMALL task319: opset11+uint8 pad trick (-741 bytes)
- `e56582d6` — OB_MID_SMALL worker session: 0 staged, 62 tasks documented
- `751fd1d6` — session 2026-05-18: accumulated task notes across worker runs
- `3c61b5d5` — Junction-relocate BatchPilotArchive/ out of tree to .notongit/
- `54ec3c87` — Mine session logs + recipe inventory + LIVE_ALGORITHM for all recipe-tasks
- `f2023f06` — Update docs + re-mine HIST (401 tasks) + fix re.sub backslash escape bug
- `e69612d9` — Mine git/session/lessons/notes history into per-task .md HIST blocks
- `6668c221` — Per-task .md folder (Task_by_Task/) + migrate task_cards/ + update docs
- `b2e3cd93` — File away scoring artifacts
- `315ed3f3` — Archive pilot artifacts
- `316414b0` — Track generated NeuroGolf artifacts and env path overrides
- _...and 9 more_

<!-- HIST:END -->


## Rule (decoded 2026-05-26)

Each input contains a uniform background colour plus exactly **3** same-colour shapes. Among those 3 shapes, one (**B**) equals another (**S**) upscaled by 2x in both axes, possibly clipped at the active-grid edges. Output is **S**'s bounding-box bitmap rendered with the input background colour (cells of `S` painted in `S.color`, holes painted in `bg`).

Tiebreaker (for the 12/267 cases where 2+ shapes satisfy the upscale relation): pick the candidate with the smallest bbox area, then smallest h+w.

Verified **267/267** via numpy reference `prototypes/task319_numpy_rule_v1.py` AND via `verify_against_live.py --tid 319 --live-zip submissions/submission.zip` (Prototype vs LIVE 267/267, vs expected 267/267).

## Algorithm sketch (numpy)

```
shapes, bg = bbox + bitmap per non-bg colour       # always 3 shapes
candidates = [i for i in shapes
              if any(contains_subrect(upscale2(shapes[i].bitmap),
                                       shapes[j].bitmap)
                     for j != i)]
picked = min(candidates, key=lambda i: (shapes[i].h*shapes[i].w,
                                         shapes[i].h+shapes[i].w))
out = where(picked.bitmap == 1, picked.colour, bg)
```

Reference implementation: [task319_numpy_rule_v1.py](task319/prototypes/task319_numpy_rule_v1.py).

## 2026-06-20 worker-319a visual/generator confirmation

Visual pass matches generator `ce602527`: active grid is top-left embedded with width/height 15..19, uniform background, two source sprites plus one clipped 2x magnified sprite. The generator's `draw()` writes `idx==0` cells into the output bitmap at the source sprite bbox size (`wide` x `tall`, 3..5 each), while the magnified copy is drawn in `magcolor` and must be partly hidden off-grid. Object roles: `colors[0]` is the source/output shape colour, `magcolor` is the clipped 2x witness, `colors[1]` is a distractor shape, and `bgcolor` fills holes in the output.

Falsifications from ARC-GEN/source:
- Not "crop the magnified object": output uses `colors[0]` and source dimensions, not `magcolor` or 2x dimensions.
- Not "pick any/smallest shape": generator adds a distractor sprite; the source is identified by the 2x clipped witness relation, with bbox-area tie-breaking only after match detection.
- Not "shape colours exclude 0": colours are random except for exclusions from background/magnified colour, so the color+1 padding sentinel remains required.

Scratch formulations considered:
- Existing v34 structural path: full-canvas color-index Conv, cropped mask, bbox/canvas render, QLinearConv match. Correct and byte-equal to LIVE, but pays a 30x30 F32 color-index plane.
- v35 #98 crop+channel-reduce Conv: use a zero-far-tap dilated Conv with dilations 11,11 to emit only the hard 19x19 active-bound color-index plane, preserving the v34 +1 colour sentinel. This deletes the full 30x30 F32 carrier and won.
- Cheaper structural/rule rewrite: bbox/count heuristic and smaller UPSCALE routes remain rejected by prior synth failures; free-output separable output is not applicable because the selected sprite bitmap is arbitrary 2D content rather than a low-rank render.

## Attempts

| Date | Approach | Result | Cost | Notes |
|---|---|---|---:|---|
| 2026-05-25 | canonical (hash-MLP lookup of 267 fingerprints → [267,25] table) | 267/267 visible, **0 LB hidden** (cherry13b §27g) | 16,552 | _lookup memorization; max latent +15.29 LB if unlocked_ |
| 2026-05-26 | `structural_v1` — 2x-upscale-match via Conv cross-corr (positive + negative kernel) + ArgMin smallest-bbox tiebreaker, canvas 12x12 / upscale 24x24 | 267/267 vs expected, 267/267 vs LIVE, 1486/1500 (99.07%) vs independent synth | **635,984** | structural rewrite; expected +11.64 LB asymmetric unlock |
| 2026-05-26 | `structural_v2` — same algorithm with canvas dropped from 12x12 → 10x10 (max observed BIG bbox dim) | 267/267 vs expected, 267/267 vs LIVE | **530,130** | -17% cost; +11.82 LB |
| 2026-05-26 | `structural_v3` — fused two Convs (positive + negative kernel) into ONE Conv with signed kernel `2*canvas - in_bbox`; match condition becomes `signed_corr == canvas_sum[j]` | 267/267 vs expected, 267/267 vs LIVE | **461,530** (polished **446,746**) | -13% cost; +11.96 LB (polished +11.99 LB) |
| 2026-05-26 | `structural_v4` — TopK(is_shape, k=3) reduces all Conv-related tensors from 10 channels to the 3 actual shape channels; Conv output shrinks from [10,10,11,11] → [3,3,11,11] | 267/267 vs expected, 267/267 vs LIVE, 1488/1500 (99.20%) vs independent synth | **337,450** (polished **325,466**) | -27% cost; +12.27 LB (polished +12.31 LB) |
| 2026-05-26 | `structural_v5` — input Slice from 30×30 → 19×19 active grid (encoder anchors at top-left; max active grid in task319 data = 19×19, matches canonical's `v9=[1,1,19,19]` slice) | 267/267 vs expected, 267/267 vs LIVE, 1489/1500 (99.27%) vs synth-19max | **279,181** (polished **267,194**) | -21% cost; +12.46 LB (polished +12.51 LB) |
| 2026-05-26 | `structural_v6` — BOOL output construction via And+Or (skip INT32 chain) + drop mask_i32/mask_f32 (use input directly for Reduce ops) | 267/267 | **160,240** (polished **148,254**) | -45% cost; +13.02 LB (polished +13.09 LB) |
| 2026-05-26 | `structural_v7` — INT32 indices (vs INT64), BOOL canvas pipeline (And vs Mul) | 267/267 | **136,120** (polished **124,254**) | -15% cost; +13.18 LB (polished +13.27 LB) |
| 2026-05-26 | `structural_v8` — drop input_crop intermediate (Cast input → UINT8 mask directly, use input for Reduce ops); rewrite output construction via single color_tensor + Equal | 267/267 | **118,868** (polished **107,915**) | -13% cost; +13.31 LB (polished +13.41 LB) |
| 2026-05-26 | `structural_v9` — Cast input directly to UINT8 (skip BOOL intermediate) | 267/267 | **109,867** (polished **98,914**) | -8% cost; +13.39 LB (polished +13.50 LB) |
| 2026-05-26 | `structural_v10` — ConvTranspose for upscale (vs Tile+Reshape) + BOOL output dtype (skip final Cast) + pre-shape inits | 267/267 | **94,861** (polished **83,908**) | -14% cost; +13.54 LB (polished +13.66 LB) |
| 2026-05-26 | `structural_v11` — pre-shape ch_arange_b as [1, 10, 1, 1] + name Equal output directly as "output" (skip Reshape) | 267/267 | **76,845** (polished **74,905**) | -11% cost; +13.75 LB (polished +13.78 LB) |
| 2026-05-26 | `structural_v12` — TopK gather to 3 shape channels BEFORE canvas building (shrinks canvas pipeline tensors 10→3 channels) | 267/267 | **62,430** (polished **60,490**) | -19% cost; +13.96 LB (polished +13.99 LB) |
| 2026-05-26 | `structural_v13` — skip Cast(row_has, INT32) — ArgMax works on FLOAT directly | 267/267 | **60,030** | -4% cost; +14.00 LB |
| 2026-05-26 | `structural_v14` — ReduceMax-first match (vs Equal-first) skips [3,3,11,11] BOOL intermediate; gather mask via mask_full → mask_full_3 (avoid mask_3d_full duplicate) | 267/267 | **53,383** (polished **51,429**) | -14% cost; +14.11 LB (polished +14.15 LB) |
| 2026-05-26 | `structural_v15` — ArgMax on [1,10,30] directly (skip row_has Squeeze intermediate) | 267/267 | **51,303** (polished **49,349**) | -4% cost; +14.15 LB (polished +14.19 LB) |
| 2026-05-26 | `structural_v16` — reuse canvas_4d_f as BOTH ConvTranspose input AND Conv kernel; build signed_kernel directly in 4D form | 267/267 | **48,015** (polished **47,263**) | -4% cost; +14.22 LB (polished +14.24 LB) |
| 2026-05-26 | `structural_v17` — Slice canvas_4d_f to canvas_5 (5x5 SMALL portion) for ConvTranspose; explicit Pad to 16x16; signed_corr shrinks from [3,3,11,11] → [3,3,7,7] | 267/267 | **45,216** (polished **41,378**) | -12% cost; +14.28 LB (polished +14.37 LB) |
| 2026-05-26 | `structural_v18` — ArgMax(select_last_index=1) for row_max/col_max (skip the entire Gather(rev_idx) → ArgMax chain) | 267/267 | **42,625** (polished **38,787**) | -6% cost; +14.34 LB (polished +14.43 LB) |
| 2026-05-26 | `structural_v19` — UINT8 ReduceMax for row_has/col_has (Rule 11, Onnx_Advanced/techniques.md): operate on mask_full_u8 instead of FLOAT input | 267/267 | **40,825** (polished **36,987**) | -5% cost; +14.38 LB (polished +14.48 LB) |
| 2026-05-26 | `structural_v20` — ConvInteger replaces FLOAT Conv: signed kernel built via Where chain in UINT8 (values {0, 1, 2}, weight_zp=1 for effective {-1, 0, 1}); kernel + ConvTranspose path stays UINT8 throughout | 267/267 | **33,853** (polished **33,108**) | -17% cost; +14.57 LB (polished +14.59 LB) |
| 2026-05-26 | `structural_v21` — Resize (mode=nearest, scales=[1,1,2,2], coord=asymmetric) replaces Unsqueeze+Tile+Reshape chain for 2x upscale | 267/267 | **33,470** (polished **32,725**) | -1% cost; +14.58 LB (polished +14.60 LB) |
| 2026-05-26 | `structural_v22` — 4D output construction (Less on [1, 1, 30, 1] / [1, 1, 1, 30] arange inits, skip ct_b Reshape) + BOOL Pad on picked_canvas_b (skip bitmap_30x30_u8 → Greater chain) | 267/267 | **31,617** (polished **31,328**) | -4% cost; +14.64 LB (polished +14.65 LB) |
| 2026-05-26 | `structural_v24` — flatten GatherElements to single 2D op (combined_idx = row*W + col, mask_flat [3, 361]); skips rows_gathered intermediate | 267/267 | **31,472** (polished **31,183**) | -1% cost; +14.64 LB (polished +14.65 LB) |
| 2026-05-26 | `structural_v25` — bbox computation MOVED after shape Gather (only on 3 channels via ReduceMax(mask_3d, axes=[2/3]) + ArgMax); skips row_min/col_min Gather ops | 267/267 | **30,042** (polished **29,753**) | -5% cost; +14.69 LB (polished +14.70 LB) |
| 2026-05-26 | `structural_v26` — skip mask_3d Squeeze intermediate (Reshape mask_3_cropped [1, 3, 19, 19] directly to mask_flat [3, 361]) | 267/267 | **29,055** (polished **28,766**) | -3% cost; +14.72 LB (polished +14.73 LB) |
| 2026-05-26 | `structural_v27` — consolidate canvas BOOL pipeline: skip canvas_3 (UINT8) + canvas_3_4d (UINT8) + canvas_3_4d_bool intermediates; do Where chain for shifted_kernel in 3D then Unsqueeze once | 267/267 | **28,002** (polished **27,715**) | -4% cost; +14.76 LB (polished +14.77 LB) |
| 2026-05-26 | `structural_v28` — QLinearConv (UINT8 output) replaces ConvInteger (INT32 output) — same effective signed-kernel computation (scale=1, w_zp=1, y_zp=0). Saves signed_corr 1764→441 bytes | 267/267, 1490/1500 (99.33%) vs synth | **26,648** (polished **26,360**) | -5% cost; predicted **+14.81 LB** (polished **+14.82 LB**) |
| 2026-05-26 | `structural_v29` — output built at [1, 10, K_H, K_W]=[1, 10, 10, 10] then Pad to "output" (skipped from memory). Shrinks 4 output tensors from [1, 1, 30, 30] → [1, 1, 10, 10] | 267/267 | **24,496** (polished **24,205**) | -8% cost; +14.89 LB |
| 2026-05-26 | `structural_v30` — K_OUT=5 (max picked = 5x5 in data), UPSCALE=14 (max picked offset = 4), removed dead is_shape_i64, use canvas_5_3d_bool for picked extraction | 267/267 | **22,730** (polished **22,439**) | -8% cost; +14.98 LB |
| 2026-05-26 | `structural_v31` — Conv-based color-index (Onnx_Advanced techniques.md §23): Conv with weight=[0..9] computes color_idx_f in one shot, replacing mask_full_u8 (9000) + mask_full_3 (2700) with Conv 3600 + cast 900 + equal 2700 | 267/267 | **18,151** (polished **17,860**) | -20% cost; +15.21 LB |
| 2026-05-26 | `structural_v32` — Path B: Cast Conv→UINT8 immediately, Slice on small UINT8 (361), Equal on cropped → BOOL [1, 3, 19, 19]. Skips the [1, 3, 30, 30] BOOL Equal intermediate (2700) | 267/267 | **16,694** (polished **16,403**) | -8% cost; +15.29 LB |
| 2026-05-26 | `structural_v33` — Pad-Resize reorder: pre-pad canvas_5 to [3,1,7,7] BEFORE Resize so Resize directly outputs [3,1,14,14]. Skips upscale_10 intermediate (300) | 267/267 BUT **synth 80.6% FAILS gate** | **16,541** (polished **16,250**) | -1% cost; +15.30 LB but UNSAFE — bugs uncovered: (1) Conv weights [0..9] make padding (all-channels-zero) indistinguishable from color-0 shape cells; 35.3% synth pairs have shape color 0; (2) UPSCALE=14 misses picked offset > 4 (synth has offsets up to 6) |
| 2026-05-26 | `structural_v34` — **Fixed v33 bugs**: Conv weights shifted to [1..10] so padding → color_idx=0 ≠ any active color; Add shape_indices+1 in INT32 for Equal compatibility. UPSCALE reverted to 16 (covers offsets [0,6]); synth distribution shows 99.93% coverage with offset≤6. | 267/267, **1490/1500 (99.33%) synth PASSES** | **17,007** (polished **16,716**) | +3% cost vs v33; **+15.28 LB SAFE** |
| 2026-06-20 | `structural_v35` — techniques #98: replace v34's full 30x30 color-index Conv+Cast+Slice with one dilated zero-far-tap Conv that emits [1,1,19,19] directly | visible 267/267; truth 197397/200000 same as v34; AB vs submission.zip 200267/200267 byte-equal; profile==static | **13,969** (polished **13,681**) | -18.2% vs v34; +0.2004 points vs team-best 16,716 |
| 2026-06-24 | `structural_v36` (worker w17) — opset 13->20 + direct `ReduceMax(BOOL)` for the bbox row/col-presence: delete the full UINT8 `mask_3_cropped` [1,3,19,19] (1083 B), Cast only the tiny [1,3,19] reduced results to UINT8 for ArgMax | visible 267/267; truth 197397/200000 identical to v34/v35; **AB vs v35 team-best parent AND vs submission.zip both 200267/200267 byte-equal (0 div)**; profile==static | **13,003** (polished **12,714**) | -7.1% vs v35 team-best (13,681); +0.0732 points; sha 6DA41F43 |

## Cost analysis

Best so far: **`structural_v36` polished at cost 12,714 → +15.549 points predicted; byte-equal to the v35 team-best parent AND to submission.zip over the full 200,267-pair A/B (0 divergences)**. Staged at `staged_submits/task319_dilated_cropconv_boolreduce_cost12714_sha6DA41F43_gatesgreen/` (sha 6DA41F43A2D86EA37CB3B1AEF831A511B5A66E7144D9069F5890EC60BDF99BB5). v36 = v35 + opset 20 `ReduceMax(BOOL)` deleting the 1083 B full UINT8 bbox mask (-967 B vs v35). Prior `structural_v35` at 13,681 is superseded.

**Don't-do (v36 floor notes):** the GatherElements repositioning index pair (`combined_idx` [3,10,10] + flat [3,100] = 2400 B) is NOT trivially removable — GatherElements indexes one flattened axis so both the 3D broadcast and its flat reshape are load-bearing (#143: the map is already smaller than the canvas it deletes). Dynamic Slice (#146) cannot replace it because the 3 shapes need 3 different per-channel shifts and padding the mask to allow a uniform window costs more (2352 B). Canvas/kernel 10x10, UPSCALE=16, SMALL=5 are all measured floors.

**v33 was bugged**: it cost 16,250 but failed synth at 80.6% — see the trap notes above. v34 is the correct pre-v35 structural parent.

Milestones only (full per-version detail in the Attempts table above):

| version | cost | points | key change |
|---|---:|---:|---|
| canonical (LIVE) | 16,552 | 15.29 (0 on hidden) | hash-MLP lookup [267,25] |
| v1 | 635,984 | 11.64 | initial structural draft |
| v18 polished | 38,787 | 14.43 | ArgMax(select_last_index=1) — skip Gather(rev_idx) |
| v28 polished | 26,360 | 14.82 | QLinearConv (UINT8) replaces ConvInteger |
| v31 polished | 17,860 | 15.21 | Conv-based color-index (techniques.md §23) |
| v34 polished | 16,716 | 15.28 | Fixed color-0 bug + UPSCALE=16; synth 99.33% PASS |
| v35 polished (worker-319a) | 13,681 | 15.48 | #98 dilated crop+channel-reduce Conv; AB byte-equal over 200,267 |
| **v36 polished (worker w17)** | **12,714** | **15.55** | opset 20 `ReduceMax(BOOL)` deletes 1083 B full UINT8 bbox mask; AB byte-equal to v35 parent + submission.zip |

## COMPRESSED HISTORY

The 2026-05-27 RESUME EXAMINATION (v34 formulation-relative floor analysis, the 5 rejected directions, v33 color-0 bug postmortem, and v1/v18-era op-dominator notes) is archived → `_archive/task319_history.md`. Superseded by v35 (#98 dilated crop, 13,681) and v36 (opset-20 ReduceMax(BOOL), 12,714). The live floor notes are in the Cost analysis "Don't-do (v36 floor notes)" block above.

## Don't-do log

- **Do not** ship a 12x12 sub-rect "exact equality" check (`2*corr == self_corr + canvas_sum`) — the BIG shape is clipped, so its canvas (with 0-padding outside its bbox) does NOT match a sub-rect of upscale(SMALL) exactly. The correct condition restricts the comparison to BIG's bbox area via either (a) two Convs (positive + `in_bbox - canvas` negative kernel), or (b) **one Conv with signed kernel `2*canvas - in_bbox`** (267/267 either way; signed kernel saves ~50k cost — see v3).
- **Do not** assume "pick smallest-bbox shape" alone. That's only ~178/267 — the upscale-match property is required and identifies the rule uniquely 255/267, with the bbox-area tiebreaker resolving the remaining 12/267.
- **Do not** treat the canonical as positive-weight. cherry13b §27g confirmed task319 contributes 0 LB on hidden at the current canonical (16,552). Any working structural ship is upside; downside is 0.
- **Do not** keep all 10 colour channels through the heavy ops (Conv, upscale, etc.). The input always has exactly 3 shape colours; `TopK(is_shape, k=3)` cleanly identifies them, and Gather-then-process on the 3 cuts Conv-tensor size by 11× (`[10, 10, 11, 11]` → `[3, 3, 11, 11]`). See v4: -27% total cost.
- **Do not** "exact-match within the full canvas" — the SMALL upscaled extends BEYOND BIG's bbox (those are the cells that got clipped off in the input). The match must be restricted to BIG's bbox area (via in_bbox kernel or signed kernel).

<!-- SPREAD:BEGIN -->
<!-- generated by tools/spread_annotate_notes.py from 20260518 scan; manual edits OUTSIDE this block -->

## Spread-knowledge scan (20260518)

Current canonical cost: **19184** (memory=11585, params=7599, points≈15.138168).

Pattern sweep verdicts (each pattern measured INDEPENDENTLY vs the original — patterns overlap, so the sum overstates the real cumulative savings):

| Pattern | Verdict | Δ cost (vs original) | Notes |
|---|---|---:|---|
| `clip_mul_to_min` | — | — | no Clip[0,1] + Mul(mask) pairs detected |
| `const_to_init` | — | — | no Constant nodes with a 'value' attribute to convert |
| `crop_to_size_advisory` | ? | -21560 | max active grid is 19×19 but graph has 1 [...,30,30] intermediate(s). Crop-pr... |
| `fp16_selective` | ✗ verify | — | fp16-converted; driver must verify cost actually drops and output is within t... |
| `onnxsim` | — | 0 | onnxsim found additional simplifications \| cost unchanged |
| `safe_opt` | — | — | no changes from safe-opt + extended passes |
| `scalar_broadcast_ones` | — | — | no uniform-valued float initializers consumed only by broadcasting ops |
| `slice_axis_collapse` | — | — | no Slice nodes with redundant axis-0 entries to drop |

**Sequential pipeline result**: 0 bytes saved — already at the floor for the patterns currently in `tools/spread_patterns/`.

**Advisory** (crop_to_size_advisory): max active grid is 19×19 but graph has 1 [...,30,30] intermediate(s). Crop-process-pad could save ~21560 bytes (rough est). — estimated savings if applied manually: ~21560 bytes.

_Re-run: `.venv/Scripts/python.exe tools/spread_scan.py` (updates JSON in `Logs/spread_scan_results/`); then `.venv/Scripts/python.exe tools/spread_annotate_notes.py`.
<!-- SPREAD:END -->

_(Two stale 2026-05-20 OB_MID_SMALL worker notes — f16 NO-WIN and the opset11 pad-uint8 -741 B rebuild — archived → `_archive/task319_history.md`.)_

<!-- BEGIN AUTO-GRAPH-DIAG -->

## Graph Diagnostic (auto-generated)

**Cost**: 16,716 | **Memory**: 16,566 | **Params**: 150 | **Points**: 15.2759
**Nodes**: 97 | **Initializers**: 42 | **Opset**: 13 | **Graph output dtype**: BOOL
**Max active grid (from `task319.json`)**: 19×19

### Quick-look flags

- has_Conv: ✓
- has_MatMul: ✗
- has_ReduceSum: ✓
- has_ReduceMax: ✓
- has_Where: ✓
- has_Gather: ✓
- has_OneHot: ✗
- first_node: `Conv`
- last_node: `Pad`
- ends_with_Cast(F16→F32)→output: ✗
- first_node_is_Slice: ✗

### Op-type histogram

| op | count |
|---|---:|
| `Cast` | 16 |
| `Unsqueeze` | 9 |
| `Add` | 7 |
| `Reshape` | 6 |
| `ArgMax` | 5 |
| `And` | 5 |
| `Mul` | 5 |
| `Gather` | 5 |
| `Equal` | 4 |
| `ReduceMax` | 4 |
| `Squeeze` | 4 |
| `Less` | 4 |
| `Where` | 4 |
| `Sub` | 3 |
| `Greater` | 2 |
| `Slice` | 2 |
| `Clip` | 2 |
| `Pad` | 2 |
| `Conv` | 1 |
| `ReduceSum` | 1 |
| `Not` | 1 |
| `TopK` | 1 |
| `GatherElements` | 1 |
| `Resize` | 1 |
| `QLinearConv` | 1 |
| `ArgMin` | 1 |

### Initializers (top 10 by params)

| name | dtype | shape | params |
|---|---|---|---:|
| `color_weight` | F32 | [1, 10, 1, 1] | 10 |
| `c_arange_i64` | INT64 | [10] | 10 |
| `ch_arange_b` | UINT8 | [1, 10, 1, 1] | 10 |
| `arK_u_40` | INT32 | [1, 10] | 10 |
| `i_idx_u_57` | INT32 | [1, 10, 1] | 10 |
| `j_idx_u_58` | INT32 | [1, 1, 10] | 10 |
| `eye_3` | INT32 | [3, 3] | 9 |
| `pre_pad_amt` | INT64 | [8] | 8 |
| `pads_output` | INT64 | [8] | 8 |
| `arange_K_out_r_4d_93` | INT64 | [1, 1, 5, 1] | 5 |
| ... (32 more) | | | |

### Top 5 largest intermediates (static value_info bytes)

| name | dtype | shape | bytes |
|---|---|---|---:|
| `color_idx_f_1` | F32 | [1, 1, 30, 30] | 3,600 |
| `combined_idx_48` | INT32 | [3, 10, 10] | 1,200 |
| `combined_idx_flat_49` | INT32 | [3, 100] | 1,200 |
| `mask_3_cropped_b_19` | BOOL | [1, 3, 19, 19] | 1,083 |
| `mask_3_cropped_20` | UINT8 | [1, 3, 19, 19] | 1,083 |

Note: grader charges max(static, runtime-on-real-inputs)/tensor — techniques.md §39 MECHANISM.

*Technique routing: `Onnx_Advanced/by_symptom.md` (+ INDEX verdict list). Flags above feed it.*

*Node list (97 nodes) omitted — the .onnx is canonical; run `tools/agent_helpers/inspect_graph.py task319` for the full sequence.*

<!-- END AUTO-GRAPH-DIAG -->
```
