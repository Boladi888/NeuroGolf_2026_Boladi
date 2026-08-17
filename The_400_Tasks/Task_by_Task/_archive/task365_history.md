# task365 — archived history (moved from task365.md 2026-06-08, task233 convention)

Full optimization ladder, floor-proof derivation, and dead-end experiments for task365.
The live `task365.md` keeps the rule, current best (v16 = 3,354), and a 1-line floor verdict.

## Full ladder (LIVE 29,489 → 3,354)

| Date | Candidate | Cost | Verdict |
|---|---|---:|---|
| 2026-05-19 | crop10x10 (tail-swap of LIVE double-Gather) | 29490 | superseded (LIVE blend 68599→29490) |
| 2026-06-08 | peel_int32 (uniform int32/int64) | 32299 | rejected — int64 casts 16,992 B, worse than LIVE |
| 2026-06-08 | peel_u8 W=10 | 9988 | superseded |
| 2026-06-08 | peel_u8 v1 W=6 | 8080 | superseded |
| 2026-06-08 | peel_u8 v2 (row-r0 width + Min/Max winner) | 6887 | staged SAFE control (no einsum/QLC) |
| 2026-06-08 | peel_u8 v3 (scalar-track winner, drop boxA + win mask) | 5797 | superseded |
| 2026-06-08 | peel_u8 v4 (einsum red-count, drop Ri32 + MatMul i32) | 5112 | superseded |
| 2026-06-08 | peel_u8 v5 (rank-4 channels) | 4804 | staged FALLBACK-A (einsum, no QLC) |
| 2026-06-08 | peel_u8 v6 (QLinearConv-as-output placer #85) | 4536 | superseded by v8 |
| 2026-06-08 | peel_u8 v7 (colidx-einsum read + Equal-tail) | 4810 | REJECTED — Equal-tail + int64 overhead made this colidx form worse |
| 2026-06-08 | peel_u8 v8 (= v6 + int32 indices) | 4308 | superseded |
| 2026-06-09 | peel_u8 v9 (colidx + QLC + int32) | 4050 | superseded |
| 2026-06-09 | peel_u8 v10 (v9 + Min occupancy) | 3951 | superseded |
| 2026-06-09 | peel_u8 v11 (v10 + dynamic output Slice/Pad) | 3856 | superseded |
| 2026-06-09 | peel_u8 v12 (dynamic red-count slices) | 5200 | rejected — cost worse + real-input ORT shape-reuse crash |
| 2026-06-09 | peel_u8 v13 (v11 + F16 red-count path) | 3512 | superseded |
| 2026-06-09 | peel_u8 v14 (narrow CumSum to int16/f16) | n/a | rejected — ONNX shape inference rejects both int16 and f16 `CumSum` |
| 2026-06-09 | peel_u8 v15 (affine QLinearConv score output) | 3284 | rejected — raw visible 0/266; gate requires exact one-hot bytes |
| 2026-06-09 | **peel_u8 v16 (3D work grids + F16 red + dynamic output)** | **3354** | **PRIMARY — visible/truth/AB/profile clean** |
| 2026-06-09 | seed_firstred v2 static one-peel floor | 2066 | rejected probe — visible 117/266 |
| 2026-06-09 | seed_firstred v3 fixed 6x6 dynamic Slice | 1996 | rejected probe — profile=static, visible 52/266 (neighbor fragments) |
| 2026-06-09 | seed_firstred v4 variable Slice + dynamic Pad | profile 1972 (static 2036) | rejected probe — visible 117/266 |
| 2026-06-09 | seed_rowmax MatMulInteger row-count seed | profile 2012 (static 2076) | rejected probe — visible 175/266 |

## Floor proof — <1000 IMPOSSIBLE, <2000/<3000 INFEASIBLE (MEASURED)

Every component measured, not estimated. The earlier "1,200 B read is forced" was an overstatement (the colour read CAN drop to 810 B); corrected below.

- **Colour read ≥ 810 B (MEASURED) — v9/v10 prove colidx is viable only with QLC + int32.** A `[1,10,10]` colour-index plane via `Einsum('bchw,c,hH,wW->bHW', input, colorvec[10], sel[30,10], sel[30,10])` (the `sel` crop selector SHARED across both spatial axes) scores 500 mem + 310 params = 810, beating 3× F32 slices (1,200); the internal `[1,30,30]` is a scorer-invisible in-kernel temporary (#89, verified and gate-5 clean). v7 still lost at 4,810 because it used Equal-tail + int64 overhead. v9 keeps the colidx read, derives a separate red plane, uses int32 stacks, and decodes only {1,2,8} through QLC for 4,050. v10 then replaces `Greater(ci,0)->Cast(u8)` with exact `Min(ci,1u8)`, landing at 3,951.
- **CumSum run-detection ≈ 480 B is FORCED (MEASURED 68%).** A cumsum-free peel (all occupied rows in the leftmost column) differs from correct on 54,821/80,000 (68%) fresh samples — boxes routinely share the leftmost column (row-separated) and top row (col-separated). `CumSum(1-occ)==r0` (int32; uint8 rejected) is unavoidable.
- **Per-box red count ≥ ~400 B + data-dependent selection.** The winner is identified ONLY by "most reds" (boxes have no size/position bias). Cheapest red operand: `R_f32` [10,10] ≈ 400 B (free-input red Einsum needs [30] vecs = 720 B, worse).
- **Output ≥ 108 B** (QLinearConv placer) or ≥ 360 B (Equal-tail).
- **ABSOLUTE FLOOR ≈ 2,000–2,200 B** (color read/red operand ≥800 + forced CumSum 480 + QLC output 108 + irreducible locate/bookkeeping). `<2000` infeasible; `<1000` impossible (read ≥810 + CumSum 480 ≥ 1,290).

## v16 = 3,354 is the current verified floor (v8/v10/v11/v13 floor claims retracted)

The locate cannot be cheapened:
- (a) **flood-label** (#88) costs MORE — color read still needed, flood adds ~900–1,500 B, post-flood per-label red aggregation adds ~400 B that the 3-peel's row/col-vector red count avoids → est. 5,000+.
- (b) **corner-detect + run-length** (no work-removal) needs down-run + right-run for all cells (~1,200 B iterated) + integral image (~800 B) → est. ~4,100, worse.
- (c) **marginal-profile** detection (task049 748 / task291 1,756) CANNOT isolate per-box red counts (a row/col crosses multiple boxes) — only locates a SINGLE known region. No shipped task selects-among-multiple-objects-and-crops under ~9,600 (task384 9,617; task205 10,518).

The 3-peel + colidx/Einsum red count is the cheapest verified structure for "pick the max-of-a-count object and crop it." v11 saves 95 vs v10 by replacing final static Gather+extent-mask output with variable object Slice + dynamic Pad to 6x6 before QLC, with full truth/A-B/profile clean. v13 saves another 344 by casting the red plane and row/column red-count masks to F16; red counts are ≤4 so F16 is exact, and `Einsum`/`ReduceSum`/`ArgMax` accept it. v16 saves 158 more by keeping the peel work grids as `[1,10,10]`, eliminating the duplicate `[10,10]` `ci` tensor and some scalar-stack bookkeeping while staying byte-exact. `<3000` is still below the structural floor; `<1500` impossible (read 810+ + forced CumSum 480 ≥ 1,290 before locate/output).

## 2026-06-09 under-2k/risky probe results

The prompt asked for an under-2k push. The one-peel output floor can cross 2,000, but only after dropping exact winner selection:

- `build_task365_seed_firstred_v1.py` / `task365_seed_firstred_v1_W6.onnx`: static cost 2,351; visible 117/266; truth 846/2000. Baseline one-peel using first red as seed.
- `build_task365_seed_firstred_v2_floor.py` / `task365_seed_firstred_v2_floor_W6.onnx`: static cost 2,066; visible 117/266. Uses ci3 directly, row-projection first-red seed, and reverse-mask height/width instead of F32 sums.
- `build_task365_seed_firstred_v3_dynslice_floor.py` / `task365_seed_firstred_v3_dynslice_floor_W6.onnx`: static=profile=1,996; visible 52/266. Fixed 6x6 dynamic Slice is cheap but leaks neighboring box fragments outside the selected object's extent.
- `build_task365_seed_firstred_v4_varslice_floor.py` / `task365_seed_firstred_v4_varslice_floor_W6.onnx`: static 2,036; real-input profile 1,972 (memory 1,606 + params 366); visible 117/266. Variable object Slice + dynamic Pad to 6x6 restores exact output for the chosen seed and proves the one-peel output floor is under 2k, but the seed is still wrong.
- `build_task365_seed_rowmax_v1_varslice.py` / `task365_seed_rowmax_v1_varslice_W6.onnx`: static 2,076; real-input profile 2,012; visible 175/266. `ReduceSum` on uint8 is rejected, but `MatMulInteger(redu3, ones10)` works for row red counts. It is still far below deterministic safety.

Corpus seed-sweep summary (20k): first red 41.6%; row max 61.0%; rowmax+colmax 67.4%; best row/col 71.8%; density radius-2 89.8%; best small linear density/row formula found 95.3%; exact 100% only when each red seed is scored by the red count of its recovered component. That exact component count is essentially the missing work in the 3-peel v10 line, so do not stage the under-2k probes as submit candidates.

## Don't-do log (full)

- 2026-05-19 | mul_reorder: Mul(v250,v243)->temp then Mul(temp,v246) — v250 is [1,10,30,30] not [1,10,1,1]; added a duplicate [1,10,30,30] intermediate. Cost 100999 (+32400).
- 2026-05-20 [OB_BIG worker]: no candidate (whole_f16, cast variants, drop_final_cast all delta≥0 / verify-fail / ORT crash).
- 2026-06-08 | "Rule unknown" was WRONG — the prior note described the LIVE graph, not the rule. The rule IS decodable (max-red box → crop to origin).
- 2026-06-08 | UUID glob trap: `glob *365*` false-matches `task_9b365c51.py` (UUID contains "365", a DIFFERENT in-place-recolor task). Correct generator: NG N → `sorted(training stems)[N-1]` = `e50d258f`. Always compute the index map.
- 2026-06-08 | int32/int64 uniform lowering = 32,299 (WORSE than LIVE): int64 casts for CumSum/Equal cost 800 B each (16,992 B). Keep grids UINT8 (100 B), cast only the tiny [10] cumsum vectors to int32, cast ArgMax SCALARS (not grids) for Equal.
- 2026-06-08 | ORT uint8 op support (probed): Min/Max/ReduceMax/ArgMax/Equal/Greater/Gather/Clip/Pad OK; Mul/Add/Sub/ReduceSum/CumSum NOT (use Min=AND, Max=OR, Cast(Equal(x,0))=NOT; int32 for sums/cumsum). OneHot needs int64 indices + float/int64 values; out-of-range index → all-off (sentinel works) — but plane-Concat/QLC avoids OneHot and is cheaper.
- 2026-06-08 | gather-clamp smear: `clip(r0w+arange,0,9)` repeats row 9 when the box touches the bottom edge → win mask bottom row smears. v3+ uses extent-mask `(arange<tall)&(arange<wide)` instead — the box always fits (`r0w+tall<=10`) so in-box gathers are never clamped.
- 2026-06-08 | einsum red-count beats MatMul (#89, gate-5-confirmed): `Einsum('bchw,h,w->b', s2, rm_f32, cm_f32)` — the [1,1,10,10] masked product is a scorer-invisible in-kernel temporary. Dropped Ri32 (400 B) + MatMul operands; static==profile. Row/col masks must be F32 (match s2); they double as the tall/wide ReduceSum inputs.
- 2026-06-08 | scalar-track the winner: tracking (r0,cL,tall,wide,redA) as scalars + Gather-by-red-ArgMax dropped boxA (3×100 B) + winner mask machinery (~500 B). redA via einsum needs only rm/cm; work_next via `Min(work, Max(¬rmc,¬cmr))`.
- 2026-06-08 | QLinearConv-as-output placer (#85): `QLinearConv([1,3,6,6] u8, W[10,3,1,1] routing, scales=1 zps=0, pads=[0,0,24,24]) -> output` routes 3 colour planes to channels {1,2,8} + pads to 30×30 free; 108 B vs 360 B Concat. NO bias (BIAS-LEAK LAW). y_zero_point MUST be u8. gate-5 clean.
- 2026-06-09 | v7 colidx Equal-tail is a DEAD END, not colidx itself. v9 combines colidx read + int32 bookkeeping + 3-plane QLC output and wins at 4,050. Do not repeat v7 unless preserving both fixes.
- 2026-06-09 | occupancy from colidx should be `Min(ci,1u8)`, not `Greater(ci,0)->Cast`; v10 saves 99 net and stays byte-equal.
- 2026-06-09 | dynamic output Slice/Pad is GOOD: v11 = 3,856, full truth/A-B/profile clean. This is safe because only one variable object Slice is present and it is padded to fixed 6x6 before QLC.
- 2026-06-09 | dynamic red-count Slices are BAD: v12 = 5,200 static and crashes real-input ORT profiling with repeated variable Slice shape-reuse mismatch (`{1,5,4} != {1,4,6}`). Do not repeat unless replacing variable red slices with fixed padded windows or unique shape annotations that profile safely.
- 2026-06-09 | F16 red-count path is GOOD: v13 = 3,512, full truth/A-B/profile clean. F16 halves `redf3` and the six row/col mask vectors; no numeric risk because red-count sums are small integers.
- 2026-06-09 | CumSum narrowing is blocked: ONNX shape inference rejects both int16 and f16 `CumSum`; int32 vectors remain forced.
- 2026-06-09 | affine QLinearConv score output is not shippable: it scores 3,284 but raw visible is 0/266 because `ng_gate` compares exact one-hot tensors, not argmax-equivalent channel scores.
- 2026-06-09 | 3D work grids are GOOD: v16 = 3,354, full truth/A-B/profile clean. Keeping the peel as `[1,10,10]` removes the duplicate `ci` reshape and scalar-stack overhead without touching the exact rule.
- 2026-06-09 | bool work grids are blocked: `ReduceMax(bool)` is rejected by ONNX shape inference, so a bool occupancy peel would need u8 casts and loses the intended savings.
- 2026-06-09 | dynamic Slice value_info is required: clearing the annotated `cibox` value_info from v3 makes `score_network_v2` return None. Annotated v3/v4 profile cleanly, but both visible-fail due seed/neighbor issues.
- 2026-06-09 | cheap seed heuristics are not enough. Row-count `MatMulInteger` is legal but only 175/266 visible; density/rule-score heuristics remain deterministic bugs even if they look high-percent on corpus.
- 2026-06-08 | cumsum-free peel 68% wrong. flood-label / corner-detect / marginal-profile all measured/estimated worse.

---

## 2026-07-12 consolidation — CumSum-peel line archived (superseded by corner-detect frontier)

The team frontier moved to a **corner-detect** structure (Boladi 2430 → my offset-fuse v2 2360); the entire CumSum projection-peel line below is now a personal-ceiling relic kept for history. All these rungs are ABOVE live 2430 and are NOT shippable.

**SUPERSEDED Algorithm (colidx projection-peel):** 3 unrolled peels — read a 10×10 color-index plane via `Einsum('bchw,c,hH,wW->bHW')`; find leftmost occupied column (`ReduceMax`+`ArgMax`); topmost contiguous run via `CumSum(1-occ)==r0`; read box width from the same trick on the top row; count reds via `Einsum` over F16 red plane + F16 row/col masks; subtract box, repeat; `ArgMax` over 3 red counts picks winner; dynamic Slice the crop, pad to 6×6, decode {1,2,8}, `QLinearConv` to free output. Work grids UINT8 `[1,10,10]`; CumSum int32-forced. **The "forced CumSum run-detection ≈480 B" floor claim from this line is INVALIDATED by the corner-detect route, which uses QLinearConv corner detection + offset-probe extent instead of CumSum.**

**Full CumSum ladder (all superseded):** crop10x10 29490 → peel_u8 W6 8080 → v2 6887 → v3 5797 → v4 5112 → v5 4804 → v6 4536 → v8 4308 → v9 4050 → v10 3951 → v11 3856 → v13 3512 → v16 3354 → polished v16 3352 → v18 scalar-Where 3262 → v19 omit-slice-steps 3260. Rejected probes: peel_int32 32299 (int64 casts), v7 colidx Equal-tail 4810, v12 dynamic red-slice 5200 (ORT crash), v14 narrow-CumSum (int16/f16 rejected), v15 affine-QLC-score 3284 (visible 0/266), direct-red-plane 3608, corner-labelling 4834 (visible 172/266). Seed floors (all visible-fail, prove exact red-count needed): firstred v2 2066 (117/266), v3 fixed6x6 1996 (52/266), v4 varslice 1972 (117/266), rowmax MatMulInteger 2012 (175/266).

**SUPERSEDED cost analysis:** old line kept work in single-channel U8 grids tracking `(r0,cL,tall,wide,redA)`, counting reds with free-input Einsum #89, placing 3 color planes via free-output QLinearConv #85. polished v16 3352 (mem 2990, par 362); v18 3262 removed 90 B via #156 scalar-Where selector.

**SUPERSEDED floor verdict:** claimed ≈2000–2200 B forced floor (color read ≥800 + forced CumSum ≈480 + QLC 108 + bookkeeping) with v18 3262 as safe floor; scalar-selector "~900" skeleton priced 1324>900 (DEAD). This whole floor model assumed the CumSum structure; Boladi's corner-detect 2430 / my v2 2360 supersede it (they never pay CumSum). Sho 2431 (sha 939c0baf…) superseded by Boladi 2430 (sha 7cc43a3d…). STILL-VALID dead formulations (do not re-explore): band-product one-shot oracle 32/266 (141/266 collide — boxes merge across a shared marginal); flood-label ~5000+; first-red/row-max seeds visible-fail.
