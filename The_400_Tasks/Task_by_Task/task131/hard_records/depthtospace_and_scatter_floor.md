## !!! CORRECTION 2026-07-11 (worker3 rework) — THE "WALL AT 3922" BELOW IS WRONG (wrong graph) !!!
Everything below (worker1's WALL / ScatterND-forced-floor / free-output-loses proofs) dissected the
WRONG file: `staged_candidates/.../task131_u8shift_cost3922/task131.onnx` (sha b5bcdca3..., opset16,
ScatterND). That is NOT the live team-best. The ACTUAL live team-best (`Team_Best_Onnx/task131.onnx`,
sha 3aa8b864..., owner Sho) is a **free-U8 output + Einsum log8-position-decode + QLinearConv
channel-expand graph @ 18x18, NO ScatterND** that really profiles **2950** (manifest 3922 is STALE,
= the old u8shift cost; Sho's score_v2=None from an all-zeros dummy crash, so the pipeline kept 3922).
So worker1's headline claims are FALSE: "ScatterND is the only sub-4k structure" (the live sub-4k graph
has none) and "free-output/Einsum loses" (the live winner IS free-U8-output + Einsum, dodging the 900B
canvas by cropping to 18x18). Sho's charged tensors: bp U8[1,2,18,18]=648; gcanvas/t1/code18
U8[1,1,18,18]=324 each; rowT/colT F32[30]=120 each. DepthToSpace kill (36000) remains correct.

**worker3 outcome = WIN + BUGFIX.** Sho's graph has a latent crash: the gcanvas-placement Pad crop
values `ntb=13-nt`,`nlb=13-nl` are Cast to UINT8, so nt/nl>13 (pattern flush to a far-edge marker)
makes -1 wrap to 255 -> Pad width 274 -> ORT CRASH on ~0.075% of ARC-GEN (Sho truth n=2000 = 1999/2000).
Fix: route the ppads path through signed INT8 (same 1-byte size) + [0,17] clamp on gt,gl,nt,nl (no-op on
real data, makes the all-zeros dummy shape-valid for a clean static score). Candidate
`staged_candidates/task131_sho_bugfix_int8pad_cost2975/` (sha 143a68db...): visible 266/266, truth
2000/2000 (fixes Sho's crash), ab byte_equal=True 2266/2266 vs submission, score_v2/profile 2975 clean,
-947 vs manifest 3922, +1 static vs Sho. status PROBE_READY (inherited QLinearConv x86 WARN).
Skinny-5x18 canvas lever does NOT pay: bp+code18 (18x18) are geometry-locked by the free-U8 output conv,
and de-canonicalizing skinny->18x18 needs an orientation transpose that re-adds an 18x18 canvas.

---
(ARCHIVED — analysis of the WRONG graph, kept for provenance)
live bar: 3922, sha256 3aa8b864ee9ae6deb15b2526843f9b590cee8eb5f75095ef1e6812d3ee8905e5 (ng_current_bar.py manifest, owner Sho, 2026-07-09). Team_Best local copy sha b5bcdca3...cdfa10235 (task-local staged u8shift). BoBrain consistency: team_manifest_cost=3922 authoritative.

session: bo_ng_cld_worker1 (Claude lane) 2026-07-11. VERDICT: WALL at 3922 (forced-component floor) — SUPERSEDED, WRONG GRAPH (see correction above).

## Rule (confirmed from data, NOT a tiling)
Color-2 FULL-LINE marker (a row or a col, spanning the skinny dim <=5). Color-3 pattern blob on ONE side. Output = marker kept in place + original pattern erased + SAME pattern rigidly TRANSLATED flush against the marker (pattern edge = marker +/-1) + a color-8 separator FULL LINE on the far side, offset from the marker by pattern_size+1. This is a 1-axis rigid TRANSLATION + one new line. It is NOT periodic tiling.

## Attempted route 1: DepthToSpace block-stamp (the prompt hint) -> INFEASIBLE (kill-rule)
Priced probe `prototypes/task131_depth2space_price_probe.onnx` (input --SpaceToDepth(2)--> P[1,40,15,15] --DepthToSpace(2)--> output): score_v2 memory=36000 params=0 cost=36000.
DepthToSpace(P,b) is a pure rearrange: P always has exactly C*H*W = 9000 elems (identical to the output). It only "compresses" when packed channels are REDUNDANT (a periodic motif). task131's output is non-periodic (ONE moved pattern + ONE marker + ONE separator), so P carries full info and is as large as the canvas. P is neither graph input nor output -> it is a counted intermediate. Even the cheapest BOOL P = 9000 B > 3922. Block sizes b in {2,3,5} all divide 30 but yield no compression. KILL-RULE ("packed motif as large as expanded canvas") triggered. DROP. Do not re-attempt DepthToSpace/SpaceToDepth for this card.

## Attempted route 2: dense channel-plane shift (v2/v4) -> LOSES
Re-priced this session: `task131_v4.onnx` cost 5251, `task131_v2.onnx` cost 5611 (both >> 3922). Dense planes pay full [1,10,30,30]-scale intermediates + two-orientation transposes + de-canonicalization. Dead vs the sparse scatter.

## Winning structure = single ScatterND(input, idx, updates) -> output (the current 3922)
Scoring (neurogolf_utils): cost = memory (dtype-weighted bytes over INTERMEDIATE tensors; graph input+output are SKIPPED/free) + params (element count of initializers, dtype-free). Any op producing a full [1,10,30,30] intermediate costs 36000 B F32 / 9000 B BOOL. Therefore the ONLY sub-4k structure is a SINGLE op input->output with sparse side-tensors. A single Gather-move is impossible: orientations split 136 col-move / 130 row-move and Gather's axis is a static attribute; chaining a row-gather and a col-gather forces a counted [1,10,30,30] intermediate (36000 B). ScatterND is the only viable single op.

## Forced-component floor (current 3922 breakdown; every major item is irreducible)
memory 3248 + params 674 = 3922.
- scatter_idx INT64 [1,50,4] = 1600 B. FORCED. ScatterND mandates INT64 indices (8 B/elem), last-dim = data rank = 4 (no batch_dims on ScatterND; squeezing batch needs a 36000 B intermediate). 50 writes forced by one-hot correctness in the non-overlapping worst case: 9 green cells x 4 writes each (erase old ch3->0 & ch0->1; write new ch3->1 & ch0->0 -- 2 channels because one-hot needs exactly one channel/cell) = 36; + separator <=5 cells x 2 (ch8->1 & ch0->0) = 10; + special 10th-green patch = 4. Data: green in {8,9,10}, sep in {4,5}. The 10-green case is in the visible TRAIN set, so dropping the special patch = visible 265/266 = certain hidden 0. TopK(10)-everywhere was already tried (top10_no_special.md): 3943, +21. A 2nd fix-up ScatterND is impossible (its input would be a 36000 B intermediate). So 50 is the static floor.
- green_probe_vals F32 [1,155] = 620 B + green_probe_idx 465 params. FORCED. Green occupies arbitrary cells across the FULL skinny region (rows0-4 x cols0-17 UNION rows0-17 x cols0-4 = 155 positions), orientation not known to a static index; each of <=10 cells' exact (row,col) is needed to place the translation. A F32 slice of the same region is MORE expensive ([1,1,18,18] = 1296 B; two skinny slices = 720 B) and still needs per-cell extraction. GatherND output dtype = input dtype = F32 (can't be read as U8). Reading fewer positions risks missing green -> truth fail. NonZero/Compress (dynamic sparse extract) are BANNED ops.
- updates72 F32 [1,50] = 200 B (50 f32 scatter values; updates dtype = data dtype = F32). scatter_idx8 U8 [1,50,4] = 200 B (cheapest u8-concat-then-cast path; building the index directly in i64 costs MORE: rows/cols i64 intermediates 800 B vs u8 100 + idx8 200). marker locate red_anchor F32[1,24]=96 + 72 params. TopK green_cand i64[1,9]=72.
Sum of forced components ≈ 3922. The staged u8shift IS the floor.

## Prior small-shave walls (all re-confirmed dead)
- top10_no_special.md: TopK(10) no special -> 3943 (+21).
- red_marker_slice.md: red-line via slices -> 3951 (+29).
- vector_free_output_rank.md: free-output vector render, moved pattern rank 1-4 (not rank-1) -> est lower-bound 3245 before scalar plumbing, deemed too close/risky; still pays the same 620 green read, replaces scatter (~2100) with ~2160 B of row/col term masks = wash, no build beat.

## SECOND PASS 2026-07-11 (forced re-attempt): #68 final-op-family enumeration + a BUILT alternative
Per techniques §68 ("floor proofs are formulation-relative; enumerate every op that can legally write `output`, price its min input set"). Every family that can emit [1,10,30,30], priced:

| final op | min pre-output cost | verdict |
|---|---|---|
| **ScatterND(input,idx,upd)** | output FREE; idx i64[1,50,4]=1600 + green 620 | **3922 — BEST.** marker+bg ride through input; no pre-output canvas |
| **Equal/OneHot-tail(color-index canvas)** | **900 B padded [1,1,30,30] MANDATORY** + base/cidx 648 + rebuild bg/lines ~650 | **BUILT + MEASURED 5635; idealized route floor ~4027. LOSES** |
| broadcast-Add/Greater/Min/Mul (A[1,10,30,1]∘B[1,10,1,30]) | separable-per-channel only | DEAD — moved green is rank 1-4 (not separable) + bg complement not separable |
| BitwiseAnd set-intersection tail | biclique cover | DEAD — every bg cell must fire ch0 = one huge biclique |
| MatMul/Einsum contraction final | separable (r,c)-coupling only (§68 discriminator) | DEAD — green coupling input-selected + rank<=4 + non-separable bg |
| Concat channel-planes / Pad(full one-hot) | >=900 B/plane or [1,10,18,18]=3240 | DEAD (task046: C*active_cells >= full_grid here) |

**The BUILT alternative** (`prototypes/build_task131_cidx_equaltail.py` -> `task131_cidx_equaltail.onnx`, genuinely different assembly): single-channel COLOR-INDEX canvas built from scratch -- marker+separator rendered FREE via `Max(rowcode,colcode)` broadcast, padding from `10*(ReduceMax(input,axes)==0)`, green scattered at NEW positions only (no erase, 1 write/cell -> `[1,10,4]` i64 = 320 not 1600), TopK(10)+duplicate-invalid-slots (no special-10 machinery) -- then `Equal(Pad(cidx,10), ch_ids) == output` (#67/#116 free-output). **VISIBLE 266/266 PASS (ng_gate + ORT), numpy reference verified 266/266. Cost 5635 (mem 4681 + par 954).** Even with perfect u8 plumbing the ROUTE FLOOR is ~4027 because the Equal-tail pays a **900 B `padded` pre-output canvas the ScatterND route does NOT** (its output is free), plus ~650 B rebuilding background/marker/separator that the scatter rides through the input for free. The move-cost saving (green scatter 320 vs 1600) is smaller than these additions. NO ship gates run (static cost already loses -- resource-safe).

## One remaining hypothesis (only thing that could break 3922)
A single legal ONNX op that (a) produces the [1,10,30,30] output directly (so NO pre-output canvas is charged), (b) applies a data-dependent 1-axis translation to a sub-region while leaving the marker fixed, for BOTH orientations, WITHOUT an INT64 per-cell scatter index and WITHOUT a full-canvas intermediate. ScatterND is the ONLY op that emits output free AND edits sparsely; it forces the i64[.,4] index. No other op in opset<=20 minus the banned set (Loop/Scan/NonZero/Unique/Compress/If-graph) satisfies (a)+(b). Absent that, 3922 is the floor.
