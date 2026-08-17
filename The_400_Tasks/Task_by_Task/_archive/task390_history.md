# task390 — archived history (superseded lineage v1→v12 + LIVE→2606)

Moved out of `Task_by_Task/task390.md` 2026-06-21 (W19) per WORKER_PROTOCOL task-md-size rule (manual content >20 KB). The CURRENT best (home staged **2601** worker390 pad-axes; sibling team-best LIVE 2904) and the W19 structural floor proof remain in `task390.md`. This file holds the superseded v1→v12 rungs and the per-version cost-analysis prose.

## Superseded attempt rows

- 2026-06-07 | v5 direct profiles + UINT8/INT32 free-output rewrite | STAGED. Gates: visible 266/266, truth 200000/200000, A/B byte-equal 200266/200266 vs `submission.zip`; LIVE truth 200000/200000; Gate-5 profile 5136 == static. | **5136** (memory 5065 + params 71) | `prototypes/build_task390_v5_direct_profiles.py`; staged `staged_submits/task390_v5_direct_profiles_cost5136/` |
- 2026-06-07 | v6 remapped 0/1/2 Equal-tail | STAGED. Same gates as v5; Gate-5 profile 4909 == static. Internal canvas uses bg=0, red=1, gray=2 and remaps `ids` at the final Equal, removing one 15x15 UINT8 plane. | **4909** (memory 4840 + params 69) | `prototypes/build_task390_v6_remap_tail.py` |
- 2026-06-08 | v11 scalar zero | STAGED PRIMARY. Gates: visible 266/266, truth 200000/200000, A/B byte-equal 3266/3266; Gate-5 profile 4785 == static. v10 plus scalar-broadcast INT32 zero for `Where`, removing 14 params. | **4785** (memory 4735 + params 50) | `prototypes/build_task390_v11_scalar_zero.py` |
- 2026-06-08 | v10 compact Slice params | STAGED. Gate-5 profile 4799 == static. v8 plus C/H/W-only Slice params. | **4799** (memory 4735 + params 64) | `prototypes/build_task390_v10_compact_slice_params.py` |
- 2026-06-08 | v8 index-select Gathers | STAGED. Gate-5 profile 4804 == static. Folds `Where(hflag, Gather(row), Gather(col))` into two 15-element INT32 index vectors + two sequential Gathers. | **4804** (memory 4735 + params 69) | `prototypes/build_task390_v8_index_select_gathers.py` |
- 2026-06-08 | risky axis-only final Conv probes | STAGED PARTIAL, not safe. Row-only 13x1: visible 123/266, truth 99713/200000, profile 1310. Col-only 1x13: visible 143/266, truth 100287/200000, profile 1310. memory 0 (Conv is graph output). Partial-credit only. | **1310** each | `build_task390_risky_axisconv.py` |
- 2026-06-08 | v12 orientation-select axis Conv branches | NOT STAGED. #82/#93 scalar orientation bit is exact (266/266) but the scoreable top-level `Where` materializes both `[1,10,30,30]` F32 branch logits = 74,736; `If` rejected (None). | 74,736 / `If` rejected | `prototypes/build_task390_v12_orientselect.py` |
- 2026-06-08 | under-3000 mixed-orientation final Conv search | NOT STAGED. Dense Conv families <=2910 params can't be exact: 5x5, 4x7/7x4, 3x9/9x3, 2x13/13x2, 1x15/15x1, 2x14/14x2, 1x29/29x1 all have channel-0 local-patch contradictions; trained probes topped at 66/266. | 2510-2910 searched | `experiment_task390_finalconv_linear.py` |
- 2026-06-08 | v7 range-valid source index / v9 UINT8 Sub tail | NOT STAGED. v7 correct but 4914; v9 4579 but officials 0/266 (UINT8 Sub saturates, gray collapses to bg). | 4914 / 4579-bad | `build_task390_v7_range_valid.py`, `build_task390_v9_u8_sub_tail.py` |
- 2026-06-07 | rung ladder v1..v5 | All passed 266 official-pair verifier; v1 7045, v2 5768, v3 5542, v4 5178, v5 5136. | 7045 -> 5136 | same prototype directory |
- 2026-05-19 | worker-7 crop15 (c27b523a / c404042d) crop-to-15x15 | 266/266 vs LIVE. Cost 32343→26805 (−5538 B). | 26805 | `.hermes/board/staged_submits/...` |

## Superseded per-version cost analysis

LIVE 22,766 was dominated by the full `[1,10,15,15]` F32/F16 crop path and a 10-channel F16 output-active tensor before `Pad`. v5 removed both: it slices only channels 2 and 5, casts them to UINT8, computes the reflected gray mask with one source-index vector, builds a one-channel UINT8 color canvas, pads that with sentinel 10, then uses final `Equal(color, ids)` as the free graph output.

v11 static/profile cost was 4,785. Largest charged tensors then: two F32 channel slices = 1,800 B; padded UINT8 color canvas = 900 B; six 15x15 UINT8 reflection/tail planes = 1,350 B; INT32 source/routing vectors = 480 B. v8's win over v6 was the gather-index fold: `Where(hflag, Gather(rows), Gather(cols))` became `row_idx = Where(hflag, source_idx, idx)`, `col_idx = Where(hflag, idx, source_idx)`, then two sequential Gathers — trading one 225-B UINT8 orientation plane for two 60-B INT32 index vectors. v10 dropped 5 params (C/H/W-only Slice params); v11 dropped 14 more (scalar-broadcast `zero_i32_15`).

Final-op family enumeration (all superseded by the QLinearConv 2->10ch free-output renderer that Aditya's 2904 / worker5 2606 / worker390 2601 use):
- Free-output Equal-tail: the v5-v11 choice. Skeleton = channel extraction (1,800 F32 + 450 U8 casts) + reflection/tail (~1,690) + 900-B padded color canvas + params 69 = 4,909.
- 10-channel Concat/Pad tail: killed by v1 (materializes [1,10,15,15] output-active 2,250 B before Pad — worse than 900-B Equal-tail canvas).
- Einsum/contraction final: killed — dynamic, orientation-dependent reflected source index needs materialized selector/index state and both axes.
- Params-only/static final Conv: gray bitmap is input-variable; correct outside placement depends on red-bar orientation + paired offset tests; no cheaper validated construction.
- Under-3k single final Conv: killed locally for exact mixed-orientation output (channel-0 contradictions across all <=29-tap geometries; trained best 66/266).
- Cross-shaped 25-tap sparse final Conv (2,510 if sparse charged): contradiction-free on channels 0/2/5 but not linearly separable as one threshold; trained 0/266; sparse initializer accounting scorer-risky.
- Under-2k axis-only final Conv: staged only as risky probes (1,310 each, ~50% truth, one orientation).
- V12 orientation-select branch: orientation bit exact via #82/#93 moment (`count*S2 - S1*S1`); `If` rejected (None), top-level `Where` materializes both branches = 74,736.
- Under-1000 dilated axis Conv (`k<=9,d<=4`): contradiction-free `k=7/8/9,d=2` not linearly separable; best 66/266 at cost 810; nothing under 1000 staged.

---

## Superseded cost-analysis prose (moved from task390.md 2026-07-12 by bo_ng_cld_worker2)

These paragraphs describe THIS LANE's Gather/table-indexselect formulation
(2606/2601/2567). They are superseded as the live bar by Sho's ENUMERATION
graph (2514 -> shaved to 2466 this batch); the enumeration floor analysis now
lives in task390.md. Kept here for provenance. Attempts-table rows + staged
manifests retain the full gate dumps.

2026-06-20 worker-task390 current staged candidate beats the sibling team-best
2904 with a byte-equal local compression to 2601. It starts from worker-5's
2606 table-indexselect graph and only changes param surface: the two U8 Pad
nodes use opset-18 axes form (`pads[4] + axes[2]` instead of full-rank
`pads[8]`), while the one affected `ReduceMax` uses an axes input required by
opset 18. Static and full real-input profile both price at memory 2470 + params
131 = 2601. No absolute floor claim; this is a clean byte-equal param shave.

2026-06-19 worker-5 staged candidate beat the sibling team-best 2904 with a
byte-equal local compression to 2606. The 2904 graph was already Aditya's
compact U8/QLinearConv renderer, not the stale home v11. The useful remaining
shave was formulation-local: the selected reflection axis only takes
`(lo, hi) = (3,10), (3,11), (4,11)` across generator and officials, so a 3x15
INT32 source table replaces the arithmetic source-index chain. Then
`row_idx = Where(hflag, source_idx, idx15)` and
`col_idx = Where(hflag, idx15, source_idx)` let two sequential Gathers replace
the old `Gather(row)`, `Gather(col)`, full-plane `Where` selector. Static and
full real-input profile both priced at memory 2470 + params 136 = 2606.

Floor verdict (Gather formulation, superseded): the gather/free-output
formulation is at structural floor for the main components. Per-charged-tensor:
F32 input reads `twos10` 400 + `fives8` 256 = 656 (floor; input is F32 one-hot,
any Slice/Gather stays F32); gray reflection pad(225)+2 gathers(450)=675 (floor;
orientation data-dependent, single-gather alternatives wash/worse); `render2_u8`
[1,2,15,15]=450 (floor; trailing 1x1 QLinearConv maps 2 masks->10 free channels
and pads 15->30). Sub-1000 needs a Conv-only detector = PROVEN DEAD (orientation
needs both branches; dense Conv has channel-0 local-patch contradictions).

## Superseded worker-09/W19 Gather cost-analysis (moved from task390.md 2026-07-12)

2026-06-24 worker-09 scalar hflag shave over W19g: `h_flag = r0_l > 0` == `ReduceMax(col5_probe_1d) > 0`
on the official/arc-gen surface (horizontal stream's first red row inside crop is idx 1 or 2; empty
vertical-stream column probe ArgMaxes to 0). Deletes only `h_probe_max` (U8 scalar, 1B). memory 2442 +
params 125 = 2567.
Progress ladder (Gather formulation): 2601 base -> 2592 (b eff-shave) -> 2587 (c scalar-chain) -> 2586
(d qzero-scalar) -> 2584 (e shared-scale) -> 2583 (f shared-xw-zero) -> 2568 (g u8idx) -> 2567 (worker-09
hflag). At 2567 the major blocks were: F32 reads 656B, U8 casts 164B, reflection pad+gather 900B,
render2_u8 450B, Cast-for-Gather 120B, scalars ~152B. These are micro-opt shaves, not architectural.
NOTE: superseded as live bar by the ENUMERATION 2363 (w4 gray-in-code) — see task390.md.
