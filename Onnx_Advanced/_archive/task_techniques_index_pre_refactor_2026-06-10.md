# task_techniques_index.md — by family, by technique class

Grouped task index. For each family, sub-grouped by technique class. Each entry:
- `taskNNN` (cost) → exemplar path if available, else `prototypes/build_*.py` file, else "no exemplar — see staged manifest".

**Usage**: locate your task ID below. Read the listed exemplar(s) AND any in the same technique class for variant detail.

**Integrity note (2026-05-23)**: this index contains a mix of current exemplars, shipped entries, historical prototype paths, and staged-manifest pointers. Treat entries under the 2026-05-23 SHIPPED section and existing `exemplars/` links as current first. If a `The_400_Tasks/.../prototypes/...` path is missing, it is a historical pointer, not proof that the technique artifact still exists.

**Legend for "Untagged" sub-sections**: tasks listed under each family's `### Untagged (...)` heading had no staged candidate and no `build_*.py` in their `prototypes/` dir at generation time (staging now lives in `staged_submits/`) — candidates for future worker attention, sorted by cost descending.

---

> Per-entry battle prose retained below pending Phase-2 compression; the dated ship LEDGER lives in `Logs/submissions_ledger.csv` + `recent_ships.md` (pointer file).

### SHIPPED 2026-06-07 evening / BATCH-16 (8 tasks, +8.56, all exact-or-law-confirmed; technique detail #87/#88/#81/#86)

- `task207` 2,387 -> 200 (mem 128 + p 72) — u8-bitand from-scratch occupancy/color rewrite; floor 200 (sub-200 u8-Mul scorer-rejected).
- `task127` 4,046 -> 900 **memory-ZERO** — single final Conv(input,w)->free output; absolute floor (free-output separability trap #87).
- `task032` 3,405 -> 915 — column-gravity count-pack; small-grid decoded-one-hot+Pad beats 900B canvas (#81 inversion).
- `task291` 4,449 -> 1,756 — bbox1map: single rowsum marginal (ArgMax=height/ReduceMax=width) + one-hot-Pad corner 933->22p (#86).
- `task240` 3,928 -> 1,706 — einsum-fold generator-invariant (task_9d9215db); 900B-canvas final-op floor, 5-row kill table.
- `task123` 4,314 -> 1,993 — leangather (direct 10-ch gather avoids 900B color-plane); floor vs task001 techniques.
- `task065` 3,803-actual -> 1,999 — geom-canvas-f32eq CREDIT-PROBE (+51% dynamic, credited exactly at profile; variable-compartment crop).
- `task359` 4,510 -> 4,158 — threshold-fold; generator-proven ABSOLUTE floor (denoising-not-symmetry, label misnomer).

### SHIPPED 2026-06-07 afternoon wave (20 tasks, +26.23, all exact — technique detail in #81-86)

- `task048` 8,831 -> 699 **CLASSBET HELD (ambiguous #8)** — row-byte bit-packed flood (#88; 72 B/round at width 8), 5 architectures in one session; k12 1,131 corpus-perfect sibling + k7 771 also held.
- `task180` 9,688 -> 210 **memory-ZERO (0m+210p)** — groupconv placer, IG one-hot-sum identity, groups-halving (#84); params-only class 3rd member.
- `task321` 8,327 -> 434 — shared priority-code vector (ARC-GEN contract 4/9/1), Max=priority, Equal tail; floor proven on paper (task321.md:115).
- `task346` 9,178 -> 523 — occupancy-pack 128-spacing + max-bound existence Equal(ReduceMax) (#86); 338-standard floor.
- `task039` 7,832 -> 710 — einsum row/col detect + F16 onehot + shared selector bank (dedup -152p); 4-route audit, residual below bar.
- `task209` 9,487 -> 5,251 — byte-equal re-plumb: i8 lanes + NIBBLE-PACK 2-cells/byte plane (#83); floor ~4.6-4.8k, Where-route discriminator-killed.
- `task265` 9,584 -> 4,713 — QLC int32-bias binarizer + in-conv dilate + crop-instead-of-mask (#85); MULTI-FORMULATION floor (11 decompositions).
- `task246` 13,391 -> ~1,369 credited — reducesum-moment probe HELD ungated (mild 1.07× surface; gold gates blind to pricing).
- `task225` ~2,160 -> 1,614 — v5 qlinear-where, no-Range sibling = MODERATE-band escape, honest-static.
- FREE-OUTPUT SWEEP ×12 (one stacked submit, +10.24): `task292` 6,301->1,310 #78 Where-as-output · `task104` 6,302->1,384 Equal+#79 LUT bank · `task001` 4,339->1,149 Equal+#80 D2S fractal (outer-MIN) · `task114` 5,219->1,451 u8 scatter-LUT · `task047` 4,950->1,687 einsum reads · `task389` 3,368->1,296 einsum k-extract · `task353` 5,411->2,781 active-mask sentinel (§ACTIVE-MASK origin) · `task081` 3,453->1,819 F16 convs · `task142` 1,520->1,059 mirror tile · `task072` 2,089->1,472 negpad XOR · `task194` 1,520->1,092 D4 quads (BL=rot90CCW!) · `task257` 1,589->1,349 priority quads. Law: #81 pad-floor crossover.

- `task148` 18,388 -> 4,461 **SHIPPED 2026-06-06 +1.42 exact** — marginals family 8-rung ladder, byte-equal FULL 200k at every rung: 3-operand free-input Einsum row reads (`'bchw,cx,w->bhx'`, temp-free, probe-verified) + INT8 signed-delta mirror gather + U8 (lo,span) bounds + u8-wraparound interval fill (`Less(Sub(c,lo),span)` — 2 planes) + negpad-Conv U8 canvas + **free-output Equal tail** (`Pad(u8,sentinel) -> Equal(ids)` AS `output`; 2,880→900). Floor 4.3–4.4k PROVEN (output-op factorization lemmas + measured alternatives in task148.md); sub-3k impossible honest-static. Spread scan: 97 tasks end Pad->output ≥1,800B pre-pad (~210KB corpus-wide) — see `.update/2026-06-06_task148_freeout_equal_einsum3_i8lanes.md`.
- `task199` 16,157 -> 3,179 **SHIPPED 2026-06-06 +1.63 exact** — dot-shift parity-fill (Filling Regions): §41 negpad-Conv color+1 encode off the free input (kills the 9,000-B Slice plane) + ALL vector logic on U8 (ReduceMax/Greater 15-B vectors; rows≤r via `Where(ind,iota1,0)→ReduceMax→LessOrEqual` — 1-BASED iota = LIVE-exact no-dot semantics, no CumSum; parity via masked-ReduceMax) + outer-product masks + U8 Where-compose + free-output Equal tail. Byte-equal FULL 200,266/200,266; truth 200k/200k; corpus contract: single dot always, r ≤ H−2, shapes ≤15×15. Floor ≈3,093 itemized (Einsum route re-priced WORSE at 3,225 — Einsum can't crop); v3 +86 over floor. `task199.md` + `staged_submits/task199_v3_u8vectors_cost3179/`.
- `task193` 55,657 -> 5,281, **+2.36 LB confirmed** — 2×2-opening pooled-predicate rewrite (Conv-erosion ==4 → pool-dilate → broadcast Where); 8-rung sub-20 crop ladder ALL failed at the exact predicted value → hidden has >17×17 grids, crop family closed (traps §VISIBLE-FAIL); floor ≈5.3k (F32 mark plane ORT-forced).
- `task244` 53,897 -> 2,252, **+3.17 LB confirmed (23.9×)** — compact-selector cell-extraction rewrite; different-architecture einsum hedge (2,354) also held hidden.
- `task287` 50,758 -> 3,016, **+2.83 LB confirmed** — point-reflection via §8b dilated 2×2 Conv (grids provably fixed 16×16) + sentinel-10 Pad + Equal-expand output; memory floor 2,948 proven exhaustively (2,500 was impossible).
- `task368` 50,484 -> 6,294 interim, **+2.08 LB confirmed** — asymmetric-kernel free row-crop + Conv weighted-sum color-decode (kills Slice+ArgMax/INT64) + cumulative ≥k Where nesting (techniques §35/§36); worker continuing toward <5k.
- `task020` 50,090 -> 2,760, **+2.90 LB confirmed** — symmetry-completion rewrite (variadic Max of flipped copies), byte-equal.
- `task240` 53,414 -> 3,928, **+2.61 LB confirmed** — 5×5-odd-lattice rewrite, byte-equal.
- `task131` 50,325 -> 5,251, **+2.26 LB confirmed** — shared-18×18 move+separator; floor residue = orientation/de-canonicalization tensors + 30×30 canvas (direct 5×18→30×30 costs more than it saves).
- `task325` 50,733 -> 5,642, **+2.20 LB confirmed** — CC-count→identity via K=5 label-propagation (proven minimum, techniques §37); 2-hop window leak-proven dead; byte-equal.
- `task122` 52,517 -> 4,816, **+2.39 LB confirmed** — marker-255 sentinel + sparse ScatterND canvas updates (techniques §38); byte-equal 20,266/20,266.
- `task279` 50,462 -> 14,067, **+1.28 LB confirmed** — ring-CC rewrite, more-correct-than-LIVE (fixes 1 LIVE GT-bug; traps §DUAL-TRUTH); depth-histogram floor proof (techniques §37); 8-conn single-step CC legalized by generator-geometry contract.
- stray-sweep fold (2026-06-04): `task153` 10,528→3,991, `task008` 19,390→8,332 (adversarial-perfect), `task170` 5,737→2,967, `task054` 48,107→47,180 — all byte-equal, **+2.49 confirmed**; found by scoring every staged dir vs the live zip (strategy meta-lesson #14a).
- `task128` 46,305 -> 5,218, **+2.19 LB confirmed** — placement outer-product (new algorithm at v4) + 2-step Where broadcast; byte-equal; floor itemized (opset-9 Pad UINT8/BOOL rejections).
- `task195` 47,787 -> 1,878, **+3.23 LB confirmed (campaign-best single gain)** — 2 Einsum anchor reductions + GatherND(batch_dims=1) 9-cell read (techniques §39); byte-equal; the 253-pattern memorizer was proven dead (failed official test pair).
- `task367` 47,225 -> 19,465 interim, **+0.89 LB confirmed** — qconv-rowtrim byte-equal; <10k routes measured dead (rowhash-4209 = 13%-truth memorizer class; 5×5 pockets in 8,518/200k); flood-rewrite second pass open.
- `task308` 49,062 -> 2,990, **+2.80 LB confirmed** — seqrect pattern-recompose, byte-equal.
- `task014` 45,690 -> 4,380 local (≈6,093 Kaggle-credited), **+2.01 LB confirmed** — dynslice-boolpad smallest-region extract, byte-equal FULL 200k; the grader-side dynamic-shape cost gap is the §39 calibration warning.
- `task328` 47,384 -> 8,367, **+1.73 LB confirmed** — 3-operand Einsum corner reads (einsum.md, −900) + Mod parity (task313 precedent, −190); plane-conservation-law floor ≈7.2k (cost_floor_reference §method); byte-equal.
- `task280` 48,080 -> 6,716, **+1.97 LB confirmed** — compact-mask extension rewrite, 36 versions, byte-equal; measured dead: 20×20-output raw-equality, uint8 TopK (ORT unsupported — op-dtype matrix updated), half-plane direction inference.
- `task364` 46,250 -> 14,162, **+1.18 LB confirmed (the 6500-crosser)** — negative-bias cliff conv (techniques §43, worker-authored); THEOREM-grade floor (cue non-separability + pre-dilation zero-trade, in task364.md); §33/§34 falsified en route (U8 dominance flood 880 B/step dominates F16 MinPool 1,760).
- `task049` 44,787 -> 748, **+4.09 LB confirmed (CAMPAIGN RECORD, 59.9×)** — Einsum 'bchw,c->bh' projection (1,200→120 B) + one Equal plane reused as Einsum selector AND And mask + per-byte floor accounting; 2nd-leanest model in the zip after task215's 110.
- `task187` 49,813 -> 21,282 interim, **+0.85 LB confirmed** — scanline-k4; shipped via the §DUAL-TRUTH ARITHMETIC shortcut (21 ab-divs = 29 LIVE-misses − 8 shared = 0 real bugs); exterior-flood rewrite + 5×5-certificate pass still open.
- `task196` 42,589 -> 4,998, **+2.14 LB confirmed** — copy-the-free-input + Where-replace-only-ring-cells (no canvas rebuild; §38 philosophy via Where) + F32→U8 direct cast + Min(dilation,ch1); byte-equal FULL.
- `task270` 38,430 -> 2,978, **+2.56 LB confirmed** — center-noop probe won over its own gated primary (3,681); per-pair-latency runtime rule discovered (904s local timeout = corpus volume, NOT grader risk).
- `task064` 39,510 -> 14,706, **+0.99 LB confirmed (bet won)** — runtime-weight Conv dot-mask; AMBIGUOUS-CLASS bet decided by §CLASS-CHECK (LIVE 99.84% on raw corpus yet full hidden credit); TopK-symbolic-K → scorer None (don't-do).
- `task025` 36,433 -> 23,495, **+0.44 LB confirmed** — k5 slot-exact; k4 99.88% PARKED (LIVE classified 100% deterministic ⇒ real bugs); annotated-dynamic scored 7,448 but BROKE >1-slot inputs (annotation-lie trap), unannotated → None.
- `task080` 49,405 -> 5,664, **+2.17 LB confirmed** — FIRST F16-Conv ship (grader-validated, halves F32 conv planes project-wide); per-axis ArgMax template reads; precomputed index-table inits; info-flow floor ≈5,050 with measured fusion don't-dos.
- `task054` 47,180 -> 26,043, **+0.59 LB (pending land)** — ScatterElements(axis=3) V-line column-writes; ARCHITECTURE FLOOR proven via grader-source read + scored probes (plane ≥4,520; detection 11,856); the max(static,runtime-on-real-inputs) mechanism discovered here.
- `task359` 40,808 -> 4,510, **+2.20 LB (pending land)** — direct-dominant rewrite, zero params; RECOVERED from staging via task .md grep (summary channel lossy — check staged_submits when workers cite "safe vN").
- `task293` 37,940 -> 838, **+3.81 LB confirmed** — batch-free Einsum; scalar derived from free-input cell replaces 1-elem initializer (§42 family); ScatterND repair lane withheld at 149/267 per axiom.
- `task374` 34,392 -> 2,831, **+2.50 LB confirmed** — U8 QLinearConv neighbor sums; Slice-not-Conv crop; opset-11 axes-ATTRS cheaper than axes-inputs.
- `task094` 39,015 -> 3,806 static, **+2.33 LB confirmed** — the §39 CONVERGENCE trilogy: dynamic rungs 313/451 local credited ≈24k/≈19k (78×/42×) while static v1 landed exact and WON; static-only rebuild open.
- `task119` 43,122 -> 3,390, **+2.54 LB confirmed** — §42 free-input Einsum marginals (grader-validated) + output-overlay/line-masks; sub-1,500 disproven (traps §14z multi-output myth + 2,400 dream-floor decomposition in task119.md); v4 sibling exact.
- `task065` 44,106 -> 3,275 local (≈3,805 credited), **+2.45 LB confirmed** — divider-region extract; absent-guard fix `And(clr_pres, not_bg)` (dummy-input was fooling a ReduceMax guard); 3rd §39 extraction-markup datapoint; teammate-687 headroom flagged.
- `task338` 45,649 -> 8,469 **CLOSED, +1.68 LB confirmed** — imos2d + NEGATIVE-PAD CONV (§41 grader-CONFIRMED, first datapoint, portfolio-wide); ~38 priced dead-ends incl. 3 non-linear routes with exact margins (§40 sizing lemma: LUT read needs ≥5-stage unbroken pointwise run); closing verdict = sub-5k needs ops ONNX lacks.
- `task252` 39,858 -> 280, **+4.96 LB confirmed (CAMPAIGN RECORD, 142×)** — zero-memory params-only palette-signrank; byte-equal FULL; with task215 proves lean graphs credit honestly (§39 counter-datapoint).
- `task398` 39,875 -> ≈13,300 grader-credited (local 364!), **+1.10 LB confirmed** — the §GRADER-GAP trap's defining datapoint (37× local-vs-grader); fallback-ladder + fixed-base probes decoded it; honest static rebuild (est. 1.5–2.5k) open in task398.md.
- `task363` 43,636 -> 3,796, **+2.44 LB confirmed** — translation-stamp with the full tie-break decode (the 95.45% no-tie-break probe family is hidden-dead); §19 runtime-weight Conv as the 361-translation correlation engine; 3 owner probes all failed at exact fail values (the §VISIBLE-FAIL axiom's 3rd proof — no_diamond was synth-200k-clean).

## 2026-06-02 submit-confirmed lessons

These are recent candidates whose Kaggle results changed the reusable-library guidance:

- `task213` 55,705 -> 2,193, **+3.23 LB confirmed** — stride-gather index-canvas; orientation select on small operands before broadcast; <1k blocked by the 970-byte grid-read + one-hot-expand sub-floor.
- `task191` 32,841 -> 24,807, **+0.29 LB confirmed** — runtime-template extraction plus positional-hash match-collapse; opset20 `ReduceMax(BOOL)` accepted by Kaggle for membership reduction. Safe opset13 fallback v7 held at 26,545.
- `task383` 55,467 -> 25,767, **+0.76 LB confirmed** — generator-truth Path-B rewrite; OOD adversarial shift misses were advisory because the generator fixes placement.
- `task238` 55,368 -> 15,085, **+1.30 LB confirmed** — small-frame scalar color-index map + sentinel 10 + final Equal expansion replaced a many-plane one-hot chain.
- `task286` 63,211 -> 55,933, **+0.12 LB confirmed** — cropped-label Conv spends params to delete larger runtime label tensors; parent-equivalent compression gates against the parent. Under-20k shallow-reach probe failed public hard.

## 2026-05-23 SHIPPED algorithm-rewrite ships (latest +22.54 LB session)

Cross-family quick-reference for the highest-yield 2026-05-23 ships. See per-task .md and per-builder file for details:

- `task330` +3.00 LB → `bbox6_plus2exact` algorithm rewrite
- `task051` +3.30 LB → `floatgeom_occ` decoded-rule rebuild
- `task175` +2.98 LB → `v12 period_sum_minus_ch0` + variadic Max + INT8 plumbing
- `task117` +2.10 LB → `topk3_u8active_rank2_tail`
- `task396` +1.70 LB → `longrun_rule` (algorithm replacement)
- `task008` +1.76 LB → `dynamic_shift_bool_validu8`
- `task387` +2.54 LB → `scatter_alg_v5d_colorgrid`
- `task090` +2.02 LB → `reduce_argmin_coords` (UINT8 MaxPool prefix/suffix scans)
- `task145` v2 +1.34 LB / v3 +0.02 LB → `maxpool_label_output_v2/v3`
- `task314` +1.09 LB → `blockwise6_convlabel_boolpad`
- `task174` +2.50 LB → `top3_topk_direct_lr` (TopK presence + LR symmetry crop)
- `task062` +1.36 LB → `bool_exactaxis_u8tail`
- `task170` +2.25 LB → `dynstride_countk_boolarea_f16tail` (Einsum projections)
- `task136` +1.79 LB → `combined_qlinear_boolpad` (QLinearConv geometric diagonal)
- `task370` +2.52 LB → `linear_valid_count1_active_hybrid_boolpad` (small-table hash NOT zero-weight)
- `task042` +2.13 LB → `grouped_qconv123_dilated` (dilated 5×5 replaces dense 9×9/13×13)
- `task075` +2.46 LB → `convcode_index` (Conv as position-encoder)
- `task168` +1.91 LB → `bgch0_f16` (ch0-crop as F16 background evidence)
- `task208` +3.15 LB → `dynamic_kernel_momentheight_tworect_qseed_varminpaint`
- `task190` +1.35 LB → `ch0_u8pool_extwhere_rowcolmin` (2×2 MaxPool over ch0 for solid-block locator)
- `task213` 55,705→2,193 (shipped 2026-06-02, +3.23 LB, 25×) → `stride_gather_idxcanvas` — read the ARC-GEN generator to learn the EXACT every-3rd-row stripe spacing, then `ArgMax(first active row)` + strided `Gather` (offset+[0,3,6,…]) replaces centroid→variance→pairwise-rank→match. Per-row/col reductions via `Einsum('bkrc,k->br')` (no `[1,10,30,*]` projection); single-channel index canvas + `Equal`-expand (ch0 vs −1 sentinel) at the free BOOL output (§8b/§23). **Orientation select on the SMALL operands**: `Where(is_h, car[1,1,7,1], car_T[1,1,1,7])` broadcasts to `[1,1,7,7]` (one tensor) instead of pre-building both `idx_H`+`idx_V`. Floor proof in `task213.md`: the two *necessary* ops — grid read (4 `Einsum[1,30]`=480) + 10-ch one-hot expand (`[1,10,7,7]`=490) — sum to 970 ≈ 1k alone, and a working model needs ~1k more (orientation 392 + offset/Sign 240 + gather glue), so <1k is unreachable for an N×N oriented one-hot.

**Confirmed zero-public-weight family** (cherry13b MANDATORY before any lookup/NN-distilled candidate): task366/324/133/182/018/157/255/066/076/277. Each scored 0.00 LB on Kaggle regardless of cost reduction. See `traps.md §14e`.

---

## Pattern_Recognition (170 tasks)

### Whole-graph rewrites

- `task382` (115,918) → [exemplars/01_whole_graph_rewrites/task382_cumsum_gather.md](exemplars/01_whole_graph_rewrites/task382_cumsum_gather.md)
- `task009` (92,939) (staged Δ=-14,600) → `The_400_Tasks/Task_by_Task/task009/prototypes/build_task009_fresh_where.py`
- `task119` (43,122) → [exemplars/01_whole_graph_rewrites/task119_diag_conv.md](exemplars/01_whole_graph_rewrites/task119_diag_conv.md)
- `task055` (31,628) → [exemplars/03_crop_and_f16/task055_per_channel.md](exemplars/03_crop_and_f16/task055_per_channel.md)
- `task071` (24,150) (staged Δ=-4,184) → [exemplars/05_static_to_dynamic/task071_ring_mask.md](exemplars/05_static_to_dynamic/task071_ring_mask.md)
- `task148` (20,904) (staged Δ=-12,616) → `The_400_Tasks/Task_by_Task/task148/prototypes/build_task148_fresh.py, build_task148_slice_col0.py`
- `task322` (1,427) (staged Δ=-81) → `The_400_Tasks/Task_by_Task/task322/prototypes/build_task322_clip_cumsum.py, build_task322_crop3x3.py`

### Bool-first family (input/Concat/Cast/Pad bool keeping)

- `task080` (165,176) (staged Δ=-9,000) → `The_400_Tasks/Task_by_Task/task080/prototypes/build_task080_scratch.py`
- `task383` (152,592) (staged Δ=-31,720) → `The_400_Tasks/Task_by_Task/task383/prototypes/build_task383_bool_or_rewrite.py, build_task383_crop24.py`
- `task234` (113,059) (staged Δ=-115,428) → `The_400_Tasks/Task_by_Task/task234/prototypes/build_task234_crop20.py, build_task234_crop20_from_work.py`
- `task191` (103,921) (staged Δ=-24,544) → `The_400_Tasks/Task_by_Task/task191/prototypes/build_task191_crop23.py, build_task191_crop23_from_work.py`
- `task328` (100,002) (staged Δ=-8,720) → `The_400_Tasks/Task_by_Task/task328/prototypes/build_task328_predopt.py`
- `task379` (94,643) (staged Δ=-4) → see staged `task379_autoapply_t_6a0bc79c_on_5759_16/manifest.json`
- `task174` (70,716) → `The_400_Tasks/Task_by_Task/task174/prototypes/build_task174_scratch.py`
- `task367` (49,772) → `The_400_Tasks/Task_by_Task/task367/prototypes/build_task367_crop20.py`
- `task117` (48,328) → [exemplars/03_crop_and_f16/task117_crop15_f16.md](exemplars/03_crop_and_f16/task117_crop15_f16.md)
- `task037` (33,913) (staged Δ=-31,707) → `The_400_Tasks/Task_by_Task/task037/prototypes/build_task037_bool_and_or.py, build_task037_vecpaint.py`
- `task132` (29,611) (staged Δ=-11,268) → `The_400_Tasks/Task_by_Task/task132/prototypes/build_task132_f16.py`
- `task075` SHIPPED 2026-05-23 +2.46 LB (28,065 → 2,386) → `The_400_Tasks/Task_by_Task/task075/prototypes/build_task075_convcode_index.py` (`code_kernel` Conv as position-encoder + ConvInteger + Gather + Equal-against-colors BOOL output)
- `task323` (26,671) (staged Δ=-62,838) → [exemplars/07_conv_reformulation/task323_conv_bool.md](exemplars/07_conv_reformulation/task323_conv_bool.md)
- `task086` (25,583) (staged Δ=-7,776) → `The_400_Tasks/Task_by_Task/task086/prototypes/build_task086_bool_int32.py, build_task086_combined.py`
- `task231` (25,026) (staged Δ=-17,506) → `The_400_Tasks/Task_by_Task/task231/prototypes/build_task231_crop5x18.py, build_task231_crop5x20.py`
- `task071` (24,150) (staged Δ=-4,184) → [exemplars/05_static_to_dynamic/task071_ring_mask.md](exemplars/05_static_to_dynamic/task071_ring_mask.md)
- `task163` (23,679) (staged Δ=-11,708) → [exemplars/04_algebraic_identities/task163_add_chain.md](exemplars/04_algebraic_identities/task163_add_chain.md)
- `task268` (22,387) (staged Δ=-2,488) → see staged `task268_cast_after_all_slices_t_df78a3ab_on_5835_30/manifest.json`
- `task105` (21,576) (staged Δ=-19,252) → `The_400_Tasks/Task_by_Task/task105/prototypes/build_task105_crop14.py`
- `task148` (20,904) (staged Δ=-12,616) → `The_400_Tasks/Task_by_Task/task148/prototypes/build_task148_fresh.py, build_task148_slice_col0.py`
- `task397` (20,389) (staged Δ=-10,516) → `The_400_Tasks/Task_by_Task/task397/prototypes/build_task397_candidates.py`
- `task356` (15,791) (staged Δ=-7,040) → `The_400_Tasks/Task_by_Task/task356/prototypes/build_task356_crop10x10.py`
- `task336` (14,670) (staged Δ=-5,608) → see staged `task336_cast_after_first_slice_t_850f8f26_on_5835_30/manifest.json`
- `task237` (14,486) (staged Δ=-7,794) → see staged `task237_cast_after_all_slices_t_5df12562_on_5835_30/manifest.json`
- `task297` (13,090) (staged Δ=-110,572) → `The_400_Tasks/Task_by_Task/task297/prototypes/build_task297_crop14x6.py`
- `task343` (12,872) (staged Δ=-12,363) → `The_400_Tasks/Task_by_Task/task343/prototypes/build_task343_crop15.py, build_task343_crop5x15.py`
- `task212` (9,970) (staged Δ=-6,502) → see staged `task212_cast_after_all_slices_t_af03946f_on_5835_30/manifest.json`
- `task176` (3,613) (staged Δ=-2,580) → see staged `task176_cast_after_first_slice_t_3c99c418_on_5835_30/manifest.json`
- `task299` (1,022) (staged Δ=-1,620) → [exemplars/02_bool_first/task299_bool_concat.md](exemplars/02_bool_first/task299_bool_concat.md)

### Crop-to-active-grid + F16 placement

- `task110` (307,174) (staged Δ=-214,276) → `The_400_Tasks/Task_by_Task/task110/prototypes/build_task110_crop29.py`
- `task092` (214,117) → `The_400_Tasks/Task_by_Task/task092/prototypes/build_task092_drop_row_f16out.py`
- `task349` (183,198) (staged Δ=-18,000) → [exemplars/04_algebraic_identities/task349_10_site_equal_fold.md](exemplars/04_algebraic_identities/task349_10_site_equal_fold.md)
- `task187` (175,624) (staged Δ=-102,958) → see staged `task187_f16_t_cc9775f2_on_5835_30/manifest.json`
- `task383` (152,592) (staged Δ=-31,720) → `The_400_Tasks/Task_by_Task/task383/prototypes/build_task383_bool_or_rewrite.py, build_task383_crop24.py`
- `task051` (138,798) (staged Δ=-56,352) → `The_400_Tasks/Task_by_Task/task051/prototypes/build_task051_crop20_eqfold.py`
- `task358` (132,871) (staged Δ=-52,478) → `The_400_Tasks/Task_by_Task/task358/prototypes/build_task358_eqfold.py`
- `task284` (118,850) (staged Δ=-20,040) → `The_400_Tasks/Task_by_Task/task284/prototypes/build_task284_onehot_pruned.py, build_task284_safeopt_delta.py`
- `task161` (117,897) → `The_400_Tasks/Task_by_Task/task161/prototypes/build_task161_crop20x25.py`
- `task234` (113,059) (staged Δ=-115,428) → `The_400_Tasks/Task_by_Task/task234/prototypes/build_task234_crop20.py, build_task234_crop20_from_work.py`
- `task191` (103,921) (staged Δ=-24,544) → `The_400_Tasks/Task_by_Task/task191/prototypes/build_task191_crop23.py, build_task191_crop23_from_work.py`
- `task387` (100,793) (staged Δ=-126,316) → `The_400_Tasks/Task_by_Task/task387/prototypes/build_task387_crop18.py`
- `task328` (100,002) (staged Δ=-8,720) → `The_400_Tasks/Task_by_Task/task328/prototypes/build_task328_predopt.py`
- `task009` (92,939) (staged Δ=-14,600) → `The_400_Tasks/Task_by_Task/task009/prototypes/build_task009_fresh_where.py`
- `task145` (81,724) (staged Δ=-43,170) → `The_400_Tasks/Task_by_Task/task145/prototypes/build_task145_crop20.py`
- `task204` (80,379) (staged Δ=-31,488) → `The_400_Tasks/Task_by_Task/task204/prototypes/build_task204_crop20.py`
- `task112` (77,270) (staged Δ=-4,564) → `The_400_Tasks/Task_by_Task/task112/prototypes/build_task112_castdtype.py`
- `task165` (76,537) (staged Δ=-30,908) → `The_400_Tasks/Task_by_Task/task165/prototypes/build_task165_crop20.py, build_task165_crop20_from_work.py`
- `task192` (74,531) (staged Δ=-1,664) → `The_400_Tasks/Task_by_Task/task192/prototypes/build_task192_crop14x18.py`
- `task162` (71,587) (staged Δ=-17,148) → `The_400_Tasks/Task_by_Task/task162/prototypes/build_task162_crop20.py, build_task162_crop20_from_work.py`
- `task215` (69,499) (staged Δ=-16,560) → `The_400_Tasks/Task_by_Task/task215/prototypes/build_task215_crop20.py`
- `task340` (67,493) → `The_400_Tasks/Task_by_Task/task340/prototypes/build_task340_f16.py`
- `task090` (63,608) → `The_400_Tasks/Task_by_Task/task090/prototypes/build_task090_crop5xfull.py`
- `task222` (58,031) (staged Δ=-32,338) → `The_400_Tasks/Task_by_Task/task222/prototypes/build_task222_f16_selective.py`
- `task131` (50,325) (staged Δ=-13,536) → `The_400_Tasks/Task_by_Task/task131/prototypes/build_task131_crop18.py, build_task131_crop18_from_work.py`
- `task367` (49,772) → `The_400_Tasks/Task_by_Task/task367/prototypes/build_task367_crop20.py`
- `task117` (48,328) → [exemplars/03_crop_and_f16/task117_crop15_f16.md](exemplars/03_crop_and_f16/task117_crop15_f16.md)
- `task219` (43,558) (staged Δ=-2,142) → `The_400_Tasks/Task_by_Task/task219/prototypes/build_task219_crop15x10.py`
- `task359` (40,808) (staged Δ=-4,324) → [exemplars/02_bool_first/task359_bool_first_where.md](exemplars/02_bool_first/task359_bool_first_where.md)
- `task037` (33,913) (staged Δ=-31,707) → `The_400_Tasks/Task_by_Task/task037/prototypes/build_task037_bool_and_or.py, build_task037_vecpaint.py`
- `task055` (31,628) → [exemplars/03_crop_and_f16/task055_per_channel.md](exemplars/03_crop_and_f16/task055_per_channel.md)
- `task168` SHIPPED 2026-05-23 +1.91 LB (31,561 → 4,673) → `The_400_Tasks/Task_by_Task/task168/prototypes/build_task168_bgch0_f16.py` (cast `ch0_crop[10×10]` directly to F16 as background evidence + 2×2 bg-score Conv kernels — works only AFTER active-grid crop)
- `task093` (30,403) (staged Δ=-13,832) → [exemplars/06_cse_dead_op/task093_dead_relu_post_maxpool.md](exemplars/06_cse_dead_op/task093_dead_relu_post_maxpool.md)
- `task281` (30,342) (staged Δ=-125,851) → [exemplars/03_crop_and_f16/task281_crop13.md](exemplars/03_crop_and_f16/task281_crop13.md)
- `task132` (29,611) (staged Δ=-11,268) → `The_400_Tasks/Task_by_Task/task132/prototypes/build_task132_f16.py`
- `task102` (28,959) (staged Δ=-13,888) → see staged `task102_cast_after_first_slice_t_77049ede_on_5835_30/manifest.json`
- `task075` SHIPPED 2026-05-23 +2.46 LB (28,065 → 2,386) → `The_400_Tasks/Task_by_Task/task075/prototypes/build_task075_convcode_index.py` (`code_kernel` Conv as position-encoder + ConvInteger + Gather + Equal-against-colors BOOL output)
- `task288` (27,190) (staged Δ=-17,834) → `The_400_Tasks/Task_by_Task/task288/prototypes/build_task288_scratch.py`
- `task076` (26,204) (staged Δ=-6,750) → see staged `task076_cast_after_all_slices_t_b6681482_on_5835_30/manifest.json`
- `task062` (26,192) (staged Δ=-17,280) → see staged `task062_cast_after_first_slice_t_268b0dce_on_5835_30/manifest.json`
- `task086` (25,583) (staged Δ=-7,776) → `The_400_Tasks/Task_by_Task/task086/prototypes/build_task086_bool_int32.py, build_task086_combined.py`
- `task099` (25,352) → `The_400_Tasks/Task_by_Task/task099/prototypes/build_task099_f16.py`
- `task231` (25,026) (staged Δ=-17,506) → `The_400_Tasks/Task_by_Task/task231/prototypes/build_task231_crop5x18.py, build_task231_crop5x20.py`
- `task232` (24,793) (staged Δ=-8,848) → `The_400_Tasks/Task_by_Task/task232/prototypes/build_task232_drop_v17mul.py`
- `task042` SHIPPED 2026-05-23 +2.13 LB (24,712 → 2,943) → `The_400_Tasks/Task_by_Task/task042/prototypes/build_task042_grouped_qconv_dilated.py` (grouped QLinearConv with dilated 5×5 kernels for n=2/n=3 pair-distance detectors, replacing dense 9×9/13×13)
- `task071` (24,150) (staged Δ=-4,184) → [exemplars/05_static_to_dynamic/task071_ring_mask.md](exemplars/05_static_to_dynamic/task071_ring_mask.md)
- `task163` (23,679) (staged Δ=-11,708) → [exemplars/04_algebraic_identities/task163_add_chain.md](exemplars/04_algebraic_identities/task163_add_chain.md)
- `task126` (23,171) (staged Δ=-32,725) → `The_400_Tasks/Task_by_Task/task126/prototypes/build_task126_crop10x20.py`
- `task061` (22,789) (staged Δ=-2,604) → `The_400_Tasks/Task_by_Task/task061/prototypes/build_task061_mod_formula.py, build_task061_reducemax_bank.py`
- `task268` (22,387) (staged Δ=-2,488) → see staged `task268_cast_after_all_slices_t_df78a3ab_on_5835_30/manifest.json`
- `task166` (21,648) (staged Δ=-6,273) → `The_400_Tasks/Task_by_Task/task166/prototypes/build_task166_gather_elim_split.py`
- `task105` (21,576) (staged Δ=-19,252) → `The_400_Tasks/Task_by_Task/task105/prototypes/build_task105_crop14.py`
- `task066` (21,408) → `The_400_Tasks/Task_by_Task/task066/prototypes/build_task066_crop20.py`
- `task251` (21,198) (staged Δ=-9,504) → see staged `task251_cast_after_first_slice_t_816d9ca9_on_5835_30/manifest.json`
- `task148` (20,904) (staged Δ=-12,616) → `The_400_Tasks/Task_by_Task/task148/prototypes/build_task148_fresh.py, build_task148_slice_col0.py`
- `task397` (20,389) (staged Δ=-10,516) → `The_400_Tasks/Task_by_Task/task397/prototypes/build_task397_candidates.py`
- `task278` (20,348) (staged Δ=-9,864) → `The_400_Tasks/Task_by_Task/task278/prototypes/build_task278_rmax_crop.py`
- `task200` (19,426) (staged Δ=-28,148) → `The_400_Tasks/Task_by_Task/task200/prototypes/build_task200_crop10.py`
- `task302` (19,321) (staged Δ=-7,776) → [exemplars/06_cse_dead_op/task302_drop_relu_shift.md](exemplars/06_cse_dead_op/task302_drop_relu_shift.md)
- `task371` (17,340) (staged Δ=-1,408) → `The_400_Tasks/Task_by_Task/task371/prototypes/build_task371_convfold.py`
- `task063` (17,147) (staged Δ=-848) → see staged `task063_cast_after_all_slices_t_927d4adf_on_5835_30/manifest.json`
- `task256` (17,062) (staged Δ=-3,488) → `The_400_Tasks/Task_by_Task/task256/prototypes/build_task256_zerofold.py`
- `task151` (17,046) (staged Δ=-3,744) → `The_400_Tasks/Task_by_Task/task151/prototypes/build_task151_crossconv.py`
- `task197` (16,252) (staged Δ=-44,180) → `The_400_Tasks/Task_by_Task/task197/prototypes/build_task197_crop10.py, build_task197_crop14x10.py`
- `task356` (15,791) (staged Δ=-7,040) → `The_400_Tasks/Task_by_Task/task356/prototypes/build_task356_crop10x10.py`
- `task012` (15,220) (staged Δ=-3,168) → see staged `task012_cast_after_first_slice_t_ada8fda9_on_5835_30/manifest.json`
- `task341` (14,849) (staged Δ=-6,608) → `The_400_Tasks/Task_by_Task/task341/prototypes/build_task341_drop_shift16.py`
- `task336` (14,670) (staged Δ=-5,608) → see staged `task336_cast_after_first_slice_t_850f8f26_on_5835_30/manifest.json`
- `task237` (14,486) (staged Δ=-7,794) → see staged `task237_cast_after_all_slices_t_5df12562_on_5835_30/manifest.json`
- `task028` (14,192) (staged Δ=-5,876) → see staged `task028_cast_after_first_slice_t_be60403f_on_5835_30/manifest.json`
- `task392` (13,546) (staged Δ=-4,488) → `The_400_Tasks/Task_by_Task/task392/prototypes/build_task392_pad10x10.py`
- `task217` (13,301) (staged Δ=-126,333) → `The_400_Tasks/Task_by_Task/task217/prototypes/build_task217_crop9.py`
- `task034` (13,273) (staged Δ=-79,002) → `The_400_Tasks/Task_by_Task/task034/prototypes/build_task034_crop9.py`
- `task297` (13,090) (staged Δ=-110,572) → `The_400_Tasks/Task_by_Task/task297/prototypes/build_task297_crop14x6.py`
- `task343` (12,872) (staged Δ=-12,363) → `The_400_Tasks/Task_by_Task/task343/prototypes/build_task343_crop15.py, build_task343_crop5x15.py`
- `task068` (12,589) (staged Δ=-5,362) → see staged `task068_cast_after_first_slice_t_5f403dfa_on_5835_30/manifest.json`
- `task260` (12,074) (staged Δ=-3,030) → see staged `task260_cast_after_first_slice_t_1c5dc6bb_on_5835_30/manifest.json`
- `task331` (11,948) (staged Δ=-55,668) → `The_400_Tasks/Task_by_Task/task331/prototypes/build_task331_crop10.py`
- `task248` (11,595) (staged Δ=-2,750) → see staged `task248_cast_after_first_slice_t_21eb3b2e_on_5835_30/manifest.json`
- `task362` (11,536) (staged Δ=-3,790) → see staged `task362_cast_after_first_slice_t_b09da5a0_on_5835_30/manifest.json`
- `task027` (11,492) (staged Δ=-3,256) → see staged `task027_cast_after_first_slice_t_e18c1a95_on_5835_30/manifest.json`
- `task298` (11,290) (staged Δ=-6,072) → `The_400_Tasks/Task_by_Task/task298/prototypes/build_task298_noop_elim.py`
- `task082` (11,245) (staged Δ=-19,917) → see staged `task082_crop15col_t_b33f4b9d_on_5806_87/manifest.json`
- `task225` (11,240) (staged Δ=-7,752) → see staged `task225_cast_after_all_slices_t_da3210cf_on_5835_30/manifest.json`
- `task381` (11,163) (staged Δ=-40,968) → `The_400_Tasks/Task_by_Task/task381/prototypes/build_task381_crop10.py, build_task381_crop10_from_work.py`
- `task357` (11,082) → see staged `task357_f16lookup_t_042ecb7b_on_5806_87/manifest.json`
- `task345` (11,069) (staged Δ=-37,208) → `The_400_Tasks/Task_by_Task/task345/prototypes/build_task345_crop10.py`
- `task320` (10,675) (staged Δ=-2,610) → see staged `task320_cast_after_first_slice_t_c81bcd76_on_5835_30/manifest.json`
- `task212` (9,970) (staged Δ=-6,502) → see staged `task212_cast_after_all_slices_t_af03946f_on_5835_30/manifest.json`
- `task258` (9,882) (staged Δ=-21,428) → `The_400_Tasks/Task_by_Task/task258/prototypes/build_task258_crop10.py`
- `task348` (9,840) (staged Δ=-1,506) → see staged `task348_cast_after_first_slice_t_dbc14661_on_5835_30/manifest.json`
- `task139` (9,614) (staged Δ=-2,592) → see staged `task139_cast_after_first_slice_t_8bcde035_on_5835_30/manifest.json`
- `task050` (9,475) (staged Δ=-900) → see staged `task050_cast_after_all_slices_t_0260455c_on_5835_30/manifest.json`
- `task045` (8,862) (staged Δ=-820) → see staged `task045_cast_after_first_slice_t_c2a48e73_on_5835_30/manifest.json`
- `task043` (8,679) (staged Δ=-440) → see staged `task043_cast_after_first_slice_t_f92bba83_on_5835_30/manifest.json`
- `task060` (8,433) (staged Δ=-3,960) → see staged `task060_cast_after_first_slice_t_1451423a_on_5835_30/manifest.json`
- `task007` (8,251) (staged Δ=-980) → see staged `task007_cast_after_first_slice_t_396a1275_on_5835_30/manifest.json`
- `task369` (7,729) (staged Δ=-5,600) → see staged `task369_cast_after_all_slices_t_9afdb627_on_5835_30/manifest.json`
- `task262` (7,433) → `The_400_Tasks/Task_by_Task/task262/prototypes/build_task262_crop3.py`
- `task106` (7,116) → see staged `task106_f16cast_t_c32f065f_on_5806_87/manifest.json`
- `task226` (5,619) (staged Δ=-3,688) → see staged `task226_cast_after_all_slices_t_f78bc3b3_on_5835_30/manifest.json`
- `task047` (4,950) (staged Δ=-2,600) → see staged `task047_cast_after_all_slices_t_9047c994_on_5835_30/manifest.json`
- `task176` (3,613) (staged Δ=-2,580) → see staged `task176_cast_after_first_slice_t_3c99c418_on_5835_30/manifest.json`
- `task322` (1,427) (staged Δ=-81) → `The_400_Tasks/Task_by_Task/task322/prototypes/build_task322_clip_cumsum.py, build_task322_crop3x3.py`
- `task266` (1,174) (staged Δ=-14) → [exemplars/08_gather_lookup/task266_scalar_broadcast.md](exemplars/08_gather_lookup/task266_scalar_broadcast.md)

### Algebraic identities (Equal-fold, Add-chain, scalar)

- `task173` (256,900) → `The_400_Tasks/Task_by_Task/task173/prototypes/build_task173_dce.py`
- `task013` (147,257) → see staged `task013_add_chain_to_sum_BIG_POLISH_c_20260520T0201Z/manifest.json`
- `task051` (138,798) (staged Δ=-56,352) → `The_400_Tasks/Task_by_Task/task051/prototypes/build_task051_crop20_eqfold.py`
- `task358` (132,871) (staged Δ=-52,478) → `The_400_Tasks/Task_by_Task/task358/prototypes/build_task358_eqfold.py`
- `task234` (113,059) (staged Δ=-115,428) → `The_400_Tasks/Task_by_Task/task234/prototypes/build_task234_crop20.py, build_task234_crop20_from_work.py`
- `task387` (100,793) (staged Δ=-126,316) → `The_400_Tasks/Task_by_Task/task387/prototypes/build_task387_crop18.py`
- `task145` (81,724) (staged Δ=-43,170) → `The_400_Tasks/Task_by_Task/task145/prototypes/build_task145_crop20.py`
- `task204` (80,379) (staged Δ=-31,488) → `The_400_Tasks/Task_by_Task/task204/prototypes/build_task204_crop20.py`
- `task378` (64,007) (staged Δ=-147) → see staged `task378_autoapply_t_6a0bc664_on_5759_16/manifest.json`
- `task102` (28,959) (staged Δ=-13,888) → see staged `task102_cast_after_first_slice_t_77049ede_on_5835_30/manifest.json`
- `task163` (23,679) (staged Δ=-11,708) → [exemplars/04_algebraic_identities/task163_add_chain.md](exemplars/04_algebraic_identities/task163_add_chain.md)
- `task397` (20,389) (staged Δ=-10,516) → `The_400_Tasks/Task_by_Task/task397/prototypes/build_task397_candidates.py`
- `task200` (19,426) (staged Δ=-28,148) → `The_400_Tasks/Task_by_Task/task200/prototypes/build_task200_crop10.py`
- `task297` (13,090) (staged Δ=-110,572) → `The_400_Tasks/Task_by_Task/task297/prototypes/build_task297_crop14x6.py`
- `task389` (3,368) (staged Δ=-9) → see staged `task389_autoapply_t_6a0bc72c_on_5759_16/manifest.json`
- `task266` (1,174) (staged Δ=-14) → [exemplars/08_gather_lookup/task266_scalar_broadcast.md](exemplars/08_gather_lookup/task266_scalar_broadcast.md)

### Static-to-dynamic / rank reduction / projection-cache

- `task127` (4,046) → see staged `task127_v5_dynamic_sep_ch0_lesshalf_pad30_t_d37da854_on_current5755_31_20260518T0700Z/manifest.json`

### CSE / dead-op elimination

- `task173` (256,900) → `The_400_Tasks/Task_by_Task/task173/prototypes/build_task173_dce.py`
- `task093` (30,403) (staged Δ=-13,832) → [exemplars/06_cse_dead_op/task093_dead_relu_post_maxpool.md](exemplars/06_cse_dead_op/task093_dead_relu_post_maxpool.md)
- `task062` (26,192) (staged Δ=-17,280) → see staged `task062_cast_after_first_slice_t_268b0dce_on_5835_30/manifest.json`
- `task302` (19,321) (staged Δ=-7,776) → [exemplars/06_cse_dead_op/task302_drop_relu_shift.md](exemplars/06_cse_dead_op/task302_drop_relu_shift.md)
- `task341` (14,849) (staged Δ=-6,608) → `The_400_Tasks/Task_by_Task/task341/prototypes/build_task341_drop_shift16.py`
- `task298` (11,290) (staged Δ=-6,072) → `The_400_Tasks/Task_by_Task/task298/prototypes/build_task298_noop_elim.py`

### Conv reformulation

- `task119` (43,122) → [exemplars/01_whole_graph_rewrites/task119_diag_conv.md](exemplars/01_whole_graph_rewrites/task119_diag_conv.md)
- `task323` (26,671) (staged Δ=-62,838) → [exemplars/07_conv_reformulation/task323_conv_bool.md](exemplars/07_conv_reformulation/task323_conv_bool.md)

### Gather/lookup compression

- `task101` (63,551) → `The_400_Tasks/Task_by_Task/task101/prototypes/build_task101_castdtype_scatterchain.py`
- `task166` (21,648) (staged Δ=-6,273) → `The_400_Tasks/Task_by_Task/task166/prototypes/build_task166_gather_elim_split.py`
- `task357` (11,082) → see staged `task357_f16lookup_t_042ecb7b_on_5806_87/manifest.json`

### Untagged (52 tasks — no transferable technique attribution yet)

- `task175` (406,404) — Filling Regions
- `task025` (252,597) — Filling Regions
- `task286` (234,282) — Filling Regions
- `task280` (207,894) — Filling Regions
- `task005` (183,854) — Tiling | Object Detection | Color Mapping | Fillin
- `task133` (161,071) — Filling Regions
- `task089` (121,519) — Filling Regions
- `task243` (117,161) — Filling Regions
- `task002` (110,030) — Object Detection | Color Mapping | Filling Regions
- `task017` (105,937) — Filling Regions
- `task285` (99,827) — Filling Regions
- `task198` (89,301) — Filling Regions
- `task224` (71,261) — Filling Regions
- `task240` (66,594) — Filling Regions
- `task306` (65,826) — Filling Regions
- `task255` (65,300) — Filling Regions
- `task181` (64,844) — Filling Regions
- `task335` (60,245) — Filling Regions
- `task265` (56,878) — Filling Regions
- `task363` (54,241) — Filling Regions
- `task287` (50,758) — Symmetry Completion
- `task020` (50,090) — Reflection | Symmetry Completion
- `task338` (45,649) — Symmetry Completion
- `task350` (28,729) — Filling Regions
- `task333` (25,690) — Filling Regions
- `task074` (21,634) — Symmetry Completion
- `task361` (21,100) — Filling Regions
- `task058` (19,826) — Filling Regions
- `task084` (19,513) — Filling Regions
- `task041` (16,327) — Filling Regions
- ... (22 more untagged tasks)

---

## Object_Based (144 tasks)

### Whole-graph rewrites

- `task031` (18,688) (staged Δ=-272) → see staged `task031_where_t_6110ca8d_on_5835_30/manifest.json`
- `task115` (13,187) (staged Δ=-135,600) → [exemplars/01_whole_graph_rewrites/task115_axis_factor.md](exemplars/01_whole_graph_rewrites/task115_axis_factor.md)
- `task021` (13,142) (staged Δ=-5,520) → [exemplars/02_bool_first/task021_mul_mask_to_where.md](exemplars/02_bool_first/task021_mul_mask_to_where.md)

### Bool-first family (input/Concat/Cast/Pad bool keeping)

- `task069` (127,412) (staged Δ=-20) → see staged `task069_polish_t_ca504325_on_5835_30/manifest.json`
- `task029` (101,347) (staged Δ=-16,200) → `The_400_Tasks/Task_by_Task/task029/prototypes/build_task029_crop25.py, build_task029_crop25.py`
- `task208` SHIPPED 2026-05-23 +3.15 LB (92,672 → 3,983) → `The_400_Tasks/Task_by_Task/task208/prototypes/build_task208_dynamic_kernel.py` (Einsum row-central-moments + QLinearConv quantized seed + variadic Min UINT8 paint-kernel)
- `task170` SHIPPED 2026-05-23 +2.25 LB (54,660 → 5,737) → `The_400_Tasks/Task_by_Task/task170/prototypes/build_task170_einsumproj_dynstride.py` (Einsum projections replace row_proj/col_proj tensors + motif-count dynamic stride)
- `task184` (23,667) (staged Δ=-28,799) → `The_400_Tasks/Task_by_Task/task184/prototypes/build_task184_nz_shortcut.py`
- `task301` (19,336) (staged Δ=-66,682) → `The_400_Tasks/Task_by_Task/task301/prototypes/build_task301_crop12x9.py`
- `task031` (18,688) (staged Δ=-272) → see staged `task031_where_t_6110ca8d_on_5835_30/manifest.json`
- `task247` (18,184) → `The_400_Tasks/Task_by_Task/task247/prototypes/build_task247_crop10x10.py`
- `task355` (17,818) (staged Δ=-2,522) → `The_400_Tasks/Task_by_Task/task355/prototypes/build_task355_crop20.py, build_task355_crop20.py`
- `task319` (16,552) (staged Δ=-741) → [exemplars/02_bool_first/task319_opset_bump_pad_uint8.md](exemplars/02_bool_first/task319_opset_bump_pad_uint8.md)
- `task143` (15,965) (staged Δ=-6,988) → see staged `task143_cast_after_first_slice_t_07e1031a_on_5835_30/manifest.json`
- `task250` (15,380) (staged Δ=-4,000) → see staged `task250_dropcast_t_38b3234a_on_5835_30/manifest.json`
- `task115` (13,187) (staged Δ=-135,600) → [exemplars/01_whole_graph_rewrites/task115_axis_factor.md](exemplars/01_whole_graph_rewrites/task115_axis_factor.md)
- `task021` (13,142) (staged Δ=-5,520) → [exemplars/02_bool_first/task021_mul_mask_to_where.md](exemplars/02_bool_first/task021_mul_mask_to_where.md)
- `task342` (12,836) (staged Δ=-4,000) → [exemplars/02_bool_first/task273_drop_pad_cast_bool.md](exemplars/02_bool_first/task273_drop_pad_cast_bool.md)
- `task263` (8,258) → see staged `task263_bool_andfold_BIG_POLISH_c_20260520T0212Z/manifest.json`
- `task006` (560) (staged Δ=-1,359) → `The_400_Tasks/Task_by_Task/task006/prototypes/build_task006_bool_v2.py, build_task006_crop3x7.py`
- `task347` (560) (staged Δ=-1,359) → `The_400_Tasks/Task_by_Task/task347/prototypes/build_task347_bool_v2.py`
- `task395` (523) (staged Δ=-1,468) → `The_400_Tasks/Task_by_Task/task395/prototypes/build_task395_bool_v2.py`

### Crop-to-active-grid + F16 placement

- `task138` (236,364) (staged Δ=-19,096) → `The_400_Tasks/Task_by_Task/task138/prototypes/build_task138_crop26.py`
- `task205` (177,538) (staged Δ=-4,000) → see staged `task205_dropcast_t_c59cc633_on_5835_30/manifest.json`
- `task153` (177,116) (staged Δ=-38,868) → `The_400_Tasks/Task_by_Task/task153/prototypes/build_task153_crop10.py, build_task153_crop12.py`
- `task008` (112,883) (staged Δ=-6,656) → see staged `task008_f16targeted_t_659601e4_on_5835_30/manifest.json`
- `task264` (111,787) (staged Δ=-73,296) → `The_400_Tasks/Task_by_Task/task264/prototypes/build_task264_conv_mask.py, build_task264_crop16.py`
- `task396` (109,351) (staged Δ=-188,772) → `The_400_Tasks/Task_by_Task/task396/prototypes/build_task396_crop18.py, build_task396_v4_channel_sparse_tail.py`
- `task216` (107,186) (staged Δ=-52,160) → `The_400_Tasks/Task_by_Task/task216/prototypes/build_task216_crop20.py, build_task216_crop20_from_work.py`
- `task202` (105,567) (staged Δ=-35,162) → see staged `task202_f16drop_t_00ffc6c9_on_5835_30/manifest.json`
- `task029` (101,347) (staged Δ=-16,200) → `The_400_Tasks/Task_by_Task/task029/prototypes/build_task029_crop25.py, build_task029_crop25.py`
- `task208` SHIPPED 2026-05-23 +3.15 LB (92,672 → 3,983) → `The_400_Tasks/Task_by_Task/task208/prototypes/build_task208_dynamic_kernel.py` (Einsum row-central-moments + QLinearConv quantized seed + variadic Min UINT8 paint-kernel)
- `task201` (84,895) (staged Δ=-47,118) → [exemplars/05_static_to_dynamic/task201_reduce_reorder.md](exemplars/05_static_to_dynamic/task201_reduce_reorder.md)
- `task159` (78,580) (staged Δ=-10,642) → `The_400_Tasks/Task_by_Task/task159/prototypes/build_task159_resize_scale.py`
- `task384` (77,211) (staged Δ=-16,200) → [exemplars/03_crop_and_f16/task384_f16_drop.md](exemplars/03_crop_and_f16/task384_f16_drop.md)
- `task036` (73,820) (staged Δ=-16,200) → see staged `task036_f16drop_t_3c37de8b_on_5835_30/manifest.json`
- `task097` (71,292) (staged Δ=-56,488) → `The_400_Tasks/Task_by_Task/task097/prototypes/build_task097_crop20.py, build_task097_crop20_from_work.py`
- `task206` (69,360) (staged Δ=-9,348) → `The_400_Tasks/Task_by_Task/task206/prototypes/build_task206_crop12.py, build_task206_crop12_f16out.py`
- `task137` (67,367) (staged Δ=-30,282) → see staged `task137_f16drop_t_9704ccd4_on_5835_30/manifest.json`
- `task377` (63,682) (staged Δ=-13,870) → `The_400_Tasks/Task_by_Task/task377/prototypes/build_task377_ringdyn.py`
- `task400` (60,395) (staged Δ=-18,000) → `The_400_Tasks/Task_by_Task/task400/prototypes/build_task400_crop24.py`
- `task213` (55,705) → `The_400_Tasks/Task_by_Task/task213/prototypes/build_task213_f16.py`
- `task193` (55,657) (staged Δ=-18,800) → see staged `task193_f16drop_t_ae28f22c_on_5835_30/manifest.json`
- `task238` (55,368) (staged Δ=-18,232) → `The_400_Tasks/Task_by_Task/task238/prototypes/build_task238_crop20.py, build_task238_crop22.py`
- `task170` SHIPPED 2026-05-23 +2.25 LB (54,660 → 5,737) → `The_400_Tasks/Task_by_Task/task170/prototypes/build_task170_einsumproj_dynstride.py` (Einsum projections replace row_proj/col_proj tensors + motif-count dynamic stride)
- `task157` (54,120) (staged Δ=-27,588) → `The_400_Tasks/Task_by_Task/task157/prototypes/build_task157_crop10x15.py`
- `task244` (53,897) (staged Δ=-1,384) → `The_400_Tasks/Task_by_Task/task244/prototypes/build_task244_crop19x19.py`
- `task122` (52,517) (staged Δ=-14,772) → `The_400_Tasks/Task_by_Task/task122/prototypes/build_task122_crop21.py, build_task122_crop21_from_work.py`
- `task325` (50,733) (staged Δ=-81,700) → `The_400_Tasks/Task_by_Task/task325/prototypes/build_task325_crop16.py`
- `task308` (49,062) (staged Δ=-2,574) → `The_400_Tasks/Task_by_Task/task308/prototypes/build_task308_crop20.py`
- `task195` (47,787) (staged Δ=-38,518) → `The_400_Tasks/Task_by_Task/task195/prototypes/build_task195_crop19.py, build_task195_crop19_from_work.py`
- `task290` (47,367) (staged Δ=-93,873) → `The_400_Tasks/Task_by_Task/task290/prototypes/build_task290_count_input_nomask.py, build_task290_crop15.py`
- `task128` (46,305) (staged Δ=-72,483) → [exemplars/03_crop_and_f16/task128_crop15.md](exemplars/03_crop_and_f16/task128_crop15.md)
- `task014` (45,690) (staged Δ=-27,432) → `The_400_Tasks/Task_by_Task/task014/prototypes/build_task014_crop15.py, build_task014_crop18.py`
- `task049` (44,787) (staged Δ=-7,636) → [exemplars/01_whole_graph_rewrites/task049_least_color_bbox.md](exemplars/01_whole_graph_rewrites/task049_least_color_bbox.md)
- `task065` (44,106) (staged Δ=-7,306) → see staged `task065_f16drop_t_1197b599_on_5835_30/manifest.json`
- `task270` (38,430) (staged Δ=-4,488) → `The_400_Tasks/Task_by_Task/task270/prototypes/build_task270_crop15.py`
- `task233` (33,057) (staged Δ=-10,482) → [exemplars/02_bool_first/task233_equal_fold.md](exemplars/02_bool_first/task233_equal_fold.md)
- `task300` (32,854) (staged Δ=-66,873) → `The_400_Tasks/Task_by_Task/task300/prototypes/build_task300_crop15.py, build_task300_crop15_from_work.py`
- `task351` (31,271) (staged Δ=-10,576) → see staged `task351_cast_after_first_slice_t_10b3444e_on_5835_30/manifest.json`
- `task011` (29,790) (staged Δ=-10,728) → `The_400_Tasks/Task_by_Task/task011/prototypes/build_task011_crop11.py, build_task011_crop11_f16out.py`
- `task365` (29,489) (staged Δ=-39,109) → `The_400_Tasks/Task_by_Task/task365/prototypes/build_task365_crop10x10.py`
- `task154` (29,015) (staged Δ=-36,438) → `The_400_Tasks/Task_by_Task/task154/prototypes/build_task154_crop15.py, build_task154_crop15_from_work.py`
- `task091` (28,989) (staged Δ=-42,663) → `The_400_Tasks/Task_by_Task/task091/prototypes/build_task091_crop15.py`
- `task178` (28,184) (staged Δ=-11,680) → see staged `task178_cast_after_first_slice_t_5f10e533_on_5835_30/manifest.json`
- `task044` (27,232) (staged Δ=-16,520) → `The_400_Tasks/Task_by_Task/task044/prototypes/build_task044_f16.py`
- `task046` (25,984) (staged Δ=-19,740) → `The_400_Tasks/Task_by_Task/task046/prototypes/build_task046_padslice.py`
- `task096` (25,364) → `The_400_Tasks/Task_by_Task/task096/prototypes/build_task096_f16decode.py`
- `task253` (23,974) (staged Δ=-9,980) → `The_400_Tasks/Task_by_Task/task253/prototypes/build_task253_crop13x13.py`
- `task228` (23,286) (staged Δ=-14,308) → `The_400_Tasks/Task_by_Task/task228/prototypes/build_task228_where.py`
- `task390` (22,766) (staged Δ=-5,538) → `The_400_Tasks/Task_by_Task/task390/prototypes/build_task390_crop15.py, build_task390_crop15_from_work.py`
- `task242` (22,355) (staged Δ=-1,780) → `The_400_Tasks/Task_by_Task/task242/prototypes/build_task242_crop16.py`
- `task109` (21,436) (staged Δ=-41,701) → `The_400_Tasks/Task_by_Task/task109/prototypes/build_task109_crop13.py`
- `task146` (20,905) (staged Δ=-16,368) → `The_400_Tasks/Task_by_Task/task146/prototypes/build_task146_crop9x3.py`
- `task310` (20,112) (staged Δ=-61,812) → `The_400_Tasks/Task_by_Task/task310/prototypes/build_task310_crop10.py, build_task310_crop8.py`
- `task301` (19,336) (staged Δ=-66,682) → `The_400_Tasks/Task_by_Task/task301/prototypes/build_task301_crop12x9.py`
- `task245` (18,360) (staged Δ=-88,068) → `The_400_Tasks/Task_by_Task/task245/prototypes/build_task245_crop10.py`
- `task247` (18,184) → `The_400_Tasks/Task_by_Task/task247/prototypes/build_task247_crop10x10.py`
- `task355` (17,818) (staged Δ=-2,522) → `The_400_Tasks/Task_by_Task/task355/prototypes/build_task355_crop20.py, build_task355_crop20.py`
- `task143` (15,965) (staged Δ=-6,988) → see staged `task143_cast_after_first_slice_t_07e1031a_on_5835_30/manifest.json`
- `task059` (15,351) (staged Δ=-5,620) → see staged `task059_cast_after_first_slice_t_d1e4138d_on_5835_30/manifest.json`
- `task183` (15,138) (staged Δ=-7,032) → `The_400_Tasks/Task_by_Task/task183/prototypes/build_task183_crop10x10.py`
- `task030` (14,534) (staged Δ=-39,840) → `The_400_Tasks/Task_by_Task/task030/prototypes/build_task030_crop10.py`
- `task057` (11,188) (staged Δ=-64,948) → `The_400_Tasks/Task_by_Task/task057/prototypes/build_task057_crop10.py, build_task057_crop8.py`
- `task180` (9,688) → `The_400_Tasks/Task_by_Task/task180/prototypes/build_task180_crop8x8.py`
- `task130` (5,148) → `The_400_Tasks/Task_by_Task/task130/prototypes/build_task130_crop9x9.py`
- `task274` (2,489) → `The_400_Tasks/Task_by_Task/task274/prototypes/build_task274_crop14x12.py, build_task274_crop14.py`
- `task073` (1,407) → `The_400_Tasks/Task_by_Task/task073/prototypes/build_task073_crop5.py`
- `task399` (1,377) → `The_400_Tasks/Task_by_Task/task399/prototypes/build_task399_crop7.py, build_task399_crop7_from_work.py`
- `task380` (1,089) → `The_400_Tasks/Task_by_Task/task380/prototypes/build_task380_crop3x3.py, build_task380_negstep.py`
- `task186` (583) (staged Δ=-215) → `The_400_Tasks/Task_by_Task/task186/prototypes/build_task186_compact_bank.py, build_task186_crop3x3.py`
- `task006` (560) (staged Δ=-1,359) → `The_400_Tasks/Task_by_Task/task006/prototypes/build_task006_bool_v2.py, build_task006_crop3x7.py`

### Algebraic identities (Equal-fold, Add-chain, scalar)

- `task138` (236,364) (staged Δ=-19,096) → `The_400_Tasks/Task_by_Task/task138/prototypes/build_task138_crop26.py`
- `task202` (105,567) (staged Δ=-35,162) → see staged `task202_f16drop_t_00ffc6c9_on_5835_30/manifest.json`
- `task170` SHIPPED 2026-05-23 +2.25 LB (54,660 → 5,737) → `The_400_Tasks/Task_by_Task/task170/prototypes/build_task170_einsumproj_dynstride.py` (Einsum projections replace row_proj/col_proj tensors + motif-count dynamic stride)
- `task157` (54,120) (staged Δ=-27,588) → `The_400_Tasks/Task_by_Task/task157/prototypes/build_task157_crop10x15.py`
- `task195` (47,787) (staged Δ=-38,518) → `The_400_Tasks/Task_by_Task/task195/prototypes/build_task195_crop19.py, build_task195_crop19_from_work.py`
- `task290` (47,367) (staged Δ=-93,873) → `The_400_Tasks/Task_by_Task/task290/prototypes/build_task290_count_input_nomask.py, build_task290_crop15.py`
- `task044` (27,232) (staged Δ=-16,520) → `The_400_Tasks/Task_by_Task/task044/prototypes/build_task044_f16.py`
- `task046` (25,984) (staged Δ=-19,740) → `The_400_Tasks/Task_by_Task/task046/prototypes/build_task046_padslice.py`
- `task319` (16,552) (staged Δ=-741) → [exemplars/02_bool_first/task319_opset_bump_pad_uint8.md](exemplars/02_bool_first/task319_opset_bump_pad_uint8.md)
- `task030` (14,534) (staged Δ=-39,840) → `The_400_Tasks/Task_by_Task/task030/prototypes/build_task030_crop10.py`
- `task021` (13,142) (staged Δ=-5,520) → [exemplars/02_bool_first/task021_mul_mask_to_where.md](exemplars/02_bool_first/task021_mul_mask_to_where.md)

### Static-to-dynamic / rank reduction / projection-cache

- `task201` (84,895) (staged Δ=-47,118) → [exemplars/05_static_to_dynamic/task201_reduce_reorder.md](exemplars/05_static_to_dynamic/task201_reduce_reorder.md)
- `task377` (63,682) (staged Δ=-13,870) → `The_400_Tasks/Task_by_Task/task377/prototypes/build_task377_ringdyn.py`

### Gather/lookup compression

- `task159` (78,580) (staged Δ=-10,642) → `The_400_Tasks/Task_by_Task/task159/prototypes/build_task159_resize_scale.py`
- `task014` (45,690) (staged Δ=-27,432) → `The_400_Tasks/Task_by_Task/task014/prototypes/build_task014_crop15.py, build_task014_crop18.py`
- `task310` (20,112) (staged Δ=-61,812) → `The_400_Tasks/Task_by_Task/task310/prototypes/build_task310_crop10.py, build_task310_crop8.py`
- `task167` (795) (staged Δ=-22) → [exemplars/08_gather_lookup/task167_opset9_attr_shrink.md](exemplars/08_gather_lookup/task167_opset9_attr_shrink.md)
- `task056` (538) (staged Δ=-22) → see staged `task056_opset9_attr_shrink_t_80d265c9_on_5770_74/manifest.json`
- `task103` (256) (staged Δ=-22) → see staged `task103_opset9_attr_shrink_t_bf9257cc_on_5770_74/manifest.json`

### Untagged (60 tasks — no transferable technique attribution yet)

- `task018` (114,578) — Translation | Shifting | Object Detection | Gravit
- `task088` (101,860) — Cropping | Object Detection
- `task085` (74,258) — Color Mapping | Object Detection
- `task209` (67,337) — Cropping | Object Detection
- `task185` (35,303) — Cropping | Object Detection
- `task079` (24,409) — Cropping | Object Detection
- `task218` (23,368) — Cropping | Object Detection
- `task189` (17,303) — Cropping | Object Detection
- `task134` (17,059) — Cropping | Object Detection
- `task078` (15,756) — Translation | Object Detection
- `task375` (15,172) — Color Mapping | Object Detection
- `task344` (15,111) — Color Mapping | Object Detection
- `task098` (14,635) — Color Mapping | Object Detection
- `task160` (13,991) — Color Mapping | Object Detection
- `task329` (13,990) — Color Mapping | Object Detection
- `task372` (13,670) — Cropping | Object Detection
- `task177` (13,566) — Cropping | Object Detection
- `task022` (12,627) — Cropping | Object Detection
- `task360` (11,386) — Cropping | Object Detection
- `task254` (9,274) — Color Mapping | Object Detection
- `task346` (9,178) — Cropping | Object Detection
- `task048` (8,831) — Cropping | Object Detection
- `task321` (8,327) — Cropping | Object Detection
- `task010` (8,242) — Object Detection | Sorting | Color Mapping
- `task039` (7,832) — Cropping | Object Detection
- `task271` (7,145) — Cropping | Object Detection
- `task100` (7,076) — Cropping | Object Detection
- `task259` (6,954) — Cropping | Object Detection
- `task188` (6,924) — Cropping | Object Detection
- `task067` (6,200) — Cropping | Object Detection
- ... (30 more untagged tasks)

---

## Color_and_Logical (45 tasks)

### Bool-first family (input/Concat/Cast/Pad bool keeping)

- `task293` (37,940) → [exemplars/02_bool_first/task293_andfold.md](exemplars/02_bool_first/task293_andfold.md)
- `task118` (20,762) (staged Δ=-30,260) → [exemplars/08_gather_lookup/task118_bool_scatter.md](exemplars/08_gather_lookup/task118_bool_scatter.md)

### Crop-to-active-grid + F16 placement

- `task366` (289,181) (staged Δ=-131,592) → `The_400_Tasks/Task_by_Task/task366/prototypes/build_task366_refactor.py`
- `task077` (142,762) (staged Δ=-123,248) → `The_400_Tasks/Task_by_Task/task077/prototypes/build_task077_crop20x21.py, build_task077_crop21.py`
- `task330` (91,750) (staged Δ=-33,988) → `The_400_Tasks/Task_by_Task/task330/prototypes/build_task330_crop10.py`
- `task364` (89,097) (staged Δ=-22,532) → [exemplars/03_crop_and_f16/task364_crop22.md](exemplars/03_crop_and_f16/task364_crop22.md)
- `task313` (74,959) (staged Δ=-27,370) → `The_400_Tasks/Task_by_Task/task313/prototypes/build_task313_crop20.py, build_task313_crop20_from_work.py`
- `task070` (64,581) → `The_400_Tasks/Task_by_Task/task070/prototypes/build_task070_crop17.py`
- `task169` (58,688) → `The_400_Tasks/Task_by_Task/task169/prototypes/build_task169_crop10.py`
- `task374` (34,392) (staged Δ=-11,988) → [exemplars/07_conv_reformulation/task374_conv_run_combined.md](exemplars/07_conv_reformulation/task374_conv_run_combined.md)
- `task118` (20,762) (staged Δ=-30,260) → [exemplars/08_gather_lookup/task118_bool_scatter.md](exemplars/08_gather_lookup/task118_bool_scatter.md)
- `task040` (16,648) (staged Δ=-8,586) → `The_400_Tasks/Task_by_Task/task040/prototypes/build_task040_vecmask.py`
- `task354` (14,135) (staged Δ=-2,820) → see staged `task354_castafterslice_t_51436551_on_5835_30/manifest.json`
- `task156` (12,991) (staged Δ=-62,508) → `The_400_Tasks/Task_by_Task/task156/prototypes/build_task156_crop10.py, build_task156_crop10_from_work.py`
- `task125` (12,905) (staged Δ=-10,800) → see staged `task125_castafterslice_t_1c6d0a5f_on_5835_30/manifest.json`
- `task023` (12,260) (staged Δ=-3,762) → `The_400_Tasks/Task_by_Task/task023/prototypes/build_task023_crop9x11.py`
- `task312` (12,211) (staged Δ=-528) → see staged `task312_castafterslice_t_e6937859_on_5835_30/manifest.json`
- `task283` (11,836) (staged Δ=-85,268) → `The_400_Tasks/Task_by_Task/task283/prototypes/build_task283_crop10.py, build_task283_scratch.py`
- `task314` (8,594) (staged Δ=-3,296) → see staged `task314_castallslice_t_796f01d5_on_5835_30/manifest.json`
- `task332` (8,590) (staged Δ=-2,090) → see staged `task332_castafterslice_t_a4d650c1_on_5835_30/manifest.json`
- `task292` (6,301) (staged Δ=-1,440) → see staged `task292_castafterslice_t_225685e6_on_5835_30/manifest.json`
- `task147` (1,805) (staged Δ=-864) → see staged `task147_castallslice_t_882e95cb_on_5835_30/manifest.json`
- `task272` (1,317) (staged Δ=-650) → see staged `task272_castallslice_t_22d03aa2_on_5835_30/manifest.json`
- `task229` (1,227) (staged Δ=-478) → `The_400_Tasks/Task_by_Task/task229/prototypes/build_task229_crop3x3.py`
- `task373` (642) (staged Δ=-320) → `The_400_Tasks/Task_by_Task/task373/prototypes/build_task373_colfirst.py, build_task373_crop2x6.py`

### Algebraic identities (Equal-fold, Add-chain, scalar)

- `task366` (289,181) (staged Δ=-131,592) → `The_400_Tasks/Task_by_Task/task366/prototypes/build_task366_refactor.py`
- `task368` (50,484) (staged Δ=-19,184) → [exemplars/02_bool_first/task368_shared_cast_equal_fold.md](exemplars/02_bool_first/task368_shared_cast_equal_fold.md)

### Conv reformulation

- `task374` (34,392) (staged Δ=-11,988) → [exemplars/07_conv_reformulation/task374_conv_run_combined.md](exemplars/07_conv_reformulation/task374_conv_run_combined.md)

### Gather/lookup compression

- `task203` (61,259) (staged Δ=-70,233) → `The_400_Tasks/Task_by_Task/task203/prototypes/build_task203_skipfind.py`
- `task118` (20,762) (staged Δ=-30,260) → [exemplars/08_gather_lookup/task118_bool_scatter.md](exemplars/08_gather_lookup/task118_bool_scatter.md)
- `task129` (451) (staged Δ=-4) → [exemplars/08_gather_lookup/task129_tile_to_edge_pad.md](exemplars/08_gather_lookup/task129_tile_to_edge_pad.md)

### Untagged (18 tasks — no transferable technique attribution yet)

- `task182` (197,257) — Color Mapping
- `task158` (182,776) — Color Mapping
- `task054` (178,511) — Color Mapping
- `task324` (160,289) — Color Mapping
- `task370` (157,750) — Color Mapping
- `task064` (131,805) — Color Mapping
- `task279` (50,462) — Color Mapping
- `task277` (46,539) — Color Mapping
- `task196` (42,589) — Color Mapping
- `task252` (39,858) — Color Mapping
- `task094` (39,015) — Color Mapping
- `task035` (35,484) — Color Mapping
- `task294` (1,855) — Color Mapping
- `task120` (910) — Color Mapping
- `task016` (10) — Color Mapping
- `task276` (10) — Color Mapping
- `task309` (10) — Color Mapping
- `task337` (10) — Color Mapping

---

## Spatial_and_Geometric (41 tasks)

### Whole-graph rewrites

- `task376` (24,626) (staged Δ=-18,210) → [exemplars/01_whole_graph_rewrites/task376_input_to_bool.md](exemplars/01_whole_graph_rewrites/task376_input_to_bool.md)

### Bool-first family (input/Concat/Cast/Pad bool keeping)

- `task239` (114,143) (staged Δ=-37,678) → see staged `task239_f16_t_9085a974_on_5835_30/manifest.json`
- `task376` (24,626) (staged Δ=-18,210) → [exemplars/01_whole_graph_rewrites/task376_input_to_bool.md](exemplars/01_whole_graph_rewrites/task376_input_to_bool.md)
- `task327` (5,997) (staged Δ=-1,440) → [exemplars/04_algebraic_identities/task327_add_chain_to_sum.md](exemplars/04_algebraic_identities/task327_add_chain_to_sum.md)

### Crop-to-active-grid + F16 placement

- `task239` (114,143) (staged Δ=-37,678) → see staged `task239_f16_t_9085a974_on_5835_30/manifest.json`
- `task275` (89,826) (staged Δ=-10,300) → `The_400_Tasks/Task_by_Task/task275/prototypes/build_task275_crop16.py`
- `task398` (39,875) (staged Δ=-19,995) → `The_400_Tasks/Task_by_Task/task398/prototypes/build_task398_f16_int32.py`
- `task376` (24,626) (staged Δ=-18,210) → [exemplars/01_whole_graph_rewrites/task376_input_to_bool.md](exemplars/01_whole_graph_rewrites/task376_input_to_bool.md)
- `task124` (15,387) → see staged `task124_f16onehot_t_e52dae1f_on_5806_87/manifest.json`
- `task316` (14,449) → `The_400_Tasks/Task_by_Task/task316/prototypes/build_task316_crop10.py, build_task316_crop10_from_work.py`
- `task327` (5,997) (staged Δ=-1,440) → [exemplars/04_algebraic_identities/task327_add_chain_to_sum.md](exemplars/04_algebraic_identities/task327_add_chain_to_sum.md)
- `task249` (5,694) → `The_400_Tasks/Task_by_Task/task249/prototypes/build_task249_crop16.py, build_task249_crop5x10.py`
- `task114` (5,219) (staged Δ=-8,588) → `The_400_Tasks/Task_by_Task/task114/prototypes/build_task114_crop5.py`
- `task155` (4,675) → `The_400_Tasks/Task_by_Task/task155/prototypes/build_task155_crop7.py, build_task155_crop8h.py`
- `task150` (4,674) → `The_400_Tasks/Task_by_Task/task150/prototypes/build_task150_crop10w.py, build_task150_crop7.py`

### Algebraic identities (Equal-fold, Add-chain, scalar)

- `task289` (21,139) → `The_400_Tasks/Task_by_Task/task289/prototypes/build_task289_equalfold.py`

### Static-to-dynamic / rank reduction / projection-cache

- `task269` (31,014) (staged Δ=-25,278) → see staged `task269_dyngridsample_t_HUGE_RSUM_e_on_5806_87/manifest.json`

### Gather/lookup compression

- `task376` (24,626) (staged Δ=-18,210) → [exemplars/01_whole_graph_rewrites/task376_input_to_bool.md](exemplars/01_whole_graph_rewrites/task376_input_to_bool.md)

### Untagged (28 tasks — no transferable technique attribution yet)

- `task107` (85,131) — Tiling
- `task004` (62,082) — Translation
- `task295` (22,748) — Tiling
- `task019` (21,779) — Tiling | Cropping | Object Detection | Color Mappi
- `task304` (11,465) — Tiling
- `task315` (10,946) — Tiling
- `task104` (6,302) — Tiling
- `task003` (4,694) — Tiling | Color Mapping
- `task221` (4,420) — Tiling
- `task001` (4,339) — Tiling
- `task123` (4,314) — Tiling
- `task083` (1,800) — Tiling
- `task108` (1,800) — Tiling
- `task223` (1,800) — Tiling
- `task307` (1,800) — Tiling
- `task142` (1,520) — Tiling
- `task152` (1,520) — Tiling
- `task194` (1,520) — Tiling
- `task211` (1,520) — Tiling
- `task087` (368) — Reflection
- `task140` (368) — Reflection
- `task116` (30) — Tiling
- `task164` (30) — Tiling
- `task172` (30) — Tiling
- `task210` (30) — Tiling
- `task311` (30) — Tiling
- `task179` (0) — Reflection
- `task241` (0) — Reflection

---
