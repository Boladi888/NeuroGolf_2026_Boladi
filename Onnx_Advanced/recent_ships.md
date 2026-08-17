# recent_ships.md — dated ship lessons + battle prose (the per-entry record behind the router)

> Moved here from `task_techniques_index.md` 2026-06-10 (refactor pass 4 — completing the Phase-2 move
> the VERIFY P2 note tracked). The authoritative submit-by-submit ledger is `Logs/submissions_ledger.csv`;
> batch trackers live in `The_400_Tasks/current_strategy.md`. This file: WHAT shipped + WHICH technique,
> newest first — read when working a task similar to a recent ship.

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
- `task328` 47,384 -> 8,367, **+1.73 LB confirmed** — 3-operand Einsum corner reads (techniques #46, −900) + Mod parity (task313 precedent, −190); plane-conservation-law floor ≈7.2k (cost_floor_reference §method); byte-equal.
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

