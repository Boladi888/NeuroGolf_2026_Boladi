# CHANGELOG (archived) — Onnx_Advanced, 2026-06-07 through 2026-06-08

Rotated out of the live CHANGELOG on 2026-06-22 to hold it under the 60 KB cap (charter / INDEX size budget). Newest-first within this slice. Continues the pre-2026-06-04 rotation in `CHANGELOG_pre_2026-06-04.md`. Historical-log warning applies: use core.md/techniques.md/ort_reality.md for current truth, not these entries.

---

### Batch-16 wave (boladi, 2026-06-07 evening): COMPLETE 20/20 +24.15, LB 6683.49 [+#89/#90/#80-inverse, §FAMILY-LABEL trap, block-upsampler]

- Ships (all byte-equal + profile==static unless noted): 207 u8-bitand 2387->200 (+2.48, from-scratch), 127 final-conv 4046->900 memory-ZERO (+1.50), 032 countmask 3405->915 (+1.31), 291 bbox1map 4449->1756 (+0.93), 240 einsum-fold 3928->1706 (+0.84), 123 leangather 4314->1993 (+0.78), 065 v7 3803-actual->1999 (+0.64, CREDIT-PROBE), 359 threshold-fold 4510->4158 (+0.08).
- **#87 FREE-OUTPUT SEPARABILITY TRAP** (task127): rank-1-separable final conv is SCORE-DEAD — splitting a free-output conv materializes a charged intermediate (task127: [1,5,30,30]=18.3k vs 900 dense). Factorize a free-output conv only if the intermediate < the param saving.
- **#81 SMALL-GRID INVERSION** (task032): when the output grid is tiny (<~9.5x9.5), a decoded [1,10,h,w] one-hot + free Pad beats the fixed ~900 B Equal-tail canvas (the crossover flips).
- **#86 BBOX single-marginal** (task291): one rowsum marginal gives both extents (ArgMax=height, ReduceMax=width) when walls are full-width; one-hot-Pad corner emit replaces a 900-param plane (933->22). Einsum-moment bbox-area is non-exact (falsified 7/8000).
- **§39 CREDIT-PROBE CONFIRMED** (task065): profile==credit law holds on a +51% dynamic gap — credited EXACTLY at profiled 1999. Law now spans static->mild->moderate->credit-probe; gate-5 profile fully predicts credit on dynamics.
- **Floor doctrine #4 GENERATOR-PROVEN ABSOLUTE** (task359): reading the ARC-GEN generator can upgrade a formulation-relative floor to absolute (denoising-not-symmetry; the family LABEL is a misnomer, the generator is truth; falsify cheaper formulations on >=3000 fresh pairs).
- **Anti-quit OFF-condition** (task291/123): the self-ping window is insurance against premature quitting, not a clock to exhaust — judge PROOF completeness (complete kill-table close is done; incomplete keeps going).
- task_state/live_costs/ledger/team-export reconciled per-ship; xlsx auto-synced; Sigma-identity re-tightened (065 mis-record corrected). Research board CLEARED.

### L2 filed — research board CLEAR (2026-06-07 evening)

- `Research/Done/l2_ort_einsum_internals.md` — L2 returned decision-grade (v1.24.4 source-cited): **in-kernel einsum temporaries are temp-space allocations, scorer-INVISIBLE** (the mechanistic why of einsum-final pricing); operand order = the contraction path (left-deep, unoptimized — kill big non-output labels early); no operand cap; split-for-10x-only rule; diagonal + output-permutation slow paths; FinalizeOutput #18654 edge. → techniques **#89**; card retired. **Every research card A→L is now delivered and integrated.**

### #88 + batch-15 close (2026-06-07 evening)

- techniques **#88 u8 flood kit** (task048 drop): row-byte bit-packed flood — #47's crossover FLIPS at grid width <=8 (8 B/lane, 72 B per 8-conn round, pack-free inside the extraction conv); Min-masked u8 MaxPool flood (mask doubles as value-coded class plane); outer-product one-hot seed (99 B vs OneHot 256); ArgMax value-priority; generator-invariant connectivity relaxation instance (8conn for 4conn when the generator resamples ties away); disjoint-boxes first/last theorem.
- traps §CLASS-CHECK roster: **048 = ambiguous-class member #8** (k6 visible-floor bet at 99.374% truth HELD; LIVE itself misses 84/200k — evidence came from the sibling rung's dual-truth attribution).
- .update queue fully drained (6/6 wave drops integrated). Batch-15 CLOSED 10/10; LB 6659.34.

### L1 research deliverable filed (2026-06-07)

- `Research/Done/l1_exact_cp_decomposition.md` — L1 returned decision-grade (5-class source tags): unrestricted-real CP rank infeasibility is ∃R-hard (no public tooling); actionable regime = bounded-integer/dyadic MIQCP with symmetry breaking + structured dictionary; flattening-rank lower bounds as instant kill-tests; permutation-support banks provably OR-incompressible (theory behind 357's discriminator). → techniques **#87**; card retired to `Research/Done/`. **L2 (ORT Einsum internals) remains the only open card.**

### Library wave #81-86 + 3 traps (2026-06-07 afternoon, batch-15 + sweep close-outs)

- **#81 PAD-FLOOR CROSSOVER LAW** (sweep worker-7): Equal-tail fixed ~900 B; dtype table for when prepad-kill pays; Where-as-output discriminator (substitution vs generated). Killed 8 conversions pre-build.
- **#82 singleton-object plane rebuild kit**: einsum scalar reads + Equal-ramp planes + outer-MIN D2S stamp + read-LIVE-inits-verbatim hygiene.
- **#83 NIBBLE-PACKED transport plane** (task209): 2 cells/byte color-index planes; packed>0 law; max-of-packed boolean-only + clip-before-pair disciplines; i8-Where-via-u8 routing; Min(vec,1) extents; einsum rank-drops.
- **#84 placer-Conv structural toolkit** (task180): IG one-hot-sum identity (replaces unsafe positive bias), groups-halving under certified purity, dilated quadrant reads. Params-only class 3rd member (180@210 memory-ZERO).
- **#85 free-output QLC toolkit** (task265): int32-bias binarizer, in-conv morphology taps, the BIAS-LEAK LAW, crop-instead-of-mask, pack-entry+BitwiseAnd residue skeleton.
- **#86 occupancy-pack + max-bound existence** (task346): (sum,occ) in one kernel at 128-spacing; aggregation-as-detector Equal(ReduceMax) existence test.
- traps: **§FREE-OUTPUT ACTIVE-MASK** (353 sentinel law + 2 immune classes), **§CLIP-DUP PHANTOM** (plane-shrink changes Clip-overflow semantics), **§SCORER DUMMY-GRID CRASH** (clamp semantic scalars once); §BOUNDARY-LEAK refined (structurally-zero last-row/col exception).
- ort_reality K2: Where(bool,i8,i8) NOT_IMPLEMENTED; Min(u8)@13; i8 arith kernel set.
- scoring_internals A.2 STALE-FLAGGED (dtype-aware params = retired counter; canonical scorer counts elements dtype-blind — task180 30-min burn prompted the flag).
- 5 worker drops integrated -> .update/_integrated (048 held, task open). Shipped this wave: 20 tasks +26.04, all exact (ledger).

### Research/ folder flattened to Todo/ + Done/ (2026-06-07)

- `Research/` now holds exactly two folders: **`Todo/`** (open card prompts: L1, L2 + the card-contract README) and **`Done/`** (all completed material: k1-k5 deliverables, K1-K5 retired cards, 5 AUDITs + VERIFY, b3/k_arc2/cull notes — 19 files, moved via git mv).
- Dissolved: `Research/notes/` and `Research/Todo/Done/` (one level too deep — the confusion this fixes).
- References rewired: techniques #66/#75-77, ort_reality K2 header, CHANGELOG k-paths, INDEX folder rows, both READMEs (root README = the 2-folder flow; Todo/README = corrected ledger with `../Done/` + `_archive/` pointers).
- Stale Phase-2-era provenance pointers fixed while in there: core.md spread-knowledge paths -> `Research_Archive/notes/`; F4 provenance -> `_archive/TodoDone_card_F4_*`; A2/A3 provenance now states the card bodies were deleted in the 2026-05-27 cull (eb74a378, git history only).


## 2026-06-06 task124 floor campaign addenda (15,387 → 2,149 staged, byte-equal)

- `techniques.md` — **#46/§42 addendum: MULTI-OPERAND EINSUM FUSION** — einsum kernel-internal scratch is not a graph tensor and is NEVER costed; einsum→einsum chains fuse into one 3-/4-operand einsum deleting the intermediate from value_info (task124 v9-v11: 3-op marginal fusion −48 B, Gram fusion −100 B, L-factored quadratic form ||L·P||² storing L instead of K=LᵀL −25p; each landed EXACTLY at static prediction). Conv-weight reuse inside einsum via size-1 subscripts (`wdxy` matches [1,10,1,1]).
- `techniques.md` — **§41 addendum: negative BEGIN Conv pads** crop from top/left too (ORT-probe-verified alignment + full_check); one shared 1×1 weight serves N crop views at zero param cost (task124 v7, superseded by v8 which needs no crops — fully gated evidence rung).
- `Logs/lessons.md` — task124 entry: hypothesis-select graph playbook (scalar-einsum equality tests, N−1 elimination, regime-conditional moment bits, MAPS-init gather vs candidate materialization, generator-proven OOB sentinel, hypothesis-invariant rows, 4-vs-5-row verification window proof).

## 2026-06-07 morning-wave docs pass (6 ships 6602.62 → 6618.40; 8 worker notes integrated)

- `techniques.md` — **#78** FREE-OUTPUT catalog v2 (placer-Conv pads-as-attr — upgrades the 97-task spread map; QLinearConv-as-output + its y_zp/8-input gotchas; BitwiseAnd i32 class-LUT ≤31 classes; Where-Y-input with the ask-#65-first caveat), **#79** reconstruction kit (Equal(iota,sum-code) skips crops; parameter-LUT three-way crossover law; one-hot-slice-as-mask; negative-Gather legality), **#80** DepthToSpace(CRD) block-stamp (extends #44); #68 discriminator (contraction folds iff (r,c)-coupling separable — 357's 4-route kill template).
- `traps.md` — **§FREE-OUTPUT BOUNDARY LEAK** (positive off-center taps fire phantoms outside the grid; passed 204k corpus, failed border stress — border-stress battery now a required gate for free-output convs).
- `ort_reality.md` — wave-2: ConvInteger grader-validated; QLinearConv y_zp u8-only + 8-inputs-mandatory; opset-14 u8/i8 probe extensions; negative Gather; i32 output decode.
- `cascade_strategy.md` — task260 trigger-block probe NULL (−90% memory, zero excess); cascade theory historical.
- 8 `.update/` notes integrated → `_integrated/`.

## 2026-06-07 L-series research cards authored (post-einsum-final gaps)

- `Research/Todo/L1` — exact CP/rank decomposition of small {0,1}/integer selector banks (the cancellation question = 312's open slots 10-18; Boolean rank for BitwiseAnd banks; exact-search tooling). HIGH.
- `Research/Todo/L2` — ORT Einsum internals source-read (contraction order, operand caps, in-kernel allocation invisibility, latency model + ordering rule). HIGH.

## 2026-06-07 K3/K4/K5 deliverables filed — ALL K-cards complete (5/5 decision-grade)

- `Research/Done/k5_accumulation_exactness.md` — **REFUTED techniques:1677's F16-Conv claim at source level** (NO native F16 Conv CPU kernel); reconciled with ship-empirics via cast-insertion→F32 compute (finals-only ≤2048 envelope); REAL hazard found: native `Gemm<MLFloat16>` rounds EVERY partial sum; QLinearConv u8 counter ≤255 source-confirmed. → techniques **#77** + line-1677 corrected + WORKER_PROTOCOL digest.
- `Research/Done/k3_unexploited_op_audit.md` — 72 residual ops audited; finds: variadic `Sum` (top — independently adopted by 256/078 workers same wave), RoiAlign static-ROI, LpPool(p=1), Hardmax; CenterCropPad CONFIRMED kernel-less (op_cost_patterns flag upgraded); NMS validator-fatal. → techniques **#75**.
- `Research/Done/k4_compression_native_minimal_nets.md` — verdict: nothing beats hand construction; one transfer = offline ESPRESSO/ABC→1×1-Conv logic-synthesis pass. → techniques **#76**.
- Duplicate K5 deliverable (sha-identical) deleted; cards retired to `Research/Done/`.

## 2026-06-07 wave docs pass (batches 11+12 BOTH complete; LB 6580.99 → 6602.62)

- `techniques.md` — added **#67** FREE-OUTPUT catalog (Equal-tail/BitwiseAnd-intersection/broadcast-Add/BOOL tails + the 97-task Pad→output spread map), **#68** formulation-relative floors (enumerate final-op families first), **#69** sign-arithmetic kit, **#70** einsum place-value/dominance packing (digit-pack + combo head + exponential first-occurrence order), **#71** moment-correlation variance-zero identity (→task206 family), **#72** QLinearConv rounding-boundary gadgets, **#73** locate-then-detect + separable ones-banks + marginal profiles, **#74** 3-operand einsum VERIFIED + INT8/U8 opset-14 lanes; §39 mild-class band extended to **1.0×** (178-v30 honest-dynamic); #62b grader facts (BitwiseAnd + QL@19 validated).
- `traps.md` — **§GENERATOR-INVARIANT vs OFFICIALS-QUIRK doctrine** (077: officials can violate every generated invariant; decode generator first, gate officials too).
- `ort_reality.md` — Mod const-divisor exactness; BitwiseAnd/QL@19/C13@12 grader-validation row.
- `.update/` — 7 promotion notes integrated → `_integrated/`.

## 2026-06-06 K1/K2 deliverables filed (research cards round-trip in one day)

- `Research/Done/k1_superoptimization_survey.md` — K1 returned decision-grade: SKIP all superoptimizer ports (TASO/TENSAT/egg = XL build, rules float-aggressive); exact-rule shortlist + Pad→Conv 40,960B worked example extracted to `techniques.md` #66.
- `Research/Done/k2_ort_exact_semantics.md` — K2 returned decision-grade (v1.24.4 source-read): TopK(sorted=0) order UNSTABLE; Where-bool kernel absent (source-confirmed); Mod/Div/Pow native-C++ integer semantics; Resize attr-pinning = #1 silent-divergence risk; NaN propagation in Min/Max. Extracted to `ort_reality.md` K2 section + `traps.md` §K2 ACT-NOW PAIR + WORKER_PROTOCOL digest line.
- Cards K1/K2 retired to `Research/Done/`. K3/K4/K5 remain open (K5 = the load-bearing F16-accumulation claim).

## 2026-06-06 docs pass (batch-11 first 5 ships; LB 6556.39 → 6568.57)

- `techniques.md` — added **#63** (Einsum×OneHot selected-channel marginals — task079's sub-1500 enabler), **#64** (Cast-BOOL occupancy→ArgMax bbox extents + ORT-rejects-`ArgMax(BOOL)` gotcha — task228 ×2 ships), **#65** (assembly-choice laws: Equal-vs-Where decided by passthrough; mild-dynamic pays only ≥600B-class statics — task166 floor campaign), **§39 CRASH ADDENDUM** (task061 v11/v12: dynamic Pad/Mod buffer-reuse crashes on REAL inputs). (task055 worker's same-day promotions renumbered #58/#58b→#62/#62b — dup vs the 06-05 P4 renumbers.)
- `traps.md` — **§CLASS-CHECK roster** += 018/187/157/**079** filtered-clean; new **§CLASS-CHECK DEPTH RECORD** (task079 v36: 98.56% corpus + visible-clean scored FULL hidden credit — deepest absorption ever, prior record 99.2%; corpus % is not a hidden predictor in either direction); **§VISIBLE-FAIL proofs #4–#5** (task086 probes landed exact fail-values); **§GRADER-GAP CRASH CLASS** (dynamic graphs can crash outright, not just misprice).
- `Research/Todo/K1–K5` — five research cards authored against verified gaps (superoptimization under memory+params, ORT exact-semantics compendium, unexploited-op audit 11–18, compression-native minimal nets, F16 accumulation exactness — the load-bearing techniques.md:1677 claim).

## 2026-06-05 morning (deferred-tail close-out — refactor program COMPLETE except whale prose)

- traps.md body-split 82→67 KB (long trap bodies → appendix, ≤5-line verdicts + pointers in main; ALL anchors count-verified; §8b-f/14z/OneHot kept whole).
- Whale compaction attempted with jewel-protection rails: rails correctly REFUSED (jewel terms matched ~every section) — per-file hand pass scoped as its own session; STATUS lines already fix buried-state globally.
- scoring sub-9k-MAC residue neutralized (VERIFY P10).
- Program totals: audits #1+#2+verifier — every mechanical item DONE; 3 tools cured of hardcoded-worktree paths; corpus 5.7→4.8 MB; every-task set ~140 KB single-spec; Logs −20 MB; zero broken anchors for in-flight workers throughout.

## 2026-06-05 Phases 1-3 (context-bloat refactor — structural 80% done; deep rewrites deferred to between-wave windows)

- **Phase 1**: traps.md 141->82 KB (dated diary -> `traps_appendix.md` 48 KB grep-only; ship-decision block -> `current_strategy.md` §ship-decision; doctrine blocks kept under a stable header — rebuild done deterministically from git after a splitter bug, all 12 anchors count-verified). SPEC 48->17 KB (self-disowned draft archived). cost_floor: 9 ship-blocks -> 32-row compact table. cascade diary -> `Logs/cascade_campaign_log.md`. task_techniques_index header -> `recent_ships.md`. INDEX: every-task set + size budgets.
- **Phase 2**: op shelf -> `op_cost_patterns.md` (salvage-before-archive; einsum.md = stub, in-flight-cited). Ort_Advanced -> **`ort_reality.md`** (matrix+quirks+errors+opset merged; all 4 old paths stubbed). alt-taxonomy archived (won 3/23); empirical-coverage spine merged into primary. QUICKSTART->INDEX; README->stub. Archived: 4 extracted deliverables, 7 Research roots, Todo/Done bodies, 16 .update DONEs, notes per audit. ort_profiling + Onnx_basics -> scoring_internals.
- **Phase 3**: techniques verdict-index front-matter (the 12 verdicts); 54-row §0 ship-table evicted; §39-stack supersession marks; scoring_internals MAC-era sections archived (~12 KB).
- **Numbers**: worker-visible 1.7 MB -> **1,223 KB**; every-task set -> **156 KB single spec** (was 127-290 KB across 3 conflicting specs); `_archive/` = 930 KB greppable provenance.
- **Deferred (content-judgment work, ~audit Phase-3 deep)**: techniques narrative compression 208->100, traps table-ization 82->10+appendix, taxonomy rewrite 46->10, tiny_architectures/mlp/cellular/symmetry deep trims, exemplar dedups. Generators NOT re-run (would clobber curated index entries — flagged for the next index rebuild).
- In-flight prompt safety: all cited paths/anchors verified intact (einsum.md + 4 Ort paths live as stubs; every §-anchor count-checked).

## 2026-06-05 Phase-0 (audit hotfixes — 6 items, references verified unbroken, zero info loss)

- `scoring_internals.md` — correction banner + ALL 7 "macs+memory+params" formula sites fixed to `memory + params` (MACs dead since 2026-05-04; bug #1).
- `QUICKSTART.md` — OneHot line now matches traps §2 (INT32+INT64 both fine, depth dtype must MATCH; bug #2).
- `INDEX.md` + `by_symptom.md` — taxonomy labels corrected by ground-truth count: primary = 18-family ConceptARC, alt = 15-family output-shape (bug #3).
- `techniques.md` — §-collisions resolved with fresh numbers + breadcrumbs: output-overlay→#48, strip-runfill→#49, crossing-parity→#50, second Merged-Conv→#51 (NOT deleted — audit's "exact dup of #24" claim REFUTED at 5% overlap), three "# N."→#52-54. Canonical #40=plane-breaker, #41=2D-imos+negpad kept (bug #4). All prompt-cited anchors (#44-47, §41-VERDICT, §40, §37-ADDENDUM, §42, §39-MECHANISM) verified unique.
- Jewel promotion — techniques **#55** (task013 trio: aspect-branch / clamped-Mod periodic fill / broadcast-And no-canvas assembly), **#56** (hash-MLP fingerprint — decode the generator), **#57** (boolean-safe reduction scaffold); traps §21 merged into §26 (true dup, verified); 4 by_symptom routes added; orphan marked INTEGRATED.
- PUA-wrapped citation tokens stripped: **894 tokens / 20 files / 48 KB freed**, zero residue, prose spot-verified.
- `agent_feedback_rules.md` §8b + `.update/README.md` — the 9-rule anti-re-bloat charter is now binding.

## 2026-06-04 night-11 (Research J1+J2 filed — Einsum diagonal reads + row-packed flood verdict; program 7-for-7)

- `Research/einsum_equation_catalog.md` (J2) — repeated-subscript diagonal READS real (spec + ORT source; `bchh->bch` 1,200 B / trace 40 B vs EyeLike 40,800 B = 34×); 9-equation adopt-list DIRECT from the free input; writes rejected (EyeLike/Trilu keep that); ORT Einsum = f32/f64/i32/i64 only; integer einsum pointless (2^24 headroom). → techniques #46.
- `Research/row_packed_uint32_morphology.md` (J1) — honest pricing: pack 3,840 + round 1,080 + unpack 4,620; crossover K=12 vs unpacked 1,800/round (the 120 B slogan false); 4-NEIGHBOR only — captain priced 8-neighbor at ≈2,040/round = LOSES; MatMul-u32 CPU-legal (dtype matrix row). → techniques #47.
- `Ort_Advanced/ort_op_dtype_matrix.md` — MatMul integer rows + Einsum dtype row.
- Research program closes **7/7 decision-grade** (G1-G3, H1-H2, J1-J2); queue empty.

## 2026-06-04 night-10 (BATCH-7 CLOSES 10/10 +21.4; four-winner round + composition 6536.07; F16-Conv validated; high 6538.23)

- `techniques.md` — §39 MECHANISM CONFIRMED (grader source: max(static, runtime-on-REAL-inputs) per tensor; static ⇒ local==Kaggle ⇒ no profitable submit-probes; symbolic → None) + F16-CONV VERDICT (first ship grader-validated exact — halves F32 conv planes project-wide) + info-flow floor style (task080) + 054's ScatterElements V-lines.
- `traps.md` — §GRADER-GAP ROOT CAUSE (the one-line mechanism) + **§CLASS-CHECK protocol** (LIVE-truth run classifies ambiguous vs deterministic; roster: filtered-clean 209/064, deterministic 025/145).
- `cost_floor_reference.md` — 054 architecture floor (grader-source grade), 080 info-flow floor, 064/025/359 floors, composition-linearity datapoint.
- `task_techniques_index.md` — 6 entries (054, 359, 080, 064, 025 + composition note).
- `Logs/lessons.md` — mechanism solved; lossy-summary-channel (359 recovery); absent-guard class.
- `current_strategy.md` — batch-7 completion record; all trackers current.

## 2026-06-04 night-9 (batch-8 run 293/374/094 — high 6531.84; the §39 CONVERGENCE upgrade)

- `techniques.md` — §39 GAP-LEDGER MAJOR UPGRADE (task094 trilogy: dynamic repricing converges to true-trace cost — 78×/42× fiction vs static EXACT winner; revised 3-rule doctrine) + cheap-byte notes (opset-11 axes-attrs; zero-initializer scalars from free-input cells).
- `traps.md` — §GRADER-GAP UPGRADE block (the convergence rule).
- `cost_floor_reference.md` — 293/374/094 floors (do-NOT-cite-313 warning).
- `task_techniques_index.md` — 3 entries; `current_strategy.md` trackers; `task094.md` reconciliation + static-only follow-up.
- Ledger: 5 rows (293, 374, 094×3).

## 2026-06-04 night-8 (batch-7 tail sweep — 119/065 entries; §42 verdict; team-trade record)

- `techniques.md` — §42 VERDICT (free-input Einsum marginals grader-validated by task119 v5's exact landing).
- `cost_floor_reference.md` — 196/119/065 floors (incl. 119's sub-1,500 disproof + 065's 3rd extraction-markup datapoint).
- `task_techniques_index.md` — 119 + 065 entries.
- `current_strategy.md` — §team-trade dated record (merge math +206/+242, gifts 209+215, reconciliation prerequisites before any merge-in).
- `Logs/lessons.md` — the absent-guard bug class (zeros-input fooling presence guards; And(clr_pres, not_bg) fix pattern).
- (verified present: worker-authored traps §14z multi-output myth + techniques #42.)

## 2026-06-04 night-7 (Research H1+H2 filed — spatial-op adoptions + the packing LAW; card queue empty)

- `Research/spatial_data_movement_catalog.md` (H1) — adopt: Slice-negstep flips/rot180 (0 intermediates, ≤8 params), Transpose (0/0), Tile, Resize-nearest, Trilu; GridSample CPU source-confirmed but 2HW-param grid tax; ReverseSequence H-param niche; runtime-k roll has no ≤2-op winner. → techniques #44.
- `Research/bit_packing_calculus.md` (H2) — the packing LAW: α·m·(n−w) > pack + g·n·unpack; plane-neutral FALSE (α=0 ⇒ strictly worse — 328/338 floors stand as the α≈0 case); U8 n=8 break-even m≈4, ≥2× at m≈8; I32 carrier trap (BitShift unsigned-only); popcount routes (LUT 5P/8P < SWAR 10P < DivMod 22P); row-packed uint32 = carry-free rows ≤30 wide, bitboard Dumb7Fill transfers → candidate flood route. → techniques #45 + cost_floor refinement note.
- `Research/Todo/README.md` — H1/H2 retired; **all 5 cards (G1-G3, H1-H2) delivered decision-grade; queue empty**.

## 2026-06-04 night-6 (task338 CLOSED with exact-margin route audit; task252 RECORD 280 B; §41 confirmed)

- `techniques.md` — §40 SIZING LEMMA (LUT read wins only over ≥5-stage unbroken pointwise runs; 4 B/elem INT32 index plane vs 625 B U8 stage on 25×25) + §39 counter-datapoint (zero-memory params-only graphs credit honestly: t252@280 +4.958 exact, t215 precedent) + §41 VERDICT (negative-pad Conv grader-CONFIRMED, first datapoint, ship-class portfolio-wide).
- `Ort_Advanced/ort_op_dtype_matrix.md` — (worker-stamped) opset-18 Bitwise*/BitShift CPU rows with wrap-semantics probes + the packed-rows dead verdict for [25,25] per-cell rules.
- `cost_floor_reference.md` — task338 CLOSED (8,469; closing verdict: sub-5k needs ops ONNX lacks) + task252 record (280).
- `task_techniques_index.md` — 338-closed + 252-record entries.
- `Research/Todo/Card_H2` — injected the confirmed Bitwise CPU facts + the 338 packed-rows datapoint as [PROMPT] facts so the researcher focuses on the open calculus.
- `Logs/lessons.md` — the exact-margin close standard.

## 2026-06-04 night-5 (task398 grader-gap — the 37× datapoint; task196 ships; H1/H2 cards posted)

- `traps.md` — **§GRADER-GAP**: local `score_network_v2` fooled 37× by a scorer-dummy/boolpad stack (task398: 364–519 local, ≈13,300 credited family-wide); local cost = lower bound for exotic graphs; ship the boring-static sibling; ladder + fixed-base probes decode credited cost.
- `techniques.md` — §39-ESCALATION operational rules (when to distrust local cost; the too-good-vs-floor tell; the ladder pattern).
- `cost_floor_reference.md` — task398 grader-credit datapoint (do NOT cite 364 as a floor); task196 was swept in night-4 family.
- `task_techniques_index.md` — task196 (+2.14, copy-input Where-replace) and task398 entries.
- `Research/Todo/` — H1 (spatial data-movement catalog) + H2 (bit-packing calculus) cards posted under the upgraded authoring contract.

## 2026-06-04 night-4 (sweep: 364/049/187-scanline/363 shipped — high 6508.33; visible-miss axiom)

- `traps.md` — §VISIBLE-FAIL upgraded to AXIOM (3rd proof: task363 no_diamond passed synth 200k/200k yet zeroed on its one visible miss; stop re-testing) + §DUAL-TRUTH arithmetic shortcut (ab_divs == LIVE_misses − cand_misses ⇒ 0 real bugs; task187 scanline shipped exact on it).
- `techniques.md` — night field-validations: §19 runtime-weight Conv as a 361-translation search engine (task363); design-time plane REUSE (task049's Equal serving Einsum-selector AND And-mask); the 'bchw,c->bh' row-projection lever.
- `cost_floor_reference.md` — 364 (theorem-grade 14,162) / 049 (per-byte 748, campaign record) / 187 (21,282 interim) / 363 (3,796) floors.
- `task_techniques_index.md` — 5 entries incl. the pending 364 one.
- `Logs/lessons.md` — the axiom + the arithmetic dual-truth shortcut.

## 2026-06-04 night-3 (Research G3 filed — plane-breaker catalog; all G-cards complete)

- `Research/nary_fanin_catalog.md` — G3 deliverable: the reliable 2-input-law breakers (Gather-LUT on INT32 code; bit-pack-MatMul→Gather-LUT; variadic Min/Max/Sum; Mod; channel-stack Conv/MatMul reductions) vs the anti-recipes (generic live-tensor mux, Concat+Gather banks, OneHot selectors, dense ScatterND); ORT-source finding: ≥3-operand Einsum executes PAIRWISE internally — trust 1-2 operands, re-verify each ≥3-op equation with score_network_v2 (task328's win held empirically, 28× local==grader).
- `techniques.md` — §40 plane-breaker catalog extracting the above.
- `Ort_Advanced/ort_op_dtype_matrix.md` — GatherND CPU indices INT64-only (8 B/elem vs Gather's INT32 4 B); Mean float-only; Sum not a bool reducer.
- `Research/Todo/README.md` — G3 retired; **G-card queue empty** (G1+G2+G3 all delivered decision-grade).

## 2026-06-04 night-2 (Research G2 filed — flood-schedule verdict; card process codified)

- `Research/log_round_flood_reconstruction.md` — G2 deliverable: doubling/log dilation provably NON-exact for masked flood (U-corridor counterexample = task325's measured gap-2 failure); exact o(K) needs graph-closure state; the ONE local exact halving = 5×5 = 2×3×3 iff corpus-certified scale-2 frontwise geodesic convexity (D 5→3, 14→7, 25→13 where certified).
- `techniques.md` — §37-ADDENDUM extracting the verdict + the measurable certificate recipe.
- Worker prompts task187 (`Task Prompts/2/`), task367 (`2/`), task198 (`4/`) — log-schedule advice replaced with the certificate recipe.
- `Research/Todo/README.md` — G2 retired; card-authoring guidance codified from observed G1/G2 quality (5-class source tags, human-readable citations, blunt-verdict requirement, captain filing protocol).

## 2026-06-04 night (task364 ships — 6500 CROSSED at 6500.94; Research G1 filed)

- `Research/ort_dynamic_shape_profiling.md` — G1 deliverable filed (ORT-source-verified): profile trace records RUNTIME-observed shapes; ORT_DISABLE_ALL still shape-infers but trace wins; Gather output shape depends only on indices SHAPE (golden static construction); zeros-drift direction rules; 5 static recipes. Card retired to Todo/Done.
- `techniques.md` — §39-UPGRADE block extracting the above (the task014 −0.33 mechanism is now understood: Greater-mask zeros-understatement vs grader inputs); §43 negative-bias cliffs (authored by the task364 worker) shipped +1.18 exact alongside two named floor theorems (cue non-separability, pre-dilation zero-trade) in task364.md.
- `task_techniques_index.md` (pending next sweep): task364 14,162 entry.

## 2026-06-04 evening (batch-6 second half — 328/280 shipped, high 6499.76)

- `cost_floor_reference.md` — the plane-conservation-law floor METHOD (task328: price every legal op family against the decision core; escapes need >2-input fan-in — see Research card G3) + 328/280 floor datapoints.
- `Ort_Advanced/ort_op_dtype_matrix.md` — TopK UINT8 unsupported on ORT CPU (task280 measurement).
- `task_techniques_index.md` — 328/280 entries (docs-credited Einsum/Mod wins; task280 dead-route list).
- `Research/Todo/` — three new internet-research cards: G1 ORT-profiler-vs-dynamic-shapes (the task014 −0.33 gap), G2 log-round exact flood, G3 n-ary fan-in catalog; each standalone paste-able.

## 2026-06-04 PM-2 (worker falsification, task364 in progress)

- `techniques.md` — §33/§34 cost-falsification appended: U8 dominance flood (880 B/step, no negation/INT64) strictly dominates F16 MinPool labeling (1,760 B/step) + ScatterND-INT64 histograms for per-component labeling on small canvases; §33/§34 reserved for true-MinPool/opset-17-histogram needs.

## 2026-06-04 PM (batch-6 first-half sweep — 128/195/367/308/014 shipped, high 6496.06)

- `techniques.md` — §39 GatherND(batch_dims=1) sparse reads (task195: read ONLY the 9 cells the rule needs, 47,787→1,878, campaign-best +3.23) + the dynamic-shape grader-cost calibration warning (task014: local 4,380 → Kaggle-credited ≈6,093, first non-exact landing after 24 exact; treat local cost of dynslice constructions as a lower bound, prefer/also-ship the static sibling).
- `cost_floor_reference.md` — batch-6 first-half floors (128/195/367-interim/308/014 incl. the grader-gap note).
- `task_techniques_index.md` — 5 new 2026-06-04 PM entries.
- `Logs/lessons.md` — dynamic-shape lower-bound rule + the task187 stale-artifact check (mtime-vs-prompt-date + cost==live+byte-equal ⇒ it IS live).

## 2026-06-04 late (batch-5 COMPLETE 10/10 sweep — +24.08 batch, 20/20 exact calibration)

- `techniques.md` — §37 corpus-measured iteration floors (task325 K-minimum proof, task279 depth-histogram proof, measured shortcut-kills, generator-geometry contracts for cheaper CC steps); §38 marker-sentinel rewrite + sparse ScatterND canvas updates (task122 10.9×).
- `traps.md` — §DUAL-TRUTH divergence attribution (run truth on BOTH candidate and LIVE; task279's lone ab-divergence was a LIVE GT-bug the candidate fixed) + task145 counter-datapoint (99%-truth fails on deterministic-class; task209 exception is ambiguous-class only) + worker-ab-window under-sampling note.
- `cost_floor_reference.md` — batch-5 completion floors (020/240/131/325/122/279 + stray fold).
- `task_techniques_index.md` — 8 new 2026-06-04 submit-confirmed entries.
- Outside the library: `current_strategy.md` batch-5 completion record + meta-lessons #15 (dual-truth) and #16 (measured floor proofs); `Logs/lessons.md` 3 entries (quantified floors, dual-truth recipe, staging-protocol contract).

## 2026-06-04 (captain docs sweep — batch-5 ships 193/244/287/368, 13/13 exact calibration)

- `techniques.md` — §35 Conv weighted-sum color decode (replaces Slice+ArgMax, no INT64) + asymmetric-kernel free single-axis crop; §36 cumulative ≥k masks with deepest-rank-outermost Where nesting; field validations for §8b dilated Conv (task287, fixed 16×16) and sentinel-10 Pad before Equal-expand.
- `traps.md` — SHIP DECISION REFERENCE: §VISIBLE-FAIL = certain hidden 0 + probe-ladder exact-bound pattern (task193 8/8) + sibling discipline (byte-equal primary ⇒ skip dominated; non-byte-equal ⇒ one arch hedge).
- `cost_floor_reference.md` — shipped-floor datapoints 193 (5,281) / 244 (2,252) / 287 (2,948 memory floor) / 368 (6,294 interim); 13/13 exact 1.0× calibration incl. fail-probes.
- `task_techniques_index.md` — 2026-06-04 submit-confirmed section (193/244/287/368).
- `diagnostics_recipes.md` — `tools/ng_fill_hang_safe.py` (hang-safe resumable corpus filler; generators 076/324/018 hang on specific seeds, 157 has healthy 20–40s slow-tail seeds; corpus now ~396/400 full and filling).
- Outside the library: `The_400_Tasks/current_strategy.md` §captain-session-2026-06-04 (live-zip audit method recovered +2.39; ladder verdict; sibling discipline; decoded-rule prompt format); `Task Prompts/_Before writing prompt.md` corpus-fill guidance; `Logs/lessons.md` captain pointer.


---

*Entries before 2026-06-04 rotated to `_archive/CHANGELOG_pre_2026-06-04.md` (2026-06-10). Rotation policy: keep ~6 weeks / ≤60 KB live; rotate the rest.*
