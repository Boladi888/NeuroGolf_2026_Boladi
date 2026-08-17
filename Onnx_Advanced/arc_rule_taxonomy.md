<!-- Refactor pass 4 (2026-06-10): textbook essays evicted to `_archive/arc_rule_taxonomy_evicted_2026-06-10.md`; snapshot: `_archive/arc_rule_taxonomy_pre_refactor_2026-06-10.md`. Phase-2: coverage spine merged as front-matter; alt taxonomy archived (won 3/23; its winning rows noted in the table). -->

# ARC Taxonomy Empirical Coverage on Shipped NeuroGolf Tasks

> Project-verified spine (2026-05-25): head-to-head of the two ARC taxonomies — **R1** = 18-family ConceptARC names (condensed below) vs **R2** = 15-family output-shape alt — on 25 shipped entries (23 unique), well-understood rules only.
> Unknown-rule task: walk the decision tree, check the table for the nearest shipped precedent, else suspect the 6 NeuroGolf-only open families. Cross-ref: `by_symptom.md` unknown-rule-family symptom.
> Verdict: R1 and R2 are **complementary, NOT competing** — both miss ~26% (6/23 NEITHER-fit) of our real task distribution.

## Classification table

Rules sourced from worker ship messages, per-task `.md` SHIPPED entries, prototype builders.

| Task | Confirmed rule | Report 1 (18-family) | Report 2 (15-family) | Verdict |
|---|---|---|---|---|
| task066 | wallpath: extend rays via coord arithmetic along wall plane until target | ExtendToBoundary (#7) ✓ clean | Line drawing (#13) approx | **R1 sharper** |
| task277 | bbox+area duplicate detector; recolor unique vs dup | Recolor by component property (#15) ✓ clean | Recolor by component property (#6) ✓ clean | **Tie** |
| task076 | sparse geodesic TopK selection + sparse render | Order and serialize (#13) approx | Object sorting and packing (#14) approx | **Both approx** |
| task101 | TopK + scatter (true structural, not lookup) | Copy and replicate (#5) approx | Object sorting and packing (#14) approx | **Both approx** |
| task118 | radius-encoded weighted cover QConv (cross neighborhoods) | NO FIT | NO FIT | **NEITHER — open family** |
| task158 | UINT8 dynamic-kernel QLinearConv + legend masks | Periodic tiling (#14) ambiguous | Period detection (#15) ambiguous | **Both ambiguous** |
| task367 v2 | structural direct-25 QLinearConv (no MPH) | NO FIT | NO FIT | **NEITHER — open family** (F4 deliverable speculatively covers, 2026-05-28) |
| task096 | row/col occupancy + padded color-index + row-run detection | Extract and crop (#8) approx | Boolean shape composition (#2) approx | **Both approx** |
| task363 v2 | enumerate shift candidates + filter via mask predicates | Copy and replicate (#5) ✓ clean | Translation/boundary (#9) approx | **R1 sharper** |
| task285 | keyfirst + countscore + twostage K=23 with reach-7 propagation | NO FIT (multi-stage) | Mixed extraction-composition (meta) approx | **R2 sharper** |
| task092 | moment-score einsum threshold | Count-driven synthesis (#6) approx | Count-parameterized repetition (#12) approx | **Both approx** |
| task005 v40 | motif ray extend with compact bbox | ExtendToBoundary (#7) ✓ clean | Line drawing (#13) approx | **R1 sharper** |
| task161 | labelboundary with dynamic slice + UINT8 | NO FIT | Connected-component selection (#3) approx | **R2 sharper** |
| task002 | rayseed extension with bit-packed shifts | ExtendToBoundary (#7) ✓ clean | Line drawing (#13) approx | **R1 sharper** |
| task017 | row14_densecrop_i8pad row-signature based | NO FIT | NO FIT | **NEITHER — row-signature open family** |
| task243 | exact row-bitpacked 28-step flood fill | InsideOutside (#11) approx | Anchor-conditioned flood fill (#1) ✓ clean | **R2 sharper** |
| task025 | bgcount==0 + colorcount>0 full-line detection | Count-driven synthesis (#6) approx | Boolean shape composition (#2) approx | **Both approx** |
| task173 | row-signature/rank-reduction (keepdims active nondiagless) | NO FIT | NO FIT | **NEITHER — row-signature open family** |
| task175 v17 | ArgMax(select_last_index=1) period detection | Periodic tiling (#14) ✓ clean | Period detection and tiling (#15) ✓ clean | **Tie** |
| task077 safe | structural qguard + max-paint scalar | NO FIT | NO FIT | **NEITHER — open family** |
| task077 rawwhere | rawwhere with hull-8 detection | NO FIT | NO FIT | **NEITHER — open family** (F4 deliverable speculatively covers, 2026-05-28) |
| task051 v23/v28 | marker-index detection + arrow direction paint | Recolor by component property (#15) approx | Recolor by component property (#6) approx | **Both approx** |
| task051 v34 | moment-based marker + arrowhead direction from centroid | Recolor by component property (#15) approx | Recolor by component property (#6) approx | **Both approx** |

**Total unique tasks classified: 23**

## Summary statistics

| Outcome | Count | Percent of 23 |
|---|---|---|
| Clean fit in R1 | 5 | 22% |
| Clean fit in R2 | 3 | 13% |
| Clean fit in both | 2 (task277, task175) | 9% |
| Approximate fit in both | 7 | 30% |
| R1 sharper than R2 | 4 (task066, task363 v2, task005, task002) | 17% |
| R2 sharper than R1 | 3 (task285, task161, task243) | 13% |
| NEITHER fits cleanly (open families) | 6 (task118, task367 v2, task017, task173, task077 safe, task077 rawwhere) | **26%** |

## Verdict

**Neither taxonomy is "the truth"** — both miss ~25% of our task distribution. The 6 NEITHER-fit rows (IDs in the stats table) expose real open families:

### Open families our shipped tasks expose (not in either published taxonomy)

1. **Radius-encoded weighted cover detection** (task118) — kernel encodes radius/weight per cell; WEIGHTED cover-shape detection, not clean "Conv-based detection".
2. **Structural-detection-via-QLinearConv-direct** (task367 v2, task077 rawwhere) — multiple QConv detectors → direct-paint output; "rule library via Conv kernels" exists in no taxonomy.
3. **Row-signature / rank-reduction** (task017, task173) — row-by-row signature matching + rank reduction; Hodel's DSL has primitives, no taxonomy names the family.
4. **Multi-stage compositional with key-first selection** (task285) — key extraction → count score → two-stage filter → reach-7 propagation; R2's "Mixed extraction-composition" closest, still its own family.
5. **Hull-8 connected-component variant** (task077 safe + rawwhere) — convex-hull / 8-connected detection; fits neither "ExtractObjects" nor "CC selection by size".

### Where the two taxonomies actually differ in usefulness

- **R1 sharper for**: extension/ray (ExtendToBoundary unambiguous: task066, task005, task002); shift-based copy (task363 v2 fits Copy cleanly).
- **R2 sharper for**: flood fill (task243's anchor-conditioned fill named only in R2); multi-stage composition (task285); CC-selection-by-property (task161 labelboundary).
- Families both name identically (Recolor by component property, Periodic tiling, Rotation/reflection) agree; only 9% fit cleanly in both — overlap real but limited.

## Recommendation

Use both as **complementary, NOT competing**: walk R1's tree first (cleaner extension/copy/translation splits, ~30% of tasks); no fit → R2 (contextual gating + mixed composition catch ARC-AGI-2-flavored tasks); both miss → one of the 6 open families above; consult `Onnx_Advanced/exemplars/` for the closest shipped example. Do not merge R1+R2 — dropping either loses ~13-17% coverage.

## Implication for the open-family backlog

All 4 F-series open-family cards complete:

- F1 (task118) → extractions in techniques.md (grep "radius-encoded"); full deliverable `_archive/radius_encoded_weighted_cover.md` (2026-05-26)
- F2 (task017, task173) → extractions in techniques.md (grep "row signature"); full deliverable `_archive/row_signature_rank_reduction.md` (2026-05-26)
- F3 (task285) → extractions in techniques.md (grep "key-first"); full deliverable `_archive/multistage_key_first_compositional.md` (2026-05-27)
- F4 (task367 v2, task077) → extractions in techniques.md (grep "detector bank"); full deliverable `_archive/direct_detection_multi_qlinearconv.md` (2026-05-28) — worked examples SPECULATIVE (from project-local labels), not project-verified.

Remaining open Todo cards (B1, B2, B3, C1, C2, C3, D2) target general ONNX-design gaps, not NeuroGolf-empirical families.

## Caveats

- 23 tasks is a small sample; ~26% "neither fits" may not generalize to all 401 tasks.
- "Approximate fit" calls are judgment calls — reviewers could shift R1/R2 counts by 1-2.
- task161 / task017 / task173 rules inferred from prototype names + brief worker notes, not a deep numpy-reference read.
- Not analyzed: task233, task219, task051 pre-v23 attempts, ~370 untouched tasks — open-family rate may differ there.

---

# Condensed 18-family reference (R1: ARC Rule Taxonomy for NeuroGolf-Style ONNX Solvers)

Provenance: card A2 synthesis (card body deleted 2026-05-27 Done/-cull, commit eb74a378 — git history only). ONNX hints are researcher-INFERRED, never project-verified. Assumes opset ≤18, ORT 1.24, NCHW one-hot planes; no Loop/Scan/NonZero/Unique/Compress. `K_ray` = max(H,W) unroll; `cc_minprop` = unrolled Neg→MaxPool→Neg min-prop CC labeling (~H²·W² footprint).

| # | Family | Recognition cue | ONNX hint |
|---|---|---|---|
| 1 | Alignment to center or anchor | same object/palette, moved to center or unique anchor | bbox via ReduceMax projections + ArgMax first/last, then Pad+Slice shift-overlay |
| 2 | Above-below relative placement | same object multiset; pairwise axis-aligned offsets change | per-object bboxes → relation offset → shift/composite |
| 3 | Cleanup and noise suppression | isolated weakly-supported pixels vanish; motif survives | Conv(occ, ones_3x3) support → Greater(thresh) → Where |
| 4 | Complete shape and symmetry completion | cells added to restore mirror/rotational symmetry; bbox fixed | reflected Slice/Transpose candidate, union via Max |
| 5 | Copy and replicate | exact duplicates of one motif at marker cells | crop source bbox; TopK marker coords; Div/Mod decode; unrolled shift-overlay |
| 6 | Count-driven synthesis | an integer (cells/objects/colors) controls output size/repetition | ReduceSum count → Range+Less prefix mask → Where |
| 7 | Extend or connect to boundary | collinear growth from seeds until edge/blocker | unrolled directional Conv/MaxPool propagation, K_ray steps |
| 8 | Extract and crop objects | output smaller; tight bbox around one component | occupancy projections → first/last active → dynamic Slice |
| 9 | Fill-state transform | hollow↔solid flip; outer contour stable | border-seeded unrolled flood fill on complement → interior mask → merge/subtract |
| 10 | Horizontal-versus-vertical orientation | branch on aspect ratio / principal axis | bbox h vs w via Greater → Where(rotate90 branch, identity) |
| 11 | Inside-outside recolor | interior vs exterior of closed contour recolored | flood-fill background from border; complement = inside; Where recolor |
| 12 | Move to boundary or target | same object translated into wall/target contact | bbox → delta-to-edge arithmetic → Pad+Slice shift |
| 13 | Order and serialize objects | same objects re-laid-out monotone by size/color/position | cc_minprop labels → per-label stats → TopK rank → unrolled slot placement |
| 14 | Periodic tiling and motif extrapolation | visible repetition; output extends/fills pattern by period | unrolled period tests → Mod(Range)+Gather |
| 15 | Recolor by component property | geometry fixed; palette tracks area/border-contact/rank | cc_minprop labels + stats → ArgMax property → Equal-mask Where |
| 16 | Rotation and reflection | discrete global 90°/flip; pixels + palette preserved | Transpose + reversed Slice branch |
| 17 | Same-different and boolean overlay | equality/intersection/difference of two structures | align → Equal/And/Not/Max → Where |
| 18 | Top-bottom stacking and occlusion | layer precedence at overlapping cells | per-layer masks → cascaded Where, top precedence first |

**Open-research families** (no efficient static-graph ONNX known; prose in evicted file): latent correspondence / analogy transfer; deep multi-stage composition with data-dependent control flow; cheap exact flood/CC in tortuous geometries; minimal 2D motif induction under phase ambiguity; general packing / assignment / collision-free layout.

## Decision tree (unknown rule → candidate family)

First yes wins; score in order: size-change → translation/layout → recolor → growth/completion → copy/count → geometric → overlap/layering.

1. Output smaller/tighter? → #8 Extract and crop objects.
2. Same objects, just moved? → to center/anchor #1; pairwise relational #2; edge-seeking #12; full re-layout #13.
3. Geometry fixed, colors change? → #15 (property rank), #11 (interior/exterior), #9 (hollow↔solid).
4. New cells collinear from seeds toward walls/blockers? → #7.
5. Exact duplicates of one motif? → #5; copy count set by an integer, not markers → also #6.
6. Rotated/mirrored/orientation-normalized? → #16; branches on aspect ratio → #10.
7. Pattern or missing-half completion? → symmetry cue #4; repetition cue #14.
8. Overlap or layer precedence operative? → pure boolean #17; visible layering #18.
9. Integer statistic controls length/multiplicity/density? → #6 (+ #13 if ranking visible).
10. Nothing fits → one of the 6 NeuroGolf-only open families; precedent: coverage table + `exemplars/`.

*Full 18-family essays + inferred ONNX sketches: `_archive/arc_rule_taxonomy_evicted_2026-06-10.md` (textbook tier — grep on demand).*
