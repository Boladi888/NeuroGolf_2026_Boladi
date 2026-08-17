# ARC Taxonomy Empirical Coverage on Shipped NeuroGolf Tasks

> **Status (2026-05-25)**: Empirical validation of the two ARC rule taxonomies (`arc_rule_taxonomy.md` and `arc_rule_taxonomy_alt.md`) against the 25 tasks we've shipped this session where the rule is well-understood. Answers the question: "which taxonomy is more useful for NeuroGolf workers in practice?"

## Methodology

For each shipped task, I classify the confirmed rule under both taxonomies using these labels:
- **Clean fit** — the family description matches the rule cleanly
- **Approximate fit** — the family is in the right neighborhood but not exact
- **No fit** — task represents an OPEN family neither taxonomy captures

Source rule descriptions come from worker ship messages, per-task `.md` SHIPPED entries, and the prototype builders.

## Classification table

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
| task118 (already counted above) | — | — | — | — |
| task158 (already counted above) | — | — | — | — |

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

**Neither taxonomy is "the truth" in any clean sense.** Both miss ~25% of our actual NeuroGolf task distribution. The 6 NEITHER-fits identify real OPEN families our worker output reveals:

### Open families our shipped tasks expose (not in either published taxonomy)

1. **Radius-encoded weighted cover detection** (task118) — neighborhood-shape detection where the kernel encodes radius/weight per-cell. Doesn't fit "Conv-based detection" cleanly because it's about WEIGHTED detection of cover shapes.
2. **Structural-detection-via-QLinearConv-direct** (task367 v2, task077 rawwhere) — multiple QConv detectors producing direct-paint output. Closer to "rule library applied via Conv kernels" but no taxonomy has that family.
3. **Row-signature / rank-reduction** (task017, task173) — Pattern_Recognition tasks using row-by-row signature matching with rank reduction. Hodel's DSL has primitives for this but no taxonomy explicitly names the family.
4. **Multi-stage compositional with key-first selection** (task285) — Report 2's "Mixed extraction-composition" meta-category gets closest, but this specific pattern (key extraction → count score → two-stage filter → reach-7 propagation) is its own family.
5. **Hull-8 connected-component variant** (task077 safe + rawwhere) — convex-hull or 8-connected-detection patterns that don't fit "ExtractObjects" or "CC selection by size".

### Where the two taxonomies actually differ in usefulness

- **R1 is sharper for**: extension/ray families (ExtendToBoundary is unambiguous: task066, task005, task002), shift-based copy patterns (task363 v2's seed-shift enumeration fits Copy cleanly).
- **R2 is sharper for**: flood fill (task243's anchor-conditioned fill has a named family in R2 but not in R1), multi-stage composition (R2's "Mixed extraction-composition" + "Contextual gating" meta-cases cover task285), CC-selection-by-property (task161's labelboundary).

**On the families both name identically** (Recolor by component property, Periodic tiling, Rotation/reflection), they agree. The 9% of tasks getting clean fits in both shows the overlap is real but limited.

## Recommendation

**Use both taxonomies as complementary, NOT competing.** Specifically:

- Workers should walk **R1's decision tree first** (it has cleaner extension/copy/translation distinctions for ~30% of tasks).
- If R1's tree returns "no obvious family fit", consult **R2** (R2's contextual-gating and mixed-composition categories catch ARC-AGI-2-flavored tasks R1 misses).
- If both return "no fit", the task is likely in one of the **6 NeuroGolf-specific open families** documented above; consult `Onnx_Advanced/exemplars/` for the closest shipped example.

**Do not consolidate the two into one document.** The complementary perspective is the actual value — eliminating either loses ~13-17% of our task coverage.

## Implication for the open-family backlog

Future research cards should target the 6 open families our shipped data identified. Specifically, these would be high-value:

- ~~New card: "Radius-encoded weighted cover detection patterns in ONNX" (motivated by task118)~~ → **F1 DONE 2026-05-26**: `radius_encoded_weighted_cover.md`
- ~~New card: "Row-signature / rank-reduction patterns" (motivated by task017, task173)~~ → **F2 DONE 2026-05-26**: `row_signature_rank_reduction.md`
- ~~New card: "Multi-stage compositional rule patterns with key-first selection" (motivated by task285)~~ → **F3 DONE 2026-05-27**: `multistage_key_first_compositional.md`
- ~~New card: "Direct-detection-via-multi-QLinearConv patterns" (motivated by task367 v2, task077)~~ → **F4 DONE 2026-05-28**: `direct_detection_multi_qlinearconv.md` (task367 v2 + task077 worked examples are SPECULATIVE inferences from project-local labels — not project-verified specifications).

These complement(ed) the existing Todo cards by adding NeuroGolf-empirical-derived families that the published ARC literature doesn't name. **All 4 F-series open-family cards complete as of 2026-05-28.** Remaining 5 open Todo cards (B1, B2, B3, C1, C2, C3, D2) target general ONNX-design knowledge gaps, not NeuroGolf-empirical families.

## Caveats

- 23 tasks is a small sample. The ~26% "neither fits" rate may not generalize to all 401 tasks.
- Several "approximate fit" classifications are judgment calls — different reviewers could shift R1 vs R2 counts by 1-2 each direction.
- "Confirmed rule" for some compression tasks (task161, task017, task173) is inferred from prototype names + brief worker descriptions, not from a deep numpy-reference reading. A more thorough validation would re-classify these after reading the actual numpy code.
- Tasks not yet shipped (task233, task219, task051 attempts before v23, plus the ~370 untouched tasks) are not in this analysis. The open-family rate may be higher or lower on the broader set.

## Open question for next iteration

Should we now produce a single MERGED taxonomy that uses R1's structure as base, adds R2's unique families (anchor-conditioned flood fill, contextual rule gating, mixed extraction-composition), and adds the 6 NeuroGolf-empirical open families? That would be a third document — `arc_rule_taxonomy_merged.md` — replacing both originals as the canonical workflow doc while preserving them as source material.

If yes, this is a ~2-hour task and produces the cleanest workflow doc.
