<!-- Textbook tier evicted from `Onnx_Advanced/arc_rule_taxonomy.md` (refactor pass 4): the full 18-family essays (definitions, ConceptARC worked examples, recognition signals, researcher-INFERRED ONNX sketches / opset floors / cost formulas — never project-verified), cross-cutting primitives table, open-research prose, full decision tree, plus the coverage-spine prose that was condensed in the live file. -->
<!-- Evicted 2026-06-10 per audits 2026-06-04 + 2026-06-10 (~90% Sonnet-class textbook content). Full byte-identical pre-refactor snapshot: `arc_rule_taxonomy_pre_refactor_2026-06-10.md`. -->
<!-- Live file (23-task empirical coverage spine + condensed 18-family table + decision tree): `Onnx_Advanced/arc_rule_taxonomy.md`. -->

# ARC Rule Taxonomy — evicted textbook tier (2026-06-10)

## Coverage-spine prose condensed in the live file (originals preserved verbatim below)
> **Status (2026-05-25)**: Empirical validation of the two ARC rule taxonomies (`arc_rule_taxonomy.md` and `_archive/arc_rule_taxonomy_alt.md` (archived)) against the 25 tasks we've shipped this session where the rule is well-understood. Answers the question: "which taxonomy is more useful for NeuroGolf workers in practice?"

## Methodology

For each shipped task, I classify the confirmed rule under both taxonomies using these labels:
- **Clean fit** — the family description matches the rule cleanly
- **Approximate fit** — the family is in the right neighborhood but not exact
- **No fit** — task represents an OPEN family neither taxonomy captures

Source rule descriptions come from worker ship messages, per-task `.md` SHIPPED entries, and the prototype builders.

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

---
# ARC Rule Taxonomy for NeuroGolf-Style ONNX Solvers

> **Status (2026-05-25)**: Canonical synthesis of card A2 (ARC-AGI rule taxonomy → ONNX patterns). Produced by an internet-research model from card A2 (card body deleted 2026-05-27 in the Done/-folder cull, commit eb74a378 — git history only). The former alt taxonomy is archived (won 3/23 head-to-head) (15 families vs 18 families; complementary ConceptARC group names).
>
> **When to read**: open at the START of any new task where the rule is unknown. The decision tree at the end narrows candidate families before you write any numpy reference. Cross-referenced from `by_symptom.md` → "I don't know what rule family this task belongs to".

## Context

This note follows the supplied Research Card A2 and therefore targets ONNX opset at or below 18, ONNX Runtime 1.24, NCHW tensors, build-time-unrolled iteration, and a score defined as intermediate-memory bytes plus parameter elements; it also respects the ban on `Loop`, `Scan`, `NonZero`, `Unique`, and `Compress`.

ARC-AGI is a few-shot grid-transformation benchmark introduced by François Chollet to measure fluid intelligence rather than benchmark-specific memorization. In the original ARC-AGI-1 materials and later ARC Prize documentation, each task is a small colored grid problem in which the solver must infer a rule from a handful of input/output demonstrations and apply it to held-out inputs; official ARC Prize reporting also emphasizes that ARC tasks are intentionally diverse, human-authored, and grounded in early “Core Knowledge” priors such as objectness, topology, geometry, and counting.

There is not one universally accepted, official “ARC family list.” Instead, the literature offers overlapping taxonomic lenses: ConceptARC organizes tasks into 16 concept groups, the recent ARC living survey groups ARC-AGI-1/2 into broader reasoning categories such as object-centric, geometric, relational, numerical, and pattern-completion reasoning, and a newer re-arc taxonomy effort reports a validated 9-category taxonomy. The 18 families below are therefore best read as a normalized synthesis for engineering use, not as a claim that the ARC community has ratified a single canonical partition.

For a NeuroGolf-style ONNX project, the practical unit is not “solve ARC end to end in one graph,” but “compile a concrete hypothesis family into a fixed acyclic graph.” ARC Prize 2024 materials, public Kaggle-linked winner code pages, Hodel’s starter notebook, and public program-synthesis writeups all reinforce that successful ARC systems work by explicit hypothesis generation, object-centric decomposition, program induction, or test-time adaptation rather than by a single monolithic feed-forward visual model.

## Methodology

I built this taxonomy by triangulating five source types. First, Chollet’s paper and official ARC Prize documentation establish what ARC is for and which cognitive priors it is meant to probe. Second, ConceptARC is the cleanest public source of recurring concept groups in the ARC domain, because it explicitly isolates families such as `Center`, `Copy`, `Count`, `InsideOutside`, `Order`, and `TopBottom3D`. Third, Hodel’s ARC-DSL and RE-ARC expose how many ARC tasks decompose into reusable object-centric primitives such as object extraction, color filtering, bordering, filling, shifting, and painting. Fourth, ARC Prize 2024 competition pages and winning-paper links show which abstractions actually mattered in the strongest open approaches. Fifth, the ONNX mappings are my own engineering synthesis from official ONNX operator semantics and versioning rules, because I did not find published work that directly specifies an ARC→ONNX family library under the exact constraints in the card.

The families below assume **external family recognition and hypothesis selection**. In other words, the graph for a family does not need to *discover* “this is a horizontal mirror” or “the largest component should become yellow”; it only needs to execute that already-chosen hypothesis over the test grid. That is the right granularity for program-synthesis-style ARC systems and is also the only realistic way to stay efficient under a graph language with no `Loop`, no `Scan`, and no sparse indexing primitives like `NonZero`.

I use the following cost shorthand throughout:

- `H`, `W`: actual puzzle height and width, each at most 30.
- `C`: color planes, usually 10.
- `b`: `bytes_per_element`.
- `K_ray = max(H, W)`: safe unroll depth for monotone row/column/diagonal propagation.
- `K_fill = H·W`: pessimistic unroll depth for masked flood fill or connected-component label propagation in serpentine cases.
- `M ≤ H·W`: explicit marker/object slots selected by `TopK`.
- Reported costs are **upper-bound graph-footprint estimates**, assuming each unrolled stage materializes once and fixed kernels are stored as parameters.

I also use three sketch macros to keep the code stubs readable:

- `first_last_active(v)`: shorthand for `ArgMax(v)` and reversed `ArgMax(Slice(v, step=-1))`.
- `shift_overlay(...)`: shorthand for “pad, slice at offset, then composite with `Max` or `Where`.”
- `cc_minprop(mask)`: shorthand for seeding active pixels with unique IDs and running unrolled min-propagation via `Neg → MaxPool → Neg` under a mask. This is an inferred ONNX pattern, not a published ARC implementation. Official operator semantics and conservative opset floors are centralized in the primitives table below.

## Family entries

The list is grouped loosely by how the output usually behaves: same-shape edits first, then layout/translation families, then size-changing families.

### Family: Alignment to center or anchor

**Definition**: These tasks preserve one salient object or object set and change only its placement, typically centering it in the canvas or aligning it to a unique anchor or reference location. This directly matches ConceptARC’s `Center` group and also fits the living survey’s relational/spatial category, where alignment matters more than recoloring or shape change.

**Mini-example** (textual ASCII):
Input:                 Output:
2 2 0 0 0              0 0 0 0 0
2 2 0 0 0              0 2 2 0 0
0 0 0 0 0              0 2 2 0 0
0 0 0 0 0              0 0 0 0 0
0 0 0 0 0              0 0 0 0 0

**Recognition signals**:
- Output shape usually equals input shape.
- Object geometry and palette are preserved.
- Input/output differ mostly by integer translation.
- The output object often has a conspicuously centered or anchor-aligned bounding box.

**ONNX implementation pattern**:
- Ops: occupancy projection, bounding-box recovery, offset arithmetic, then `Pad` + `Slice` shift/composite.
- Opset: 13 conservative floor.
- Dtype assumptions: one-hot planes in `float32`; index arithmetic in `int64`.
- Cost: approximately `(4·C·H·W + H + W)·b + 0`.
- Sketch:
  ```python
  occ = ReduceMax(X, axes=[1], keepdims=1)
  rows = ReduceMax(occ, axes=[3], keepdims=0)
  cols = ReduceMax(occ, axes=[2], keepdims=0)
  r0, r1 = first_last_active(rows); c0, c1 = first_last_active(cols)
  Y = shift_overlay(X, target_center_from_shape(Shape(X)), [r0, c0, r1, c1])
  ```

**Alternative patterns**: If a unique anchor color exists, recover that anchor bbox first and align the moving object relative to the anchor instead of to grid center.

**Known sources**: ConceptARC `Center`; Hodel’s object-centric ARC-DSL examples; ARC’s broader relational/spatial reasoning framing. The ONNX mapping is an inferred engineering pattern.

### Family: Above-below relative placement

**Definition**: These tasks preserve multiple objects and primarily change their relative arrangement, especially “A above B,” “A below B,” or similar axis-aligned relational placement. ConceptARC’s `AboveBelow` and `TopBottom2D` groups are the clearest public abstractions of this family.

**Mini-example** (textual ASCII):
Input:                 Output:
1 1 0 2 2              1 1 0 0 0
1 1 0 2 2              1 1 0 0 0
0 0 0 0 0              2 2 0 0 0
0 0 0 0 0              2 2 0 0 0

**Recognition signals**:
- Same object multiset appears in input and output.
- Output differs mostly in pairwise object offsets.
- Relative ordering along one axis is more stable than absolute positions.
- Colors are usually unchanged.

**ONNX implementation pattern**:
- Ops: per-object occupancy projection, bbox extraction, pairwise offset computation, then shift/composite.
- Opset: 13 conservative floor.
- Dtype assumptions: `float32` masks; `int64` for offsets and starts/ends.
- Cost: approximately `(5·C·H·W + H + W)·b + 0`.
- Sketch:
  ```python
  bbox_a = bbox(color_a_mask(X))
  bbox_b = bbox(color_b_mask(X))
  dr, dc = learned_relation_offset(bbox_a, bbox_b)
  Y = shift_overlay(X_obj_b, [dr, dc])  # over static reference branch
  ```

**Alternative patterns**: When the task amounts to swapping fixed halves or bands, a cheaper `Slice` + `Concat` pattern is preferable.

**Known sources**: ConceptARC `AboveBelow` and `TopBottom2D`; the ARC survey’s relational/spatial reasoning category. The ONNX mapping is inferred.

### Family: Cleanup and noise suppression

**Definition**: Cleanup tasks remove clutter, spurious pixels, or distractor fragments while preserving the meaningful foreground object or pattern. ConceptARC includes an explicit `CleanUp` group, and Hodel’s DSL/object-centric framing makes clear why ARC solvers often need to separate coherent objects from incidental noise.

**Mini-example** (textual ASCII):
Input:                 Output:
0 3 0 0 0              0 0 0 0 0
3 3 3 0 3              3 3 3 0 0
0 3 0 0 0              0 3 0 0 0
0 0 0 0 0              0 0 0 0 0

**Recognition signals**:
- Output uses a subset of the input foreground pixels.
- One dominant object or motif survives.
- Colors usually stay the same.
- Removed pixels tend to be isolated, weakly supported, or statistically atypical.

**ONNX implementation pattern**:
- Ops: local-support `Conv` or `MaxPool` over occupancy, thresholding, optional largest-component keep.
- Opset: 13 conservative floor.
- Dtype assumptions: cast binary masks to `float32` for convolution/pooling; optional component IDs in `int64`.
- Cost: approximately `(3·C·H·W)·b + 9` for the local-support variant.
- Sketch:
  ```python
  occ = ReduceMax(X, axes=[1], keepdims=1)
  support = Conv(occ, ones_3x3)
  keep = Greater(support, thresh)
  Y = Where(keep, X, 0)
  ```

**Alternative patterns**: If the demos clearly select the largest connected component rather than locally supported pixels, use `cc_minprop(mask)` plus component histograms instead.

**Known sources**: ConceptARC `CleanUp`; Hodel’s ARC-DSL object/filter/fill view; RE-ARC’s emphasis on variable object counts and object sampling. The ONNX mapping is inferred.

### Family: Complete shape and symmetry completion

**Definition**: These tasks infer missing parts of an object or pattern from symmetry, partial regularity, or an existing half/quadrant. ConceptARC names this family `CompleteShape`, while the living survey places it under pattern completion and geometric reasoning.

**Mini-example** (textual ASCII):
Input:                 Output:
0 1 0                  0 1 0
1 1 0                  1 1 1
0 1 0                  0 1 0

**Recognition signals**:
- Output adds pixels but rarely deletes the existing partial shape.
- The completed result often has mirror or rotational symmetry.
- New colors are rare; usually missing cells reuse existing foreground colors.
- Bounding boxes often stay fixed while occupancy increases.

**ONNX implementation pattern**:
- Ops: crop relevant patch, create mirror/rotation candidate with `Slice` or `Transpose`, then union with `Max` or select with `Where`.
- Opset: 13 conservative floor.
- Dtype assumptions: `float32` one-hot planes are simplest.
- Cost: approximately `(4·C·H·W + H + W)·b + 0`.
- Sketch:
  ```python
  patch = crop_active_bbox(X)
  refl  = Slice(patch, starts=[0,0,0,-1], ends=[1,C,H,-W-1], steps=[1,1,1,-1])
  Y = Max(X, paste_back(refl))
  ```

**Alternative patterns**: If the missing structure is periodic rather than symmetric, the periodic-tiling family is a better fit.

**Known sources**: ConceptARC `CompleteShape`; the ARC survey’s geometric/pattern-completion categories. The ONNX mapping is inferred.

### Family: Copy and replicate

**Definition**: Copy tasks extract one source object or motif and place one or more copies at marker positions, target cells, or regularly defined slots. ConceptARC includes a `Copy` group, and ARC DSL/object-based approaches routinely rely on extracting a patch, shifting it, and repainting it elsewhere.

**Mini-example** (textual ASCII):
Input:                 Output:
2 2 0 0 3              2 2 0 0 2
2 2 0 0 0              2 2 0 0 2
0 0 0 0 0              0 0 0 0 0
3 0 0 0 0              2 2 0 0 0
0 0 0 0 0              2 2 0 0 0

**Recognition signals**:
- Output contains exact geometric duplicates of one input object.
- Duplicate locations correlate with special colors or markers.
- Source geometry is preserved exactly.
- Marker colors often disappear or become overwritten.

**ONNX implementation pattern**:
- Ops: extract source bbox, flatten marker scores, `TopK` marker coordinates, decode with `Div`/`Mod`, then unrolled shift-overlay branches.
- Opset: 18 conservative floor.
- Dtype assumptions: `float32` source planes; `int64` flattened indices.
- Cost: approximately `((3 + M)·C·H·W + M·(H + W))·b + 0`, with `M ≤ H·W`.
- Sketch:
  ```python
  src = crop_active_bbox(source_masked_X)
  score = Reshape(Mul(marker_mask, top_left_priority), [1, H * W])
  vals, idx = TopK(score, k=M)
  rows, cols = Div(idx, W), Mod(idx, W)
  Y = unrolled_shift_overlay(src, rows, cols, vals > 0)
  ```

**Alternative patterns**: If targets form a regular lattice, skip `TopK` and use `Tile` or repeated `Pad`/`Slice` at known offsets.

**Known sources**: ConceptARC `Copy`; Hodel’s ARC-DSL shifting/painting style; ARC’s object-centric reasoning tradition. The ONNX mapping is inferred.

### Family: Count-driven synthesis

**Definition**: In count-driven tasks, a numerical property of the input—often cell count, object count, or per-color multiplicity—controls the output’s size, repetition count, or occupancy. ConceptARC’s `Count` group and the ARC survey’s numerical-reasoning category make this family explicit.

**Mini-example** (textual ASCII):
Input:                 Output:
1 0 1 0 1              2 2 2
0 0 0 0 0

**Recognition signals**:
- Output size or population varies with an integer derived from the input.
- Geometry may be simple while difficulty lies in counting correctly.
- Colors may change, but cardinality relationships are stable.
- Across demos, output length often tracks object or color multiplicity exactly.

**ONNX implementation pattern**:
- Ops: `ReduceSum` for direct counts, or component-count alternative if needed; `Range` + `Less` to make prefix masks; `Where` to paint.
- Opset: 13 conservative floor.
- Dtype assumptions: counts in `int64`; occupancy planes in `float32`.
- Cost: approximately `(2·C·H·W + H + W)·b + 0` for direct-count cases.
- Sketch:
  ```python
  n = ReduceSum(count_mask, axes=[2, 3], keepdims=0)
  xs = Range(0, W_out, 1)
  active = Less(xs, n)
  Y = Where(active, color2_row, zeros_row)
  ```

**Alternative patterns**: If “count” means connected components rather than cells, prepend `cc_minprop(mask)` and histogram component IDs before synthesis.

**Known sources**: ConceptARC `Count`; the ARC survey’s numerical reasoning category. The ONNX mapping is inferred.

### Family: Extend or connect to boundary

**Definition**: These tasks extend rays, bars, contours, or connectors until a boundary, blocker, or matching anchor is reached. ConceptARC has an explicit `ExtendToBoundary` group, and many ARC tasks use this “grow until contact” logic.

**Mini-example** (textual ASCII):
Input:                 Output:
0 0 1 0 0              0 0 1 0 0
0 0 1 0 0              0 0 1 0 0
0 0 0 0 0              0 0 1 0 0
0 0 0 0 0              0 0 1 0 0
0 0 0 0 0              0 0 1 0 0

**Recognition signals**:
- Output adds cells collinear with existing seeds.
- Extended structures often terminate at the grid edge or an obstacle.
- Color usually stays constant along the extension.
- Object count may remain constant while object size increases.

**ONNX implementation pattern**:
- Ops: directional propagation using fixed-kernel `Conv` or masked `MaxPool`, unrolled for `K_ray = max(H, W)` steps.
- Opset: 13 conservative floor.
- Dtype assumptions: propagation masks in `float32`.
- Cost: approximately `((K_ray + 2)·C·H·W)·b + 9`, where `K_ray = max(H, W)`.
- Sketch:
  ```python
  cur = seed_mask
  for _ in range(K_ray):   # graph-unrolled
      nxt = Conv(cur, directional_kernel)
      cur = And(nxt > 0, free_space_mask)
  Y = Where(cur, draw_color, X)
  ```

**Alternative patterns**: For purely horizontal/vertical extension from a known row/column, coordinate-grid masking with `Range` is cheaper than iterative propagation.

**Known sources**: ConceptARC `ExtendToBoundary`; ARC’s relational/spatial and geometric reasoning framing. The ONNX mapping is inferred.

### Family: Extract and crop objects

**Definition**: These tasks isolate one object or object set and crop the output tightly to its bounding box, often removing all background or irrelevant material. ConceptARC calls this `ExtractObjects`, and Hodel’s DSL makes object extraction central to ARC program structure.

**Mini-example** (textual ASCII):
Input:                     Output:
0 0 0 0 0                  3 3
0 3 3 0 0                  3 0
0 3 0 0 0
0 0 0 0 0

**Recognition signals**:
- Output is smaller than input.
- Output colors are a subset of input colors.
- Background is heavily reduced or eliminated.
- Output bbox often tightly hugs one salient component.

**ONNX implementation pattern**:
- Ops: occupancy projection with `ReduceMax`, first/last active row/col recovery, dynamic `Slice`.
- Opset: 13 conservative floor.
- Dtype assumptions: `float32` or `int64` masks both fine until `Slice` indices are formed.
- Cost: approximately `(3·C·H·W + H + W)·b + 0`.
- Sketch:
  ```python
  occ = ReduceMax(X, axes=[1], keepdims=1)
  rows = ReduceMax(occ, axes=[3], keepdims=0)
  cols = ReduceMax(occ, axes=[2], keepdims=0)
  r0, r1 = first_last_active(rows); c0, c1 = first_last_active(cols)
  Y = Slice(X, starts=[0, 0, r0, c0], ends=[1, C, r1, c1])
  ```

**Alternative patterns**: If the desired object is “largest component” or “border-touching component,” first do component selection, then crop.

**Known sources**: ConceptARC `ExtractObjects`; Hodel’s ARC-DSL object primitives; RE-ARC’s object-centric generator assumptions. The ONNX mapping is inferred.

### Family: Fill-state transform

**Definition**: Fill-state tasks convert outlines to solids, solids to outlines, or otherwise use whether a shape is hollow versus filled as the decisive concept. ConceptARC’s `FilledNotFilled` group isolates this notion, and Hodel’s example of identifying enclosed holes is a related ARC-DSL pattern.

**Mini-example** (textual ASCII):
Input:                 Output:
1 1 1                  2 2 2
1 0 1                  2 2 2
1 1 1                  2 2 2

**Recognition signals**:
- Outer contour or bbox is stable while interior occupancy changes.
- Output often adds or removes only interior cells.
- “Hollow vs solid” is consistent across demos.
- Colors may be preserved or systematically recolored.

**ONNX implementation pattern**:
- Ops: border-seeded flood fill on the complement to recover interior, then merge or subtract; optional erosion via `Neg → MaxPool → Neg`.
- Opset: 13 conservative floor.
- Dtype assumptions: binary masks in `float32`.
- Cost: approximately `(C·H·W + H²·W²)·b + 0` in the conservative flood-fill form.
- Sketch:
  ```python
  barrier = outline_mask
  outside = border_seed(background_mask)
  outside = unrolled_masked_dilate(outside, background_mask)   # K_fill steps
  inside = And(background_mask, Not(outside))
  Y = Where(Max(barrier, inside), fill_color, 0)
  ```

**Alternative patterns**: For strictly rectangular objects, fill can be produced more cheaply from bbox limits without flood fill.

**Known sources**: ConceptARC `FilledNotFilled`; Hodel’s enclosed-hole example in ARC-DSL. The ONNX mapping is inferred.

### Family: Horizontal-versus-vertical orientation

**Definition**: These tasks branch on an object’s orientation or normalize an oriented object into one canonical direction. ConceptARC names this family `HorizontalVertical`, and the ARC survey explicitly lists orientation-sensitive geometric transformations as a recurring challenge area.

**Mini-example** (textual ASCII):
Input:                 Output:
0 4 0                  4 4 4
0 4 0
0 4 0

**Recognition signals**:
- Pixel count stays roughly constant.
- Output often swaps height and width.
- Demos show clear branching on aspect ratio or principal axis.
- Colors are usually preserved.

**ONNX implementation pattern**:
- Ops: bbox dimensions from projections, `Greater` comparison on height versus width, branch bank such as identity versus 90° rotate.
- Opset: 13 conservative floor.
- Dtype assumptions: object masks in `float32`; dimensions in `int64`.
- Cost: approximately `(3·C·H·W + H + W)·b + 0`.
- Sketch:
  ```python
  r0, r1, c0, c1 = bbox(occ)
  h, w = Sub(r1, r0), Sub(c1, c0)
  rot = rotate90(X)
  Y = Where(Greater(h, w), rot, X)
  ```

**Alternative patterns**: If the family only selects horizontal objects and deletes vertical ones, compare orientation and mask unwanted objects rather than rotating them.

**Known sources**: ConceptARC `HorizontalVertical`; ARC’s geometric-transformation category. The ONNX mapping is inferred.

### Family: Inside-outside recolor

**Definition**: These tasks recolor pixels depending on whether they lie inside a closed contour or outside it. ConceptARC names this directly as `InsideOutside`, and Hodel’s ARC-DSL example of identifying black regions not touching the border is a near-exact operational template.

**Mini-example** (textual ASCII):
Input:                 Output:
1 1 1 1                1 1 1 1
1 0 0 1                1 2 2 1
1 0 0 1                1 2 2 1
1 1 1 1                1 1 1 1

**Recognition signals**:
- The contour itself is mostly unchanged.
- Only interior or exterior regions are recolored.
- Output often introduces a new fill color.
- Border connectivity is the key invariant across demos.

**ONNX implementation pattern**:
- Ops: border-seeded flood fill over reachable background, complement to get interior, `Where` recolor.
- Opset: 13 conservative floor.
- Dtype assumptions: `float32` binary masks; final palette in one-hot planes.
- Cost: approximately `(C·H·W + H²·W²)·b + 0`.
- Sketch:
  ```python
  bg = background_mask
  outside = unrolled_masked_dilate(border_seed(bg), bg)
  inside = And(bg, Not(outside))
  Y = Where(inside, color2, X)
  ```

**Alternative patterns**: If the contour is guaranteed rectangular, inside/outside can be implemented by bbox inequalities instead of flood fill.

**Known sources**: ConceptARC `InsideOutside`; Hodel’s hole-detection/fill example. The ONNX mapping is inferred.

### Family: Move to boundary or target

**Definition**: These tasks translate one object until it touches a designated boundary, wall, or target location without otherwise changing its geometry. ConceptARC includes a dedicated `MoveToBoundary` group, and Hodel’s second ARC-DSL example is literally an object-shift program.

**Mini-example** (textual ASCII):
Input:                 Output:
0 0 0 0 0              5 5 0 0 0
0 5 5 0 0              5 5 0 0 0
0 5 5 0 0              0 0 0 0 0
0 0 0 0 0              0 0 0 0 0

**Recognition signals**:
- The same object appears before and after.
- Translation ends in border contact.
- Colors are preserved.
- Output motion is axis-aligned in many demos.

**ONNX implementation pattern**:
- Ops: bbox extraction, compute delta to edge or target, `Pad` + `Slice` shift, then repaint.
- Opset: 13 conservative floor.
- Dtype assumptions: `float32` object planes, `int64` offsets.
- Cost: approximately `(4·C·H·W + H + W)·b + 0`.
- Sketch:
  ```python
  r0, r1, c0, c1 = bbox(occ)
  dr, dc = delta_to_boundary(r0, r1, c0, c1, chosen_side)
  Y = shift_overlay(X, [dr, dc])
  ```

**Alternative patterns**: If motion is “slide until obstacle” rather than “jump to wall,” use the extend/connect propagation family instead.

**Known sources**: ConceptARC `MoveToBoundary`; Hodel’s ARC-DSL shifting example. The ONNX mapping is inferred.

### Family: Order and serialize objects

**Definition**: Order tasks sort a set of objects by size, color, position, or another measurable property and then lay them out in a canonical sequence. ConceptARC labels this family `Order`, and it is also a natural extension of object-centric reasoning in ARC DSL systems.

**Mini-example** (textual ASCII):
Input:                    Output:
1 0 0 0 2 2               1 0 2 2
0 0 0 0 2 2               0 0 2 2
0 0 0 0 0 0

**Recognition signals**:
- Output contains the same object multiset in a more regular layout.
- Relative positions become monotone or evenly spaced.
- Sort keys are often size, color, or centroid coordinate.
- Colors and shapes are preserved.

**ONNX implementation pattern**:
- Ops: connected-component labels, per-label statistics, `TopK` ranking, then unrolled shift-overlay into slots.
- Opset: 18 conservative floor.
- Dtype assumptions: label IDs and ranks in `int64`; masks and slots in `float32`.
- Cost: approximately `(H²·W² + M·C·H·W)·b + 0`, with `M ≤ H·W`.
- Sketch:
  ```python
  labels = cc_minprop(occ)
  sizes  = label_sizes(labels)
  _, ids = TopK(sizes, k=M)
  slots  = canonical_slots(ids)
  Y = unrolled_place_by_rank(X, labels, ids, slots)
  ```

**Alternative patterns**: If there are at most two or three objects, precompile all orderings as small branch banks and gate them by comparisons.

**Known sources**: ConceptARC `Order`; object-centric ARC-DSL design; ARC’s relational/numerical reasoning categories. The ONNX mapping is inferred.

### Family: Periodic tiling and motif extrapolation

**Definition**: These tasks detect a repeating motif, minimal period, or local tile and then complete or expand the pattern across part or all of the output. The ARC living survey treats pattern completion as a core ARC category, and RE-ARC’s emphasis on broad variation over grid dimensions makes periodic extrapolation especially salient in generated ARC-like data.

**Mini-example** (textual ASCII):
Input:                 Output:
1 2 1 2                1 2 1 2 1 2

**Recognition signals**:
- Repetition is visible in rows, columns, or 2D tiles.
- Output is often larger than input or fills missing cells.
- Colors stay within the same motif palette.
- Shifts by a candidate period preserve large parts of the pattern.

**ONNX implementation pattern**:
- Ops: unrolled candidate-period validation, then closed-form indexing with `Range`/`CumSum`, `Mod`, and `Gather`.
- Opset: 13 conservative floor.
- Dtype assumptions: motif planes in `float32`; period and indices in `int64`.
- Cost: approximately `((max(H, W) + 2)·C·H·W)·b + 0`.
- Sketch:
  ```python
  p = first_valid_period(unrolled_period_tests(X))
  xs = Mod(Range(0, W_out, 1), p)
  Y = Gather(motif_patch, xs, axis=-1)
  ```

**Alternative patterns**: For pure integer upscaling without phase discovery, `Resize(mode="nearest")` is simpler and cheaper.

**Known sources**: ARC’s pattern-completion category in the living survey; RE-ARC’s broad example-generation philosophy. The ONNX mapping is inferred.

### Family: Recolor by component property

**Definition**: In this family, component geometry is preserved while color changes depend on measurable properties such as area, border contact, orientation, or relative rank. Hodel’s DSL examples and primitives such as `objects`, `colorfilter`, and `bordering` make this one of the most natural object-centric ARC families, even though it is not one of ConceptARC’s 16 named groups.

**Mini-example** (textual ASCII):
Input:                 Output:
1 1 0 2                4 4 0 2
1 1 0 2                4 4 0 2
0 0 0 0                0 0 0 0

**Recognition signals**:
- Geometry stays fixed while palette changes.
- Recolored components usually share a measurable statistic.
- Demos reveal a stable “largest,” “smallest,” “border-touching,” or “oriented” rule.
- Output colors may be a subset of input colors or introduce one new marker color.

**ONNX implementation pattern**:
- Ops: component labeling, label histograms/statistics, property selection with `ArgMax`/`ArgMin`, then recolor with equality masks and `Where`.
- Opset: 18 conservative floor.
- Dtype assumptions: labels/statistics in `int64`; working masks in `float32`.
- Cost: approximately `(H²·W² + (C + 4)·H·W)·b + 0`.
- Sketch:
  ```python
  labels = cc_minprop(occ)
  sizes  = label_sizes(labels)
  big_id = ArgMax(sizes)
  Y = Where(Equal(labels, big_id), color4, X)
  ```

**Alternative patterns**: If the property is global color frequency rather than component geometry, color-plane `ReduceSum` is cheaper than full component labeling.

**Known sources**: Hodel’s ARC-DSL object/filter/border primitives; RE-ARC’s object-centric generator writeup; the ARC survey’s object-centric category. The ONNX mapping is inferred.

### Family: Rotation and reflection

**Definition**: These tasks rotate or mirror a whole grid or a selected object by a discrete geometric transform, usually 90° steps or axis-aligned flips. The ARC survey explicitly names rotation, reflection, scaling, translation, and symmetry as recurring geometric transformations in ARC.

**Mini-example** (textual ASCII):
Input:                 Output:
7 0 0                  7 7 7
7 0 0                  7 0 0
7 7 7                  7 0 0

**Recognition signals**:
- Pixel count and colors are preserved exactly.
- Output bbox dimensions may swap.
- The transform is discrete and global, not deforming.
- Demos usually make the chosen transform unambiguous.

**ONNX implementation pattern**:
- Ops: small branch bank from `Transpose` and `Slice` for 90°/180°/270°/H-flip/V-flip.
- Opset: 13 conservative floor.
- Dtype assumptions: `float32` one-hot planes.
- Cost: approximately `(6·C·H·W)·b + 0` for a six-branch bank.
- Sketch:
  ```python
  rot90 = Slice(Transpose(X, perm=[0, 1, 3, 2]),
                starts=[0, 0, 0, -1], ends=[1, C, W, -H-1],
                steps=[1, 1, 1, -1])
  Y = rot90
  ```

**Alternative patterns**: If only one transform is hypothesized at compile time, instantiate just that branch and drop the bank.

**Known sources**: Chollet’s ARC framing plus the ARC living survey’s geometric-transformation category. The ONNX mapping is inferred.

### Family: Same-different and boolean overlay

**Definition**: These tasks depend on whether two structures are equal or different, or they build outputs from boolean relations such as intersection, union, or complementary overlap. ConceptARC names this family `SameDifferent`, and it sits naturally inside ARC’s relational/object-centric reasoning.

**Mini-example** (textual ASCII):
Input:                 Output:
1 1 0                  0 1 0
0 1 1
0 0 0

**Recognition signals**:
- Output may be smaller or sparser than either operand.
- Equality or overlap of substructures is the operative cue.
- Colors can become binary or canonicalized.
- Demos often show “same => one result, different => another” or “shared cells only.”

**ONNX implementation pattern**:
- Ops: alignment to common frame, `Equal`, `And`, `Not`, `Max`, then `Where` for the chosen boolean result.
- Opset: 13 conservative floor.
- Dtype assumptions: boolean masks or `float32` binary planes.
- Cost: approximately `(3·C·H·W)·b + 0`.
- Sketch:
  ```python
  inter = And(mask_a, mask_b)
  diff  = And(Max(mask_a, mask_b), Not(inter))
  Y = Where(mode_is_intersection, inter_color, diff_color)
  ```

**Alternative patterns**: If only a scalar verdict is needed, reduce the equality mask to a single predicate and gate between two fixed output templates.

**Known sources**: ConceptARC `SameDifferent`; ARC’s relational/object-centric reasoning categories. The ONNX mapping is inferred.

### Family: Top-bottom stacking and occlusion

**Definition**: This family resolves which object is “on top,” how overlapping structures compose, or how a 2D rendering should encode a top/bottom or near/far relation. ConceptARC exposes this family through `TopBottom2D` and `TopBottom3D`, making it one of the clearest public examples of layer reasoning in the ARC domain.

**Mini-example** (textual ASCII):
Input:                 Output:
1 1 1                  1 2 1
0 2 0                  0 2 0
0 2 0                  0 2 0

**Recognition signals**:
- Overlap or implied overlap matters.
- One layer has precedence in shared cells.
- Output preserves objects but changes visible composition.
- Demos often suggest a stable visibility rule like “middle object is on top.”

**ONNX implementation pattern**:
- Ops: build per-layer masks, then cascade `Where` from top precedence down.
- Opset: 13 conservative floor.
- Dtype assumptions: one-hot or binary `float32` planes.
- Cost: approximately `(3·C·H·W)·b + 0` for the two-layer case.
- Sketch:
  ```python
  visible = Where(top_mask, top_layer, bottom_layer)
  Y = visible
  ```

**Alternative patterns**: If “top” is equivalent to a scalar property like smaller row index or larger area, compute that property first, then order the `Where` cascade accordingly.

**Known sources**: ConceptARC `TopBottom2D` and `TopBottom3D`. The ONNX mapping is inferred.

## Cross-cutting ONNX primitives

The operator semantics and version histories below come from the official ONNX operator documentation and ONNX versioning rules. The “opset floor” is a **conservative engineering floor** for these ARC patterns, not necessarily the earliest historical version in which every individual operator ever appeared.

| Primitive | Purpose | Core ops | Conservative opset floor | Upper-bound score term |
|---|---|---|---|---|
| Color masking and compositing | Convert palette IDs to planes, test equality, overlay candidate colors or branches. | `OneHot`, `Cast`, `Equal`, `Where`, `Max`.  | 13 | `≈ 2·C·H·W·b + 0` |
| Projection, bbox, and crop | Recover active rows/cols, find first/last support, crop tight bounding boxes. | `ReduceMax`, `ArgMax`, `Slice`, `Shape`, `Pad`, `Add`, `Sub`, `Div`.  | 13 | `≈ (2·C·H·W + H + W)·b + 0` |
| Coordinate and periodic indexing | Build coordinate grids, decode flat indices, and repeat motifs by modular arithmetic. | `Range`, `CumSum`, `Mod`, `Gather`, `Reshape`, `Transpose`, `Expand`, `Tile`.  | 13 | `≈ (C·H·W + H + W)·b + 0` |
| Morphology and bounded propagation | Local support, dilation, min/ max neighborhood propagation, axis/diagonal growth. | `MaxPool`, `Conv`, `Neg`, `And`, `Not`, `Less`, `Greater`.  | 13 | `≈ (K·H·W)·b + kernel_params`, with `K` equal to the chosen unroll depth |
| Connected components and label stats | Exact-ish component IDs without `NonZero`, plus per-label counts/properties. | `Range`, `Reshape`, `Cast`, `Neg`, `MaxPool`, `Where`, `ScatterND`, `ReduceSum`, `ArgMax`, `ArgMin`, `TopK`, `Mul`.  | 18 | `≈ (H²·W² + H·W)·b + 0` |
| Layout, shift, and scale | Shift extracted patches, place them into slots, or upscale them by integer factors. | `Pad`, `Slice`, `Resize`, `Concat`, `Where`, `Max`.  | 13 | `≈ 3·C·H·W·b + 0` |

A few implementation notes fall out of these primitives. `Conv` and `Resize` are most straightforward on `float32` tensors, `MaxPool`-based min-propagation requires `Neg`, and palette recovery is easiest by channel-wise `ArgMax` after all one-hot edits are done. ONNX is also strongly typed, so mixed arithmetic between index tensors and working masks should be made explicit through `Cast` rather than relying on framework-style implicit promotion.

## Open-research families and decision tree

### Open-research families

The families above are the ones I think are most defensible as a practical ARC→ONNX library under the card’s constraints. That said, several ARC-style families remain **implementable only expensively**, or **not efficiently known at all**, when the graph language forbids dynamic looping and sparse index discovery. The frontier cases below are my inference from ARC taxonomies, RE-ARC object generators, and ONNX’s static-graph/operator-set constraints.

- **Latent correspondence and analogy transfer**. Examples include “learn the relation between object pair A/B and apply it to C/D,” where the hard part is discovering correspondences rather than applying a known transform. Efficient ONNX is unknown because the natural solution space is combinatorial over matchings, alignments, and transform hypotheses, especially when object count is variable.
- **Deep multi-stage composition with data-dependent control flow**. ARC Prize papers and surveys repeatedly point to compositional reasoning as the hardest ARC regime. A fixed acyclic ONNX graph can unroll a *chosen* sequence, but there is no efficient generic graph for “discover how many steps to take, and which operation to apply next,” especially without `Loop` or `Scan`.
- **Exact flood/CC reasoning in tortuous geometries at low cost**. The min-propagation and flood-fill tricks above are workable for 30×30 grids, but their conservative graph footprint grows like `H²·W²`; that makes them defensible for a candidate library, not elegant or truly efficient. A materially better static-graph exact formulation remains open.
- **Minimal 2D motif induction with phase ambiguity, holes, and multiple plausible periods**. The `CumSum` + `Mod` + `Gather` pattern is strong once a period is fixed, but discovering the *minimal* 2D motif under missing data and competing candidate phases still tends to require explicit search or large unrolled branch banks. That is feasible, but not yet a satisfying “efficient ONNX pattern.”
- **General packing, assignment, and collision-free layout**. Sorting a few objects into slots is manageable, but globally solving “place an arbitrary number of objects under constraints so they do not overlap and satisfy a relation” quickly becomes a combinatorial packing problem. Without sparse discovery ops and dynamic control flow, no efficient generic ONNX construction is known to me.

### Decision tree

This is an engineering decision tree synthesized from the ARC sources above rather than copied from one paper. It is meant to narrow the candidate-family set before graph instantiation.

1. **Does the output become tighter or smaller?**
   If yes, start with **Extract and crop objects**. If one specific object survives and the background is stripped away, this family should be near the top of the list.

2. **Does the output keep the same objects but move them?**
   If yes, consider **Alignment to center or anchor**, **Above-below relative placement**, **Move to boundary or target**, or **Order and serialize objects**. The key discriminator is whether the move is single-object, relational, edge-seeking, or a full re-layout.

3. **Does geometry stay mostly fixed while colors change?**
   If yes, test **Recolor by component property**, **Inside-outside recolor**, or **Fill-state transform**. Border contact, area rank, and interior/exterior status are the biggest tells.

4. **Are new cells added along existing lines or toward walls?**
   If yes, prioritize **Extend or connect to boundary**. A telltale signature is collinearity and edge contact.

5. **Do exact duplicates of one motif appear in the output?**
   If yes, test **Copy and replicate**. If the number of copies depends on an integer summary rather than visible markers, also test **Count-driven synthesis**.

6. **Does the output look like a rotated, mirrored, or orientation-normalized version of the input?**
   If yes, test **Rotation and reflection** or **Horizontal-versus-vertical orientation**. Whether the branch depends on aspect ratio is the distinguishing cue.

7. **Does the output complete a regular pattern or missing half?**
   If symmetry is the cue, test **Complete shape and symmetry completion**. If repetition or modular continuation is the cue, test **Periodic tiling and motif extrapolation**.

8. **Does overlap or layer precedence matter?**
   If yes, test **Same-different and boolean overlay** for pure logical combination, or **Top-bottom stacking and occlusion** for visible-layer composition.

9. **Is an integer statistic visibly controlling length, multiplicity, or density?**
   If yes, elevate **Count-driven synthesis**, and, if sorting or ranking is also visible, **Order and serialize objects**.

In practice, I would score candidate families in exactly this order: **size-change first**, then **pure translation/layout**, then **same-geometry recolor**, then **growth/completion**, then **copy/count**, then **geometric transforms**, then **overlap/layering**, reserving the open-research families for cases where none of the tractable families fit cleanly. That ordering is an inference, but it tracks both ConceptARC’s clean concept separations and Hodel-style object-centric decomposition.

## Verification checklist

- Covered **18 distinct ARC rule families**, which is within the requested 15–25 range.
- Every family includes a **definition**, **ASCII mini-example**, **recognition signals**, **recommended ONNX pattern**, **opset floor**, **dtype assumptions**, **symbolic cost**, **tiny sketch**, **alternative pattern**, and **known sources**.
- No family uses any banned operator: **`Loop`**, **`Scan`**, **`NonZero`**, **`Unique`**, or **`Compress`**.
- All cost expressions are stated symbolically in terms of **`H`**, **`W`**, **`C`**, and **`bytes_per_element`**.
- The note explicitly labels **open-research families** where no efficient fixed-graph ONNX pattern is known or where only expensive static formulations are available.
- The ONNX operator semantics and conservative opset floors are cited from the **official ONNX operator specs** and **ONNX versioning docs** in the primitives section.
- Taxonomy claims are grounded in **Chollet’s ARC paper**, **ARC Prize docs/reports**, **ConceptARC**, **RE-ARC**, **Hodel’s ARC-DSL**, and **competition-era public writeups/papers**.


---

## [completeness sweep 2026-06-10] snapshot lines not carried into live or the sections above

> Restored verbatim from the pre-refactor snapshot after a containment audit found the
> compression pass had dropped them from both live and this file. Mostly MAC-era points math
> and dtype-packing claims (dead under current scoring — see scoring_internals.md) plus
> stray prose; preserved here so nothing exists only in the snapshot. Lines appear in
> snapshot order, context-free; the snapshot has the surrounding text.

<!-- Phase-2: empirical-coverage spine merged as front-matter; alt taxonomy archived (won 3/23 head-to-head; its 3 winning rows noted in the coverage table). -->
| task118 (already counted above) | — | — | — | — |
| task158 (already counted above) | — | — | — | — |
