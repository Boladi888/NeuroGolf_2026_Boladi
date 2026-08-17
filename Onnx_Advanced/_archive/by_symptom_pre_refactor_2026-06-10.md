# by_symptom.md — flat exemplar index by graph signature

You see a symptom in your graph. Find it below; jump to the matching exemplar.

This file is the COMPLEMENT to `task_techniques_index.md`. The router maps **task ID → exemplar**. This maps **graph pattern → exemplar**. Use whichever entry point matches what you have.

---

## By rule-decode stage (READ FIRST when working on a NEW task)

### SYMPTOM: I don't know what RULE family this task belongs to

→ `arc_rule_taxonomy.md` — 18-family taxonomy with definitions, recognition signals, recommended ONNX patterns per family, and a Mermaid decision tree for unknown puzzles. Synthesized from Chollet 2019, ARC Prize 2024/2025, ConceptARC, BARC, Hodel ARC-DSL, RE-ARC.

→ `_archive/arc_rule_taxonomy_alt.md` (archived — primary covers it) — alternative 15-family output-shape taxonomy (`Center`, `Copy`, `Count`, `InsideOutside`, `MoveToBoundary`, etc.). Use if the primary doc's families don't fit.

The taxonomy short-circuits Phase 1 (rule decoding) — instead of inventing a numpy rule from scratch, walk the decision tree to identify the candidate family in <15 minutes, then use that family's recommended ONNX pattern as the starting template.

### SYMPTOM: I have the rule decoded but don't know what ONNX ops to use

→ The per-family entries in `arc_rule_taxonomy.md` include ONNX implementation patterns with code sketches per family.
→ For specific ops not in your vocabulary, see `op_cost_patterns.md (uncommon-ops salvage)` (quick-ref) and the dedicated op files in `Research/`.

---

## By cost band

### SYMPTOM: cost > 200,000

→ Probably has many F32 `[1, 10, 30, 30]` intermediates OR uses MatMul/Gemm with large weights.
→ Likely actions:
- `crop_to_active_grid` if max active grid < 30 (read `cost_floor_reference.md` to confirm worth it).
- F16 placement on intermediate tensors (see exemplars below).
- For algorithm-family rewrite, see `01_whole_graph_rewrites/`.

### SYMPTOM: cost 30,000-200,000 (most polished tasks)

→ This is the "post-F16-placement, post-onnxsim" band. Further savings need structural changes.
→ Read `01_whole_graph_rewrites/task376_input_to_bool.md` first — if eligible, biggest leverage.
→ Otherwise per-task radical rewrite (see `09_periodicity_detection/task110_periodicity.md` as case study).

### SYMPTOM: cost < 5,000 (near-floor)

→ DO NOT DISMISS. task299 dropped a 2,678-cost task to 1,058 (60% reduction) via bool-Concat keeping.
→ Open the graph. Look for bool-Concat / Pad chains. If present:
→ `exemplars/02_bool_first/task299_bool_concat.md`

---

## By LIVE architecture signature (added 2026-05-23 — solve-from-scratch unlock pattern)

### SYMPTOM: LIVE has `[N≈266, K]` lookup table init + cherry13b bulk-tested at 0.00 LB delta

→ LIVE-contribution = 0 confirmed (LIVE memorized visible, fails Kaggle hidden).
→ Cost compression of LIVE-equivalent variants is **dead lever** — proven via bulk cherry13b 2026-05-23T19:45Z (11 tasks × cost-0 stub = 0.00 LB delta).
→ **Path to LB gain**: decode rule + TRUE structural rewrite (no `[N≈266, K]` tables, no trained weights) + ship.
→ **2 confirms (100% rate)**:
   - task066 wallpath_rule (21,408 → 11,769) = **+15.62 LB** at 2026-05-23T18:33Z
   - task277 bbox_area_min_edges (46,539 → 16,748) = **+15.27 LB** at 2026-05-23T20:50Z
→ Both landed at exact theoretical max: `25 - log_e(new_cost)` (= 15.62 and 15.27 respectively).
→ Family ceiling: 10 tasks still at LIVE-contribution=0 (76, 96, 101, 118, 158, 219, 233, 285, 363, 367) × +12-17 LB each = **+120-150 LB ceiling** if pattern reproduces.
→ Required reading: `traps.md §14e` 2026-05-23T20:50Z entry + `The_400_Tasks/Task_by_Task/task066.md` SHIPPED entry + `task277.md` SHIPPED entry.

### SYMPTOM: LIVE is a small trained MLP (3+ MatMul with F16 weight tables) + cherry13b=0

→ Same lookup-LIVE pattern but architecture is gradient-descent-distilled instead of explicit `[N, K]` table.
→ Same dead-lever / unlock pattern applies: structural rewrite (no trained weights) = +12-17 LB.
→ task277 example: LIVE was `v7 [21, 264]` + `v10 [264, 52]` + `v11 [52, 150]` 3-layer F16 MLP; bbox/area duplicate detector replaced it cleanly at +15.27 LB.
→ NN/MLP/CAE family is closed by *policy* (not by zero-weight proof). Decode rule + ship structural.

### SYMPTOM: structural-LIVE (rule-shaped algorithm, no [N≈266,K] table) but cost > 100k

→ NOT a solve-from-scratch task — LIVE already contributes positive LB.
→ Expected gain = **raw log_e × 1.0×** exact match (post-2026-05-23 calibration).
   - task092 (198,818 → 4,352): predicted +3.82, actual +3.83 LB
   - task005 (183,854 → 16,738): predicted +2.40, actual +2.40 LB
→ Formula: `delta_LB = log_e(LIVE_cost / new_cost)` — workers can quote this number with ±0.05 LB accuracy.
→ Don't bank on solve-from-scratch unlock surprises (task066/277 pattern doesn't apply when LIVE is genuinely structural).
→ Vectors: Einsum projection compression (task170/092), separable masks, F16 cast placement (task055), compact bbox containers replacing full masks (task005 v40).

---

## By graph shape

### SYMPTOM: graph has Concat with multiple `Cast(bool, F32)` inputs

→ All Concat inputs are bool-eligible? → KEEP THE CONCAT IN BOOL.
→ `exemplars/02_bool_first/task299_bool_concat.md` (rule 7 of bool-first family)

### SYMPTOM: graph ends with `Concat → output_f16 [F16] → Cast(F32) → output`

→ F16-output trick — drop the final Cast.
→ `exemplars/03_crop_and_f16/task384_f16_drop.md`
→ Stacks beautifully with crop-to-active-grid (`task117_crop15_f16.md`).

### SYMPTOM: graph has `Mul(Cast(bool, F32), Cast(bool, F32))`

→ `bool_andfold` rule 1.
→ `exemplars/02_bool_first/task293_andfold.md`
→ Detector: `pass_bool_andfold.py` auto-applies.

### SYMPTOM: graph has `Mul(Cast(bool, F32), V)` (mask × value)

→ `mul_mask_to_where` rule 2 → rewrite as `Where(bool, V, 0)`.
→ `exemplars/02_bool_first/task021_mul_mask_to_where.md`

### SYMPTOM: graph has `Cast(bool, F32) → Pad → output`

→ Drop the Cast; let Pad operate on BOOL (ORT-specific behavior).
→ `exemplars/02_bool_first/task273_drop_pad_cast_bool.md` (rule 5)

### SYMPTOM: opset-10 with `Pad(attr form) + Cast(bool, F32)`

→ Bump opset 10→11 and switch Cast target to UINT8.
→ `exemplars/02_bool_first/task319_opset_bump_pad_uint8.md` (rule 6)

### SYMPTOM: parallel `Sub→Mul(self)→Less<1` chains

→ Equal-fold + share Cast across N chains.
→ `exemplars/02_bool_first/task368_shared_cast_equal_fold.md`
→ Pointwise variant: `exemplars/02_bool_first/task233_equal_fold.md`

### SYMPTOM: many `Add` nodes in a chain (each Add's first input = prior Add's output)

→ `add_chain_to_sum` — collapse to one `Sum` node.
→ `exemplars/04_algebraic_identities/task327_add_chain_to_sum.md`
→ Auto-detector: `pass_add_chain_to_sum.py`.

### SYMPTOM: max active grid < 30, but graph operates at 30×30 throughout

→ Crop-to-active-grid + dimension-tied init rewrites + Pad back.
→ `exemplars/03_crop_and_f16/task117_crop15_f16.md` (the canonical cornerstone)
→ Also see other crop variants: task128 (H=15), task281 (H=13), task364 (H=22).
→ **WARNING**: if first node is already a Slice to a small region, this is a false positive — pre-crop adds overhead. See `traps.md` §21.

### SYMPTOM: graph Gathers individual channels and does F16 ops on them

→ F16 placement: cast PER-CHANNEL, not pre-cast the full input.
→ `exemplars/03_crop_and_f16/task055_per_channel.md`

### SYMPTOM: graph has many duplicate Slice nodes with identical params

→ Slice CSE — onnxsim misses these on NeuroGolf graphs.
→ `exemplars/06_cse_dead_op/task306_slice_cse.md`
→ Auto-detector: `pass_slice_cse.py`.

### SYMPTOM: `Relu(Sub(MaxPool(X, sym-pad, stride=1), X))`

→ Provably dead Relu. Drop it.
→ `exemplars/06_cse_dead_op/task093_dead_relu_post_maxpool.md`

### SYMPTOM: 2D mask init where every row is identical (or every column is identical)

→ `separable_mask_collapse` — store as 1D, broadcast at use.
→ `exemplars/01_whole_graph_rewrites/task115_axis_factor.md` (runtime version)
→ Auto-detector for static-init case: `pass_separable_mask_collapse.py`.

### SYMPTOM: `Slice(axis=A) → Slice(axis=B)` chain with A ≠ B

→ Merge into one `Slice(axes=[A, B])`.
→ Auto-detector: `pass_slice_chain_to_multiaxis.py`.

### SYMPTOM: large lookup table at `[..., H, W]` shape with computable structure

→ Static table → dynamic compute at rank-2.
→ `exemplars/05_static_to_dynamic/task071_ring_mask.md` — the rank-2-vs-rank-4 insight.

### SYMPTOM: `template_sets [N1, N2] INT64` initializer dominates params (N1, N2 ≈ 10²–10⁴)

→ Template-vote → sparse-stamps decomposition. Transpose data layout from per-output-position to per-input-pair.
→ `exemplars/01_whole_graph_rewrites/task366_template_vote_to_sparse_stamps.md`
→ **Always run cherry13b "replace with Identity" diagnostic FIRST** — if the task has zero public weight, the technique transfers but the LB gain doesn't.

### SYMPTOM: graph needs all-zero OneHot output when no pattern matches

→ Multiply OneHot `on_value` by `ReduceMax(matches)` instead of using `Where` mask before OneHot.
→ `exemplars/01_whole_graph_rewrites/task366_template_vote_to_sparse_stamps.md` sub-technique 2.
→ Saves ~3,600 bytes per "no-match-fallback" requirement.

### SYMPTOM: many `padslice_shift_conv` pairs producing perpendicular shifts of a sparse mask

→ Project the source mask to its perpendicular axis FIRST via `ReduceMax(axis=ray_axis, keepdims=1)`, then shift on the compact `[1, 1, 1, W]` or `[1, 1, H, 1]` tensor. Materialize full grid only at the broadcast point.
→ `techniques.md §22` (compact-mask projection for perpendicular thickening)
→ `exemplars/01_whole_graph_rewrites/task280_padslice_to_compact_masks.md`
→ Saves `H×` (or `W×`) per perpendicular shift. **task280 saved -15,200 bytes** via this.

### SYMPTOM: multi-object detector using shift-based thickness check

→ Use INCREMENTAL contiguous AND chain instead of naive `And(edge, shift_K(rect))`. Naive check has silent cross-rectangle false positives that can pass visible verify but break on multi-object hidden inputs.
→ `traps.md §14b`
→ `exemplars/01_whole_graph_rewrites/task280_padslice_to_compact_masks.md` sub-technique 4

### SYMPTOM: graph uses opset 11 `Cast(F16→F32) → CumSum(F32) → Cast(F32→F16)` roundtrip

→ Bump opset 11→14. F16 CumSum is supported natively. Saves the F32 intermediate + 2 Cast ops per CumSum (~3,200 B per block on NeuroGolf shapes).
→ `Ort_Advanced/ort_opset_support.md` (CumSum row)
→ **Common falsified claim**: "F16 CumSum is unsafe if output feeds Gather as indices" — verify the Gather connection first. `Gather(data, indices)` only cares about precision on the `indices` input. F16 data + INT64 indices is exact. task280 v1 saved -25,600 B after debunking this wrong claim.

### SYMPTOM: algorithm processes per-cell color (not per-channel BOOL), uses one-hot `[1, 10, H, W]` throughout

→ Switch to scalar color-index `[1, H, W] F16` representation via ArgMax+Cast at entry. Materialize one-hot only at output via `Equal(out_idx_f16, channel_values_f16 [1, 10, 1, 1]) → BOOL`. Skip INT64 OneHot index cast entirely.
→ `techniques.md §23`
→ Empirical: task138 v2 went 150,728 → 58,877 (-61%).
→ Trap: don't use `ReduceSum(masked_color_index)` for marker recovery — sums repeated positions; use ReduceMax or per-color presence.

### SYMPTOM: graph has paired `Conv(sum_kernel) == W` + `Conv(boundary_kernel) == 0` then And

→ Merge into single Conv with negative-flanked kernel `[-1, 1×W, -1]` of length W+2 + single `Equal(conv_out, W)`. The merged Conv computes `−boundary_left + sum(W cells) − boundary_right = W` iff both conditions hold.
→ `techniques.md §24`
→ Empirical: task349 v5 saved ~18,000 B across 5 W-iterations.

### SYMPTOM: graph uses F16 MaxPool on 0/1-valued tensors

→ Bump opset 10/11 → 12. `MaxPool(UINT8)` is accepted at opset 12+ (rejected at 10, 11; BOOL rejected at all opsets). Cast to UINT8 throughout the MaxPool chain.
→ `techniques.md §1 Rule 11` UINT8 MaxPool variant.
→ Empirical: task025 saved ~76k bytes via opset 12 + UINT8 MaxPool on directional propagation.

### SYMPTOM: graph has 4-8 directional Pad+Slice probes per axis × distance

(Pattern: testing "is there a marker at distance ≤K in direction D" by Pad+Slice pairs per direction)

→ Replace each `Pad+Slice` chain with a single `MaxPool` having an asymmetric kernel + pad. For "distance ≤K to EAST": `MaxPool(kernel=[1, K+1], pad=[0,K, 0,0])`. Mathematically equivalent: MaxPool with right-pad K and kernel width K+1 at position (r,c) sees max over (r, c..c+K) — True iff any cell in that range is set.
→ Keep mask logic in BOOL/UINT8 (Rule 11 dtype, opset 12+).
→ This is LIVE-equivalent structural compression (NOT algorithm replacement).
→ Empirical: **task383 saved -97,125 B (+1.01 LB CONFIRMED Kaggle 2026-05-22)** via directional Pad/Slice → MaxPool windows. See `.update/DONE_2026-05-22T204400Z_task383_directional_padslice_to_maxpool.md` for surgery walkthrough.

### SYMPTOM: graph output is one-hot BOOL/UINT8 after final ops but has F32 declaration

(Pattern: graph ends with `... → Equal/And/Or → Pad → Cast(F32) → output` or similar)

→ Drop the final `Cast(F32)` and declare graph output as BOOL or UINT8. `run_network`'s `(out > 0.0).astype(float)` thresholding is identity on {True, False}.
→ This is the **Track B** generalization of `task376_input_to_bool` — works even when internals contain Conv, MatMul, F32 arithmetic. Only the output declaration needs to switch dtype.
→ `techniques.md §3` F16-output / BOOL-output trick.
→ Empirical:
  - task376 (MID_GATHER, no Conv on input): Track A whole-graph BOOL, +0.55 LB
  - **task069 (Crop+Conv+ConvTranspose internals): Track B output-only BOOL, +2.54 LB CONFIRMED Kaggle 2026-05-22**

### SYMPTOM: graph enumerates `[N, N]` pair-validity matrix when only K active channels can contribute

→ Active-subset projection: select active K via ArgMax/Gather first, then enumerate only `[K, K]` cross-product. Deletes the large pair matrix initializer + intermediates.
→ `techniques.md §25`
→ Empirical: task153 v2 saved -173k bytes (177k → 4k) via 2-active-color projection on a 9×9 pair matrix.

### SYMPTOM: memorized-lookup classifier uses `Flatten(input) → MatMul(big_weight)` for hashing

→ Replace with `ReduceSum(input, axes=[2,3]) → MatMul(small_weight)` when inputs are uniquely identifiable by color histogram. Verify with `np.unique(histograms, axis=0).shape[0] == n_examples`. Critical: ensure projection `min_gap > sqrt(threshold)`.
→ `techniques.md §26`
→ Empirical: task158 v2 saved -45k via this.

### SYMPTOM: many `Cast(BOOL → F32) + Sum + Greater(>0)` accumulating "is any mask active"

→ Replace with `Or(Or(b0, b1), b2)...` chain in BOOL. Use the Or-result directly in `Where(BOOL, F32, F32)` (NOT in `Where(BOOL, BOOL, BOOL)` — ORT rejects).
→ `techniques.md §27`
→ Empirical: task158 v2 saved -54k.

### SYMPTOM: per-cursor row-range AND col-range cursor check creates two `[H, W] BOOL` intermediates per cursor

→ Reorder And-chain: compute `row_range [H, 1]` and `col_range [1, W]` separately, then `And(row_range, col_range) [H, W]` ONCE per cursor. Commutativity/associativity preserves semantics.
→ `techniques.md §28`
→ Empirical: task158 v2 saved -16k via this on 18 cursors.

### SYMPTOM: ScatterND needs NOOP slot but every output position is valid foreground

→ Extend canvas to N+1, use position N as NOOP sentinel, Slice off after ScatterND.
→ `techniques.md §29`
→ Empirical: task182 v1 used this.

### SYMPTOM: memorized-lookup discriminator MatMul + F16 conversion gives 0/267 verify

→ Hash sums exceed F16 exact-integer ceiling (2048). Keep discriminator in F32. Cast to F16 ONLY after the threshold check.
→ `techniques.md §30` + `Ort_Advanced/ort_quirks.md §11`
→ Empirical: task182 F16 conversion failed catastrophically at sums ~25k.

### SYMPTOM: graph needs both per-channel slices AND a single `Max`/`Or` across channels

→ Skip the staged `[1, N, H, W]` intermediate: slice each channel directly from input, then `Max(ch1, ch2, ..., chN)`. Saves `N*H*W` bytes when downstream uses per-channel slices anyway.
→ `techniques.md §31`
→ Empirical: task187 saved -5,625 B at 25×25 / N=9.

### SYMPTOM: model's final output assembly is `Cast(BOOL→F16) → ReduceSum(channel-axis) → Greater → Concat → output (F16)`

→ INT8 spatial chain at opset 12+: Cast BOOL→INT8, ReduceMax(INT8, axis=1), Greater(INT8, int8_zero), Concat(INT8, INT8) → output INT8. NeuroGolf decoder threshold `> 0.0` treats INT8 {0,1} identically to F16. Saves -9,900 B per pipeline on [1, 9, 30, 30].
→ `techniques.md §1 Rule 11` INT8 output assembly variant.
→ Empirical: task092 v2 saved -9,900 B; combined with Or-fold INT8 ReduceMax = -15,300 B total.

### SYMPTOM: wide `[1, C, H, W]` intermediate consumed only by reductions/gathers

→ Projection-cache: compute reductions from input directly, skip the intermediate.
→ `techniques.md §12`
→ Cross-ref `exemplars/05_static_to_dynamic/task201_reduce_reorder.md`.

### SYMPTOM: long Add-chain ending in ReduceSum, or geometric position computation

→ Replace with CumSum (opset 11+).
→ `exemplars/01_whole_graph_rewrites/task382_cumsum_gather.md`

### SYMPTOM: Conv with 0/1 weights (counting/neighborhood)

→ Reformulate as bool counting via Slice/Concat/ReduceSum.
→ `exemplars/07_conv_reformulation/task323_conv_bool.md`

### SYMPTOM: Conv followed by `Mul(per-channel constant)` + linear ops

→ Fold downstream linear ops into Conv weights.
→ `exemplars/07_conv_reformulation/task374_conv_run_combined.md`

### SYMPTOM: pattern detection via Conv chain + AND-of-thresholds

→ Single Conv with custom kernel of the pattern's exact shape.
→ `exemplars/01_whole_graph_rewrites/task119_diag_conv.md`

### SYMPTOM: ScatterND with F32 0/1 output canvas

→ Switch data + updates to BOOL (opset 11+).
→ `exemplars/08_gather_lookup/task118_bool_scatter.md`

### SYMPTOM: graph is one-hot input → Gather/Reduce chain (no Conv, MatMul, F32 arithmetic on input)

→ `input → BOOL` whole-graph propagation — THE highest-leverage technique.
→ `exemplars/01_whole_graph_rewrites/task376_input_to_bool.md` (rule 8 of bool-first family)
→ -18,210 bytes, +0.55 LB on task376.

---

## By algorithm class

### SYMPTOM: rule involves periodic patterns / repeating tiles with missing cells

→ Periodicity detection + iterative fill.
→ `exemplars/09_periodicity_detection/task110_periodicity.md` — full algorithm template.

### SYMPTOM: rule involves bounding-box of an object

→ `ReduceMax(axes=[2]) + ReduceMax(axes=[3])` → row/col presence → outer-product bbox.
→ `exemplars/01_whole_graph_rewrites/task049_least_color_bbox.md`
→ `Algorithm_Library/algorithm_patterns.md §2`.

### SYMPTOM: rule is color permutation π

→ Single `Gather(input, π_inv, axis=1)`.
→ Threshold: require ≥ 7 of 10 colors observed.
→ `Algorithm_Library/algorithm_patterns.md §1`.

### SYMPTOM: rule depends on "is foreground / non-bg" mask

→ Use `ReduceSum-Sub-Greater` (NOT `1-ch0`).
→ See `core.md` "1-ch0 trap" + `traps.md` §9.

### SYMPTOM: rule needs whole-grid summary (hash of entire input grid)

→ Single Conv with weights = `2^(i*W+j)` at active positions.
→ `Algorithm_Library/algorithm_patterns.md §9`.

### SYMPTOM: rule needs to count cells / find mode color

→ Per-channel histogram via `ReduceSum(input, axes=[2,3], keepdims=False)` → `[1, 10]` → ArgMax.
→ `Algorithm_Library/algorithm_patterns.md §8`.

### SYMPTOM: every visible pair has the same input/output shape

→ Single-fixed-shape hardcoded transform via `Gather` or `Transpose`. Cost 0.
→ `Algorithm_Library/algorithm_patterns.md §6`.

### SYMPTOM: task uses connected-component analysis

→ Approximations are silent-fail-prone (Unique is banned). See `traps.md` §13.
→ For "isolated cell" rules, verify 4-conn vs 8-conn from ≥ 3 visible pairs first.
→ `Algorithm_Library/algorithm_patterns.md §7`.

### SYMPTOM: stamp-forest of N Conv/ConvTranspose detectors for rectangle/shape enumeration

(Pattern: 20-50+ Conv kernels each detecting a specific rectangle size or shape variant, fed into a large selection step.)

→ Replace the enumeration with **MaxPool prefix/suffix scans**: pool the blocker mask with kernels covering each candidate dimension; a pooled 0 marks a top-left where the rectangle fits entirely in the background. Pick coordinate via `ReduceMin/ArgMin` over (row, col) keys.
→ Empirical: **task090 saved -55,134 B (+2.02 LB Kaggle confirmed 2026-05-23)** via 23 UINT8 MaxPool blocker pools replacing 246-node F32 Conv/ConvTranspose stamp forest. **task145 saved -65,101 B (+1.34 LB)** via 4 MaxPool prefix/suffix boundary scans replacing explicit run chains.
→ Cross-ref: `traps.md §14e` for the bitpack-pruned-to-visible-set hidden-bad alternative (e.g., task090's `bitpack_u32_pruned` at 4,984 was a -13 LB hidden-bad family member — skipped in favor of `reduce_argmin_coords` at 8,474 with real algorithm).

### SYMPTOM: geometric ray/diagonal drawing from anchor blocks

(Pattern: input has a small geometric anchor — a 2x2 block, a corner cell, a marker — and output draws rays/diagonals/extensions from it using F16/F32 Conv with custom kernel weights.)

→ Use a **single QLinearConv with documented integer kernel weights**: encode the geometric extension directly as kernel weight pattern (e.g., weight 4 at the anchor, weight 1 on each diagonal footprint cell), then `Greater(score, threshold)` to get the foreground bool mask. Cap each footprint cell at 1 to prevent overlap-induced false positives; reserve high integer weights only for the anchor.
→ QLinearConv keeps channel buffers/scores at UINT8, saving ~400 B vs equivalent F16 Conv path. Compose multiple QLinearConvs (one per output color) via `Or` + `Not` + `Concat` + BOOL `Pad`.
→ Empirical: **task136 saved -10,631 B (+1.79 LB Kaggle confirmed 2026-05-23)** via 2 QLinearConv kernels with diagonal block-footprint weights replacing the F16 anchor-detect + 9x9 ray-Conv approach.
→ Don't-do (per task136.md): sparse-kernel initializers verify in ORT but the official scorer rejects them (`score_network_v2` returns `None`). QLinearConv zero-point inputs cannot be omitted as empty optional inputs. Opset-10 attribute `Pad` rejects BOOL — use opset-13 `Pad` for BOOL output.

### SYMPTOM: large-receptive-field detector with sparse pattern (9×9, 13×13 dense kernels)

(Pattern: graph uses 9×9 / 13×13 / larger dense Conv/ConvTranspose kernels to detect specific distance/anchor relationships. Most kernel weights are zero or unused because the pattern is sparse.)

→ Replace dense kernel with **5×5 dilated** Conv (dilation=2 covers 9×9 receptive field; dilation=3 covers 13×13). Params drop from 81/169 → 25.
→ For grouped per-class detectors, use **grouped QLinearConv** so each output channel gets its own small dilated kernel rather than one shared dense kernel.
→ Empirical: **task042 saved -21,769 B (+2.13 LB Kaggle confirmed 2026-05-23)** via grouped QLinearConv with dilated 5×5 kernels for n=2/n=3 pair-distance detectors, replacing dense 9×9/13×13. 92.1% generic adv + 95.6% task-shaped adv.
→ Don't-do (per task042.md): sparse QConv weights run/verify in ORT but `shape_inference` strict mode rejects them (scorer can't type them) — dense dilated is the only scoreable form.

### SYMPTOM: Conv used as position-encoder rather than feature-detector (marker → cell-index)

(Pattern: input has unique anchor/marker, you need to read which neighboring cell of a 3×3 patch the marker is at, then index into a small palette.)

→ Use a **3×3 Conv with position-encoding weights** `[[9,8,7],[6,5,4],[3,2,1]]` (or any unique-per-cell numbering). Convolving against the binary marker mask produces a unique integer "code" (1-9) at each output pixel, identifying which of the 3×3 neighbors held the marker. Combine with `Gather(palette, code)` or `Equal(code, k)` to look up the corresponding stamp value.
→ This pattern uses `Conv` algebraically — not as a feature-detector but as a tiny address-decoder. Saves the cost of multiple Pad+Slice shifts.
→ Empirical: **task075 saved -25,679 B (+2.46 LB Kaggle confirmed 2026-05-23)** via 3×3 `code_kernel` Conv + `ConvInteger` + `Gather` + `Equal`-against-colors final BOOL output. Cost 28,065 → 2,386.

### SYMPTOM: foreground = `1 - ch0` pattern in cropped active grid

(Pattern: graph computes a foreground mask by `Sub(1, ch0)` then casts that to F16 to use as evidence. Carries both `ch0` and `fg` intermediates.)

→ When the task contract guarantees active-grid size ≤ N and you've already cropped to [N, N], just **cast `ch0_crop` directly to F16 and use it as BACKGROUND evidence** in subsequent convolutions. Background-score kernels (high weight where ch0=1) work symmetrically to foreground-score kernels (high weight where ch0=0). Saves one bool/cast tensor + one parameter.
→ **CRITICAL: only valid AFTER cropping to active grid.** Outside the active grid (padded cells), ALL channels including ch0 are 0 → `1-ch0` mistakenly marks padding as foreground. See `core.md` "The 1-ch0 trap" and `traps.md §9`. If the task contract permits the active grid to vary in size, the crop must be dynamic.
→ Empirical: **task168 saved -26,888 B (+1.91 LB Kaggle confirmed 2026-05-23)** via `Cast(F16)(ch0_crop[10×10])` + 2×2 background-evidence Conv kernels. 500/500 task-shaped adv = LIVE-equivalent. Task168's documented contract is "every input is 10×10", so the static crop is safe.

### SYMPTOM: dense row/column projection tensors charged for `[1, C, H, 1]` and `[1, C, 1, W]`

(Pattern: graph materializes full per-channel row projections and column projections as intermediates to support downstream reductions.)

→ Replace materialized projection tensors with **`Einsum` reductions** that compute the same answer in one node without storing the full projection. Multiple reductions sharing the same input can each `Einsum` independently. MACs are no longer scored, so Einsum is cost-free relative to the inputs.
→ Empirical: **task170 saved -48,923 B (+2.25 LB Kaggle confirmed 2026-05-23)** via four `Einsum` reductions replacing the full row_proj/col_proj tensors. **task208 used Einsum row-central-moments** to derive rectangle height (part of the +3.15 LB ship).

### SYMPTOM: detect a solid K×K block of foreground in an otherwise-sparse mask

(Pattern: input has exactly one solid K×K block of foreground; you need its top-left coordinate to drive downstream geometry. Typical naive approach: K×K Conv with all-ones kernel + threshold = K², or per-channel match.)

→ Use **MaxPool with kernel K×K over ch0 (background channel)**: the pool output is 0 only where the K×K window is entirely background-free (i.e., the solid foreground block). Recover the block's top-left as `(ArgMin(ReduceMin(pool, axis=W)), ArgMin(ReduceMin(pool, axis=H)))`. No Conv weights, no threshold, UINT8 throughout.
→ Saves both the Conv weight init (K²+1 params) AND the F32 intermediate the Conv would produce. ArgMin/ReduceMin on UINT8 is supported at opset 13+.
→ Empirical: **task190 saved -8,297 B (+1.35 LB Kaggle confirmed 2026-05-23)** via `Cast(ch0_crop, UINT8) → MaxPool(2×2) → ReduceMin axis=W → ArgMin axis=H` for the 2×2 block locator.
→ Works because ch0 = "is background" semantically in NeuroGolf encoding — a foreground block has ch0=0 everywhere inside, so the MaxPool window-max is 0 exactly at the block. Pre-crop to active grid first to avoid the padding-cell trap (`traps.md §6`).

### SYMPTOM: building a dense one-hot output to overlay on an unchanged input

(Pattern: rule says "extend rays from anchor / draw mask / paint a few cells onto input"; naive graph builds the full output one-hot from scratch then writes to `graph.output`.)

→ Use **`Where(extension_mask, computed_overlay, input)` as graph output** when only a small region of the output differs from the input. The original full input one-hot becomes the unchanged branch; the unchanged-branch Where result is the graph output and not counted as intermediate memory.
→ Saves rebuilding the full N-channel output tile (typically 900-3,600 B) plus any Equal-expansion from color-index to one-hot.
→ Empirical: **task090 final-Where saved ~5,500 B** (`Where(fill_full, six_value, input)`); **task190 Where-overlay route shaved ~2,000 B** vs Equal-expansion from F16 color-index. Combined with active-grid crop, the only counted cost is the small extension/fill mask itself.

### SYMPTOM: every output channel is a row-band × col-band product (bands / sections / crosses / dividers)

(Pattern: the rule partitions rows and cols into bands — divider lines, grid sections, quadrants — and each output channel is "rows in band A × cols in band B" (possibly plus a line-cross union). Naive graph builds per-channel `[1,1,30,30]` planes and Concats 10 of them.)

→ **techniques #58b bitand bit-detectors** (preferred): one U8 BIT per (row-set × col-set) product, per-axis bitmask vectors built by nested `Where(bool, u8-scalar, u8-scalar)`; `output = BitwiseAnd(BitwiseAnd(x, K[10,1,1]), y)` — channel expansion 300 B, unions free (multi-bit K), zero planes. Band boundaries from strict-max markers → ArgMax pair instead of CumSum. Opset 18.
→ **techniques #58 staircase-code compare-form** (no-opset-18 fallback): stack per-channel U8 row codes `X [1,10,30,1]` and col thresholds `Y [10,1,30]`, make **`Greater(X, Y)` the graph output**. Unions of two outer products fit via 3-level codes. Pair either form with §42/#46 free-input Einsum marginals for the per-axis counts.
→ Empirical: **task055 31,628 → 1,174 (26.9×, #58b) / 2,386 (13.3×, #58), both byte-equal-full 200k** (2026-06-06). Construction-floor tables in `task055.md`.
→ Limit: when MANY cells differ from input, the condition mask cost grows back. Break-even is roughly when the changed region > ~25% of the output grid.

---

## By error message

- ORT error messages (load failures, type mismatches): `Ort_Advanced/ort_errors.md`
- "Model loads, wrong output": `traps.md` runtime section
- "Expected savings didn't materialize": `traps.md` scoring section

---

## When nothing here matches

Either you need a fundamentally new algorithm (read `Algorithm_Library/rule_discovery_methodology.md` + `algorithm_patterns.md`), or your task's optimization is genuinely novel — contribute via `.update/` when done. `techniques.md` is the comprehensive toolkit; browse by name as fallback.

**"Generator has a transpose/xpose branch"** → `techniques.md` #55a (aspect-ratio branch — 2 scalars, no Div/Mod SIGFPE risk).
**"Periodic stripes / data-dependent tiling period"** → `techniques.md` #55b (clamped Mod + one GatherElements; never enumerate distances).
**"Is this MLP just memorizing? / distillation keeps failing"** → `techniques.md` #56 (hash-MLP fingerprint — decode the GENERATOR instead).
**"Boolean reduction rejected by ORT"** → `techniques.md` #57 (Cast→U8 scaffold; ReduceMax(BOOL) ok at opset 20 only).
**"Graph ends `X -> Pad -> output` — is converting to Equal-tail worth it?"** → `techniques.md` #81 (pad-floor crossover law: fixed ~900 B; dtype table; Where-as-output for sparse substitution).
**"Iterated flood/morphology on a narrow (≤8-wide) grid"** → `techniques.md` #88 (row-byte packed flood, 72 B/round) · #83 (nibble-pack for ≤30-wide color-index planes, 2 cells/byte).
**"Officials fail 0/N but every intermediate verifies"** → `traps.md` §FREE-OUTPUT ACTIVE-MASK (canvas 0 decodes ch0 outside the active grid) · §CLIP-DUP PHANTOM (plane-shrink changed Clip-overflow targets).
**"score_network_v2 returns None / crashes on staging"** → `traps.md` §SCORER DUMMY-GRID (clamp semantic scalars once — the scorer runs a synthetic input).
**"Selector bank params dominate an einsum-final"** → `techniques.md` #87 (flattening-rank kill test, then bounded-dyadic MIQCP; permutation banks provably OR-incompressible).
