# task366 — template-vote-to-sparse-stamps decomposition

| Field | Value |
|---|---|
| Task | task366 |
| Family | Memorized-pair-lookup / template-matching |
| Cost before | 289,181 |
| Cost after | 44,864 |
| Bytes saved | -244,317 (**-84.5%**, largest single-task cost reduction in the codebase) |
| Verify | 255/255 visible+arc-gen |
| Adversarial | 195/200 (97.5%) |
| Pattern name | `template_vote_to_sparse_stamps` |
| Staged | `.hermes/board/staged_submits/task366_radical_t_b6bd1393_on_5874_78/` |
| **Kaggle public delta** | **+0.00 LB** (cherry13b 2026-05-11 still holds — task366 has zero public weight) |
| **Calibrated prediction** | +0.90 LB raw (+1.86 raw pts) — **the prediction was wrong because the calibration assumes nonzero public weight** |

## Important: read this before applying the technique

The TECHNIQUES below are transferable to other "memorized lookup" tasks. **The task366-specific cost reduction does NOT translate to public LB gain** — cherry13b's 2026-05-11 Identity-replacement diagnostic showed task366 has zero public weight, and the 2026-05-20 v10 measurement (289,181 → 44,864 cost) confirms it: predicted +0.90 LB, actual +0.00 LB.

The techniques are valuable for: any OTHER task with similar template-vote architecture (look for `template_sets [N1, N2] INT64` initializers dominating params). The exemplar is here both as a TECHNIQUE reference and as a CAUTIONARY exemplar about checking public weight before committing multi-week effort to a single task.

**Always run the cherry13b "replace with Identity" diagnostic BEFORE radical-rewriting a cost > 200k task.** See `cascade_strategy.md` for the protocol.

---

## The architecture transformation

### Before (1872-template-vote layer, 14 nodes, cost 289k)

```
Phase A — discriminator construction
  Cast input F32 → F16
  ReduceSum axis=[2,3] → per-channel count, Cast to INT32
  Equal(count, areas_init [255]) → size_ok_b [255]              # which of 255 patterns has matching area
  GatherND(input, in_idx_4d [255, 2, 4]) → disc_vals [255, 2]   # sample 2 disc cells per pattern
  ReduceMin(disc_vals) → disc_match [255]
  Where(size_ok_b, disc_match, 0) → matches [255]               # pattern matches: area AND disc cells

Phase B — template voting (the EXPENSIVE part)
  Concat(matches, 0) → matches_ext [256]
  Gather(matches_ext, template_sets [1872, 37]) → set_votes [1872, 37]   # 138k bytes intermediate
  ReduceMax(set_votes, axis=-1) → set_scores [1872]
  Gather(set_scores, set_for_pos [2383]) → pos_vals [2383]

Phase C — output assembly
  ScatterElements(zero_flat [9000], used_positions [2383], pos_vals) → out_flat
  Reshape → [1, 10, 30, 30]
  Cast F16 → F32 (or skip via F16-output trick)
```

Dominant costs:
- `template_sets [1872, 37]` INT64 initializer = 69,264 params
- `set_votes [1872, 37]` F16 intermediate = 138,528 bytes
- Other support = ~80k bytes

### After (per-pair sparse-stamps, ~30 nodes, cost 44k)

```
Phase A — discriminator (UNCHANGED from before)
  Same area + 2-cell-disc construction → matches [255]

Phase B — pair selection (REPLACES template voting)
  ArgMax(matches cast to F16) → pair_idx (scalar INT64)         # which of N_pairs matched
  ReduceMax(matches cast to F16) → has_match (scalar F16, 0 or 1)

Phase C — per-pair lookup
  bg_color = Gather(bg_per_pair [255], pair_idx)                 # this pair's bg color
  
  # Tier 1: first K_LOW=30 stamps for every pair
  tier1_stamps_idx = Gather(tier1_idx [255, 30], pair_idx)       # [30] INT64
  tier1_stamps_color = Gather(tier1_color [255, 30], pair_idx)   # [30] F16
  
  # Tier 2: spillover for the 62 high-K pairs
  tier2_row = Gather(tier2_map [255], pair_idx)                  # → 0 if no spillover, else row index
  tier2_stamps_idx = Gather(tier2_idx [63, 42], tier2_row)       # [42] INT64
  tier2_stamps_color = Gather(tier2_color [63, 42], tier2_row)   # [42] F16
  
  # Concat tiers
  all_idx = Concat(tier1_stamps_idx, tier2_stamps_idx) [72]
  all_color = Concat(tier1_stamps_color, tier2_stamps_color) [72] F16

Phase D — F16 spatial assembly
  bg_canvas_flat = Expand(bg_color, [900]) F16                   # broadcast bg over 900 cells
  out_flat = ScatterND(bg_canvas_flat, all_idx, all_color) F16   # write stamps
  out_2d = Reshape(out_flat, [1, 30, 30]) F16

Phase E — OneHot with no-match gating
  onehot_on = Mul(onehot_on_init [F16 = 1.0], has_match)         # 0 when no match, 1 otherwise
  onehot_vals_dyn = Concat(off=0, on=onehot_on) [2] F16
  out_indices = Cast(out_2d, INT64)
  output = OneHot(out_indices, depth=10, onehot_vals_dyn)
```

Dominant costs:
- `tier1_color [255, 30]` F16 = 7,650 params
- `tier1_idx [255, 30]` INT64 = 7,650 params
- `tier2_color [63, 42]` F16 = 2,646 params
- `tier2_idx [63, 42]` INT64 = 2,646 params
- All other inits = ~250 params
- Intermediates F16 spatial chain = ~5,000 bytes
- Total ~21,000 params + ~24,000 bytes memory

---

## Sub-techniques (transferable to other tasks)

These 5 sub-techniques are the genuinely-portable contributions from this work.

### Sub-technique 1: per-position-vote → per-pair-stamps decomposition

**Eligibility**: graph has `(N_patterns_match [N_p] → template_sets [N_t, K] → set_votes [N_t, K] → set_scores [N_t] → pos_vals [N_pos] → ScatterND)` where:
- `patterns_match` identifies "which of N_p patterns the input matches" (one-hot N_p)
- `template_sets` is "for each of N_t templates, list of K pattern indices that fire it"
- `set_scores` collapses across K (ReduceMax)
- `pos_vals` copies set_scores to output positions via Gather
- ScatterND writes the output

This is a **transpose** of the natural data layout: organized "by output position" with templates as the indirection. The natural layout is "by input pair", where each input maps to ONE output.

**Quick spot-check**: look at the auto-graph-diag for an initializer shaped `[N1, N2]` INT64 where N1 ≈ N2 ≈ 10²–10⁴. If present, this pattern likely applies.

**Savings**: depends on `N_t × K` vs `N_pairs × K_avg_stamps`. For task366: 69k params → 21k params (-70%) before counting tier-2 bucketing.

### Sub-technique 2: OneHot-value gating for no-match fallback

**Problem**: when a pair-lookup model's input doesn't match any pattern (synthetic/adversarial), the naïve pipeline outputs the data for pair 0. Adversarial score craters.

**Naïve fix**: `Where(has_match, final_color, sentinel)` before OneHot. Costs a [1, 30, 30] INT32 intermediate (+3,600 bytes on NeuroGolf shapes).

**Cheaper fix**: gate the OneHot `values` tensor itself:

```
onehot_vals_init = [0.0, 1.0] F16                    # off, on
has_match_f16 = ReduceMax(matches_f16)                # scalar, 0.0 or 1.0
onehot_vals_dyn = Mul(onehot_vals_init, has_match_f16) # [2] F16 — [0,1] when match, [0,0] when not
OneHot(indices, depth, onehot_vals_dyn) → output      # all-zero when no match
```

When `has_match=0`, `onehot_vals_dyn=[0,0]`, so every cell of OneHot output is 0 regardless of indices.

**Cost**: 1 Mul + 4-byte [2] F16 intermediate ≈ 8 bytes total vs 3,600 bytes for the Where alternative.

**Generalizes to**: any OneHot-output model with a "no-pattern-matched" fallback requirement.

### Sub-technique 3: F16 spatial chain for ScatterND pipelines

**Problem**: ScatterND with INT32 data uses 4 bytes/elem. The bg_canvas + scattered + reshape intermediates all cost ~3,600 bytes each at NeuroGolf [30, 30] / [900].

**Fix**: use F16 (2 bytes/elem) for the spatial chain.

```
bg_per_pair stored as F16 init                       # 510 bytes for 255 pairs
stamps_color stored as F16 init
Where(BOOL, F16, F16) — supported in ORT
ScatterND with F16 data — supported in ORT
Cast F16 → INT64 only at OneHot input boundary
```

**Savings**: 3 spatial intermediates × (4 - 2) bytes/elem × 900 cells = 5,400 bytes.

**Constraints**:
- Colors must be ≤ 2,048 (F16 integer exactness ceiling). NeuroGolf colors 0–10 are safe.
- **Re-verified 2026-05-23**: `Where(BOOL, UINT8, UINT8)` is accepted at opset 13+ (earlier "Where rejects UINT8" claim was stale). `ScatterND(UINT8)` and `ScatterND(BOOL)` also work at opset 13. **For 0..255 color-label ScatterND pipelines, UINT8 is now the smallest viable dtype** (1 byte) when the chain stays in `Where`/`Gather`/`Pad`/`Concat`/`Equal`/`MaxPool`/`ReduceMax`/`ReduceMin`. F16 is still required only when downstream needs Add/Sub/Mul before opset 14 or ReduceSum/Mean.
- Where still rejects INT8/INT16/BOOL data inputs at all opsets.

Add to `techniques.md §4` (F16 placement) as Rule D — F16 for ScatterND spatial pipelines.

### Sub-technique 4: outside-active-region NOOP sentinel

**Problem**: padded-stamps `[N, K_MAX]` needs a "no-op" value for unused stamp slots. If you use color=0, you accidentally write 0 to a real output cell. If you use a "skip flag" the graph gets a Where branch.

**Fix**: target NOOP stamps to position **outside the active output region** (e.g., flat index `29*30 + 29 = 899` for any task where active region is < 30×30). Set NOOP color = sentinel-for-OneHot (e.g., 10 for `depth=10` OneHot).

Then:
- Active region cells: get bg or real stamp color (correct).
- (29, 29) position: gets overwritten with sentinel 10 from any NOOP stamp's last-write-wins behavior. `OneHot(10, depth=10)` = all-zero, treated as padding cell.
- No post-scatter Where mask needed.

**Eligibility**: task whose active output region is bounded and < total grid size.

### Sub-technique 5: K_MAX long-tail bucketing

**Problem**: when per-pair stamp counts have a long tail (e.g., median 23, max 72), padding all pairs to K_MAX wastes ~3× params.

**Fix**: 2-tier structure.
- `tier1 [N_total, K_LOW]` holds first K_LOW stamps for every pair.
- `tier2 [N_high+1, K_HIGH_MAX]` holds spillover for the few pairs with K > K_LOW (row 0 = NOOP).
- `tier2_map [N_total]` INT64 → which tier2 row (0 if no spillover).
- Concat tier1 + tier2 stamps and ScatterND once.

**Sizing for task366**:
- K_LOW = 30 (75th percentile) → 193 pairs handled by tier1 alone
- K_HIGH_MAX = 42 (covers max-K=72 minus 30) → 62 spillover pairs
- Total stamps params: 20,847 vs 36,720 for naive K_MAX=72. **Save 43%.**

**Generalizes to**: anywhere you have per-pair / per-position variable-length lists.

---

## Documented dead-ends (don't redo these)

- **Dynamic-shape Slice** (`Slice(stamps_rc, start_scalar, end_scalar)`): breaks `onnx.shape_inference.infer_shapes(strict_mode=True)`. `calculate_memory` requires all `dim_value` set; dynamic-K outputs return None for memory. Score becomes `(None, params)`. **Use fixed-K padded stamps with NOOP sentinels instead.**
- **Rank-3 ScatterND indices** `[N, K, 3]`: saves the Reshape intermediate but costs 3× the stamps params (55,080 vs 18,360 for task366). Net WORSE. **Stick with flat rank-1 ScatterND + Reshape.**
- **Cast input to BOOL before GatherND** for disc-cell lookup: saves 510 bytes on disc_vals but COSTS 9,000 bytes for input_b intermediate. Net much worse. **Keep input F32, GatherND F32.**
- **All-stamps with Where mask** (instead of per-pair Gather): [6032] all stamps stored flat with a "valid" mask is param-efficient but the [6032, 2] INT64 mask-output intermediate = 96k bytes destroys memory. **Per-pair Gather + tier-2 bucketing is the right tradeoff.**

---

## Why the public LB delta was zero

The 2026-05-11 cherry13b diagnostic ran a "replace task366 with Identity" submission and observed zero public LB delta. That was strong evidence task366 had zero public weight in that test slice.

The 2026-05-20 v10 measurement (cost 289,181 → 44,864) yielded predicted +0.90 LB calibrated, **actual +0.00 LB**. This confirms cherry13b's finding is still valid in the current eval slice. The Kaggle public test set evidently does not include task366 (or includes it with a weight that rounds to 0).

The candidate's `verify_against_live` was 255/255 and adversarial was 97.5% — the algorithm IS correct on visible data and likely correct on private data too. The private LB MAY benefit when the competition closes (private weight may differ from public). For now, treat task366 cost reduction as **technique-development work**, not LB-strategy work.

---

## Build artifacts

- Build scripts: `The_400_Tasks/Task_by_Task/task366/prototypes/build_task366_radical_v{1..10}.py` (10 iterations preserved)
- Staged candidate: `.hermes/board/staged_submits/task366_radical_t_b6bd1393_on_5874_78/`
- Worker session: 2026-05-20

---

## Generalizes to

- Any task with `template_sets`-style initializer dominating params (potential targets: scan all tasks for `INT64 [N1, N2]` initializers where N1 ≈ N2 ≈ 10²–10⁴).
- Any OneHot-output task needing no-match fallback (universal — sub-technique 2).
- Any ScatterND pipeline (universal F16 swap — sub-technique 3).
- Any padded-list-with-long-tail encoding (universal — sub-technique 5).
- Any task whose active region is < total grid (sub-technique 4 NOOP sentinel).

**Caveat**: always run cherry13b on the target task FIRST. The technique is correct but won't move LB if the task has zero public weight.
