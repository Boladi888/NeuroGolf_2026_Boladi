# DONE 2026-05-22T205000Z — task358 organic cross-fill techniques (locally under 10k cost, awaiting submit authorization)

## Result

- Builder: `The_400_Tasks/Task_by_Task/task358/prototypes/build_task358_organic_cross.py`
- Prototype: `The_400_Tasks/Task_by_Task/task358/prototypes/task358_organic_cross.onnx`
- Staged: `.hermes/board/staged_submits/task358_organic_cross_under10k_t_5f00e7cb_on_current/`
- ONNX SHA256: `5F00E7CB8F2B5B4F7F82BE882A74734D501621ABB43079A0A90B10AF0D7E8ACD`

Cost: 132,871 → 9,814 (memory 8,679 + params 1,135). Δ = **−123,057 bytes (−92.6%)**

Validation:
- `verify_against_live --tid 358`: 265/265 proto-vs-LIVE
- `adversarial_test --tid 358`: 6979/7027 = 99.3% aggregate; `add_isolated_cell` worst 747/795 (94.0%); all other perturbation types 100%
- Synthetic official-style zero-padded random one-hot raw LIVE comparison: **1000/1000**

**Status: not yet submitted to Kaggle. Awaiting user authorization. Local +2.61 LB expected via raw log_e(132871/9814).**

## Techniques (all transferable)

### Technique 1 — Direct crop Conv (skip 30×30 materialization)

**Old**: extract color-index plane at full 30×30 with Conv on full input, then Slice to 21×20 active grid. Materializes one [1,1,30,30] color-index tensor.

**New**: Direct Conv with kernel that ALREADY crops to the active grid by encoding kernel position. Output is [1,1,21,20] directly. Saves the [1,1,30,30] intermediate (~3,600 B F16, ~7,200 B F32).

Eligibility: any task where (a) max active grid is fixed at compile time, (b) only the active grid is consumed downstream. Common in NeuroGolf where workers know the active rectangle.

### Technique 2 — ArgMax + Gather + Squeeze (skip per-row residue scoring tensors)

**Old**: score periodic period existence at full [1,P,H,W] resolution, then ReduceMax to find period.

**New**: 
1. ArgMax to find row/column with unique 3/4-cell foreground marker
2. Gather to extract just that row/column's color vector
3. Squeeze to scalar period score per candidate period
4. ReduceMax over the (small) scalar tensor

Avoids ever materializing [1,P,H,W] scoring intermediates. task358 went from 18,349 → 14,439 via this step.

### Technique 3 — Concat-repeat for periodic patterns (not stored residue masks)

**Old**: store per-period residue masks as [P, H, W] initializers (P × H × W params).

**New**: build periodic row/column patterns at runtime via Concat of scalar residue scores. Each Concat = period-step replication. For P=2..5, period=P pattern is `[score, score, ..., score]` Concat'd 30/P times.

Saves ~1,000+ params on tasks with 4-5 candidate periods × 20-30 grid extent.

### Technique 4 — Sentinel-10 UINT8 label + Pad

**Old**: emit final F16 [1, 10, 30, 30] one-hot via Where/Equal across channels.

**New**:
1. Use F16 sentinel value 10 for "outside active grid" (10 is not a valid color index, so no one-hot channel matches)
2. Cast to UINT8 (sentinel becomes 10_u8)
3. Pad with UINT8 10 (= sentinel padding)
4. Equal-compare UINT8 input to UINT8 color codes [0..9]
5. Outside active grid: 10 ≠ any 0..9 → all-zero one-hot

Saves the explicit "outside-grid → 0" mask materialization. The sentinel 10 inherently produces all-zero one-hot via the Equal mechanism.

### Technique 5 — Sparse Conv weights (REJECTED in current scorer, but worth re-testing)

The worker tested replacing dense Conv kernel with sparse_tensor(float) Conv weights. Cost reduction would have been substantial. **Rejected** because `score_network_v2` doesn't support ONNX shape inference on sparse_tensor(float). Dense Conv retained.

Library implication: sparse Conv weights are a documented dead-end for the current scorer. Don't waste worker hours on this until scorer support lands.

## Sorter note: techniques.md — add as Rule "direct_crop_conv"

Generalizes Technique 1. Pattern: any task where (a) max active grid is < 30, (b) only active grid consumed downstream. Replace full-input Conv + post-Slice with kernel-encoded-crop Conv. Saves the [1,1,30,30] F16/F32 intermediate.

## Sorter note: techniques.md — add as Rule "argmax_gather_squeeze_for_periodic_scoring"

Generalizes Technique 2. Pattern: tasks computing periodic-score per candidate period across full grid. ArgMax+Gather+Squeeze the scoring axis BEFORE the per-grid materialization. Saves the [1,P,H,W] scoring tensor.

## Sorter note: techniques.md — add as Rule "concat_repeat_for_periodic_patterns"

Generalizes Technique 3. Replace stored [P, H, W] periodic-mask initializers with Concat-of-scalar runtime construction. Saves P × H × W params at the cost of P-1 Concat ops.

## Sorter note: techniques.md — add as Rule "sentinel_uint8_label_pad_for_outside_grid"

Generalizes Technique 4. Use UINT8 10 as sentinel for "outside active grid", Pad with 10, Equal-compare to color codes [0..9]. Inherent all-zero one-hot outside the active grid. Saves the explicit outside-mask materialization.

## Sorter note: cascade_strategy.md — add task358 sub-10k as cascade-aware win

cost 132,871 → 9,814 = −123,057 (raw log_e +2.61 LB single-task wildcard). When (and if) submitted as wildcard on 5874.78 base, predicted Kaggle LB delta = +2.61 LB clean.

## Sorter note: traps.md — add "sparse Conv weights rejected by current scorer"

scorer (`score_network_v2`) doesn't support ONNX shape inference on sparse_tensor(float). Workers should NOT attempt sparse Conv weight optimizations until scorer support lands. task358 worker tested and rejected on 2026-05-22.
