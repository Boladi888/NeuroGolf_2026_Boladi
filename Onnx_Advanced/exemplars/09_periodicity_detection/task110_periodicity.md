# task110 — periodic-fill via per-direction scoring + iterative fill

| Field | Value |
|---|---|
| Task | task110 |
| Family | Pattern_Recognition / Filling Regions |
| Cost before | 307,174 |
| Cost after | 212,871 |
| Bytes saved | -94,303 (-30.7%) |
| Kaggle delta | **+12.73 LB** (predicted +0.18 — see Calibration anomaly) |
| Verify | 266/266 |
| Adversarial | 6,575/8,809 (74.6%) |
| Pattern name | `radical_rewrite` (periodicity_detection) |
| Staged | `.hermes/board/staged_submits/task110_radical_t_2fa872e0_on_5862_05/` |

## Rule (algorithm)

task110 is **periodic-fill**: input has a periodic pattern with rH ∈ {2, 4, 5, 6, 7, 8, 9} (vertical period) and rW ∈ {same set} (horizontal period). Some cells are missing (bg, color 0). Output fills the missing cells using the inferred period.

**264 of 265 train+arc-gen pairs have rH == rW** (symmetric period). One test pair has asymmetric periods (rH=9, rW=18) — this is the case that breaks joint V+H ArgMax (see Anti-pattern below).

## Op chain (after rewrite)

9 phases, ~245 ops, ~210k bytes intermediates:

```
# Phase 1: convert one-hot to color index (cheap)
# IMPORTANT: Conv binds X/W/Y to the same type parameter. F32 input + F16 weights FAILS at ORT load.
# Either Cast(input, F16) first, OR keep F32 weights with F32 intermediates.
input_f16 = Cast(input, F16)                                       # required if input is F32
Conv(input_f16, weights=[0,1,2,..,9] F16) → color_idx F16 [1,1,30,30]

# Phase 2: per-candidate-k validity gating (vertical AND horizontal independently)
for k in [2, 4, 5, 6, 7, 8, 9]:
    # vertical: shift color_idx by k along axis 2
    shifted_v = Gather(color_idx, shift_indices_k_V, axis=2)
    # boundary validity (avoid wrap-around)
    mvp_v = boundary_validity_mask_V_k                          # BOOL [1,1,30,30]
    # mismatch detection
    bg = Equal(color_idx, 0)                                     # BOOL — missing cell, "valid by absence"
    ok_v = Or(Equal(color_idx, shifted_v), Or(bg, Not(mvp_v)))   # BOOL
    valid_v[k] = Equal(ReduceSum(Cast(Not(ok_v), INT32)), 0)     # scalar BOOL

    # horizontal: same, axis=3
    ...same construction...
    valid_h[k] = ...

# Phase 3: pick smallest valid k via Where chain (separate per direction)
selected_k_V = Where(valid_v[2], 2, Where(valid_v[4], 4, Where(...)))
selected_k_H = Where(valid_h[2], 2, Where(valid_h[4], 4, Where(...)))

# Phase 4: iterative fill (2 iterations of vp/vn/hp/hn directions)
filled = color_idx
for _ in range(2):
    filled = fill_step(filled, selected_k_V, selected_k_H)

# Phase 5: padding detection — set fi=10 (out-of-range OneHot index) → all-zero output
input_active = Greater(ReduceSum(input, axes=[1], keepdims=1), 0)
fi = Where(input_active, filled, 10)                              # 10 is out of OneHot depth

# Phase 6: OneHot expand to [1,10,H,W]
output = OneHot(fi, depth=10)
```

## Why each phase is the way it is

- **Conv with weights [0..9]** is the cheapest one-hot → integer-index conversion. F16 weights save 2 bytes/elem vs F32 with no precision loss on integer-valued outputs. **Conv requires X/W/Y same dtype** — for F32 input you must `Cast(input, F16)` before the Conv; mixed-dtype Conv fails ORT load.
- **Per-direction scoring** (NOT joint V+H ArgMax) is required: joint scoring systematically favors V when V pair count ≥ H pair count, masking H periodicity. Confirmed empirically (task110 v4 lesson, 2026-05-13).
- **`Or(Equal, Or(bg, Not(mvp)))`** is the validity check: a cell is "ok" if (a) it matches its k-shifted neighbor, OR (b) it's missing (bg) and doesn't need to match, OR (c) it's outside the boundary mask (don't penalize wrap).
- **Where chain for smallest-valid-k** instead of ArgMin: ArgMin requires INT output and downstream Gather. Where chain with the 7 candidate periods inlined as constants is cheaper for small N.
- **2 fill iterations** is the sweet spot: 1 iteration leaves some cells unfilled in dense bg regions; 3+ iterations don't change the result but add cost.
- **Padding via fi=10 + OneHot(depth=10)**: out-of-range indices produce all-zero OneHot rows, which is what we want for cells outside the active grid. Cleaner than masking post-hoc.

## Anti-patterns avoided (worker hit during development)

1. **`Where(bool, bool, bool)`** — ORT does NOT support this. Worker initially built selected_k_V via `Where(valid_v[2], one_bool, Where(...))` and got an ORT load failure. Rewrote with And/Or/Not chain. See `ort_reality.md §2`.
2. **OneHot wrapping negative indices to channel 9** — worker initially used `fi = Where(input_active, filled, -1)` expecting OneHot to produce all-zero for -1. Instead, OneHot wraps -1 modulo depth → channel 9. Switched to `fi = 10` (out-of-range, produces all-zero).
3. **Joint V+H ArgMax** — produced wrong period selection on the one asymmetric test pair. Switched to per-direction scoring.

## Calibration anomaly

The agent's local prediction was +0.18 LB (calibrated). Actual Kaggle delta was **+12.73 LB** — 70× more. Hypothesis: task110 at cost 307,174 was at or near the cross-task memory cascade threshold; reducing to 212,871 unblocked Kaggle-cascade-locked neighbor tasks from ERROR back to scored.

task110 is not in the documented trigger task set `{45, 129, 130, 171, 172, 210, 211, 220, 239-242, 256-262, 309}` but the cascade is heuristic — high-memory ONNX in adjacent slots may have been ERRORing tasks that recovered when task110 dropped.

Implication: tasks with cost > 200k may have outsized Kaggle impact when reduced, beyond what `cost_floor_reference.md` log calibration predicts.

## v9 follow-on optimizations (2026-05-20)

A second-pass worker stacked two further reductions on top of the shipped v8 (212,871 → **198,764**, -14,107 bytes). Both are independently transferable techniques that have been promoted to `techniques.md`:

### Opt 1: padding detection via row+col ReduceSum + Or (-3,300 bytes)

Phase 5's `Greater(ReduceSum(input, axes=[1]))` produces an F32 `[1,1,30,30]` intermediate (3,600 bytes) before the BOOL Equal. Equivalent compact form when interior rows/cols all have ≥1 active cell:

```
row_active = ReduceSum(input, axes=[1, 3], keepdims=1)   # F32  [1,1,30,1]  =   120
col_active = ReduceSum(input, axes=[1, 2], keepdims=1)   # F32  [1,1,1,30]  =   120
row_pad_b  = Equal(row_active, 0_f32)                     # BOOL [1,1,30,1]  =    30
col_pad_b  = Equal(col_active, 0_f32)                     # BOOL [1,1,1,30]  =    30
is_padding = Or(row_pad_b, col_pad_b)                     # BOOL [1,1,30,30] = 900 (broadcast)
```

Total 1,200 bytes vs the old 4,500. See `techniques.md §9` "Compact form: row+col reductions + Or".

### Opt 2: padding-sentinel in F16 before Cast→INT64 (-10,800 bytes)

The original Phase 5 ends:
```
fi          = Cast(filled_f16, INT64)        # 7,200
fi_masked   = Where(is_padding, neg1_i64, fi) # 7,200
fi_3d       = Squeeze(fi_masked)              # 7,200
```

Reordered: keep the Where in F16 (sentinel = `10.0`, exactly representable; out-of-OneHot-range = all-zero one-hot row), Cast last:
```
cur_masked    = Where(is_padding, ten_f16, filled_f16)   # F16 1,800
cur_masked_3d = Squeeze(cur_masked)                       # F16 1,800
fi_3d         = Cast(cur_masked_3d, INT64)                # INT64 7,200
```

10,800 bytes vs 21,600. See `techniques.md §1 Rule 9` "padding_sentinel_before_cast".

**Note on sentinel choice**: must be `10.0` (or any value ≥ depth=10), NOT `-1.0`. OneHot wraps negative indices modulo depth, mapping `-1 → channel 9`. This is documented in the anti-pattern section above.

### v9 shipped 2026-05-20

`.hermes/board/staged_submits/task110_radical_t_0e8c49b0_on_5862_05/` — cost 198,764, verify 266/266, adversarial 6,575/8,809 (74.6%).

**Kaggle delta: +0.07 LB** (5874.78 → 5874.85). Calibrated prediction was +0.21 LB; v9 underperformed by ~3×.

**Interpretation**: the v8 cascade-unlock anomaly (+12.73 LB on a +0.18-predicted change, 70× over) did NOT repeat for v9. Most plausible reason: v8 already unblocked the adjacent task110 cascade neighbors when cost dropped from 307,174 → 212,871. v9's further reduction (212,871 → 198,764) did not cross any new cascade-unlock threshold, so the gain reverted to ordinary log-curve-with-undershoot territory. This pins down a useful boundary condition: cascade-unlock is a step function tied to specific cost thresholds, not a smooth scaling on every reduction.

## Generalizes to

`Algorithm_Library/algorithm_patterns.md` §4 (periodic-fill). Apply this template to any task whose rule is "fill missing cells using inferred period".

**Eligibility**:
- Input has periodic structure (you should verify by computing `input[r, c] == input[r+P, c]` for various P).
- Missing cells are marked by a known sentinel (bg/0 in NeuroGolf).
- Period candidates form a small finite set (here 7 candidates: {2, 4, 5, 6, 7, 8, 9}).
- Output is the input with missing cells filled.

## Cost-floor analysis

The 9-phase algorithm dominates at ~190k bytes minimum for 7-candidate × 2-direction × full-grid scoring:
- 7 candidates × 2 directions × ~9,000 byte BOOL scoring tensor each ≈ 126,000.
- 4-direction fill chain (vp/vn/hp/hn × 2 iters) ≈ 43,000.
- Conv + OneHot ≈ 40,000.

Below 190k requires algorithmic change. Two paths were tried by the v9 worker (2026-05-20) and **both failed**:

- **Row-signature hashing** (compute per-period signature `ReduceSum(color_idx × pos_weights, axis=3)` → row signatures `[30]` per period, then compare vectors instead of per-cell): provably **unsound under partial visibility**. Empirical test on the test pair (rH=9): 17 of 20 row pairs `(y, y+9)` have different signatures even when the rows are "compatible at common visible cells", because missing cells contribute 0 to the sum and different fill patterns give different hashes. Two-hash variants (different weight vectors) and mask-aware variants (Mul by visibility first) fail the same way — they reject valid periods, breaking 100% verify.
- **F16-arithmetic scoring** (`Sub → Mul-self → vis-mask → ReduceSum`, all F16): measured **13,500 bytes/k vs the current 7,200 BOOL-chain bytes/k — 87% worse.** BOOL intermediates beat F16 per byte. See `techniques.md §1` calibration paragraph for the general rule.

The 7-candidate × 2-direction × per-cell BOOL scoring floor (~75,600 bytes for the validity tensors alone) is the hard limit for this algorithm structure. Below 190k requires a fundamentally different algorithm — e.g., predicting period from cheap input statistics with a small lookup table, conditional on the input passing a periodicity-detection trigger.

## Artifacts

- Build prototypes: `The_400_Tasks/Task_by_Task/task110/prototypes/build_task110_*.py` (v1-v8 progression, plus the final `task110_radical_2FA872E0.onnx`).
- task notes: `The_400_Tasks/Task_by_Task/task110.md` (worker's hour-by-hour log + 9-pattern attempt list).
- Staged: `.hermes/board/staged_submits/task110_radical_t_2fa872e0_on_5862_05/manifest.json`.

## Reproducibility

To apply this technique to another periodic-fill task:

1. **Verify periodicity** — compute per-row `input[r] == input[r+P]` agreement for P in candidate set. Identify candidate periods that hold across train pairs.
2. **Check for asymmetric periods** — examine the test pair specifically. If rH ≠ rW for any pair, MUST use per-direction scoring (no joint ArgMax).
3. **Identify the bg / missing-cell sentinel** — usually color 0 in NeuroGolf.
4. **Build the 9-phase template** above, adjusting:
   - Candidate period set (replace [2, 4, 5, 6, 7, 8, 9] with your task's candidates).
   - Iteration count (start with 2; increase if verify fails on dense-bg pairs).
   - Padding detection if the grid size is fixed (skip Phase 5).
5. **Avoid `Where(bool, bool, bool)`** — use And/Or/Not chains for boolean Where equivalents.
6. **Verify**: 100% M/M on visible+arc-gen is the only hard gate. Adversarial-against-live: run + report, advisory only per `core.md` 2026-05-22 update (empirical: task064 at 9.7% adv shipped +1.20 LB clean).
