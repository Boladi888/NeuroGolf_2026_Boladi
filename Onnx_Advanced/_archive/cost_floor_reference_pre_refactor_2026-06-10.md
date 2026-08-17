# cost_floor_reference.md — price-shop your algorithm before building

You're about to build an ONNX. Before writing code, estimate the cost floor of your op chain. This file is the price list.

## The formula

```
cost = memory + params
points = max(1.0, 25 - log_e(max(1.0, cost)))
```

Lower cost = higher points. **Cost 0 (e.g., `input → Identity → output`) and cost 1 both yield the 25-point ceiling** via the inner `max(1.0, cost)` guard. At cost=1000, points=18.1. At cost=10,000, points=15.8. At cost=100,000, points=13.5. At cost=1,000,000, points=11.2.

Implication: the gain from 1,000,000 → 100,000 (-900k bytes) ≈ +2.3 pts. From 10,000 → 1,000 (-9k) ≈ +2.3 pts. From 100 → 10 ≈ +2.3 pts. **Log scale means small absolute reductions at low cost are worth as much as huge reductions at high cost.**

## Full-canvas-plane floor taxonomy (2026-06-01 — the irreducible cost class)

For per-cell-rule tasks whose grids reach 30×30, the floor is set by how many **full 30×30 planes** the rule structurally needs — and ORT forces those planes to a 4-byte dtype (a U8 plane needs a ≥plane-bytes F32/INT cast *source*, so dtype-narrowing never nets out). Before targeting sub-2k/sub-3k, count the unavoidable planes AND check the grid max (`corpus.build(N)['in_shapes']`): if grids are < 30, crop to the real max (task070 grids are hardcoded 17×17; task400 are exactly 24×24).

| Task | Irreducible plane(s) | Why it can't shrink | Floor |
|---|---|---|---:|
| task203 | `mid_oh` `[1,10,1,30]` F32 = 1,200 | 10-channel F32 read of the middle row (the recolor permutation is *derived from* it); channel-collapse-first needs the 3,600 full canvas; the sub-1,200 read (dynamic Slice → `[1,10,1,18]`) scores `None` (traps §8b) | 2,188 |
| task400 | `canvas` `[1,1,24,24]` F32 = 2,304 | dilated crop-collapse Conv output; ORT forces Conv→F32; grids are *exactly* 24×24; extracting the 5×5 from the 10-ch one-hot instead costs 4,800 | ~3,300 |
| task138 | `canvas` `[1,1,25,24]` F32 = 2,400 | frame-raycast crop must read per-cell colors from absolute source coords; normalized crop planes can be `[24,23]`, but the robust source canvas must cover rows `0..24`, cols `0..23`; ORT forces crop-collapse Conv→F32. The official-only negative-pad source `[24,23]` scored lower (12,899) but failed adversarial edge-frame parity, so the clean floor keeps `[25,24]` | 13,204 |
| task224 | `border_full_b` `[30,30]` BOOL = 900 | full-canvas condition for the output-overlay `Where`; output is 30×30 byte-equal to LIVE | ~1,700 |
| task137 | two `[1,1,30,30]` planes (axis-select 900 + 3-state canvas 900) | 2D Chebyshev ring needs *both* axes (1 plane); 3-state output ring/bg/padding needs the sentinel-UINT8 canvas (1 plane); grids reach 30 | ~2,500 |
| task383 | F32 color-decode Conv (3,600) + F16 B-count (1,152) + U8 canvas/bridge (1,800) | bordered-rectangle rule needs per-cell color AND global row/col B-counts. **Driven 55,467→13,112 (2026-06-02, hidden-confirmed rule)** via: corner/center Gather for A/B (0/200k), vector-level cleared, **canvas+`Equal` BOOL output** (kills the OneHot INT32 index — make the Equal the graph output, don't Cast it), **MatMul-cleared** on the F16 B-mask. Conv+count (4,752) are ORT-forced; one-hot-free output measured WORSE (24,977). <3k unreachable. | ~13,000 |
| task286 | flood depth × per-step (geodesic) + I/O ≈ 23,210 | reach-2sym is `(kh,kv)`-sweep-optimal (open runs short: median 1, p90 5). Crop-label proved the parent-equivalent floor can move by spending params to delete runtime label planes (63,211 -> 55,933, hidden-held); the under-20k m2 shallow-reach probe failed badly, so sub-20k is not supported by the current algorithm | 55,933 parent-equivalent; sub20 probe hidden-bad |

**Real plane-floor vs beatable "floor":** task203's "2,404 floor" fell to 2,188 (a center-Gather fold + opset-18 `ScatterElements` removed plumbing *around* the plane); task215's "5,730 floor" fell to 110 (a from-scratch rewrite removed the plane *entirely*). The plane-bytes themselves are the wall; everything else is negotiable. Verification checklist in current_strategy.md §captain-session-2026-06-01 #2.

## Calibration: local prediction vs Kaggle reality

Local prediction of LB gain on stacked submits:

```
predicted_lb_gain ≈ Σ (log_e(orig_cost_i / new_cost_i))   # across all changed tasks
```

This OVERESTIMATES Kaggle reality. Empirical calibration multipliers:

- **Large stacks (100+ tasks)**: actual ≈ 0.48 × predicted. (megastack153 on 2026-05-20: predicted +54.20, actual +26.13.)
- **Small stacks (~30 tasks)**: actual ≈ 0.84 × predicted. (megastack33: predicted +0.74, actual +0.62.)
- **Single-task wildcards near cascade threshold**: actual can be **10-70× predicted** ON FIRST CROSSING. (task110 v8 2026-05-20: predicted +0.18 calibrated, actual +12.73 — first crossing of the cascade-unlock threshold for task110's neighbors.)
- **Same-task follow-on reduction after cascade unlock**: reverts to log-curve **with undershoot**. (task110 v9 2026-05-20, immediately after v8 shipped: predicted +0.21 calibrated, actual **+0.07** — ~3× undershoot. v8 had already unlocked cascade neighbors; v9's further cost reduction crossed no new threshold.)
- **Zero-public-weight tasks**: calibrated prediction can land at **literal 0.00 actual** if the task is not in the public test slice. (task366 v10 2026-05-20: predicted +0.90 calibrated from cost 289,181 → 44,864, actual **+0.00**. Cherry13b's 2026-05-11 Identity-replacement diagnostic showed task366 had zero public weight in that test slice; this v10 measurement confirms it still holds. **Check the cherry13b-style "replace with Identity" diagnostic outcome before counting a task's reduction toward expected LB.**)
- **Single-task wildcards: the 0.48× calibration multiplier is WRONG — empirical multiplier is ~1.0× (raw log_e directly)**. **Seven confirmed datapoints (four 2026-05-20 indirect + three 2026-05-22 exact)**:

  **2026-05-22 datapoints — direct raw log_e prediction, exact match**:
  - **task383 bool_u8_maxpool**: cost 152,592 → 55,467. Predicted +1.01 raw log_e. **Actual: +1.01 LB EXACT MATCH** (5874.78 → 5875.79).
  - **task069 scalar_squarecode_sub10k_bool_pad**: cost 127,412 → 9,999. Predicted +2.54 raw log_e. **Actual: +2.54 LB EXACT MATCH** (5874.78 → 5877.32).
  - **task358 organic_cross_under10k**: cost 132,871 → 9,814. Predicted +2.61 raw log_e. **Actual: +2.61 LB EXACT MATCH** (5874.78 → 5877.39).
  - **3-stack (task383+task069+task358) on 5874.78 base**: Predicted +6.16 LB additive (sum of individual deltas). **Actual: +6.16 LB EXACT MATCH** (5874.78 → 5880.94). Confirms perfect additivity across 3 mixed-family candidates (LIVE-equivalent + 2 algorithm rewrites).
  - **21-stack (10-stack + 18-stack-adds + 20-stack-adds + task358) on 5874.78 base**: Predicted ~+20+ LB sum of individual deltas. **Actual: +20.90 LB** (5874.78 → 5895.68). Beats prior 20-stack high water mark of 5893.21 by +2.47. Demonstrates additivity holds out to 21 mixed-family candidates with no cross-task interference (mix of LIVE-equivalent + algorithm rewrites + memorized-lookup-derived). The 21-stack includes 3 today's new wildcards (task383 maxpool, task069 sub10k, task358 organic_cross) replacing/adding to the 20-stack composition.
  - **task175 v12 wildcard ON TOP of 5895.68 (22-stack effectively)**: cost 63,857 → 3,231 (period_sum_minus_ch0 + variadic Max + INT8 plumbing). Predicted +2.98 raw log_e. **Actual: +2.98 LB EXACT MATCH** (5895.68 → 5898.66). Now 5 in a row exact matches at 1.0× multiplier. Stack additivity holds at this size. **Total session gain from 5874.78 → 5898.66 = +23.88 LB** across 7 candidate submissions.
  - **task382 + task284 + task077 3-stack on 5900.63**: predicted +5.90 LB. **Actual: +3.15 LB** (5900.63 → 5903.78). task382 +2.17 EXACT + task077 +0.98 EXACT, but **task284 +2.75 predicted contribution evaporated to 0.00 LB**. New failure mode: "whole-tail replacement" candidates can land at 0.00-LB-cost-decoupling on hidden-correctness-sensitive tasks even when 266/266 visible + 569/569 on-contract synthetic pass. NOT a regression (LB went up not down) but the predicted cost-based LB gain didn't materialize. This is a NEW data point — distinct from zero-public-weight (consistent across candidates) and distinct from -X LB hidden-bad (visible-correct, hidden-wrong). See `cascade_strategy.md` task284 detailed note.
  - **task008 + task051 + task157 3-stack on 5903.78**: predicted +5.06 to +6.86 LB. **Actual: +5.05 LB** (5903.78 → 5908.83). task008 +1.76 EXACT (LIVE-equivalent dynamic-shift; 684/684 task-shaped synth-LIVE) + task051 +3.30 EXACT (algorithm replacement at 74.8% adv held up) + **task157 +0.00** (6th confirmed memorized-lookup-family zero-weight task). Sub-prediction at lower bound = task157 zero-weight; task008+task051 hit predicted contributions. Memorized-lookup family base rate now **6/6** (task366/324/133/182/018/157).
  - **task209 + task234 2-stack on 5908.83**: predicted +1.96 to +4.50 LB. **Actual: +2.53 LB** (5908.83 → 5911.36). task209 ~+1.96 EXACT (LIVE-equivalent sparse GatherND + tiny palette) + **task234 top2 ~+0.57 partial** (vs +2.54 predicted; ~22% materialized). NEW data point: PARTIAL cost-decoupling distinct from task284's full decoupling. 0/100 adversarial → 22% credit on hidden, not 0%. The fraction of cost-gain that survives depends on what proportion of hidden inputs the candidate's structural assumptions actually hold on.
  - **task255 splitkey_equal_matmul wildcard on 5911.36**: predicted +1.13 LB raw log_e. **Actual: 0.00 LB** (5911.36 → 5911.36 unchanged). **task255 = 7th confirmed memorized-lookup zero-public-weight family member**, joining task366/324/133/182/018/157. Notable: the LIVE-equivalent compression candidate (100/100 synthetic-LIVE) returned 0.00 — confirms the zero-weight verdict is a HIDDEN-DISTRIBUTION property of the task, not a property of any specific candidate. Architectural pattern: `binary_probe + packed_rects + MatMul-union` is the 3rd canonical lookup-family signature documented.

  **2026-06-01/02 continuation datapoints — raw log_e still holds for single-task and follow-on ships**:
  - task265 byte-equal single-channel: 11,237 → 9,584. Predicted `ln(11237/9584)=+0.16`; actual **+0.16** (6446.86 → 6447.02).
  - task138 crop-collapse v7: 58,877 → 14,838. Predicted `ln(58877/14838)=+1.38`; actual **+1.38** (6447.02 → 6448.40).
  - task013 mod/gather rewrite: 58,000 → 2,774. Predicted `ln(58000/2774)=+3.04`; actual **+3.04** (6448.40 → 6451.44).
  - task165 signed/capped follow-on: 10,440 → 8,920. Predicted `ln(10440/8920)=+0.16`; actual **+0.15** (6451.44 → 6451.59).
  - task138 crop-collapse v9 follow-on: 14,838 → 13,204. Predicted `ln(14838/13204)=+0.12`; actual **+0.12** (6451.59 → 6451.71).
  - task213 stride-gather index-canvas: 55,705 → 2,193. Predicted `ln(55705/2193)=+3.23`; actual **+3.23** (6451.71 → 6454.94).
  - task191 opset20 hashmatch: 32,841 → 24,807. Predicted `ln(32841/24807)=+0.28`; actual **+0.29** (6454.94 → 6455.23). Safe opset-13 v7 at 26,545 also held at 6455.16.
  - task383 rule rewrite: 55,467 → 25,767. Predicted `ln(55467/25767)=+0.77`; actual **+0.76** (6455.23 → 6455.99).
  - task238 scalar color-index map: 55,368 → 15,085. Predicted `ln(55368/15085)=+1.30`; actual **+1.30** (6455.99 → 6457.29).
  - task286 crop-label parent compression: 63,211 → 55,933. Predicted `ln(63211/55933)=+0.12`; actual **+0.12** (6457.29 → 6457.41). The m21/m22/m23/m24 ladder tracked near monotonic; m2 sub20 failed public hard.

  These are useful because they include *same-task follow-ons after a hidden-correct ship* (task165, task138 v9), not just first replacements. The follow-on gain remained raw-log additive; no special discount was observed for shaving an already-shipped single task.

  **2026-05-20 datapoints — ratio 2.05–2.09× over calibrated prediction (same finding, less precise)**:
  - **task173 v6**: cost 256,900 → 167,464. Predicted +0.21 calibrated. Actual **+0.43 LB**. Ratio 2.05×.
  - **task286 v11**: cost 234,282 → 119,906. Predicted +0.32 calibrated. Actual **+0.67 LB**. Ratio 2.09×.
  - **task205 v6**: cost 177,538 → 71,912. Predicted +0.43 calibrated. Actual **+0.90 LB**. Ratio 2.09×.
  - **task138 radical_v1**: cost 236,364 → 150,728. Predicted +0.22 calibrated. Actual **+0.45 LB**. Ratio 2.05×.

  All seven submitted as single-task wildcards on the 5874.78 / v8-canonical base. The 2026-05-22 datapoints span 3 different LB magnitudes (+1.01, +2.54, +2.61) and 2 different lineage classes (LIVE-equivalent structural rewrite for task383; algorithm-level rewrite for task069/task358) — exact match in all three confirms **the 1.0× multiplier is well-calibrated across the full single-task wildcard regime**. **For single-task wildcards: use the raw log_e prediction.** The 0.48× was empirically fit on 100+ task megastacks; the megastack regime has compounding effects (verify failures cross-contaminating, ORT memory pressure, etc.) that single-task submits don't trigger.

  **Original "partial cascade" interpretation was probably wrong**: the consistent 2.05× ratio across 4 independent tasks with very different cost-band positions is too uniform to be cascade-recovery events. It's calibration model error.

- **Stacking confirmed-positive single-task wildcards is PERFECTLY ADDITIVE** — three consecutive measurements:
  - 4-stack (2026-05-20 evening): t110 v9 + t173 v6 + t286 v11 + t366 v10. Predicted 5874.78 + 0.07 + 0.43 + 0.67 + 0.00 = **5875.95**. Actual **5875.95** — exact match.
  - 6-stack (2026-05-20 night): + t205 v6 + t138 radical_v1. Predicted 5874.78 + (0.07 + 0.43 + 0.67 + 0.00 + 0.90 + 0.45) = **5877.30**. Actual **5877.30** — exact match.
  - 7-stack (2026-05-20 night): + t092 radical_v1. Predicted 5877.30 + 0.026 = **5877.33**. Actual **5877.33** — exact match.

  Measured single-task wildcard deltas stack cleanly in small bundles. The safer interpretation is independent task-point gains via the raw cost curve, not multiple cascade unlocks. **Workflow**: single-submit each candidate first to measure (use raw log_e for prediction), then stack confirmed-positives to compound.

- **Architecture floor for 9-channel span-fill graphs: ~208k bytes**. Discovered via task092 radical_v1 worker (2026-05-20). For tasks with the pattern "detect between-leftmost-and-rightmost for N=9 channels of [1,9,30,30]": minimum F16 intermediates ≈ 99,000 B (1 full input + 1 sliced nonbg + 4 directional CumSum/Mul outputs), minimum BOOL intermediates ≈ 72,900 B (comparison + And/Or chain). Total architectural floor ~172k; task092's measured floor is 208,718 B after INT8 ReduceMax Or-fold optimization. Breaking below 150k requires moving to per-color [1,1,30,30] (9 independent CumSum chains per direction), which replaces batched [1,9,30,30] tensors but adds 9× nodes and complex per-color recombination. Empirical: **the rank-reduce trick of replacing CumSum with ReduceMin/ReduceMax DOES NOT WORK** — comparison broadcasts force full [1,9,30,30] BOOL intermediates regardless. The falsification evidence is folded into this paragraph so workers do not need the missing task092 prompt note.
- **Do not label ordinary additive single-task gains as cascade.** The 2026-05-20 stack additivity datapoints show that measured single-task wildcard deltas can be stacked without loss; they do not prove multiple cascade unlocks. Cascade remains a rare step-function exception, with task110 v8 as the only confirmed full unlock.

**LB prediction by regime (2026-05-23 — supersedes old 0.48× blanket rule)**:
- **Single-task wildcards**: raw `log_e(cost_before/cost_after) × 1.0`. Confirmed by ~21 single-task ships across 2026-05-22 to 2026-05-23 (5874.78 → 5949.01); empirical actuals matched raw log_e predictions within ±0.05 LB.
- **Small stacks (2-7 tasks)**: still ~raw log_e additive. Perfect additivity confirmed at 3-stack (+6.16), 5-stack (+9.68), and 7-stack levels.
- **Large megastacks (100+ tasks)**: empirically ~0.48× (megastack153 on 2026-05-20). The compounding loss is from verify cross-contamination, ORT memory pressure, and many tasks living below per-task floors.
- **Cascade unlock**: rare step-function exception. Only confirmed event was task110 v8 (+12.73 LB). Not the default; most single-task wildcards land at raw-log estimate alone.
- **Zero-public-weight tasks (10 confirmed)**: task366/324/133/182/018/157/255/066/076/277. Any cost reduction via lookup/NN-memorization scores 0.00 LB regardless. See `traps.md §14e`.
- **Hidden-cost-decoupling tasks (task284 full, task234 partial)**: candidates that pass 266/266 visible + on-contract synthetic but the cost-gain partially or fully evaporates on Kaggle. Distinct from zero-weight (consistent across candidates) and from -X LB hidden-bad (visible-correct but hidden-wrong).

The old "0.48× multiplier" was empirically fit on 100+ task megastacks; that regime has compounding effects single-task submits don't trigger. **Do NOT apply 0.48× to single-task wildcards** — it underpredicts by ~2×.

## Tensor byte sizes (the units of memory)

| Shape | F32 (4 B/elem) | F16 (2 B/elem) | INT32 (4 B/elem) | UINT8/BOOL (1 B/elem) | INT64 (8 B/elem) |
|---|---:|---:|---:|---:|---:|
| `[1, 10, 30, 30]` | 36,000 | 18,000 | 36,000 | 9,000 | 72,000 |
| `[1, 1, 30, 30]` | 3,600 | 1,800 | 3,600 | 900 | 7,200 |
| `[1, 10, 15, 15]` | 9,000 | 4,500 | 9,000 | 2,250 | 18,000 |
| `[1, 1, 15, 15]` | 900 | 450 | 900 | 225 | 1,800 |
| `[1, 1, 9, 9]` | 324 | 162 | 324 | 81 | 648 |
| `[10]` | 40 | 20 | 40 | 10 | 80 |
| `[1]` (scalar value_info) | 4 | 2 | 4 | 1 | 8 |
| `[]` (scalar value_info) | 4 | 2 | 4 | 1 | 8 |

**Scalars cost `dtype.itemsize` bytes**, not always 4. `calculate_memory` does `num_elements × dtype.itemsize` where shape `[]` yields `num_elements = 1`. `BOOL []` = 1 B, `F16 []` = 2 B, `F32/INT32 []` = 4 B, `INT64 []` = 8 B. The old "4 bytes minimum" claim only applies to 4-byte dtypes. Use `BOOL`/`UINT8` scalars where possible for the cheapest sentinel/threshold value_info entries.

## Cost contribution per op

These are LOWER BOUNDS assuming the smallest valid intermediate dtype and minimal `value_info` count.

| Op chain | Contributes to `cost` | Notes |
|---|---|---|
| Pure shape-only (`Slice/Gather/Pad/Reshape/Concat/Transpose/Identity/Cast`) on small intermediates | ~1,500 floor | Cost from intermediate value_info bytes + ~30 params for indices |
| One Conv (3×3, no padding, no F16) | +36,000 if intermediate is `[1, 10, 30, 30]` F32; less if narrowed | Conv itself is not a memory hit; the activations are |
| One Conv (1×1, F16 intermediates) | +18,000 for F16 [1,10,30,30] or proportionally less | F16 halves memory |
| ReduceMax / ReduceSum on `[1, 10, 30, 30]` axes=[2,3] keepdims=1 | +40 (`[1, 10, 1, 1]` F32) | Tiny |
| ReduceMax / ReduceSum on `[1, 10, 30, 30]` axes=[2] keepdims=1 | +1,200 (`[1, 10, 1, 30]` F32) | Per-column reduction |
| ReduceSum of bool → INT64 | output dtype is INT64, count bytes accordingly | Cast to BOOL if needed downstream |
| OneHot with `[1, 1, 30, 30]` indices, depth=10 | output is `[1, 10, 30, 30]` F32/F16/INT64; +36,000 F32 or +18,000 F16. **BOOL/UINT8 values are NOT supported by ORT's OneHot kernel** (verified opset 11/13). Use `Equal(label, channel_constants) → BOOL` if you need a BOOL multi-channel mask. | OneHot is 0-MAC but the output tensor counts |
| Where(c, x, y) | output shape = broadcast of all 3; ORT charges max of profiler-observed | `Where(BOOL, BOOL, BOOL)` rejected; `Where(BOOL, INT8/INT16, ...)` rejected; **`Where(BOOL, UINT8, UINT8)` works at opset 13+**; `Where(BOOL, F16, F16)`, `Where(BOOL, F32, F32)`, `Where(BOOL, INT32, INT32)`, `Where(BOOL, INT64, INT64)` all work |
| Equal / Greater / Less | output is BOOL of broadcast shape | `Equal(BOOL, BOOL)` works at all opsets ≥ 1; `Equal(F32/F16, F32/F16)` requires opset 11+ (NOT opset 10). `Greater(BOOL)`/`Less(BOOL)` always rejected. `Greater/Less(UINT8/INT8/INT16)` work at opset 10+ — see `Ort_Advanced/ort_quirks.md §4` |
| Constant node with value `[1, 10, 30, 30] F32` | params: 9,000 element count | NOT bytes; element count |
| Initializer `[1, 10, 30, 30] F32` | params: 9,000 | Same — element count |
| Initializer `[1, 10, 30, 30] BOOL` | params: 9,000 | Element count is SAME as F32; dtype doesn't help params |
| `[1, 10, 30, 30]` BOOL initializer in `memory`? | 0 | Initializers don't count in `memory`. They count in `params`. |

## Cost bands (verified by shipped tasks)

### Cost ≤ 100 (theoretical floor)
Two distinct sub-cases:
- **Pure structural identity** (`input → Identity → output`, `input → Transpose → output`, `input → Cast → output` when input/output share the same shape): cost can be **literal 0** because the model has zero `value_info` intermediates and no `params`. These yield the 25-point ceiling via `max(1.0, cost)` guard.
- **Constant-output via initializer/Constant node**: params = element count of the output buffer the initializer provides. A `[1, 10, 30, 30] BOOL` constant-output costs 9,000 params, NOT ≤100. Only genuinely small constant outputs (small grids, scalar) can hit the sub-100 band this way.

**Realistic floor for any task with non-trivial per-cell output**: ~500-1,500 bytes.

### Cost 100-3,000 — shape-only + minimal cast
- task299 bool-Concat: cost 1,058. Algorithm uses Concat of bool intermediates + Pad. No Conv, no MatMul.
- task129 tile_to_edge_pad: cost in this range.
- Fixed rotation via Slice→Gather→Pad sandwich: 1,091 (was 36,030 with full-canvas Gather, -34,939 bytes from refactor).

Worth chasing this band when the algorithm is purely "rearrange the input pixels".

### Cost 3,000-30,000 — narrow intermediates + small Conv
- task374 conv_run_combined: 36,315 → 24,587 after polish. One Conv path with F16 intermediates.
- task343 crop15 + onnxsim: 12,872 (was 19,108). Crop-to-active-grid + F16 placement combined.
- task071 ring-dyn at rank-2: 24,150 (was 26,859, params 2,140 → 133).
- task115 axisfactor: 14,087 (was 149,687, -135k via factorization).

### Cost 30,000-300,000 — full-canvas F32 intermediates
- Most "polished" tasks on the current canonical sit in this band.
- task110 after radical rewrite: 212,871 (was 307,174). Periodicity detection over 7 candidates.
- task366 with F16 placement: 289,181 (was 422,810).
- task187 with F16: 175,624 (was 278,586).
- Single-Conv on the full input + scoring overhead lives here.

### Cost > 300,000 — many F32 [1,10,30,30] intermediates OR large LUTs
- Above this is "haven't been F16-converted yet" or "uses MatMul/Gemm" territory.
- Usually the first 30 minutes of work brings these into the 30k-300k band.

## Decision tree

When designing an ONNX for a new task, ask in order:

1. **Can the output be a constant?** (Is the answer purely a function of the task ID, not the input?) → ~0 cost. Rare but worth checking.

2. **Can the algorithm be pure shape-only?** Slice + Gather + Pad + Concat + Reshape + Transpose. → ~1,500 floor. Applies when the task is "rearrange input pixels" with no per-cell decision.

3. **Can it be done in BOOL throughout?** All masks, no F32 arithmetic. → ~9,000 floor (one `[1, 10, 30, 30]` BOOL intermediate). See techniques.md rule 8 (input→BOOL whole-graph).

4. **Does it need one Conv on a narrowed (cropped) input?** Crop to active grid first → narrow intermediates. → 10,000-30,000 band.

5. **Does it need multi-stage decision (period detection, CC analysis, etc.)?** → 50,000-200,000 band. task110 is the exemplar — 7 candidates × 2 directions × scoring tensors.

6. **Does it need MatMul/Gemm/big lookup table?** → likely > 200,000. Consider if the table is compressible (separable_mask_collapse), packable (bit-pack), or replaceable by dynamic compute (rank reduction).

## Price-shopping examples

### task110 radical rewrite (worker analysis 2026-05-20)

Algorithm: period-detection over 7 candidates with per-V/H scoring + 2-iter fill.

Cost breakdown (post-rewrite, 212,871 total):
- 7 candidates × 2 directions × ~9,000-byte BOOL mismatch tensors = ~126,000
- Iterative fill chain (2 iters, 4 directions) = ~43,000
- Conv + OneHot expand = ~40,000
- **Lower bound** for ANY 7-candidate × 2-direction × full-grid algorithm: ~210,000

To get below 50,000 would require replacing per-cell mismatch tensors with row-signature hashes — per-period scalar/vector signatures instead of per-cell [1, 1, 30, 30] tensors. Risky for correctness, hard but possible.

### Anti-example: pre-cast full input F16

Tempting: `Cast(input [1,10,30,30] F32, F16)` adds an F16 intermediate of 18,000 bytes. If downstream only uses 1-3 channels: pre-cast added 18,000 bytes vs the alternative of per-channel cast (5,400 bytes for 3 channels). Net loss: 12,600.

See `techniques.md` §4 (F16 placement rules).

## Constants and initializers — separate budget

The above table is `memory` (intermediates). `params` is a separate budget:

- Each non-zero-dim initializer element counts as 1 param (dtype-agnostic).
- Constant nodes' value attributes also count.
- A `[1, 10, 30, 30] F32` initializer = 9,000 params (NOT 36,000 bytes; element count).
- A `[1, 10, 30, 30] BOOL` initializer = 9,000 params. Same. Dtype doesn't help params.

If your algorithm needs a 30×30 lookup table:
- Stored as F32 init: 900 params.
- Stored as BOOL init: 900 params. **No savings via dtype.**
- Stored as `[1, 30]` separable + broadcast at use: 30 params. **30× savings via shape.**
- Stored as a fully computed-on-the-fly value: 0 params, but adds compute nodes.

This is why `separable_mask_collapse` is high-leverage: shape compression saves params; dtype compression doesn't.

## Recolor / channel-permutation floor = C (proven 2026-05-30)

Any recolor whose rule is a color permutation π on the C=10 one-hot channels has a **hard floor of cost = C = 10**, realized by a single `Gather(input, idx[C], axis=1)` (memory 0, params C). This is provably minimal — no sub-10 exists:
- Every data-carrying intermediate is `[1,k,30,30]` ≥ 900 B, so a cost-≤9 graph can have **no** intermediate; the last op must read `input` directly.
- Only `Gather` emits a channel-permuted `[1,10,30,30]` cheaply; its index must have C entries (C params). Transpose can't permute within an axis without a ≥9000-B reshape intermediate; MatMul/Conv permutation matrices are ≥100 params; Slice/Concat reassembly is ≥900 B.
- The one sub-10 accounting path (sparse index, NNZ = C−1) is **categorically rejected** — Gather forbids sparse indices (3 scorer gates). See `core.md` sparse note + `task337.md` Floor proof.

**Implication: do NOT sub-floor-hunt GENERIC single-Gather recolor tasks.** Verified 2026-05-30: 016 `[0,5,6,4,3,1,2,7,9,8]`, 276 `[0,1,6,3,4,5,2,7,8,9]`, 309 `[0,1,2,3,4,7,6,5,8,9]`, 337 `[0,1,2,3,4,8,6,7,5,9]` are all generic permutations already at floor 10. The Reshape+Transpose "block trick" never beats a 10-param Gather (it carries a ≥9000-B intermediate).

**The ONE clean sub-10 exception — channel REVERSAL.** If a recolor's index is the exact reversal `[9,8,7,6,5,4,3,2,1,0]`, use `Slice(input, starts=[9], ends=[-11], axes=[1], steps=[-1])` instead of Gather: 4 one-element index inits, memory 0 ⇒ **cost 4 (23.61 pts)**, byte-identical, no accounting gimmick (verified). Transpose can't reverse within an axis; opset-1 attribute-Slice has no `steps`; so 4 is the reversal floor. **Quick test before shipping any recolor: `idx == list(range(9,-1,-1))` ⇒ reversal ⇒ Slice-reverse (cost 4); else ⇒ generic ⇒ Gather floor 10; `idx == list(range(10))` ⇒ identity ⇒ cost 0.** Full proof: `task337.md`.

## Single-channel color-index canvas = 900 B is the floor for a [1,10,30,30] output (proven 2026-05-30)

When a task's OUTPUT is the full `[1,10,30,30]` one-hot canvas, the cheapest *counted* way to produce it is a single-channel `[1,1,30,30]` U8 color-index canvas fed to a final `Equal(canvas, channel_ids[1,10,1,1])` — the BOOL `[1,10,30,30]` Equal result is the graph OUTPUT (outputs aren't counted), so the only counted cost is the **900-B U8 canvas** (+ a tiny `[10]` channel-ids init). This is the canvas+broadcast pattern (task387 / 009 / 054 / 388 family).

**Non-obvious corollary — do NOT try to eliminate the canvas itself.** task388 v6 tried to skip the `[1,1,30,30]` canvas and emit a BOOL `Equal` directly off a smaller `[1,10,12,12]` intermediate: it scored **2,546 — WORSE** than v5's 1,998, because the `[1,10,12,12]` BOOL counted intermediate (1,440 B) is *larger* than the `[1,1,30,30]` U8 canvas (900 B). The 900-B single-channel canvas is the architectural floor whenever the active region ≳ 10×10; only when the active region is genuinely tiny (task384-style ≤10×10 crop, output via the fold-mask-into-gather trick) does a cropped BOOL output beat it. **Price both before choosing:** full canvas = `30*30*1 = 900 B` (U8) vs cropped-BOOL-output = `10 * h * w` B (crosses 900 at h·w = 90, i.e. ~9×10).

**Empty-cell sentinel gotcha (task275).** When you collapse a 10-channel one-hot to a single-channel color-index canvas, all-zero (background / empty) cells decode to color 0 unless you inject an explicit out-of-range sentinel: `canvas = color_index + (1 - ReduceSum(onehot, axis=ch)) * 10` makes empty cells map to value 10 (≥ depth) so `Equal(canvas, [0..9])` yields all-zero one-hot there, matching the canonical. Missing this is a silent OOD bug — task275 v1 caught it on 1/2000 random pairs only.

**Canvas+Equal BEATS ScatterND channel-patch for partial-recolor outputs (proven 2026-05-31, task070).** When the rule recolors only K of the 10 output channels (e.g. color-1→color-3 = 2 channels) and leaves the rest = input, it is tempting to skip the Conv color-read entirely with `output = ScatterND(input, [[0,c1],[0,c2],…], updates)` — input and output are both free, so the 8 unchanged channels cost 0. **It loses to the canvas.** `ScatterND.updates` must match the **F32** `input` dtype, so a K-channel update is `[1,K,30,30]` F32 = `K*3600 B` (K=2 → 7,200), plus its BOOL precursor `[1,K,30,30]` (the masks are BOOL → must Cast) = another `K*900`. That ~9,000-B anchor (K=2) alone exceeds the entire single-channel-canvas cost (~6,300-B core: Conv-F32 3,600 + Cast-U8 900 + recolor Pad 900 + Where 900). Measured: task070 channel-patch = **15,956** vs canvas = **8,958**. The Equal-expand gets ALL 10 channels free; ScatterND pays F32 for every patched channel. (You can't shrink the update dtype: casting `input`→U8 to use U8 updates costs a `[1,10,30,30]` U8 base = 9,000 itself.) **Rule: for any "recolor/patch a few channels of a one-hot output" task, use the single-channel canvas + `Equal(canvas, ids)`, not ScatterND.** Marginal canvas win: apply the recolor `Where` on the active-region crop and `Pad` the *recolored canvas* once (saves the full-size Where) — but that zeros the >crop region, so only do it when the task's grid size is provably fixed (task070: ARC-GEN hardcodes 17×17) — otherwise keep the full-canvas Pad-bool+Where for byte-equality on all sizes.

## Shift / translation floor = gathered-axis length (proven 2026-06-01, task053)

A "translate the grid by (dr,dc), zero-fill" rule (frame shift) has a hard floor of **cost = the length
of the shifted axis** (e.g. **30** for a 1-row down-shift on the 30-row canvas), realized by a single
`Gather(input, idx[30], axis=2)` whose index sources an *existing background row* into the vacated band
(memory 0 — Gather output IS `output`; params = index length). task053 (down-shift-by-1) is the
exemplar: `idx=[2,0,1,29,…,29]` (row 2 is always background → becomes the new top row), **cost 30,
proven optimal**.

**Why you can't go cheaper (and the trap that looks like you can):**
- The tempting `Pad`-shift (opset-10 `pads` attribute ⇒ 0 params, output direct ⇒ 0 memory ⇒
  **cost 0, 25.0 pts**) is **byte-WRONG** under the standard ch0-hot background encoding: it
  diverges on ~100% of inputs (task053: 2060/2060). A scalar pad fill can't synthesize the `ch0=1`
  background row, and it drags the active-region background row into the padding band. See
  `traps.md §15`. **`Pad`-shift is correct ONLY when the encoder makes background all-zero.**
- Single-op alternatives fail: `Slice` gives only strided-arithmetic row orders (a shift's
  source order is non-arithmetic and needs a synthesized bg row); `Transpose` can't move within an
  axis. **The obstruction is not param-count** — a depthwise `Conv` (`[10,1,2,1]`) shifts with just
  **20 params** (<30) but DIVERGES, because every *geometric* shift zero-fills and zero-fill ≠ the
  `ch0=1` background the vacated band needs (and a per-channel bias can't target row 0 only). Only a
  primitive that **relocates an existing background row** (Gather/Slice of a real bg row) is correct,
  and a single Gather doing that for a 30-row output needs a 30-element index ⇒ 30 params.
- Multi-op alternatives exceed the floor: any data intermediate that assembles a 30-wide-row,
  10-channel output is ≥ `[1,10,1,3]` BOOL = 30 B (realistically `[1,10,3,3]` F32 = 360 B; measured
  Gather3+Pad = 3,603, Slice+Concat = 7,205); a *computed* `idx[30]` intermediate is ≥120 B (INT32).
  A sub-30-B intermediate can't enable the shift at all.
- Sparse Gather index (drop the one `0` → NNZ 29) is **rejected** by the scorer (`score_network_v2`
  → None). So the dense full-length index is the floor.

**Implication: do NOT sub-floor-hunt single-`Gather` shift/translation tasks** (the ~5–15 frame-shift
tasks in the set). The Gather-with-full-length-index IS the floor; the only "cheaper" candidates are
byte-wrong (Pad) or rejected (sparse). Full proof + probe harness: `task053.md` +
`Task_by_Task/task053/prototypes/build_and_probe.py`.

## Quick gut checks

- `[1, 10, 30, 30]` F32 intermediate = 36,000 bytes. **The largest common full-canvas F32 activation** — anything larger (e.g., `[1, 10, 30, 30]` INT64 = 72,000 bytes, or `[1, 10, 30, 30, K]` for any K > 1) is avoidable and almost always the cost dominator if present. Aim to keep no `[1, 10, 30, 30]` F32 intermediate unless the output assembly requires it.
- Crop to `[1, 10, 10, 10]` = 4,000 bytes. **9× smaller.**
- BOOL `[1, 10, 30, 30]` = 9,000 bytes. **4× smaller.**
- Drop the channel dim entirely (`[1, 30, 30]`) = 900 bytes F32. **40× smaller.**
- A single scalar = 4 bytes. **9000× smaller** than full canvas F32.

For a task at cost 200,000, asking "where are my biggest intermediates and can each one shrink 4-10×?" is the right question. Inventory them with the diagnostic snippets in `Ort_Advanced/ort_errors.md`.

## The plane-conservation-law floor method (task328, 2026-06-04 — promote-worthy proof style)

Stronger than per-route rejection: define the task's per-cell DECISION CORE (e.g. "2 ternary
decisions x leg-compare x parity x 3-state output"), then price EVERY legal op family against
it — task328 found every 2-input formulation lands within ~150 B of the same ~19-plane total
(even the Sign/Clip-code + wrap-Gather-LUT route ties exactly at 3,240 before losing on LUT
build cost; packed payloads are plane-neutral because each pack costs one Div/Mod unpack).
Escapes must come from ops with >2-input fan-in per materialization (3-operand Einsum corner
reads bought −900 there; see Research card G3 for the systematic catalog). Write the law with
numbers in the task .md — it converts "couldn't go lower" into "no legal op can".

## task398 grader-gap datapoint (2026-06-04 night)

- task398 anti-diagonal expansion: 39,875 → **≈13,300 GRADER-CREDITED** (v8/v12/v15 all local
  364–519 — the 37× §GRADER-GAP trap; +1.10 banked, byte-equal). The honest static floor
  (fixed 25×25 canvas + Range/Equal stripes) is estimated 1.5–2.5k and remains OPEN — see
  task398.md follow-up. Do NOT cite 364 as a real floor anywhere.

## Packing-law refinement of the conservation floors (2026-06-04, Research H2)

The task328/task338 "packed payloads are plane-neutral" findings are the α≈0 special case of
the general law (`techniques #45`): when no consumer executes on packed words, packing is
strictly worse — those floors STAND. But graphs with ≥4 packed-executable consumer sites (U8,
n=8) ESCAPE: re-audit conservation-law floors only if the decision core is rich in pure
AND/OR/XOR stages that could run on packed words.

## task054 architecture floor (2026-06-04 night, grader-source-grade proof)

- task054 legend/role: 47,180 → **26,043 SHIPPED** (byte-equal FULL; ScatterElements(axis=3)
  column-writes replace transpose pairs, −1,647). **<10k/<1000 PROVEN impossible**: scored
  probes — any one materialized plane ≥4,520; 8-plane detection = 11,856 alone; the 14
  surviving planes each load-bearing (4-D point-scatter regresses to 30,299 measured). First
  floor proof backed by reading `calculate_memory` itself.


## task080 information-flow floor (2026-06-04 night — batch-7 closes 10/10)

- task080 lattice: 49,405 → **5,664** (+2.166 exact; FIRST F16-Conv ship, grader-validated).
  Info-flow floor ≈5,050 (canvas 900 + decode 800 + select ~500 + two conv passes ~1,500 +
  assembly ~1,100 + probes ~250); every fusion alternative measured worse (don't-dos in
  task080.md). <4,000 below the rule's floor; <1,000 below the canvas law alone.



## Floor verdicts — compact ledger (Phase-1 consolidation 2026-06-05)

Full dated ship-blocks: `_archive/cost_floor_full_pre_phase1.md`. One row per shipped floor.

| task | floor / note |
|---|---|
| task193 | 55,657 → **5,281**. F32 `mark` `[1,1,20,20]` Conv plane is ORT-forced; |
| task244 | 53,897 → **2,252** (compact selector; einsum-arch sibling 2,354 also |
| task287 | 50,758 → **3,016**; **memory floor 2,948 proven exhaustively** (6 |
| task368 | 50,484 → **6,294 interim** (asym-kernel crop + Conv color-decode + |
| task020 | 50,090 → **2,760** (byte-equal). |
| task240 | 53,414 → **3,928** (byte-equal). |
| task131 | 50,325 → **5,251** (byte-equal; 18×18 orientation tensors + 30×30 |
| task325 | 50,733 → **5,642** (byte-equal; K=5 proven minimum — K=4 fails |
| task122 | 52,517 → **4,816** (byte-equal; marker-255 sparse-Scatter §38). |
| task279 | 50,462 → **14,067** (more-correct-than-LIVE: fixes 1 LIVE GT-bug; floor |
| task128 | 46,305 → **5,218** (byte-equal; placement outer-product; opset-9 Pad |
| task195 | 47,787 → **1,878** (byte-equal; §39 9-cell GatherND read — the |
| task367 | 47,225 → **19,465 interim** (byte-equal qconv-rowtrim; <10k routes |
| task308 | 49,062 → **2,990** (byte-equal seqrect). |
| task014 | 45,690 → **4,380 local / ≈6,093 Kaggle-credited** (byte-equal |
| task328 | 47,384 → **8,367** (byte-equal; einsum-corners + mod-parity; |
| task280 | 48,080 → **6,716** (byte-equal, 36 versions; dead under-5k: |
| task364 | 46,250 → **14,162** (byte-equal negbias-cliff §43; THEOREM-GRADE floor: |
| task049 | 44,787 → **748** (byte-equal; CAMPAIGN-RECORD 59.9×; per-byte |
| task187 | 49,813 → **21,282 interim** (scanline; dual-truth-attributed 0-real-bugs, |
| task363 | 43,636 → **3,796** (byte-equal full 200,265; the tie-break decode held — |
| task338 | 45,649 → **8,469 CLOSED** (negpad v8, +1.685 exact). Closing verdict: |
| task252 | 39,858 → **280 (CAMPAIGN RECORD +4.958, 142×)** — zero-memory |
| task196 | 42,589 → **4,998** (byte-equal FULL; copy-the-free-input + |
| task119 | 43,122 → **3,390** (byte-equal; §42 Einsum marginals grader-validated; |
| task065 | 44,106 → **3,275 local / ≈3,805 credited** (byte-equal; 3rd |
| task293 | 37,940 → **838** (+3.81 exact; batch-free free-input Einsum, zero-initializer scalar). |
| task374 | 34,392 → **2,831** (+2.50 exact; QLinearConv neighbor sums + Slice crop + opset-11 |
| task094 | 39,015 → **3,806 static** (+2.33 exact; v1 center-cross). The 313/451 dynamic rungs |
| task064 | 39,510 → **14,706** (+0.99 bet won; filtered-clean class; lower routes: v2 ScatterND |
| task025 | 36,433 → **23,495 k5-exact** (k4 19,355 @99.88% parked — deterministic class; |
| task359 | 40,808 → **4,510 zero-params** (+2.20; direct-dominant Cast-active masks; the 1,330 |


## Salvage from empirical-cost-probes.md (Phase-2)
- \* Memory column **as reported by onnx-tool**. For single-Conv graphs the only intermediate is the output tensor, which the validator excludes; effective memory contribution = 0. For multi-node graphs the memory term is real and additive.
- This works for single-Conv graphs because their only memory entry is the OUTPUT tensor, which the validator excludes (per the brief: "memory metric is custom: sum of `value_info` intermediate-activation byte sizes + declared IO (excluding `'input'`/`'output'` themselves)"). For **multi-node** graphs, intermediate-tensor memory is real and additive, so the COVERED table's `cost = macs + params` shortcut is **misleading** outside the single-Conv regime.

## task270 floor (2026-06-05)

- task270: 38,430 → **2,978** (+2.56 exact; center-noop probe beat the gated tri15 primary 3,681).
  Runtime lesson banked: local truth-full timeout at 904s = 4.5 ms/pair × corpus volume — grader
  unaffected (1k pairs ≈ 4.5s). Gate slow candidates with --n 20000, compute per-pair first.

## task367 EXACT floor (2026-06-05 — closed at 19,465)

- task367: **19,465 proven exact** (§A: 25 distinct thresholds × all corpus sizes × spatial
  minimum = irreducible 9,000 B candidate plane; composed-kernel v2 measured 0/266 — the
  inter-stage non-linearity is structurally irreplaceable; #47 flood/G2/Einsum/negpad all
  priced N/A or worse; F16 doubles the candidate; chunked QConv still needs all 25 bits live
  at stamp time). Proof table: task367.md:134-164. The 338-standard close.
