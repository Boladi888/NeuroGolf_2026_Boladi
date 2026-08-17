# cost_floor_reference.md — evicted bodies (2026-06-10 refactor pass 4)

> Dated calibration datapoint ledger + price-shopping worked examples, evicted from the live file.
> The regime conclusions live on in cost_floor_reference.md (calibration section) + traps.md §25.
> Full pre-refactor file: `_archive/cost_floor_reference_pre_refactor_2026-06-10.md`.

## [evicted] Calibration datapoint ledger (2026-05-20 … 2026-06-02, verbatim)

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

## [evicted] Price-shopping worked examples (verbatim)

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
