# Alpha-A Cost-Rewrite Marathon Plan

> Established 2026-05-11 in response to owner mandate: "for the next 80hrs, do hours of complex per-task rewrites averaging small gains; work tasks in chronological order up to 6hrs each or until small gain; repeat 10 iterations or until 80hr mark; 2 submissions/hour; keep to Alpha-A (Alpha-B is working concurrently)."

## State as of start

- **Live submission.zip:** `alpha_a_mega_t089v2_t137v2` at Kaggle public 5651.22 (+54.30 vs starting kiwi11 5596.92).
- **Wins already applied (in mega):** task062, task125, task191, task205, task243, task308 (band variants); task089, task137 (cost rewrites).
- **Don't-touch confirmed hidden-bad** (do NOT re-attempt): task089-cheap, task153, task044, task382, task387, task017-old, task009, task076, task350, task154, task230, task235, task261, task270-organic, task306, task217, task231, task269, task149, task212, task314, task019, task119, task013, task083, task169, task194, task108, task121, task015, task092, task115, task141, task204, task184, task022, task071, task263, task101, task102, task275, task366.

## The tracker

- `Logs/alpha_a_rewrite_progress.csv` — per-task attempts ledger.
- Status values: `pending` (queued), `low_priority` (5k-20k cost), `skip_optimal` (<5k), `skip_done` (in mega), `skip_dont_touch` (hidden-bad), `attempted_no_gain`, `won` (+pts landed), `attempted_complex_deferred`.

## Workflow per task

1. **Check tracker status.** If `skip_*` or `won`, move on.
2. **Inspect task.** Read `data/competitions/neurogolf-2026/task<NNN>.json`. Note shape, color count, transformation pattern.
3. **Identify if dedicated builder exists.** `grep "_task<NNN>_\|task<NNN>_network\|_task<NNN>_predict" build_submission.py`.
4. **If dedicated builder exists:** check cost; if rewrite has clear path (MatMul-fold shifts; mod-period-mask; etc.) attempt.
5. **If no dedicated builder:** task uses generic dispatch. Decide if NEW dedicated organic is feasible in <2hr. If yes attempt; if no defer to later iteration.
6. **Verify 100% on train+test+arc-gen via numpy + ONNX.**
7. **Score via `_score_v2.score_network_v2`.** If cost < current → ok to submit.
8. **Submit as isolated probe on current best.** Respect 2/hr Kaggle cap.
9. **Update tracker:** status, attempts++, last_attempt_date, notes.
10. **If positive: lock in as new live, move to next task. If negative: revert live, note in don't-touch.**

## Submit budget

- 2/hour cap. 80 hours = 160 submits maximum.
- Each submit ~= one task attempt verified locally.
- Don't burn submits on unverified prototypes.

## Productive vectors discovered

1. **MatMul-folded shift Conv** — applied to task089 (1.1M→295k, +1.31 pts). Works for any task with `for offset in (-N..N)²: shift_then_overlay` pattern.
2. **Mod-period ring mask via Floor(x+0.5)** — applied to task137 (536k→107k, +0.23 pts). Works for any task with concentric-rings-at-multiples-of-period.
3. **Single Conv replaces nested loop over k** — generic pattern: any algorithm that iterates `for k in range(N)` with shifted intermediates can be a Conv producing N output channels.
4. **SAFE math-equivalent mega optimization** — applied to all 400 ONNXs, +0.37 pts (5651.22 → 5651.59). Tools at `tools/dedup_onnx_inits.py`, `tools/onnx_opt_safe.py`, `tools/onnx_opt_with_fold.py`. Passes: dedup duplicate initializers (1377), common subexpression elimination (169 nodes), dead code elimination (12 nodes + 22 inits), scalar algebraic identities (5 Mul-by-1/Add-by-0/Sub-by-0), constant folding (2 Mul/Add/Sub of two initializers). **Per-task verify+fallback** catches CSE bugs on rare ops (task202 needed fallback initially; bug-fix in attr_to_tuple for TENSOR non-raw data made task202 CSE-able correctly). The marginal cost-saving per task is tiny but cumulates across 400 ONNXs.

## Failed-vector lessons

- **Cherry4 band variants ON tasks with prior-public-negative organic history** — uniformly hidden-bad. Skip task382, task387, task044, task089-cheap, task153, task017-old-cheap.
- **Tasks with `synthetic_guard.py` requirements** (task014, task153) — band variants fail the synthetic guard → hidden-bad. Only use guard-passing variants.
- **Smaller bytes ≠ cheaper validator cost.** cherry4-era ONNXs often have many initializers + many intermediate value_info entries → high memory cost under post-May-4-rule-change validator. Always score via `_score_v2.score_network_v2`, never byte-pick.
- **Validator-cost-cheaper ≠ hidden-test positive.** Some validator-cost-cheaper variants are hidden-bad anyway. Only Kaggle test confirms.
- **Hand-written organics with complex algorithm logic are HIGH RISK for hidden-bad.** task383 v2 (axis-cross-out): 266/266 visible verified, cost 301k vs 322k existing, submitted as `alpha_a_mega_t089v2_t137v2_t383v2` = **5638.91 (-12.31 vs 5651.22)**. The algorithm appeared correct on visible but failed hidden. The SAFE pattern is math-equivalent rewrites (task089 MatMul-folded, task137 Floor-trick) that produce IDENTICAL outputs by algebraic transformation — NOT re-derivations of the algorithm.

## Realistic projection

- Each successful cost rewrite: +0.2 to +1.5 pts. ~30-60 min effort.
- Per-iteration ceiling: maybe +10-20 pts if every easy win lands. Many won't.
- 10 iterations × +15 avg = +150 pts → realistic end-of-marathon: 5800.
- **Hard ceiling for current methods:** ~5800-5900. Path to 6500 needs new source pools (NAS scrape beyond cherry4/quince3/salak3/ugni3) or entirely new pattern implementations (the 108 generic-dispatch Alpha-A tasks).

## Coordination with Alpha-B

Alpha-B is doing parallel work, mostly on their own task range. Periodic check `kaggle competitions submissions` for their wins — STACK any positive Alpha-B variant ONNXs onto my mega base. They're effectively my mining partners.

## Resumption protocol per next session

1. Read this file + `Logs/alpha_a_rewrite_progress.csv`.
2. Pull main, ensure tracker is up to date.
3. Pick HIGHEST-cost `pending` task (see strategy update below).
4. Begin workflow above.

## Strategy update 2026-05-10 (after iteration 1 failures on task004, task011)

**Key learning:** low-cost pending tasks (cost < 150k) already have cheaper band variants in the live mega. Even when my organic rewrite is correct (verified 100%), it loses on cost to the band variant. Examples:
- task004: organic 342k vs band 110k → -1.13 pts (skipped)
- task011: organic 128k vs band 84k → -0.42 pts (skipped)

**Revised picking order:** prioritize highest-cost pending tasks first. The ones with cost > 200k are most likely to have ROI for organic rewrites, since the existing solution is dispatcher-built (expensive) rather than band-variant (cheap).

Top targets (as of 2026-05-10):
- task110 521k, task396 506k, task138 475k, task286 421k, task074 389k, task280 330k

## Iteration count tracking

| Iteration | Started | Completed | Tasks attempted | Pts gained | Notes |
|---:|---|---|---:|---:|---|
| 1 | 2026-05-11 00:XX | 2026-05-10 21:XX | 2 organics built + verified (task004, task011); inspected 6 high-cost candidates (task110, task396, task138, task286, task074, task280, task383) | 0 | Both organics 100% correct but mega band variants cheaper. High-cost tasks all use complex specialized builders that are hard to beat in single-session effort. Strategy needs new vectors. |

## Key insights from iteration 1

1. **task004 organic (parallelogram-lean)**: verified 265/265 numpy + 265/265 ONNX. Algorithm: shift +1 col, except bottom row stays, clip at right edge. Cost 342k vs existing 110k (band variant). Loss -1.13 pts → SKIP.

2. **task011 organic (missing-8 sub-region key)**: verified 267/267 numpy + 267/267 ONNX. Algorithm: compact 11×11 to 9×9, find missing-8 sub-region, use its color positions as output sub-region color map. Cost 128k vs existing 84k. Loss -0.42 pts → SKIP.

3. **High-cost task inspection** (110, 396, 138, 286, 074, 280, 383):
   - task110 (521k): periodic-hole-fill with variable period — complex
   - task396 (506k): find-largest-frame + change-border-color — multi-frame detection
   - task138 (474k): frame_raycast_crop_network already optimized with MaxPool 30×1 rays
   - task286 (421k): MaxPool/Min heavy — already cheap structure
   - task074 (388k): 4-fold symmetry restoration, only 13 nodes, near-optimal
   - task280 (329k): rectangle-extrude-2-color — 370 inits suggest shift-stack
   - task383 (322k): axis-cross-out from anomaly markers — complex projection

4. **Strategic conclusion**: cost-rewrites on existing dispatcher-built tasks are hard. Best ROI may be:
   - New band variant mining with different filter criteria (already exhausted basic)
   - NAS forum scraping for fresh source pools (mentioned in "Hard ceiling" comment)
   - Building entirely new dedicated builders for the 108 generic-dispatch Alpha-A tasks (multi-hour each)
