# task228 delta packet — t_e06c43ad

Date: 2026-05-17
Worker: neurogolf-worker-7

## Verdict

No submit candidate. The best fresh local rewrite found in this sweep is exact but still above the inherited/current task228 cost.

## Baselines and attempts

- Inherited/current base from card: cost 41194, points 14.374.
- Original organic builder (2026-05-08): cost 68040; strict-clean train 3/3, test 1/1, arc-gen 262/262; not submit-eligible over cheaper base.
- Prior submit-captain package `task228_v1_no_pad_ch0_from_fg_on_5739_36_20260517T145712Z`: public-negative, Kaggle 5724.99 vs 5739.36 lineage (-14.37); do not resubmit this family unchanged.
- Fresh ArgMax/nonzero-stack rewrite: cost 53492 (score tuple `(0, 53357, 135)`), dispatcher points 14.1127; strict-clean train 3/3, test 1/1, arc-gen 262/262; still above base, so not submit-eligible.
- Safe/extended optimizer over the fresh rewrite: no cost change, tuple remains `(0, 53357, 135)`.

## Evidence

- Built local ONNX during verification: `submissions/task228.onnx` (ignored workspace artifact; after final recheck it is the original organic dispatcher build, cost 68040).
- Dispatcher audit for fresh temporary rewrite: `task228 ok frame-inner-markers-to-outer-opposite (53492 / 14.1127 pts)`.
- Silent-fail CSV for fresh temporary rewrite: `Logs/scans/2026-05-17_task228_fimoo2.csv` with train 3/3, test 1/1, arc-gen 262/262.
- Final original-dispatcher recheck: `Logs/scans/2026-05-17_task228_recheck_original.csv` with train 3/3, test 1/1, arc-gen 262/262 and build cost 68040.
- Task ledger updated: `The_400_Tasks/Task_by_Task/task228.md`.

## Recommendation

Keep task228 closed/no-submit unless a new structural family beats ~41194 local cost and is not the no-pad/ch0-from-fg public-negative family.
