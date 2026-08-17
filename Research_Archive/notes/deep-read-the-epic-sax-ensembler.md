# Deep-read: `the-epic-sax-ensembler.ipynb`

> Beta-track deep-read, 2026-05-03 20-hr session. Source: `Research/primary_sources/the-epic-sax-ensembler.ipynb` (7 cells, 27 KB).

## Summary

This is a **pure submission-zip ensembler** — no patterns, no training, no inference logic. The notebook scans `/kaggle/input/` for all `submission*.zip` files, extracts ONNX models from each, and merges them into a single `submission.zip` using a "highest-safe-task-coverage source first" strategy.

**Strategic value to NG**: 0 novel patterns (architecture). Real value is the **`VALIDATOR_BAD_TASK_IDS` skip list** — a curated set of 22 tasks the author declared unsafe.

## VALIDATOR_BAD_TASK_IDS (22 tasks)

```
task008, task014, task016, task019, task050, task064, task101, task137,
task177, task185, task206, task244, task263, task276, task291, task309,
task337, task355, task359, task368, task384, task389
```

The author treats these as "verify_subset passes locally but Kaggle private-test scores zero" — same family as our `silent_fail_check`-based watchlist for task363.

### Cross-reference vs our 50 SOLVED set

**5 of our 50 SOLVED appear on this skip list**: `016, 050, 276, 309, 337`. Combined with the prior `03_PROJECT.md` "Russian SKIP_TASKS" list (53, 129, 266, 385), the union of our SOLVED that competitors flag is:

| Task | Our pattern | Competitor source flagging |
|---|---|---|
| 016 | color-perm | epic-sax (this notebook) |
| 050 | enclosed-region-fill | epic-sax |
| 053 | fixed-shift_v_1 | Russian SKIP_TASKS |
| 129 | const-dominant-color | Russian SKIP_TASKS |
| 266 | marker-corners | Russian SKIP_TASKS |
| 276 | pixel-perm | epic-sax |
| 309 | color-perm | epic-sax |
| 337 | color-perm | epic-sax |
| 385 | shape-changing-pixel-perm | Russian SKIP_TASKS |

**9 of our 50 SOLVED tasks are flagged by at least one external competitor.** All have arcgen-passing organic implementations + matched local-vs-Kaggle prediction within ~0.84% on every submission, so we're not visibly losing points on these. But this **constitutes a watchlist for future regression testing**.

### Cross-reference vs our truly-unsolved 150

The skip list also contains tasks 019, 177, 263 — which we **just identified as candidates** in Phase 4 mining (task263 as color-filter-bbox-crop, task177 as bbox-crop with object-filter). If competitors view these as "silent-fail risk", we should be extra-careful with our `verify_subset` gate before submitting any ONNX for these. Specifically, **`silent_fail_check --min-arcgen-N 100`** (vs default 30) is the conservative move.

## Strategy: "highest-safe-task-coverage source first"

The notebook prioritizes sources (zip files) in this order:
1. Most safe-task members (excluding VALIDATOR_BAD_TASK_IDS).
2. Smaller zip size (tie-break, smaller is faster).
3. Lexical zip path (deterministic tie-break).

For each task, the FIRST source that has it (by this priority) wins. No cost-comparison across sources — just first-come-first-served.

This is **inferior to our `external_blend.py`** which compares per-task COSTS across sources to pick the cheapest valid candidate. The notebook's strategy is simpler but leaves points on the table when one source has a cheaper version of the same task.

## What's NOT in the notebook

- No novel ONNX patterns or builders.
- No new cost-formula insights.
- No new task-specific recipes.
- No new mining angles.

## Action items

1. **Cross-reference the 9 competitor-flagged SOLVED tasks** against `silent_fail_check` results. If any of them fail the `--min-arcgen-N 100` threshold, regression-test before any future submission.
2. **For tasks 019, 177, 263** (Beta's recipe candidates that competitors flag as "bad"): when these are eventually implemented, run `silent_fail_check --min-arcgen-N 100` BEFORE submitting. If they fail, gate the submission.
3. **No backlog additions** from this notebook (no novel patterns).
4. **No new tools** needed.

## Closed sub-question

**"Is the-epic-sax-ensembler a useful pattern source?"** — No. It's an ensembler, like our `external_blend.py` but with a simpler source-priority rule. Its value is the validator-bad skip list, not the merging logic.

---

*Author: Beta-track 2026-05-03 deep-analysis session. Cross-reference: `Logs/sessions/2026-05-03_1830_beta-20hr-session.md`.*
