# Logs/

Archived project lessons and submit history for the finished NeuroGolf run.

## What survived

**Top-level files:**
- `lessons.md` - durable engineering lessons, newest first.
- `submissions_ledger.csv` - Kaggle submit ledger. It is row-by-row through 2026-06-08, then a final 2026-06-21 `_RESYNC` row records the confirmed end state instead of fabricating missing rows.
- `cascade_campaign_log.md` - retained because `Onnx_Advanced/cascade_strategy.md` points here for the campaign diary and threshold-watchlist history.

**Subdirectories:**
- `_archive/lessons_pre0601.md` - pre-June lesson archive split out on 2026-06-05.
- `_archive/CASCADE_WORK_LOG.md` - earlier cascade/hardening diary referenced by old task notes.
- `research/` - 20 structured research cards and data files (`R1` through `R6`) that are higher signal than the deleted scan dumps.

## Archived convention note

This folder used to document a live process: one session log per run, dated scan buckets, and a `runs.jsonl` telemetry file. That convention decayed by 2026-05-25, while hundreds of task scratch files landed at `Logs/` top level despite the old rule.

During the 2026-07-15 archive cleanup, the dead session logs, scans, empty `runs.jsonl`, active-lane ledgers, and per-task scratch dumps were removed. Treat the old convention as a useful pattern with a caution attached: convention docs need audits, or they become stale claims.

## Pointers for future readers

- Current/end scoring state: `../CURRENT_STATE.md` plus the final row of `submissions_ledger.csv`.
- Durable lessons: `lessons.md` and `_archive/lessons_pre0601.md`.
- Deep technique library: `../Onnx_Advanced/INDEX.md`.
