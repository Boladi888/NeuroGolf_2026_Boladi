# Archived 2026-06-21 — stale `The_400_Tasks/` top-level files

These were moved here (not deleted) because the per-task **cost/candidate tracking moved to
`K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`** (+ `current_leaderboard_state.json`,
`Staged_*`, `submit.py`) around 2026-06-08. The files below froze at that point and are superseded.

Filed under `The_400_Tasks/_archive/retired_top_level_ledgers_2026-06-21/` on 2026-07-11 during the repo cleanup pass.

| File | Last updated | Why archived |
|---|---|---|
| `submission_candidates.csv` | 2026-05-10 | per-candidate-zip blend-ladder ledger; submission flow is now `staged_submits/` → `Staged_Boladi/` → `submit.py`. 6 weeks stale. |
| `submission_candidate_tasks.csv` | 2026-05-10 | per-task-within-candidate pool for the same retired blend ladder. (`SCHEMA.md` documents only these two.) |
| `latest_package_costs_2026-06-02.csv` | 2026-06-02 | one-off dated cost snapshot. |
| `CASCADE_ANALYSIS.md` | 2026-06-04 | auto-generated report; regenerate fresh anytime via `tools/cascade_report.py`. |
| `SCHEMA.md` | 2026-06-06 | documented ONLY the two `submission_candidate*.csv` above — moved here with the data it describes. |

**Still live in `The_400_Tasks/` (NOT archived):** `task_state.csv` + `NeuroGolf_Task_Status.xlsx`
(54-col metadata ledger — note its *cost* columns are stale; `manifest.csv` is the live cost truth),
`current_strategy.md`, `live_costs.json` (read by `context_pack`/`generate_task_mds`), `Task_by_Task/*.md`,
`README.md`.

**To restore any file:** move it from `The_400_Tasks/_archive/retired_top_level_ledgers_2026-06-21/` back to `The_400_Tasks/` (their generator tools still exist under `tools/`).

Also deleted 2026-06-21 (git-tracked, recoverable via history): the `_archive` *file* (no extension) — the
retired alpha sub-lane assignment, lane system retired 2026-06-05.
