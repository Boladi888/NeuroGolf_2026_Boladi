> ⚠ **task_state.csv is ~1 MB (~272k tokens). NEVER Read it whole — query one row: `python tools/agent_helpers/show_task.py --task N`.**
> Note (2026-06-01 reconciliation): `solved_status` / unlock / cherry13b columns are ADVISORY-ONLY relics — every task scores (see current_strategy header).

# The_400_Tasks/ schema reference

Column documentation for the CSV ledgers in this folder.

## `task_state.csv` (one row per task — 401 rows)

Primary task ledger. `NeuroGolf_Task_Status.xlsx` is the rendered view and includes the same canonical column docs on its **Schema** sheet (54 columns after the 2026-05-23 live-arch/cherry13b tracking update).

Headline columns:

| Column | Meaning |
|---|---|
| `task` / `task_id` | Canonical ID (taskNNN / integer 0-400) |
| `family` | Semantic family (Pattern_Recognition, Spatial_and_Geometric, ...) |
| `complexity_1_10` | DeepSeek/karnak estimated complexity; blank only where source has no row |
| `status` | Workflow state (WIP, recipe-ready, locked-in, blocked, ...) |
| `best_known_status` | Per-task scoring verdict from prior runs |
| `priority` | H/M/L or numeric |
| `assigned_to` | Agent / branch on this task |
| `last_attempted_at` | **NEW** — ISO 8601 datetime of last *active attempt* (not just inspection) |
| `time_spent_attempting` | **NEW** — Cumulative duration (e.g., "2h30m") |
| `current_cost` / `current_src` / `current_points` | Our best per-task ONNX cost + source + LB points |
| `best_ext_cost` / `best_ext_src` / `ext_savings_pct` | Cheapest external observed + gap warning |
| `floor_proxy` / `gap_to_floor` | Cheapest legal op chain (sparse) |
| `macs` / `memory` / `params` | ONNX op profile |
| `recipe_exists` / `recipe_hint` / `recipe_path` | Recipe note presence |
| `prototype_path` | Semicolon-joined task-local prototype artifact paths (`.py`/`.onnx`) if any exist |
| `live_arch` | Auto-detected current/LIVE ONNX architecture: `lookup`, `lookup_confirmed_zero`, `trained_nn`, or `structural` |
| `cherry13b_tested` | Manual status, default `not_yet`; update after Identity/conv-identity public-weight diagnostic |
| `structural_attempt` | Manual status, default `not_yet`; update when a worker is assigned or a structural rewrite ships |
| `unlock_potential` | Computed: `HIGH` when `live_arch=lookup` and `cherry13b_tested` is `not_yet`/`regresses`, else `LOW` |
| `structural_kaggle_delta` | Manual Kaggle delta from structural-rule ship/probe |
| `verdict` / `next_action` | Human-readable status + what to do |
| `team_split` | **NEW 2026-06-06** — OnnxlyFans team assignment: `boladi` (our active 100), `aditya` (teammate's 100 — do NOT prompt/submit), `deferred` (200 covered by the team merge), `excluded_broken` (task000). Source: `Task Prompts/OnnxlyFans_TeamTasks.csv` (self-documented header) |
| `candidate_zips` | Comma-separated zip names containing this task |
| `local_payload_path` | Path to local ONNX selected |
| `reference` | Pointer to session log / digest |
| `n_batch_runs` / `last_batch` / etc. | Batch experiment metadata |
| `notes` | Free text |

To regenerate the CSV from the xlsx (after editing the xlsx):

```bash
python tools/merge_task_status.py
```

## `submission_candidates.csv` (one row per candidate zip — ~140 rows)

| Column | Meaning |
|---|---|
| `priority` | Sort order. Higher = more relevant. Negative = superseded but kept for lineage. |
| `candidate` | Candidate zip identifier (e.g., `fig11_elderberry11_tasks062_125_strict`). |
| `role` | Role in the ladder. Values: `current_valid_best`, `previous_valid_best`, `tested_public_positive_isolated`, `tested_public_positive_ladder`, `tested_public_negative_hidden_bad`, `tested_public_negative_pipeline_bad`, `tested_non_improving`, `superseded_stale_package`. |
| `local_payload_path` | Local path to the submission.zip. |
| `tasks_metadata_path` | Path to the candidate's tasks.txt manifest. |
| `task_count` | Number of tasks this candidate touched relative to its base. |
| `estimated_result` | Kaggle score + verdict. |
| `known_risks` | Free-text risk notes (cascade, hidden-bad, pipeline-bad, etc.). |
| `next_action` | What to do with this candidate. |
| `reference` | Pointer to session log / digest. |

## `submission_candidate_tasks.csv` (one row per task-within-candidate — ~1665 rows)

| Column | Meaning |
|---|---|
| `candidate` | Which `submission_candidates.csv` row this entry belongs to. |
| `task` | Task ID this entry covers. |
| `delta` | Per-task local score/cost delta. |
| `cost` | Per-task estimated cost. |
| `source` | Source supplying the task's payload (organic, external source name, etc.). |
| `payload_path` | Path to the per-task ONNX or zip extract. |
| `tasks_metadata_path` | Mirror of `submission_candidates.csv` row. |
| `base` | Base candidate (parent in the lineage) this candidate extends. |
| `candidate_estimated_local_delta` | Estimated local delta for the candidate as a whole. |
| `candidate_exclude_note` | If non-empty, the reason this row is excluded from the candidate's effective payload set. |

## Generated digests (Logs/)

- `Logs/submission_candidate_digest_<date>.{md,csv}` — generated by `tools/summarize_submission_candidates.py`.
- `Logs/next_isolated_probe_recommendations_<date>.{md,csv}` — generated by `tools/recommend_isolated_probes.py`.

Regenerate after CSV changes:

```bash
python tools/summarize_submission_candidates.py --date YYYY-MM-DD --out-prefix Logs/submission_candidate_digest_YYYY-MM-DD
python tools/recommend_isolated_probes.py --date YYYY-MM-DD --out-prefix Logs/next_isolated_probe_recommendations_YYYY-MM-DD
```

## `documentation_only_solved` status

A `best_known_status` value of `documentation_only_solved` covers tasks where the local ONNX exists but is hidden-bad on Kaggle (e.g., the silent-fail trap). Treat as not-SOLVED for scoring purposes but keep the row for evidence.
