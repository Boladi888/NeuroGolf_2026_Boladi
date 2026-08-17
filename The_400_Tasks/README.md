# The 400 Tasks

Active coordination surface for NeuroGolf scoring work. Keeps the small, reviewable records that collaborators (humans + AI agents) need; generated ONNXs, submission zips, profiler dumps, and competition artifacts stay local-only.

## Layout

### Per-task structured data (one row per task)

- **`task_state.csv`** — **source of truth.** Diffable, programmatic task ledger. **401 task rows × 54 columns** after the 2026-05-23 live-arch/cherry13b tracking update.
- **`NeuroGolf_Task_Status.xlsx`** — rendered, human-readable view of the CSV/task ledger. Color-coded master spreadsheet; the Schema sheet documents every column. Group of columns:
  - **Identity** — `task`, `task_id`
  - **Classification** — `family`, `transformations`, `complexity_1_10`, `grid_resized` (filled by `tools/classify_task_family.py`, 401/401 family rows classified; complexity imported from local karnak primitives where available)
  - **Workflow state** — `status`, `best_known_status`, `priority`, `assigned_to`, `last_attempted_at`, `time_spent_attempting`
  - **Cost / scoring** — `current_cost`, `current_src`, `current_points`, `best_ext_cost`, `best_ext_src`, `ext_savings_pct`, `floor_proxy`, `gap_to_floor`, `macs`, `memory`, `params`
  - **Verdict** — `verdict`, `next_action`
  - **Recipe / prototype pointers** — `recipe_exists`, `recipe_hint`, `recipe_path`, `prototype_path`
  - **Cherry13b / structural-unlock tracking** — `live_arch`, `cherry13b_tested`, `structural_attempt`, `unlock_potential`, `structural_kaggle_delta`
  - **Batch metadata** — `n_batch_runs`, `last_batch`, `last_batch_reviewed`, `best_train_acc`, `best_test_acc`, `n_external_candidates`
  - **Artifact links** — `candidate_zips`, `local_payload_path`, `reference`
  - **Free text** — `notes`
  - **Relational (auto)** — `ops_count`, `op_signature`, `unique_ops`, `graph_name`, `producer`, `similar_tasks`, `similar_task_count`, `current_src_shared_with`, `solved_status`, `hidden_bad_history`

### Per-task working notes (one .md per task)

- **`Task_by_Task/taskNNN.md`** — 401 files (000-400). Technical notes for AI agents: rule, algorithm sketch, attempts history, cost analysis, don't-do log. Auto-detects and links related `Research_Archive/notes/recipe-taskNNN-*.md` and `Research_Archive/organic_prototypes/taskNNN_*.py`. See `Task_by_Task/README.md` for the format and `tools/generate_task_mds.py` to regenerate the AUTO blocks.

### Historical ledgers / archives

- **`_archive/retired_top_level_ledgers_2026-06-21/`** — retired candidate-zip blend-ladder ledgers (`submission_candidates.csv`, `submission_candidate_tasks.csv`), their `SCHEMA.md`, a dated cost snapshot, and the old `CASCADE_ANALYSIS.md`. Current candidate/shipping truth moved to the OnnxlyFans `Team_Submission` manifest/staging flow.
- **`_archive/artifact_manifests/`** — provenance for the local scoring artifact archive (separate concern from task state). See its own README inside.

### Reports

- **`tools/cascade_report.py`** — can generate a fresh `The_400_Tasks/CASCADE_ANALYSIS.md` when a new cascade-risk snapshot is needed. The old checked-in snapshot is archived under `_archive/retired_top_level_ledgers_2026-06-21/`.

### Other

- For `task_state.csv` / `.xlsx` column docs, see the `Schema` sheet inside the xlsx.
- *(alpha_sub_lanes.md archived 2026-06-05 — lane system retired)*

## Tooling

All scripts live in `tools/` and produce / regenerate files in `The_400_Tasks/`. Run after editing the xlsx or after a new Kaggle submission lands.

### Source of truth + format

```bash
# Edit task_state.csv for programmatic changes, or NeuroGolf_Task_Status.xlsx for manual review.
python tools/merge_task_status.py        # regenerate the color-coded xlsx and CSV mirror
```

### Auto-classifiers / enrichment

```bash
python tools/classify_task_family.py [--rerun] [--mark-low-confidence]
# Fills 'family' and 'transformations' columns from:
#   1. Exact local overrides from durable Task_by_Task notes (task000)
#   2. Research_Archive/notes/recipe-taskNNN-*.md filename keywords (high confidence)
#   3. Input/output pair heuristics (medium confidence, marked yellow if --mark-low-confidence)
# Also fills blank 'complexity_1_10' values from:
#   data/competitor_artifacts/karnak-task-transformation/arc_primitives.csv
# Preserves the original user-trusted task001-task020 taxonomy rows.

python tools/enrich_task_state.py [--zip PATH]
# Recomputes relational columns plus live_arch/unlock_potential from submission.zip + filesystem:
#   ops_count, op_signature, unique_ops, graph_name, producer
#   similar_tasks, similar_task_count, current_src_shared_with
#   solved_status, hidden_bad_history, live_arch, unlock_potential, prototype_path
# Initializes manual tracking fields if blank:
#   cherry13b_tested, structural_attempt, structural_kaggle_delta
# Maintain the HIDDEN_BAD_TIDS set at the top of the file when a new task
# is confirmed in the silent-fail trap (see Logs/lessons.md).
```

### Per-task .md regeneration

```bash
python tools/generate_task_mds.py
# Regenerates the AUTO:BEGIN..AUTO:END block in every Task_by_Task/taskNNN.md
# (Pointer to xlsx row, auto-detected Research_Archive/notes/ + task-local prototype links.)
# Manual edits OUTSIDE the AUTO block are preserved.
```

### Reports

```bash
python tools/cascade_report.py            # writes a fresh The_400_Tasks/CASCADE_ANALYSIS.md if needed
# split_alpha_lanes.py archived — lane system retired 2026-06-05
# summarize_submission_candidates.py is historical; it reads the archived 2026-05 candidate CSVs directly.
python tools/recommend_isolated_probes.py --date 2026-05-XX
```

### Full refresh sequence (run after a Kaggle submission lands)

```bash
python tools/enrich_task_state.py    # 1. recompute ops / cascade signals from submission.zip
python tools/classify_task_family.py # 2. backfill family for any new tasks (optional --rerun)
python tools/merge_task_status.py    # 3. sync CSV mirror
python tools/cascade_report.py       # 4. refresh cascade analysis
python tools/generate_task_mds.py    # 5. refresh AUTO blocks in Task_by_Task/
```

## Workflow

1. Use the OnnxlyFans `Team_Submission` manifest / current leaderboard state for current candidate and shipping truth. The archived `submission_candidate*.csv` ledgers are historical blend-ladder evidence only.
2. Check one row via `tools/agent_helpers/show_task.py --task N` (⚠ never Read task_state.csv whole — 1 MB) before changing a task. Respect `avoid_or_retest` rows unless fresh evidence overturns the old result.
3. Open `Task_by_Task/taskNNN.md` for prose notes about the task's algorithm + attempt history.
4. **At the end of any focused attempt session**, update the row's `last_attempted_at` (ISO 8601 datetime) and accumulate `time_spent_attempting`. These two columns are the only ones that require manual hand-tracking — everything else is auto-populated or imported from logs.
5. After changing the xlsx or CSV, re-run `python tools/merge_task_status.py` so the two stay in sync.
6. For task153 replacements, run `python tools/_archive/task153_synthetic_guard.py <task153.onnx>` in addition to bundled verification. The `date9` task153 cost-displacement passed visible+arc-gen but failed this guard and scored hidden-bad.
7. Keep generated payloads out of git. Commit ledgers, helper scripts, and these tables.

## Cascade-risk discipline

The `current_src_shared_with` column is the single most actionable signal for prioritizing work:

- **High value** (e.g., 144 tasks share `thisray4808`): if you replace that source with a SAFE alternative on any one task, the lesson applies to up to 143 others. Conversely, if `thisray4808` causes a Kaggle cascade error, all 144 fail at once.
- **Use `tools/cascade_report.py`** to see the current top-N risk pools and the per-task cost / point distribution within each pool.

## When to add a new task .md vs a Research_Archive/notes file

- **`Task_by_Task/taskNNN.md`** — short prose: 1-3 sentences for the rule, a small attempts table, ruled-out approaches.
- **`Research_Archive/notes/recipe-taskNNN-*.md`** — multi-page algorithmic exploration; numpy verifier; cost-rewrite leads. Link to it from the `.md`.
- **`Research_Archive/organic_prototypes/taskNNN_*.py`** — actual ONNX builder + numpy reference. Link to it from the `.md`.

The auto-link generator finds these by filename — name files with the task ID and they'll appear in the AUTO block of the matching `Task_by_Task/taskNNN.md`.

## Migration history

- **2026-05-11** — Old `task_cards/` directory removed; 4 detailed files migrated into `Task_by_Task/`. User-started `NeuroGolf_Task_Status.xlsx` merged with `task_state.csv` into a single 38-column source of truth. New columns: `last_attempted_at`, `time_spent_attempting`.
- **2026-05-11** — Classifier filled `family` + `transformations` for 380 previously-empty rows (now 400/401).
- **2026-05-11** — Enrichment added 11 relational columns (ops_count, op_signature, similar_tasks, current_src_shared_with, solved_status, hidden_bad_history, prototype_path, etc.). xlsx column count now 49.
- **2026-05-23** — Backfilled `grid_resized`, `local_payload_path`, `live_arch` / cherry13b tracking, task-local prototype artifacts, and karnak `complexity_1_10` values. The ledger remains 54 columns.

No SQL database is needed for this layer yet. Four hundred task rows are easy to review and diff as CSV/xlsx; revisit SQLite only if the evidence turns into repeated joins across tens of thousands of verdict rows.
