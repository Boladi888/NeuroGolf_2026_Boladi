# Agent C Cleanup Report

Scope: `The_400_Tasks/` plus this report. No runtime project commands were run; this
worktree has no `.venv/` or `data/`, by instruction.

## Inventory Reconciliation

Tracked files under `The_400_Tasks/` before cleanup: **12,435**.

Final projected tracked files after staging additions: **1,609**.

| Category | Files | Disposition |
|---|---:|---|
| `Task_by_Task/task*/prototypes/**` | 7,048 | Deleted. Attempt builders, ONNX payloads, zips, and gate dumps were scratch artifacts. |
| `Task_by_Task/task*/staged_candidates/**` | 1,732 | Deleted. Payload ONNXs were not tracked; manifests were summarized in the index. |
| `Task Prompts/_archive/**` original files | 2,008 | Deleted. One linked prompt path was replaced with a tombstone for `lib_lint`. |
| `Task Prompts/70/**` | 50 | Deleted. Kept round `71/` as the single worked prompt example. |
| Prompt selection summaries | 3 | Deleted. They pointed at deleted prompt rounds; durable lessons live in prompt research. |
| `task076/candidates/task076_table_lookup_fp16_t_e1ad7998/**` | 3 | Deleted. One 988,758-byte scratch candidate bundle with a tracked ONNX and zip. |
| **Deleted total** | **10,844** | Original tracked files removed from the final archive. |
| Tracked moves | 21 | 9 staged-candidate exemplar files, 11 archive-debris files, and `Task_by_Day/task349.md`. |
| New integrated archive files | 18 | 17 new `taskNNN_history.md` files plus `current_strategy_session_log_2026-07-15.md`. |
| Tombstone replacements | 2 | Kept exact paths for an old Boladi work-plan link and one archived prompt link so `lib_lint` did not regress. |

Reconcile: `12,435 - 10,844 + 18 = 1,609`.

Filesystem after cleanup: **1,609 files / 2,367 items** under `The_400_Tasks/`.

## Lint Counts

Observed baseline in this worktree differed from the prompt's reference numbers:

| Linter | Before | After |
|---|---:|---:|
| `python tools/repo_lint.py` errors | 32 | 5 |
| `python tools/repo_lint.py` warnings | 859 | 75 |
| `python Onnx_Advanced/.update/tools/lib_lint.py` errors | 2 | 2 |
| `python Onnx_Advanced/.update/tools/lib_lint.py` warnings | 586 | 586 |

`repo_lint` target was met: the 27 `task-md-size` errors are gone. The remaining
5 `repo_lint` errors are `Onnx_Advanced` dead-tool references outside Agent C
scope. `lib_lint` stayed at the observed baseline after adding two compatibility
tombstones for out-of-scope Agenters links.

## Kept Because It Earns Its Place

- **All 401 `Task_by_Task/taskNNN.md` files**: these are the durable per-task
  lessons. The 27 oversized files now keep current rule/conclusion material in
  the main file and point to `_archive/taskNNN_history.md` for long history.
- **All 877 `hard_records/` files**: high-value wall proofs, bars, and
  do-not-retry evidence.
- **`Task_by_Task/STAGED_CANDIDATES_INDEX.md` plus 8 exemplar candidate dirs
  (9 files)**: the index is now the ledger; examples preserve representative
  proof-record shapes without keeping 1,700+ payloadless manifests.
- **`Task Prompts/71/`, prompt templates, `_Before writing prompt.md`, README,
  and `PROMPT_RESULTS_RESEARCH_2026-07-10.md`**: one complete worked prompt
  wave plus the reusable authoring doctrine.
- **`current_strategy.md`**: converted from stale append-only session log into a
  retrospective doctrine file. Full old body is archived at
  `The_400_Tasks/_archive/current_strategy_session_log_2026-07-15.md`.
- **Two tiny tombstones**: kept only to avoid increasing `lib_lint` warnings
  from outside-scope Agenters links.

## Decisions / Recommendations

- I did **not** strip generated AUTO/HIST/SPREAD/AUTO-GRAPH blocks from task
  docs. The prompt recommended against it, and the lint errors measured manual
  content only.
- I did **not** edit out-of-scope Agenters or Onnx_Advanced links. Agenters run
  history still has many references to deleted prompt waves; another agent owns
  that cleanup.
- After the Agenters cleanup merges, the owner can probably delete the two
  tombstones if `lib_lint` no longer needs them:
  - `The_400_Tasks/Task Prompts/_WORK_PLAN_Boladi_2026-06-10.md`
  - `The_400_Tasks/Task Prompts/_archive/ran_20260621/Prompt_2026-06-21T0004Z_Task083_ScatterMirrorVsDeepEinsum.md`

## Out Of Scope Noted

- `Agenters/Prompter/README.md` still links to the retired Boladi work-plan path.
- `Agenters/Orchestrator` run history still links to many historical prompt
  files and staged-submit paths.
- The required linters update `Logs/lint_last.md` outside Agent C scope; I did
  not stage that file.
