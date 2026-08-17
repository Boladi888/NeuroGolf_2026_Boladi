# Agent A Cleanup Report

## Scope

Owned paths handled: root-level files, `Competitor_Notebooks/`, and the root `.notongit/` decoy if present.

## Inventory and Reconciliation

Tracked starting inventory from `git ls-files`:

| Area | Before |
|---|---:|
| Root-level tracked files | 13 |
| `Competitor_Notebooks/` tracked files | 21 |
| `.notongit/` tracked files | 0 |
| **Total owned tracked files** | **34** |

Tracked outcome:

| Outcome | Count | Notes |
|---|---:|---|
| Deleted | 22 | 20 competitor payload/scaffold files plus `NG_Gate_Queue_Watch.bat` and `NG_OnnxlyFans_TaskStatus.xlsx`. |
| Moved/integrated | 1 | The useful competitor survey moved from `Competitor_Notebooks/competitor_notebooks_2026-05/00_INDEX.md` into `Competitor_Notebooks/README.md`; the old path is now a pointer. |
| Kept | 11 | Root files retained because they are visitor-facing, legal, config, or compatibility anchors. |
| **Reconciled total** | **34** | 22 deleted + 1 moved/integrated + 11 kept. |

Filesystem checks:

| Area | Before | After |
|---|---:|---:|
| `Competitor_Notebooks/` recursive entries with `-Force` | 23 | 3 |
| Root `.notongit/` decoy entries | 0; path absent | 0; path absent |
| Requested untracked root strays (`NUL`, root `task*.onnx`, ORT profile JSON) | 0 found | 0 found |

The requested root strays were not present in this worktree, so there was nothing to delete for that part. I verified the root and `Competitor_Notebooks/` parents had no `LinkType` entries before considering directory cleanup.

## Lint Counts

The live lint baseline in this worktree did not match the prompt's master baseline. I used the live before counts as the hard gate.

| Linter | Before | After |
|---|---:|---:|
| `python tools/repo_lint.py` errors | 32 | 32 |
| `python tools/repo_lint.py` warnings | 859 | 859 |
| `python Onnx_Advanced/.update/tools/lib_lint.py` errors | 2 | 2 |
| `python Onnx_Advanced/.update/tools/lib_lint.py` warnings | 586 | 586 |

`repo_lint.py` writes `Logs/lint_last.md`; I restored that generated side effect because `Logs/` is outside Agent A scope.

## What I Kept

- `README.md` now presents the repo as a completed archive instead of a fresh-clone working loop.
- `AGENT_TASK_START.md`, `.ai_instructions/`, `The_400_Tasks/current_strategy.md`, `Logs/lessons.md`, and `Onnx_Advanced/INDEX.md` remain the recommended read path for a future agent.
- `WHERE_DATA_LIVES.md` remains the junction and artifact safety map, with the four-junction recovery section made self-consistent.
- `CURRENT_STATE.md` remains a pointer stub; I did not restore the stale generated snapshot.
- `.gitignore` remains the shared cleanup contract, with stale comments fixed for `.ai_instructions`, `submissions/`, and `BatchPilotArchive/`.
- `build_submission.py` remains as a small retired-stub anchor so inbound filename references still resolve without keeping a 942 KB half-disabled dispatcher at the root.
- `Competitor_Notebooks/README.md` keeps the valuable defensive survey: competitor workflow, cost-metric exploit recon, silent-fail risk, and follow-up artifact slugs.
- `Competitor_Notebooks/competitor_notebooks_2026-05/00_INDEX.md` remains only as a compatibility pointer to the moved survey.
- `LICENSE` and `NOTICE` stay at root as the legal first-stop.

## Licensing Decisions

I deleted the 14 third-party Kaggle notebooks and 5 competitor zip files from `Competitor_Notebooks/` because all notebook metadata was `lic: NONE` and root `NOTICE` otherwise implied Apache-2.0 coverage over the whole tree. I kept only the analytical survey, rewritten as historical notes rather than a vendored-source index.

The onnx-tool reference dump under `tools/onnx_tool_1.0.1_reference/` is outside Agent A scope, but `NOTICE` is Agent A scope. Final NOTICE attribution is pending Agent F's exact proposed text.

## Could Not Decide / Recommendation

- `build_submission.py`: I chose the smallest root-facing cleanup: replace the huge retired dispatcher with a stub that points to git history. If the owner later wants the old script visible in-tree again, restore it under an archive/tools location owned by Agent F, not at root.
- `_cleanup/proposals-F.md`: not present yet at the time this report was written, after repeated polling. Add the exact onnx-tool attribution to `NOTICE` before final closeout.

## Out-of-Scope Issues Noticed

- `Handoff/Research/ng_gate_queue_visualizer_2026-07-09/implementation_notes_2026-07-09.md` still mentions the deleted root `NG_Gate_Queue_Watch.bat`. That file is outside Agent A scope.
- `Agenters/Janitor/research_archive_cleanup_plan.md` still discusses deleting/migrating `Competitor_Notebooks/`; that plan is now partly stale and outside Agent A scope.
- The required linters still report many pre-existing out-of-scope issues; counts did not increase.
