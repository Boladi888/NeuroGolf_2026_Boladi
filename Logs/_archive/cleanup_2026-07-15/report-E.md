# Agent E Cleanup Report

Date: 2026-07-15
Worktree: `<CODEX_WORKTREES>\c187\K-NeuroGolf`
Owned paths: `Agenters/`, `Handoff/`, `GUI/`, `.update/`, `.agents/`, `.codex/`, `.hermes/`

## Inventory and Reconciliation

System of record for tracked files was `git ls-files -- <path>`. Disk browse checks used `Get-ChildItem -Recurse -Force`. This worktree did not contain the main checkout's ignored `Handoff/logs/` spill or `.hermes/` tree.

| Path | Before tracked | Deleted tracked | Moved | Integrated/added | Kept tracked |
|---|---:|---:|---:|---:|---:|
| `Agenters/` | 1,453 | 1,381 | 0 | 0 | 72 |
| `Handoff/` | 431 | 351 | 0 | 0 | 80 |
| `GUI/` | 432 | 400 | 0 | 1 | 33 |
| `.update/` | 46 | 0 | 0 | 0 | 46 |
| `.agents/`, `.codex/`, `.hermes/` | 0 | 0 | 0 | 0 | 0 |
| **Total** | **2,362** | **2,132** | **0** | **1** | **231** |

Reconcile: `2,362 before - 2,132 deleted + 1 added = 231 kept`.

Tracked byte footprint for the owned paths went from 124,255,523 bytes to 4,274,209 bytes, a net reduction of 119,981,314 bytes. The final disk entry counts in this worktree are: `Agenters/` 95, `Handoff/` 94, `GUI/` 42, `.update/` 48.

## Deletes

- Deleted 60 closed `Agenters/Orchestrator/runs/` run directories. Kept `runs/_template/`, `runs/README.md`, and `runs/beta_p8c_claude_cli/`.
- Kept `beta_p8c_claude_cli` because it has no `RUN_CLOSED.md`; its `task_claims.csv`, `reviewer_leases.csv`, and `final_report.csv` are header-only, but the launch packet says not to close while unresolved.
- Deleted `Handoff/logs/`, per-run `Handoff/config/runner_*_pNN.json` files, `Handoff/config/aborted_batches/`, implemented/archive proposal folders, and two stale `Handoff/archive/batch30/` runner snapshots.
- Deleted 400 tracked `GUI/client/public/tasks/task*.json` payloads and added `GUI/client/public/tasks/README.md` as the regeneration breadcrumb.
- `.agents/`, `.codex/`, and `.hermes/` had 0 tracked files and were absent on disk in this worktree.

## Lint Gate

The prompt's master snapshot said `repo_lint` was 32 errors / 869 warnings and `lib_lint` was 0 errors / 573 warnings. The live local baseline on this worktree before edits had already drifted:

| Linter | Local before | After cleanup | Delta |
|---|---:|---:|---:|
| `python tools/repo_lint.py` | 32 errors / 859 warnings | 32 errors / 859 warnings | 0 / 0 |
| `python Onnx_Advanced/.update/tools/lib_lint.py` | 2 errors / 586 warnings | 2 errors / 107 warnings | 0 / -479 |

Both linters returned exit code 1 before and after; counts above are from the summary lines, not process exit code. `repo_lint.py` rewrites `Logs/lint_last.md`; I restored that out-of-scope generated file after each run.

## Kept and Why

- `Agenters/README.md`, role READMEs, role `SPAWN_PROMPT.md` files, Orchestrator machinery, and run template stay because they are the portable multi-agent doctrine: floor claims, reviewer affinity, leases, lane routing, and parallel write-scope safety.
- `Agenters/Submit_Captain/captain_ctl.py`, queue READMEs, and Janitor retirement plans stay because they are the reusable submit/captain and cleanup-control knowledge.
- `Handoff/runner/` and `Handoff/scripts/` stay because they are real client code for the external HandOff platform. The retained docs are dated 2026-07-03 snapshots; I recommend keeping them as frozen API evidence while this client remains in the archive.
- `Handoff/config/public.json`, `runner.json`, and `runner_claude.json` stay as durable examples. I removed the machine-specific `<USER_HOME>/...K-NeuroGolf-OnnxlyFans` path from `runner_claude.json`.
- `Handoff/_proposals/2026-07-10_runner_fastmode_fable_passthrough.md` stays because it was the only possibly open proposal called out in scope.
- `GUI/` stays because NeuroGolf Lab is the most demo-able surface in this repo: a localhost ARC task viewer, ONNX graph editor, and Doors Maze mode. I added a server-side `/tasks/taskNNN.json` fallback so the app can serve task data from the local `data/` junction or `NEUROGOLF_DATA_DIR` instead of tracking 92 MB of duplicated JSON.
- `.update/` stays unchanged for now. It is drained, but its 46 archived files are unique relative to `Onnx_Advanced/.update/`; consolidation belongs after Agent B's provenance fix.

## Could Not Fully Decide

- `beta_p8c_claude_cli` looks stale but not formally closed. Recommendation: owner should either add a closure note or explicitly archive it after confirming no pending reviewer/worker state matters.
- Hermes is inconsistent. In this worktree, both `.hermes/` and `_hermes_archive/` are absent and 0 files are tracked. `CURRENT_STATE.md` still claims retirement to `_hermes_archive/`, while scope says the main checkout has a `.hermes/board/submit_credit_state.json` modified today. Recommendation: do not delete or ignore `.hermes/` until Agent F or the owner reconciles the writer and the remaining `tools/` Hermes references.
- GUI runtime was not verified. This worktree has no `.venv/` and no `data/` junction, so I only ran `python -m py_compile GUI/server.py` and JSON parsing on retained configs.

## Out of Scope Noticed

- `Agenters/Supervisor/NEXT_SUPERVISOR_START_HERE.md` still points at deleted `Handoff/logs/codex2_p42_aborted_resource_pressure.md` and `Handoff/logs/codex_p43_aborted_tooling_fix.md`. Supervisor is outside Agent E scope; those decisions should be folded into the Supervisor doc or left to git history.
- `CURRENT_STATE.md` is outside scope and has the stale Hermes retirement claim.
- `tools/` Hermes references are Agent F scope.
- `Handoff/Research/` manifests contain dated absolute paths and references to deleted log/proposal files. I kept them as an audit packet rather than rewriting archived evidence.
- `Handoff/skills/README.md` still contains absolute example paths. I left it untouched because the prompt explicitly said to leave the skills breadcrumb alone.

## Commits

- `5ee9084b cleanup(agent-e): drop stale run filings`
- `f642df0a cleanup(gui): drop generated task payloads`
- `a02e7f39 cleanup(agent-e): document retained runtime surfaces`
