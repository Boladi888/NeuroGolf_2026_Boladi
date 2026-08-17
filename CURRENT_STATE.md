# CURRENT_STATE.md — pointer stub (2026-06-06)

> The auto-generated snapshot that used to live here went stale repeatedly (last regen
> 2026-05-25, flagged by three audits), and its generator `tools/update_current_state.py`
> is hermes-coupled (reads `.hermes/board/`, shells `hermes kanban`). That board tooling
> is retired, so the generator needs a patch before it can run again. Rather than another
> stale snapshot, this stub routes you to sources that are
> always current:

| Want | Source |
|---|---|
| Current Kaggle LB + last submits | last rows of `Logs/submissions_ledger.csv` |
| One task's state (cost/points/family/history) | `python tools/agent_helpers/show_task.py --task N` |
| Strategy, team split, batch rosters | `The_400_Tasks/current_strategy.md` (header + §TEAM-SPLIT) |
| Active worker prompts | `The_400_Tasks/Task Prompts/1/` and `/2/` |
| Staged candidates awaiting ship | `The_400_Tasks/Task_by_Task/task*/staged_candidates/` |
| Per-task costs (machine-readable) | `The_400_Tasks/task_state.csv` |
| Filesystem layout / junctions | `WHERE_DATA_LIVES.md` |
| Single-task session router | `AGENT_TASK_START.md` |

If a future `refresh_all.py` revives an auto-generated snapshot here, it must stamp a
`generated-at + source-hash` header and be wired into the captain's post-ship flow —
otherwise it will go stale again (it has, three times).
