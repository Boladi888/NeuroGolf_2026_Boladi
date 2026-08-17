# AGENT_TASK_START — single-task session router (20 lines, read this first)

Assigned "work task NNN"? This is your whole read path. Do not browse archives, logs, or
adjacent tasks unless your prompt says so.

1. **Your prompt** (usually `The_400_Tasks/Task Prompts/1/` or `/2/`) — target, direction, gates.
2. **Worker contract**: `Onnx_Advanced/WORKER_PROTOCOL.md` — read order, gates, staging. If a prompt conflicts with it, the protocol wins.
3. **Task state + history in one command**: `python tools/agent_helpers/context_pack.py --task NNN` — ~14 KB curated pack (state row, rule, don't-dos, prompt, staged, cited §s). Or piecewise: `show_task.py --task NNN` + `The_400_Tasks/Task_by_Task/taskNNN.md`. NEVER read task_state.csv whole (1 MB).
4. **Library**: `Onnx_Advanced/INDEX.md` → `core.md` → only the §-sections your prompt names.

Build in `The_400_Tasks/Task_by_Task/taskNNN/prototypes/`. Gate with `tools/ng_gate.py`.
Score with `tools/_score_v2.py`. Stage to `The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/` (ONNX + manifest). **Never submit
to Kaggle — the captain ships.** Never create git branches/worktrees. Never run `git push`.

Filesystem layout (junctions, what's where): `WHERE_DATA_LIVES.md` — only if you hit a path problem.
Tool catalog: `tools/README.md` + `Onnx_Advanced/diagnostics_recipes.md` — check before writing a new script.
Team split (who works which tasks): `The_400_Tasks/Task Prompts/OnnxlyFans_TeamTasks.csv` header.
