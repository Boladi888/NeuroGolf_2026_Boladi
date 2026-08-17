# Next supervisor start here

Created UTC: 2026-07-03T00:44:19Z

## Current safety state

- No HandOff runner/monitor process was found for the active lanes after cleanup.
- No live `task125`, `task157`, `task268`, `runner_cdx2`, `ng_gate.py`, `Task_by_Task`, or `search_task` process was found from this workspace in the final process sweep.
- Do not restart Codex2 batch 42.
- Do not restart Codex batch 43 until runner/tooling fixes below are in place.
- Do not stage to `Staged_Boladi`.
- Do not run live Kaggle submit.
- Do not probe unless the user explicitly asks and the shared home zip/Kaggle account is idle.

## Stopped lanes

### Codex2 batch 42

- PID file: `Handoff/private/runner_loop.codex2.pid.json`
- Final PID-file state: `stopped`
- Exit reason: `manual_stop_resource_pressure_task125_task157_respawn`
- Stop note: `Handoff/logs/codex2_p42_aborted_resource_pressure.md`
- Config: `Handoff/config/runner_codex2_p42.json`
- State: `Handoff/private/runner_state_codex2_p42.json`
- State at manual stop: `ship_packeted=3`, `drop=4`, `worker_running=3`, `todo=2`
- Useful packets written before stop: `task008`, `task086`, `task245`
- Resource-pressure tasks: `task125`, `task157`, `task369`

Critical details:

- `task125` launched `The_400_Tasks/Task_by_Task/task125/prototypes/search_task125_scalar_code.py --build`.
- `task125` first survived a visible Codex child kill because its local Python search had escaped the process tree.
- `task157` launched `tools/ng_gate.py truth --task 157 ... --full`.
- Killing a hot task child while the runner was still alive caused the runner to respawn more work. Pause/stop the runner before killing children in future incidents.

### Codex batch 43

- PID file: `Handoff/private/runner_loop.codex.pid.json`
- Final PID-file state: `stopped`
- Exit reason: `manual_abort_for_handoff_tooling_fix`
- Stop note: `Handoff/logs/codex_p43_aborted_tooling_fix.md`
- Parked config: `Handoff/config/aborted_batches/runner_codex_p43_aborted_20260703T003847Z.json`
- Parked state: `Handoff/private/aborted_batches/runner_state_codex_p43_aborted_20260703T003847Z.json`
- Restart prompt note: `The_400_Tasks/Task Prompts/43/_RESTART_BATCH43_AFTER_TOOLING_FIX.md`
- Completed/drop/excluded prompts were moved to: `The_400_Tasks/Task Prompts/43/_archived_after_abort_20260703T003847Z`
- Restartable prompts remain in: `The_400_Tasks/Task Prompts/43`

Packets already handed to alpha lane before p43 abort:

- `task050`
- `task139`
- `task153`
- `task161`
- `task214`

Reviewed drops/no-gain before p43 abort:

- `task229`
- `task248`

Worker timeout/failed before p43 abort:

- `task074`
- `task224`

Interrupted/restartable p43 tasks:

- `task071`
- `task232`
- `task260`
- `task290`
- `task305`
- `task355`
- `task384`
- plus timeout-retry tasks left in the prompt folder

## Reports to read before launching anything

ONNX/team win-rate critique:

- `Agenters/_proposals/p42_resource_pressure_onnx_win_actions_2026-07-02.md`
- `Agenters/_proposals/codex_p43_onnx_win_rate_actions_2026-07-02.md`
- `Agenters/_proposals/onnx_win_rate_improvement_actions_2026-07-02.md`

HandOff/platform feedback:

- `Handoff/_proposals/codex2_p42_handoff_platform_feedback_2026-07-02.md`
- `Handoff/_proposals/codex_p43_handoff_platform_feedback_2026-07-02.md`
- `Handoff/_proposals/codex_p39_handoff_platform_feedback_2026-07-02.md`

## Required tooling fixes before another high-concurrency wave

Do not start another 3-worker HandOff runner until these are addressed or explicitly waived:

1. Attempt process containment: every worker/reviewer attempt must run under a killable Windows Job Object or equivalent process-group wrapper.
2. Gate wrapper: no raw `tools/ng_gate.py ... --full`, `--no-throttle`, exact-search, or MILP-style command from worker shells.
3. Task-level stop/park: add a safe command that kills one task's attempt tree and marks it terminal so the runner cannot respawn it.
4. Batch pause: add a way to freeze scheduling before killing hot child processes.
5. Structured reviewer rework budgets: reviewer `KEEP_WORKING` must include machine-enforced wall-clock, gate, candidate, and forbidden-command limits.
6. Lease-based board ownership: stop board self-healing from turning locally active work back to `todo`.
7. Orphan gate scan: monitor must find dead-parent or escaped `ng_gate.py`/search processes automatically.
8. Idempotent stop scripts: `Stop-HandoffBatch.ps1` must stamp pid files even if taskkill reports an already-dead child.

## First commands for the next supervisor

Run these read-only checks before any launch:

```powershell
Get-Content Handoff\private\runner_loop.codex.pid.json
Get-Content Handoff\private\runner_loop.codex2.pid.json
Get-Content Handoff\private\runner_loop.claude.pid.json
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'Start-HandoffRunner|Start-HandoffMonitor|handoff_runner.py|handoff_monitor.py|ng_gate.py|Task_by_Task|search_task' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine
Get-ChildItem 'The_400_Tasks\Task Prompts\43' -Force
Get-Content 'The_400_Tasks\Task Prompts\43\_RESTART_BATCH43_AFTER_TOOLING_FIX.md'
```

If any hot gate/search process is found, identify its task and parent chain before killing it. If a runner is alive, stop or pause the runner first so it cannot respawn the task.

## Submit Captain state

Top-level `Agenters/Submit_Captain/Updates_Codex` still contained packets at the last check, including:

- `20260702T221853Z-task050.json`
- `20260702T215202Z-task214.json`
- `20260702T214058Z-task161.json`
- `20260702T205150Z-task153.json`
- `20260702T203323Z-task245.json`
- `20260702T202745Z-task139.json`

Do not infer staged/accepted status from presence in `Updates_Codex`; alpha owns final staging. Use Captain digest/preflight only if the user asks and shared probe resources are idle.

## Restart recommendation

The safest next move is not another broad runner. The next supervisor should either:

1. implement the runner/gate containment fixes, then resume p43's restartable prompt folder at low concurrency; or
2. run only a single low-risk micro-shave task manually with guarded gates and no reviewer rework.

Do not resume p42.

