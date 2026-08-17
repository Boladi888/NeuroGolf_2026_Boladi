# NeuroGolf HandOff Runner

This runner is the local automation layer for `NeuroGolf_HO`. HandOff stores board/project state; this script runs the actual Codex role sessions and writes operational records.

Default behavior is safe:

- no Codex worker/reviewer is launched unless `-RunCodex` is passed;
- no Kaggle-affecting captain command is launched unless `-AllowKaggleSubmit` is also passed;
- raw Codex transcripts go under `Handoff/logs/raw/runner/` and are gitignored;
- task session ids live in `Handoff/private/runner_state.json` and are gitignored.

## Workflow

1. A worker agent takes one task and produces an initial result.
2. The configured `worker_attempts_before_review` decides whether the task goes directly to review (`1`, current default) or gets another worker pass first.
3. A reviewer agent checks the worker packet.
4. If reviewer says `KEEP_WORKING`, the runner resumes the same worker session with reviewer notes.
5. If reviewer says `SHIP`, the runner validates the packet schema and writes it into the lane's Submit Captain inbox.
6. The captain path uses `captain_ctl.py` only after enough packets accumulate.
7. Captain probe/stage commands remain behind `-AllowKaggleSubmit`.

## Commands

Render a batch config from a prompt folder, creating/reusing board tasks with stable external keys:

```powershell
python Handoff\runner\prepare_handoff_batch.py --lane Codex --batch 35 --prompt-folder "The_400_Tasks/Task Prompts/35" --sync-board --force-all-prompts --expected-count 30
```

Prompt-folder renders are exact by default. `--exclude` is refused unless
`--allow-excludes` is also passed, and `--force-all-prompts` ignores any
inherited exclude list. Use `--expected-count` for 30-prompt waves so a board or
history filter cannot silently shrink the batch.

Dry-run one decision:

```powershell
.\Handoff\scripts\Start-HandoffRunner.ps1 -Once -DryRun
```

Run one real worker/reviewer step:

```powershell
.\Handoff\scripts\Start-HandoffRunner.ps1 -Once -RunCodex -SyncHandoff
```

Run the loop without Kaggle submits:

```powershell
.\Handoff\scripts\Start-HandoffRunner.ps1 -Loop -RunCodex -SyncHandoff -PollSeconds 90
```

Run a rendered batch and exit the loop automatically when every configured task is terminal:

```powershell
.\Handoff\scripts\Start-HandoffRunner.ps1 -Lane Codex -Loop -RunCodex -SyncHandoff -ConfigPath Handoff/config/runner_codex_p35.json -Batch 35 -PollSeconds 90 -ExitWhenDrained
```

Enable the captain's Kaggle-affecting probe path only after the dry-run and the first packets look right:

```powershell
.\Handoff\scripts\Start-HandoffRunner.ps1 -Loop -RunCodex -SyncHandoff -AllowKaggleSubmit -PollSeconds 90
```


## Scheduling, retries & safety (updated 2026-06-30)

The runner uses a **refilling concurrent scheduler** instead of a fixed
per-step batch. Up to `worker_max_active` workers and `reviewer_max_active`
reviewers run at once, and a freed slot is refilled the instant a task finishes
— a single slow worker no longer holds the other slots idle until the whole
batch drains (the batch31 stall). Workers and reviewers pipeline: a task a
worker pushes to `pending_review` is picked up by a reviewer as soon as a
reviewer slot frees. One real run (`-Loop`) drains the batch to terminal; the
loop just re-invokes to catch stragglers.

Tunable knobs in `Handoff/config/runner.json` → `workflow`:

| Key | Default | Meaning |
|---|---|---|
| `worker_max_active` | 3 | concurrent worker sessions; `prepare_handoff_batch.py` normalizes Codex/Codex2 batches to 3 unless the config is intentionally edited afterward |
| `reviewer_max_active` | 2 | concurrent reviewer sessions |
| `worker_attempts_before_review` | 1 | successful worker passes before review. `1` = single pass then review (the cost-saving default, set 2026-06-30). Set `2` to force a second improvement pass before review (~2× worker cost, may raise win rate on hard tasks) |
| `worker_max_failures` | 2 | failed worker attempts retried (fresh session) before a task is marked `worker_failed` |
| `reviewer_max_failures` | 1 | failed/unparseable reviews retried before `review_failed`/`review_parse_failed` |
| `max_rework_cycles` | 3 | `KEEP_WORKING` rework rounds before the task is dropped (prevents an infinite reviewer↔worker loop — important now that one run drains tight, with no inter-step sleep) |

`force_first_rework` is **unused** by the code (set `false` to match reality);
the forced second pass is controlled entirely by `worker_attempts_before_review`.

A failed or timed-out worker no longer strands the task: it retries from a fresh
session up to `worker_max_failures`, then settles on `worker_failed`. The per-call
timeout is **role-aware and generous** (2026-07-02, `_timeout_for()`): workers use
`worker_timeout_seconds` (default `14400` = 4h), reviewers use
`reviewer_timeout_seconds` (default `10800` = 3h), both falling back to
`timeout_seconds` then a `14400` default. The brief `2700` (45m) cap was reverted
because it was reaping *productive* workers mid-gate; the ceiling only bites a
*hung* attempt (a productive worker returns as soon as it finishes), and timed-out
sessions are still killed with their whole process tree (`taskkill /T /F`).

### Process ownership and orphan cleanup

Runner-launched Codex sessions now export `NEUROGOLF_HANDOFF_LANE`,
`NEUROGOLF_HANDOFF_BATCH`, `NEUROGOLF_HANDOFF_TASK`, `NEUROGOLF_HANDOFF_ROLE`,
`NEUROGOLF_HANDOFF_AGENT`, `NEUROGOLF_HANDOFF_RUN_ID`, `NEUROGOLF_HANDOFF_RAW_DIR`,
and `NEUROGOLF_HEAVY_GATE_N=8000` into every worker/reviewer process. `ng_gate`
copies that owner metadata into intent and lock sidecars under the shared throttle
directory before and during gate acquisition, so a waiting or active gate can be
attributed without guessing from command lines.

Each batch writes an append-only process registry at
`Handoff/private/process_registry_<lane>_p<N>.jsonl` with `start`, `exit`,
`terminate_attempt`, `orphan_sweep`, and `cleanup_verified` events keyed by
lane/batch/task/role/run id. After every worker/reviewer return and at scheduler
shutdown, the runner sweeps only owner-matched gate processes for that task; it
does not kill other lanes/batches. `write_closeout.py` reads this registry and
refuses normal closeout if no cleanup audit was recorded.

`Start-HandoffRunner.ps1 -Loop -Run*` has a single-instance guard scoped to the
lane, batch, and normalized config path. A live matching runner refuses a
duplicate start; a dead or command-line-mismatched pid file is treated as stale
and overwritten on the next launch.

### Codex sandbox (Windows)

codex 0.142.4's *unelevated* restricted-token sandbox fails closed on this box
("cannot enforce split writable root sets"), which blocked **all** worker file
writes in batch31. The posture is now controlled via `config.json` → `codex`:

- `bypass_sandbox: true` (current setting) → passes
  `--dangerously-bypass-approvals-and-sandbox`. Reliable on a trusted
  single-user machine; the agent already runs `approval_policy=never`. Verified:
  a worker writes files with no sandbox refusal.
- `bypass_sandbox: false` + `config_overrides: ["sandbox_workspace_write.exclude_tmpdir_env_var=true", "sandbox_workspace_write.exclude_slash_tmp=true"]`
  → keep the sandbox, collapse to a single writable root (untested on 0.142.4).
- Or set `[windows] sandbox = "elevated"` in `~/.codex/config.toml` (one-time UAC).

### Monitoring

`Start-HandoffMonitor.ps1` watches **artifacts** (state freshness + task
statuses), not just process liveness, and now emits on every terminal condition
— `complete` (all tasks terminal), `interrupted` (runner pid dead with work
remaining), `stalled` (alive but state unchanged past `--stale-seconds`, default
5h/18000s, kept above the 4h worker `timeout_seconds` so the runner's own
reap+retry fires first — a tail worker can legitimately run for hours). Pass `--exit-on-terminal` to stop the
loop on completion/crash. Both the runner loop and the monitor stamp their pid
files with `state`/`stopped_utc`/`exit_reason` on exit, so a dead loop is itself
detectable (the old loops left a stale `running` pid file — the batch31 monitor
silently orphaned).

Pass `-CheckBoard` to `Start-HandoffMonitor.ps1` when you want read-only
board/local drift detection (it forwards the lane `-ConfigPath` as `--config` so
the monitor can read the board project id). Each pass reads the project task list
**once** via `GET /api/v1/requests/{project_id}/tasks` — the same route the MCP
`list_tasks` tool uses; the bare `GET /api/v1/tasks/{id}` route 404s with the SPA
fallback — and looks each local task up by `external_key` first, then
`handoff_task_id`. Mismatches are emitted as `board_drift` entries to the JSONL
and latest report. Board problems get their own console tag — `drift` for a real
status/assignee mismatch, `board_unreachable` when the list can't be read — and
never masquerade as `stalled`.

The monitor also reads the read-only `ng_gate` queue snapshot and reports
lane/batch-owned orphan flags as `gate_orphan_alerts`. This is alert-only:
automatic process killing remains inside the runner's owner-metadata cleanup
path, or in explicit operator stop/rescue scripts.

### Managed Rescue Verification

Use `Start-HandoffRescue.ps1` for review-rescue queue cleanup instead of an
ad-hoc background `powershell -Command ... rescue_verify.py ...` wrapper. It
scopes the rescue verifier to one lane/batch, reads the batch config task list
when `-Task` is omitted, defaults the full gate to `-HeavyN 8000`, writes
`Handoff/private/rescue_verify_<lane>_p<batch>.pid.json`, and refuses to start a
second live verifier unless `-ForceRestart` is passed.

```powershell
.\Handoff\scripts\Start-HandoffRescue.ps1 -Lane Codex2 -Batch 67 -ConfigPath Handoff/config/runner_codex2_p67.json
```

`rescue_verify.py` auto-resolves rows whose candidate already has a Submit
Captain packet, records heavy-gate failures as resolved queue rows, compacts the
append-only rescue JSONL to one latest row per canonical task/candidate, and
flushes progress to the rescue stdout/report files. For manual cleanup, run:

```powershell
python Agenters/Submit_Captain/captain_ctl.py rescue-compact --apply
```

The compactor writes a timestamped `.bak` beside
`Handoff/private/review_rescue_queue.jsonl` before rewriting. `Stop-HandoffRescue.ps1`
kills only the managed rescue verifier tree after checking the pid command line,
then stamps the rescue pid file stopped. `Stop-HandoffBatch.ps1` now calls that
rescue stop path by default; pass `-KeepRescue` only when you deliberately want a
rescue verifier to outlive a runner/monitor stop.

## HandOff Task Writes

Task updates use:

```text
PUT https://handoff.socnet.lol/api/v1/tasks/{task_id}
```

Rendered configs may also include `external_key`. If a task ID is blank, or a
sync gets a stale-resource `404`/`410`, the runner tries
`GET /api/v1/requests/{project_id}/tasks/resolve?external_key=...`, records the
repointed `handoff_task_id` in local state, and retries the sync once.

For status transitions, include the assigned agent credential:

```text
Authorization: Bearer <owner bootstrap token>
X-Agent-Id: bo_ng_worker1
X-Agent-Key: <bo_ng_worker1 agent key>
Content-Type: application/json
```

The owner bearer token can read tasks and perform no-op writes, but task status transitions returned `403` without the assigned worker key. The runner loads the key from `Handoff/private/bo_ng_worker1.env`; do not copy that key into shareable logs or docs.

## Proposed between-waves changes (2026-07-02 p37/p38 token-efficiency review)

These touch launch-critical runner code, so they are specced here rather than hot-patched mid-run. Land them when **no lane is live** and dry-run (`-Once -DryRun`) before a real batch. Motivation: a ~9h p37 opus run produced only ~2 genuinely-new wins; most spend was workers re-confirming known floors, re-solving already-owned tasks, and one hard-task tail.

1. **Per-task total session budget** (stops the tail-grind: p37 `task286` ≈5 sessions/~3h, p38 `task369` had to be killed). Phase caps (`worker_max_failures`, `max_rework_cycles`, `reviewer_max_failures`, `timeout_seconds`) *stack* — one hard task can eat a quarter of a run. Add to `workflow`: `max_task_sessions` (e.g. 3, total worker+reviewer) and `max_task_wallclock_minutes` (e.g. 90). Track per-task `session_count`/`first_started_utc` in runner state; in `step()`/`select_worker_task`, when a task exceeds either, mark it `wall_budget_exhausted` (terminal — add to the drained/terminal set) with a one-line certificate and move on. Both `handoff_runner.py` + `handoff_runner_claude.py`.
2. **Cheap scout gate before the expensive worker** (biggest token lever). A pre-worker pass reads the prompt + `Task_by_Task/taskNNN.md` don't-do log + live manifest row + staged/consumed history → `{park_wall, park_owned, needs_specialist(levers[])}`; only `needs_specialist` reaches the pinned worker model. Start scripted/no-LLM or a cheap model. It may emit an excluded-by-reason report, but owner-selected prompt folders should still render exactly unless the launcher passes `--allow-excludes`.
3. **Batch-selection pruning before prompt creation, not during render.** Cross-check candidate ideas vs the live manifest + `Staged_Boladi`/`_consumed` before writing a prompt folder. Once prompts exist, `prepare_handoff_batch.py` treats the folder as the owner-approved task list; any partial render must be explicit with `--allow-excludes`.
4. **Do-not-repeat context block** injected into `{task_prompt}`: current live best cost+sha, known floors/WALL records, last-2 failed levers, consumed-packet outcomes. Cuts tokens; stops re-discovering known floors.
5. **Compact packets/verdicts**: machine fields + ≤5 lines rationale; full analysis to `taskNNN.md` only on a shipped win (the p32 state file was ~106k tokens for 17 tasks).

**Landed 2026-07-02 (safe, additive — already in tree):** `captain_ctl.py reconcile-banked` (staged-but-not-banked leak audit; a first run surfaced ~26 tasks / ~+4.8 pts of historical un-merged staged gold — verify before re-merge), `Handoff/scripts/Stop-HandoffBatch.ps1` (surgical safe-abort: runner-tree kill + pid stamp + close-out note, stale-pid guarded), and strict-gain-vs-**LIVE-manifest** + dup/tie rejection + narrow-rework hardening in both reviewer prompts (`prompts/reviewer.md`, `prompts_claude/reviewer.md`). **Rejected:** a min-gain *skip* on probes (would discard real cumulative gains — grace-batch multiple candidates into one shared submit instead).
