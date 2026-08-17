# Orchestrator troubleshooting

This file records process-layer failures so future Orchestrators do not rediscover them from scratch.

## Subagent launch failures

When worker/reviewer spawning fails or behaves unexpectedly, stop the wave, report the exact failure, and record it in
`Agenters/Orchestrator/runs/<orchestrator_id>/issues.md` with timestamp, requested role/model/concurrency, launch path,
error text, created subagents, and safe next action. Do not silently switch to doing all worker tasks yourself unless the
owner explicitly approves.

## Current known constraints

- Default long-wave mode is `cli-orchestrator-native-workers`: a top-level `claude -p` Orchestrator uses native
  subagents while Supervisor reads ledgers/finals.
- `main-session-native` is for short one-sitting waves where the main session owns the ledger and native subagents.
- A spawned Orchestrator subagent may not have any child-spawn tool. It must verify its surface before promising a worker
  wave.
- If a spawned Orchestrator cannot spawn workers directly, return a blocker so the parent can choose parent-spawn recovery
  or explicitly approve CLI-worker mode.
- Do not shell out to `codex exec` when native subagent spawning is available.
- Reviewer affinity still applies after recovery: if a reviewer was already assigned to a task, preserve that pairing in `reviewer_leases.csv` or in the recovery handoff.

## 2026-06-10 live run launch lessons

During `live_2026-06-10_p1`, the Orchestrator attempted to launch three Codex worker sessions.

Failed launch paths:

- Do **not** use `--dangerously-bypass-approvals-and-sandbox` for autonomous workers. It grants too much unsandboxed command/write capability and was rejected by safety review.
- Do **not** put the approval flag after `exec` as `codex exec -a ...`; Codex returned
  `error: unexpected argument '-a' found`. When needed, use the known working top-level placement:
  `codex -a never exec ...`.
- Do **not** force `-m gpt-5` on this account; Codex returned `The 'gpt-5' model is not supported when using Codex with a ChatGPT account.`

Fallback shell launch pattern for spawned-Orchestrator environments without native child-spawn when the Codex sandbox
helper is working:

```text
codex exec -C <repo> -s workspace-write --output-last-message <file> --json -
```

Pass the worker prompt on stdin. Omit model override unless the parent/owner has verified that model is supported in the current service.

A future launcher wrapper should only wrap this fallback pattern; it is not needed when native subagents are available.

### Codex Windows sandbox helper missing

On this host, sandboxed commands can fail before PowerShell starts:

```text
codex-windows-sandbox-setup.exe ... program not found
```

That is not by itself a reason to abandon a Codex Orchestrator wave. If `codex.exe` is on PATH and unsandboxed PowerShell
commands are approved, use the known working launcher pattern from `beta_p7e_codex_cli`:

```text
codex -a never exec -C <repo> -s danger-full-access --json -o <run_dir>/orchestrator_final.md -
```

Pass the Orchestrator launch packet on stdin. Capture stdout to `orchestrator_events.jsonl`, stderr to
`orchestrator_err.txt`, and write `exit_code.txt` / `ended.txt` in `finally`. Start it from a small PowerShell wrapper
with `Start-Process -WindowStyle Hidden`. Record the policy in `launch_mode.md`:

```text
sandbox_policy: use -s danger-full-access for spawned Codex CLI processes on this Windows host
```

Use this only for Codex CLI child processes in this host-specific fallback. Do not use the long
`--dangerously-bypass-approvals-and-sandbox` flag; it was rejected in the 2026-06-10 live run. Do not reuse an existing
closed run folder; bump the id, for example from `beta_p7e_codex_cli` to `beta_p7f_codex_cli`.

## Subagent process orphan leak — CLI-orchestrator mode (claude_p4_cli, 2026-06-10)

Confirmed live: when a `claude -p` orchestrator spawns native worker/reviewer subagents, their `claude.exe` processes may
orphan after completion. The folder-4 run grew from ~6 to 21 `claude.exe` (~9.5 GB) over ~2 hours.

- **Cleanup duty (orchestrator at run end; supervisor as backstop):** sweep stale subagent processes only after every
  worker/reviewer is terminal.
- **NEVER blanket-kill `claude.exe`** — that kills the supervisor session and any currently-`building` worker. Reap only
  processes you can positively associate with this run AND confirm are terminal/idle. Protect: the supervisor's own
  session, and anything you can't confirm belongs to this run.
- **Mid-run** (only if memory is approaching a system limit): reap *confirmed-idle* orphans, never a busy `building`
  worker. Otherwise wait for run end.
- **Likely root cause:** a harness-level reaping gap for native subagents under `claude -p`. The sweep is a mitigation,
  not a fix — worth reporting upstream.

## SAFE_BATCH handoff rule

Do not treat `SAFE_BATCH_QUEUE` as permission to copy directly into
`K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi/`.

Correct flow: reviewer returns `SHIP SAFE_BATCH_QUEUE`; Orchestrator writes a JSON packet to
`Agenters/Submit_Captain/Updates_<Lane>/`; Submit Captain runs `captain_ctl.py probe-isolated` (the default 0-base probe; `probe-batch` is legacy) / `probe-collect`; only after
delta match or owner approval does `captain_ctl.py stage` copy to `Staged_Boladi/`.

If a SAFE_BATCH candidate is copied early, remove/quarantine it from `Staged_Boladi/`, mark the task as
`PREMATURELY_STAGED_UNPROBED`, and do not count the expected gain as confirmed.

`RISK_SINGLE_PROBE` is separate: a sound risky-probe packet may still be individually probed per the Submit Captain playbook.

## Run-folder issue note template

```text
## <UTC timestamp> - <short title>

- role trying action:
- requested task(s):
- requested concurrency:
- command/tool attempted:
- exact error:
- artifacts already created:
- active subagents, if any:
- safe recovery:
```
