# Prompts/Handoff/ — HandOff supervisor launch prompts

Four operating contracts, one per **tier × backend**, for driving the **HandOff runner**
(`Handoff/runner/`) + board (`handoff.socnet.lol`) instead of the local `prep_run.py` orchestrator.
They are the HandOff analog of the four prompts in `Prompts/` — the producer is the hardened runner,
and the ALPHA Part-B staging is identical (`captain_ctl.py`, same `Updates_*` inboxes, same
`Staged_Boladi`).

| file | tier | backend | stages to team? |
|---|---|---|---|
| `ALPHA Supervisor Codex.txt`  | ALPHA | Codex (`codex exec`) | yes — Part B drains BOTH lanes → `Staged_Boladi` |
| `ALPHA Supervisor Claude.txt` | ALPHA | Claude (`claude -p`) | yes |
| `BETA Supervisor Codex.txt`   | BETA  | Codex  | no — feeds `Updates_Codex` for ALPHA |
| `BETA Supervisor Claude.txt`  | BETA  | Claude | no — feeds `Updates_Claude` for ALPHA |

You spawn **one ALPHA** total (Codex *or* Claude); it produces its own lane and stages every lane's
`probe_ok` gold. BETAs are optional extra producers of the other lane. The runner already collapses the
orchestrator → workers → reviewers → captain pipeline into one process per lane, so there is no prompt
per role — one supervisor prompt per lane launches the runner, which spawns the rest.

## How HandOff differs from the local `Prompts/`

- **Producer** = `Handoff/scripts/Start-HandoffRunner.ps1 -Lane <Codex|Codex2|Claude> -Loop -RunCodex|-RunClaude`
  (the runner spawns its own worker/reviewer sessions), NOT `prep_run.py` + `start_orchestrator.ps1`.
- **Coordination** = the HandOff board (project/team/task IDs, status sync via `--sync-handoff`). Sync is
  **best-effort** — the runner treats board failures as non-fatal, so a lane still produces locally if the
  board is down.
- **Config** = each batch gets a rendered `Handoff/config/runner_<lane>_p<N>.json` (tasks[] + lane-specific
  `private_state` / `operation_log`); use `Handoff/runner/prepare_handoff_batch.py` so prompt folders,
  external keys, board task IDs, and lane paths are generated consistently. The launchers take `-ConfigPath`.
- **Lane isolation** = lane-specific pid files (`runner_loop.<lane>.pid.json`) and monitor logs
  (`monitor_<lane>_batch<N>.*`) so Codex + Claude can run concurrently without collision.
- **Context** = the runner gives every TASK a fresh session (resume only within a task). No cross-task
  bleed; nothing to "reset." Agents are board identities, not long-lived processes.
- **Cost default (2026-06-30)** = `worker_attempts_before_review:1` (single worker pass then review). Set
  `2` in the rendered config to force a second improvement pass on a hard batch.

## Short-form launch request (preferred for routine launches)

```text
Apply Prompts/Handoff/<ALPHA|BETA> Supervisor <Codex|Claude>.txt.
prompt folder:
- The_400_Tasks/Task Prompts/<N>
batch tag: p<N>
exclude: <taskNNN, ... | none>
special overrides: <none | ...>
```

The supervisor then:
1. Renders `Handoff/config/runner_<lane>_p<N>.json` from the prompt folder:
   `python Handoff/runner/prepare_handoff_batch.py --lane <Codex|Codex2|Claude> --batch <N> --prompt-folder "The_400_Tasks/Task Prompts/<N>" --sync-board`.
   Omit `--sync-board` only for local dry prep; with board sync on, the helper create-or-resolves the
   per-batch goal/tasks using stable external keys.
2. Collision-checks against local runs (`Agenters/Orchestrator/runs/*`) and the board.
3. Starts `Start-HandoffRunner.ps1 -Lane <lane> -Loop -RunCodex|-RunClaude -SyncHandoff -ConfigPath ... -Batch <N> -ExitWhenDrained`
   and `Start-HandoffMonitor.ps1 -Lane <lane> -Loop -Batch <N> -ConfigPath ... -CheckBoard` (both background).
4. (ALPHA only) runs the `captain_ctl.py` drain → probe → stage → reconcile loop across BOTH lanes.
5. Closes when the monitor reports `complete`; writes `Handoff/logs/<lane>_p<N>_closed.md`.

The authoritative runner contract (scheduler, retries, sandbox/auth, monitor terminal states, all tunable
knobs) is `Handoff/runner/README.md`. The live HandOff participation contract is
`https://handoff.socnet.lol/api/v1/docs/swarm`; fetch it at launch if board behavior looks different.
The long `.txt` contracts remain the fallback — the short form just avoids re-pasting them.
