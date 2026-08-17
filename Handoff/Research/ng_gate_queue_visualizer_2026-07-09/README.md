# ng_gate Queue Visualizer Research

Created: 2026-07-09

Status: research only. No implementation changes were made to `ng_gate.py`, `throttle.py`, runners, launchers, or monitor loops.

## Purpose

The current `ng_gate` throttle keeps the machine from melting, but the user cannot easily see:

- which gate owns the scarce heavy slot;
- which full gates are waiting behind it;
- how long each active or waiting gate has been there;
- whether a gate process looks orphaned;
- how much reviewer rescue work is piling up because full gates could not run.

This packet gives the next agent a reviewable design before implementation.

## Key Finding

`tools/ng_gate_status.py` already gives a read-only view of active lock holders, but active locks are not the whole queue.

`tools/ng_harness/throttle.py` writes lockfiles only after a process successfully acquires a slot. A process waiting inside `Slot.acquire()` has no lockfile yet. Therefore a lockfile-only UI can show "heavy slot busy" but cannot show "task350 has been waiting 12 minutes" unless it also reads live process trees or the throttle code starts writing explicit wait records.

## Current Safe Data Sources

- `%TEMP%\ng_throttle\*.lock`: active normal and heavy slot holders.
- `tools/ng_gate_status.py --json`: current read-only active-slot summary.
- `Get-CimInstance Win32_Process`: live `ng_gate.py` process trees, including waiters without lockfiles.
- `Handoff/private/runner_state_*.json`: runner-local task status and worker/reviewer assignment.
- `Handoff/logs/monitor_*_batch*.latest.txt`: compact monitor state and drift.
- `Handoff/private/review_rescue_queue.jsonl`: backlog of strict-gain candidates waiting for full gates or reviewer recheck.
- `Handoff/logs/operations_*.jsonl`: recent `GATE_WAIT`, timeout, worker/reviewer transitions.

## Current Gaps

- Waiters have no first-class queue record.
- Queue age is currently inferred from process creation time, not an explicit "started waiting" timestamp.
- Venv launchers can create duplicate `ng_gate.py` command lines in a parent/child chain. A visualizer must group a whole process tree before deciding whether something is active or waiting.
- Orphan detection cannot be perfectly certain from command lines alone. It should flag "suspect" and show evidence, not kill anything.
- The existing status tool does not show command line, candidate path, parent chain, runner lane, reviewer/worker identity, or rescue queue backlog.

## Recommended First Implementation

Build a read-only snapshot command first:

`python tools/ng_gate_queue_snapshot.py --json`

It should:

1. Parse active lockfiles without deleting or reclaiming them.
2. Query live process command lines.
3. Build parent/child process trees.
4. Group duplicate `ng_gate.py` command lines into one logical run.
5. Classify each logical run:
   - `active_heavy`: tree owns a heavy lock.
   - `active_light`: tree owns only a normal lock.
   - `waiting_heavy`: full `truth`/`ab`/`battery`/`synth`, no heavy lock yet.
   - `waiting_normal`: no normal lock yet, rare unless normal slots or CPU guard block it.
   - `orphan_suspect`: no known runner ancestry or state link, parent chain is broken, or runtime is abnormal.
6. Join optional Handoff context by task number and current runner state.
7. Include rescue queue summary and oldest open rows.

Do not add stop buttons or auto-kill behavior in the first pass.

## Recommended Visual Window

Use a VizTerms-style visible terminal, not a hidden service, for the first version.

Suggested shape:

- `Handoff/scripts/Start-NGGateQueueWatch.ps1`
- opens a visible terminal or runs in the current one;
- title: `NG Gate Queue Watch - CLOSE WINDOW TO STOP`;
- polls `tools/ng_gate_queue_snapshot.py --json` every 2 to 5 seconds;
- clears and redraws a compact table;
- prints the exact command line for any selected/suspect item only when requested.

The first UI can be plain terminal output with ANSI color. No browser server is needed yet.

Suggested sections:

- `HEAVY SLOT`: holder, task, mode, age, PID, candidate basename.
- `WAITING FULL GATES`: task, mode, age waiting, parent runner, candidate basename.
- `LIGHT GATES`: active normal-slot work.
- `RUNNER BACKLOG`: worker/reviewer running tasks and open `GATE_WAIT` rescue rows.
- `ORPHAN SUSPECTS`: processes or locks that do not reconcile cleanly.

## Optional Later Instrumentation

After review, consider adding explicit wait records inside `throttle.acquire_gate()`:

- write `wait_<pid>_<nonce>.json` before entering the acquire loop;
- include mode, task label, n, heavy/light expectation, command line, cwd, created timestamp;
- heartbeat/update the file while waiting;
- remove it immediately after the slot is acquired or the process exits;
- never let a wait-record write failure block the gate.

This would make queue age exact and avoid relying only on process creation time. It is a behavior-adjacent change and should be reviewed before landing.

## Hard Boundaries For Implementation

- Do not change `NEUROGOLF_HEAVY_GATE_K` as part of the visualizer.
- Do not use `--no-throttle` to "test" the queue.
- Do not kill gate processes from the dashboard in v1.
- Do not call helper APIs that reclaim/delete lockfiles unless that behavior is intentional and reviewed.
- Do not start a hidden server. If a web dashboard is later desired, launch it from a visible VizTerms window.
- Avoid broad process kills by name; any future stop path must be pid-scoped or path/command-line scoped.

## Files In This Packet

- `README.md`: this overview and recommendation.
- `live_snapshot_2026-07-09.md`: evidence gathered from the current machine without disturbing active runs.
- `observer_model.md`: proposed classification and data model for the next agent.
- `review_checklist.md`: specific questions to answer before implementation.

