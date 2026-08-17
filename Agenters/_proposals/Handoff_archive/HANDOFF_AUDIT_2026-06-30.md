# HandOff Audit: NeuroGolf Runner/Board Integration

Date: 2026-06-30
Author/context: Codex lane supervisor, NeuroGolf_HO batch 33
Status: ARCHIVED 2026-07-02. Local runner/prompt-owned fixes were integrated; remaining board/platform ideas are historical context unless the owner explicitly reopens them.
Scope: HandOff board/runtime behavior observed while running a local multi-agent Codex runner. This does not include private agent keys, wallet details, or Kaggle/submission staging.

## Executive Summary

The local HandOff runner worked: batch 33 reached terminal state with 8 SHIP packets and 1 DROP. The board integration was the weak point. The board showed active NeuroGolf work as "needs you" / open, and all agents as offline, even while local Codex workers were running. Board history showed active task claims being reset by `self-healing` roughly one second after successful runner status updates.

This creates a dangerous split-brain:

- The local runner believes tasks are claimed and running.
- The board can show the same tasks as open/unassigned.
- Another lane or human can pick up duplicate work.
- The owner sees "needs you" for work that is actually in an agent queue.
- The only observed workaround was synthetic agent presence, which the owner does not want to run just to preserve board truth.

The core fix is to make task execution a lease owned by a runner/batch, not by the web presence status of a human-visible agent identity.

## Observed Evidence

Batch: `33`
Project: `NeuroGolf_HO`
Lane: `Codex`
Local config: `Handoff/config/runner_codex_p33.json`
Local state: `Handoff/private/runner_state_codex_p33.json`
Local operation log: `Handoff/logs/operations_codex.jsonl`

Terminal local result from monitor:

```text
Counts: drop=1, ship_packeted=8
ALERT: batch complete - all tasks terminal
```

The runner produced SHIP packets under:

```text
Agenters/Submit_Captain/Updates_Codex/
```

During launch, the runner successfully set the first three batch 33 board tasks to `in_progress` with worker assignees. Board task history then showed `self-healing` reverting them to `todo` with no assignee approximately one second later:

```text
task333: in_progress by bo_ng_worker1 at 2026-06-30T05:24:48Z
task333: todo / assignee null by self-healing at 2026-06-30T05:24:49Z

task368: in_progress by bo_ng_cdx_worker2 at 2026-06-30T05:24:48Z
task368: todo / assignee null by self-healing at 2026-06-30T05:24:49Z

task280: in_progress by bo_ng_cdx_worker3 at 2026-06-30T05:24:48Z
task280: todo / assignee null by self-healing at 2026-06-30T05:24:49Z
```

Starting a synthetic presence heartbeat for the agents and re-syncing the same three tasks made the board retain `in_progress`. Presence was then stopped because the owner explicitly does not want to spend resources just to keep presence alive.

## Findings

### P0: Self-healing resets fresh active claims

Problem:
The board reset tasks from `in_progress` to `todo` immediately after a valid task update. This appears coupled to the assigned agent being considered offline. A successful status update itself should be enough evidence that the runner is alive, at least for a lease window.

Impact:
This breaks the board as a coordination ledger. It makes duplicate work likely and makes the UI contradict the runner's real state.

Recommended fix:
Introduce task leases. A runner claim should write a lease record that self-healing respects.

Suggested fields:

```json
{
  "runner_id": "codex:host:batch33:pid18720",
  "lane": "Codex",
  "batch": "33",
  "claim_id": "uuid",
  "claimed_by": "bo_ng_worker1",
  "claimed_at": "2026-06-30T05:24:48Z",
  "lease_expires_at": "2026-06-30T06:14:48Z",
  "last_runner_sync_at": "2026-06-30T05:24:48Z"
}
```

Self-healing should only release active work when all of these are true:

```text
status is active
AND lease_expires_at is in the past
AND no accepted runner/task update has arrived within the grace window
AND no active runner heartbeat exists for runner_id
```

For long-running local agents, the default lease TTL must exceed the worker call timeout. In this NeuroGolf run the worker timeout was 2700 seconds, so a reasonable default would be 50-60 minutes unless the runner renews leases periodically.

Acceptance criteria:

- If an offline-looking agent successfully claims a task through the API, self-healing does not reset it for at least the lease TTL.
- If the runner renews the lease every N minutes, the task remains active indefinitely.
- If the runner dies and no renewal occurs, self-healing releases the task after TTL plus grace.
- Task history records the exact reason for release, including expired lease ID and timestamps.

### P0: Presence is doing coordination work it should not do

Problem:
The board treated "agent online" as a precondition for keeping task claims alive. The owner does not want to run presence just to prevent board drift. Presence should be a UX signal, not the source of truth for task ownership.

Impact:
Users are incentivized to run fake heartbeat loops. That wastes resources, makes online counts misleading, and couples operational correctness to a social/visibility feature.

Recommended fix:
Separate three concepts:

- Agent identity: who is allowed to act.
- Runner lease: what process/batch currently owns a task.
- Presence: whether an agent is socially/visibly online.

The UI can still show "offline", but an offline agent with an active runner lease should render as something like:

```text
In progress by bo_ng_worker1 via Codex batch 33
Runner lease active, last sync 2m ago
```

Acceptance criteria:

- Turning off presence does not make active leased tasks reopen.
- The dashboard can show `0 / 15 agents online` while still showing task leases as healthy.
- Presence heartbeats are never required to preserve valid task state.

### P1: "Needs you" conflates human action with agent queue state

Problem:
The dashboard showed NeuroGolf tasks under "Needs you" as `open` or `verify` while the local runner had them queued, running, or pending reviewer. In this workflow, `pending_verification` usually means "reviewer agent should run", not "owner must manually verify".

Impact:
The owner gets false interrupts and may manually intervene in work that is already scheduled.

Recommended fix:
Split owner-required states from agent-required states.

Suggested status buckets:

```text
owner_needs_action
agent_queue
runner_active
external_blocked
terminal
```

For example:

- `todo` with no lease: owner/team can assign.
- `todo` with a scheduled runner claim: agent queue.
- `in_progress` with active lease: runner active.
- `pending_verification` with reviewer assignee and active lease: reviewer queue.
- `pending_verification` with no reviewer lease and stale age: owner needs action.

Acceptance criteria:

- A task with `pending_verification`, a reviewer assignee, and an active runner lease does not appear as "Needs you".
- A task only appears under "Needs you" when no agent/runner can make progress without owner action.

### P1: Board needs compare-and-swap task writes

Problem:
Task writes are simple status updates. When two lanes exist, stale writes and self-healing writes can race with runner writes. There is no visible version/etag/claim precondition in the current update path.

Impact:
Last-writer-wins can silently corrupt coordination state.

Recommended fix:
Add optimistic concurrency to task updates.

Suggested request shape:

```json
{
  "status": "in_progress",
  "assignee": "bo_ng_worker1",
  "claim_id": "uuid",
  "expected_version": 42,
  "lease_seconds": 3600,
  "result": "initial worker pass started by runner"
}
```

Suggested behavior:

- Reject stale writes with `409 Conflict`.
- Reject updates that do not match the active `claim_id`, unless the actor is an owner/admin or an explicit force-release endpoint is used.
- Include current task version and active lease metadata in the conflict response.

Acceptance criteria:

- A self-healer cannot overwrite a task updated after the self-healer's scan started.
- Two runner lanes claiming the same task cannot both succeed.
- A stale runner cannot mark a task verified after another claim has superseded it.

### P1: Add idempotent task creation/upsert by external key

Problem:
The runner must discover whether board tasks already exist, then create missing tasks and record IDs in a local config. This is fragile across retries and multiple lanes.

Recommended fix:
Support server-side upsert with a stable external key.

Suggested key:

```text
project_id + lane + batch + task_slug
```

or, if tasks should be shared across lanes:

```text
project_id + task_slug
```

Suggested endpoint:

```text
PUT /api/v1/projects/{project_id}/tasks/by-external-key/{external_key}
```

Acceptance criteria:

- Re-running config generation for the same batch does not create duplicate board tasks.
- The response always returns the canonical task ID.
- The server stores `external_key`, `lane`, `batch`, and `source_prompt_file`.

### P1: Monitor should detect board/local drift

Problem:
The local monitor correctly reported local progress, but it did not alert that the board had reset active tasks to `todo`. The owner only caught it visually in the HandOff UI.

Recommended fix:
Add an optional board reconciliation check to the monitor:

```text
for each task with handoff_task_id:
  read board task
  compare board.status/assignee to local status/agent
  warn if local active and board open/unassigned
  warn if local terminal and board non-terminal
```

This should be best-effort and non-fatal, matching the existing "board as ledger" model.

Acceptance criteria:

- If local state says `worker_running` and board says `todo/null`, monitor latest file contains a `board_drift` alert.
- Drift is written to JSONL with task ID, local status, board status, local assignee, board assignee, and timestamps.
- The monitor redacts credentials and never logs agent keys.

### P2: Runner loops should be able to auto-exit when drained

Problem:
The runner loop keeps idling forever after all tasks are terminal. The monitor can report `complete`, but the runner loop must still be stopped by an operator.

Recommended fix:
Add an explicit batch mode that exits when all configured tasks are terminal, or let the monitor stop the paired runner when `--exit-on-terminal` is set and a `--stop-runner` flag is provided.

Acceptance criteria:

- Batch runner launched with `--exit-when-drained` exits with code 0 after all tasks are terminal.
- Runner pid file is stamped `state: stopped`, `exit_reason: complete`.
- Existing daemon/continuous mode remains available for users who want an always-on queue.

### P2: Log and response redaction needs a server/client contract

Problem:
The runner logs API responses and task histories. No keys were exposed in this audit, but the system should guarantee that server responses never echo secrets and client logs redact any field that looks like a key/token.

Recommended fix:

- Server: never return `X-Agent-Key`, `AGENT_KEY`, bootstrap tokens, or private env-derived values from task/update endpoints.
- Client: redact keys by field name and by pattern before writing operation logs.
- Tests: include fake `ak_...` and bearer-like values in mock responses and prove logs redact them.

Acceptance criteria:

- Operation logs do not contain agent keys, bearer tokens, or wallet private material.
- CI test fails if a sensitive field appears in a serialized log fixture.

## Proposed Agent Work Tickets

### Ticket 1: Implement task leases and self-healing guardrails

Goal:
Prevent self-healing from resetting fresh valid runner claims.

Tasks:

1. Add lease fields to task model.
2. Add claim/renew/release semantics to task update endpoint.
3. Modify self-healing to respect `lease_expires_at` and `last_runner_sync_at`.
4. Record self-healing reason in task history.
5. Add tests for fresh claim, renewed claim, expired claim, and stale self-healer race.

Done when:

- A task claimed by an offline-looking agent remains `in_progress` until its lease expires.
- Expired tasks are released with a clear audit trail.

### Ticket 2: Decouple presence from task ownership

Goal:
Make presence a display signal only.

Tasks:

1. Audit all places where offline status changes task status.
2. Replace presence-based release logic with lease-based release logic.
3. Update UI badges to show both presence and runner lease health.

Done when:

- The board can show agents offline while active leased tasks remain correctly assigned.
- No synthetic presence process is required for correct task coordination.

### Ticket 3: Add board/local drift detection

Goal:
Catch split-brain before the owner notices it manually.

Tasks:

1. Add optional board read mode to monitor.
2. Compare local runner state with board task state.
3. Emit `board_drift` alerts to latest text and JSONL.
4. Keep network failures non-fatal.

Done when:

- A local active task that board shows as open produces a clear alert.
- Drift alerts include enough fields for a repair agent to act.

### Ticket 4: Fix "Needs you" classification

Goal:
Only page the owner for true owner-required action.

Tasks:

1. Add derived UI buckets for owner action vs agent queue vs runner active.
2. Treat `pending_verification` with reviewer lease as reviewer queue, not owner verification.
3. Add stale thresholds for when an agent queue becomes owner action.

Done when:

- Active/reviewer-queued NeuroGolf tasks no longer appear as owner "Needs you".
- Stale unleased tasks still surface clearly.

### Ticket 5: Add idempotent task upsert

Goal:
Make batch config generation safe to rerun.

Tasks:

1. Add `external_key` to tasks.
2. Add upsert endpoint by external key.
3. Return canonical task ID on every upsert.
4. Add duplicate-prevention tests.

Done when:

- Re-running the same batch generation creates zero duplicate tasks.
- Multiple lanes can safely discover the same existing task.

## Suggested Implementation Order

1. Task leases and self-healing guardrails.
2. Presence/task ownership decoupling.
3. UI "Needs you" classification.
4. Board/local drift monitor.
5. Idempotent task upsert.
6. Auto-exit drained runner mode.
7. Log redaction tests.

The first two items are the real correctness fixes. The remaining items reduce operator confusion and make future incidents easier to diagnose.

## Notes From This NeuroGolf Run

- Local runner completion was healthy despite board drift.
- The board was reachable and accepted task updates.
- The split happened after successful writes, not because sync failed.
- Synthetic presence made the board retain `in_progress`, which strongly suggests presence/self-healing coupling.
- The owner explicitly does not want to run presence as a resource workaround.
- No live Kaggle submit or team staging happened in this lane.
