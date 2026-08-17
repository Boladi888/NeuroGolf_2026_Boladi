# Plan 1 - Local File-Backed Kanban

## Goal

Replace the current run-folder/inbox sprawl with a small repo-local Kanban
system under `Agenters/00_Kanban/`. This plan works without Hermes. It is best
for manual Codex/Claude waves, short local runs, or as a fallback when the
Hermes gateway is not running.

## Source Of Truth

Use one card file per work item plus append-only event files. Do not keep the
whole board in one mutable JSON file.

Proposed layout:

```text
Agenters/00_Kanban/
  README.md
  local_kanban_plan.md
  handoff_mcp_plan.md
  board/
    cards/
      KAN-000001.md
      KAN-000002.md
    events/
      KAN-000001/
        20260611T210000Z-created.json
        20260611T211500Z-moved.json
    reports/
      daily_20260611.md
    config.json
```

Card files are human-readable Markdown with strict YAML front matter:

```yaml
---
id: KAN-000001
type: worker | review | captain | librarian | janitor | infra
task_id: task110
title: Restage task110 onto canonical base
status: ready
assignee: worker-1
priority: 80
service_class: expedite | fixed-date | standard | intangible
created_utc: 2026-06-11T21:00:00Z
started_utc:
finished_utc:
blocked_reason:
parent_ids: []
child_ids: []
source_paths: []
artifact_paths: []
expected_delta:
terminal_owner:
---
```

The body holds the task-specific prompt, acceptance criteria, and latest
handoff summary. Scripts read the YAML front matter; humans read the body.

## Workflow

Columns:

1. `intake` - rough idea or owner request, not ready for an agent.
2. `ready` - specified, scoped, and allowed to be pulled.
3. `running` - an agent has started work.
4. `review_required` - worker is done, but not terminal.
5. `rework` - reviewer found a concrete next attempt.
6. `captain_probe` - reviewer-approved candidate awaiting captain validation.
7. `librarian` - durable lesson awaiting library integration.
8. `blocked` - cannot move without a named unblock condition.
9. `done` - terminal and auditable.
10. `archived` - done and no longer shown by default.

Started means a card enters `running`, `review_required`, `captain_probe`, or
`librarian`. Finished means it enters `done` or `archived`.

## Work Item Types

- `worker`: one task shrink/build attempt. Owns only task-local files and
  scratch artifacts.
- `review`: independent skeptical review of a worker handoff.
- `captain`: own-stack probe, staging, and submit-relevant validation.
- `librarian`: integrate durable lessons into `Onnx_Advanced/`.
- `janitor`: inventory or cleanup proposal, report-first.
- `infra`: board/tool/process maintenance.

## WIP Limits

Initial conservative limits:

- Worker lane: 1 active card per worker across `running`, `review_required`,
  `rework`, and `blocked`.
- Global workers: owner-selected per wave, default 3.
- Review queue: max 4 cards in `review_required`; reviewer max 1 active card
  each.
- Captain: max 1 card in `captain_probe` or staging.
- Librarian: max 1 active library integration card.
- Janitor: max 1 cleanup card and no direct deletes without explicit approval.
- Blocked: no numeric limit, but every blocked card must have an unblock owner
  and service-level expectation.

If a limit is hit, do not raise it first. Swarm the bottleneck: review cards,
clear blocked work, or finish captain/librarian work.

## Explicit Policies

- Owner-supplied exact task lists are hard scope. Cards outside the allowlist
  cannot be pulled into a wave.
- A positive expected delta cannot close as `held`. It must be `probe_ok`,
  `staged`, `drop`, `rejected`, `probing`, or `external_owner` with a named
  owner and next action.
- A worker result that needs judgment moves to `review_required`, not `done`.
- `blocked` is for missing input, stale base ambiguity, crashed tooling, or
  human decision. "I am done" is not a blocked reason.
- Captain and Librarian stay single-writer. Other roles create cards or packets
  for them; they do not edit captain/librarian state directly.
- Workers may not edit `Onnx_Advanced/**`, team submission paths, shared
  ledgers, or another task's folder unless the card grants exclusive ownership.
- Every worker handoff must include structured metadata compatible with
  `tools/schemas/worker_handoff.schema.json` where applicable.
- Every status move writes an event file before or with the card update.

## Tooling To Build

Start tiny:

```text
tools/local_kanban.py
  init
  create
  list
  show
  move
  assign
  comment
  report
  lint
```

Implementation notes:

- Use atomic write: write `*.tmp`, then rename.
- Use one card file per item to avoid shared-file conflicts.
- Store event files per card to make audit history append-only.
- Keep `board/config.json` for WIP limits and status names.
- Make `lint` enforce WIP limits, missing handoff fields, stale blocked cards,
  duplicate task claims, and unowned positive deltas.
- Reuse parsing ideas from `tools/neurogolf_kanban_python.py`, but do not
  depend on Hermes.

## Migration Steps

1. Freeze new Agenters direct-run starts for one planning window.
2. Import current active run CSVs from `Agenters/Orchestrator/runs/*/` into
   local cards, but only for non-terminal work.
3. Import `Agenters/Submit_Captain/Updates_*` packets as `captain` cards.
4. Import `Agenters/Librarian/Updates_*` lesson files as `librarian` cards.
5. Import the current Hermes blocked task110 card as either `review_required`
   or `captain_probe`, depending on the desired next owner.
6. Run `tools/local_kanban.py lint`.
7. Pilot a 3-worker wave:
   - 3 worker cards max.
   - 2 reviewer slots max.
   - 1 captain slot.
   - no new worker pull while review or captain is over WIP.
8. Update Agenters role docs to say: "If local Kanban is enabled, card status is
   the source of truth; run CSVs are derived reports."

## Metrics

Generate a daily report with:

- WIP by column and by assignee.
- Throughput: cards finished per day, by type.
- Work item age: active cards sorted oldest first.
- Cycle time: started to finished, by type.
- Blocked age and reason.
- Positive expected delta cards not terminal.
- Captain queue age.
- Librarian queue age.

## Risks

- File-backed Kanban will need careful locking if multiple local agents mutate
  cards at the same time.
- It duplicates capabilities Hermes already provides.
- Without a dispatcher, someone still has to spawn workers and enforce pull.
- If `board/config.json` becomes another giant policy file, this repeats the
  old orchestration bloat.

## Success Criteria

- A new operator can answer "what is active?" with one command.
- No worker has more than one active card.
- No positive delta is unowned.
- Blocked cards are reviewed daily or have a named unblock owner.
- Captain and Librarian queues are visible and bounded.
- Closing a wave requires no manual archaeology through transcripts.

