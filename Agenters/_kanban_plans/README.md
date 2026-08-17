# Agenters Kanban Plans

Created: 2026-06-11

**Status: PARKED — not in use (owner, 2026-06-11).** The file-inbox/run-folder system is the system of record;
nothing depends on adopting these. Kept as design references (renamed out of the top-sort, excluded from doc_lint).

This folder records two replacement orchestration plans for the current
Agenters workflow:

- `local_kanban_plan.md` - repo-local, file-backed Kanban with no Hermes
  dependency.
- `handoff_mcp_plan.md` - Hermes/Handoff-style Kanban using the durable Hermes
  board as the source of truth, with MCP/subagent execution treated as worker
  leases.

## Recommendation

Use the Handoff/Hermes plan as the primary route if the goal is autonomous,
multi-role work that survives restarts. Hermes already has the right primitives:
durable board state, profile lanes, comments, task links, blocked/retry
semantics, dispatcher history, and audit logs. The current board is almost
idle, with `hermes kanban stats` showing 0 ready/running cards, 1 blocked card,
and 1018 done cards, so this is a reasonable point to restart with clearer
policies.

Keep the local Kanban plan as a fallback and as a compatibility layer for
Codex-native/manual runs. It is simpler and easier to review in git, but it
will need more custom tooling to enforce WIP, stale claims, handoff completeness,
and audit history.

## Key Local Findings

- `Agenters/00_Kanban/` existed but was empty before this plan.
- The live Agenters docs already encode useful Kanban-like ideas: exact task
  claims, reviewer leases, captain/librarian single-writer lanes, no unowned
  positive expected deltas, and role-local run folders.
- Current Agenters orchestration is split across run CSVs, role inboxes,
  prompt files, and human-readable docs. That makes the true workflow state
  harder to query than it should be.
- `tools/` already contains Hermes/Kanban helpers:
  - `tools/neurogolf_kanban_python.py`
  - `tools/neurogolf_board_policy.py`
  - `tools/neurogolf_dependency_serializer.py`
  - `tools/auto_triage_blocked.py`
  - `tools/submit_gate.py`
  - `tools/update_current_state.py`
  - `tools/schemas/worker_handoff.schema.json`
- Several tools assume old file paths like `.hermes/board/*.json`, but this
  checkout does not currently have a `.hermes/board` directory. The active
  Hermes board is SQLite-backed under Hermes home and exposed through
  `hermes kanban ...`.
- Hermes is installed on PATH at
  `<HERMES_HOME>\hermes-agent\venv\Scripts\hermes.exe`.
- The old `.notongit/NeuroGolf_nogit/orchestrator` system used useful
  filesystem-message ideas: atomic JSON inbox/outbox files, one task per
  worker, TTLs, explicit locks, candidate preservation before approval, and
  lineage-aware submit approvals. Its weak point was a giant mutable
  `state.json` as the working memory.
- I did not read `.env.handoff` because it may contain secrets and is not
  required for this planning pass.

## External References Used

- Kanban Guide v2025.5:
  https://kanbanguides.org/the-kanban-guide/2025.5/
- Atlassian Kanban boards overview:
  https://www.atlassian.com/agile/kanban/boards
- Atlassian WIP limits guide:
  https://www.atlassian.com/agile/kanban/wip-limits
- Hermes Kanban multi-agent board docs:
  https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban
- Hermes Kanban worker lanes docs:
  https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban-worker-lanes

## Kanban Principles To Preserve

The plan borrows only the parts that matter for this repo:

- Define the workflow explicitly: what a work item is, when it starts, when it
  finishes, which states it moves through, and who can move it.
- Limit WIP visibly: one active task per worker lane, one captain, one librarian,
  bounded reviewer backlog, and no hidden "held" positive candidates.
- Make blocked work a signal, not a parking lot.
- Treat every handoff as durable data: summary for humans, structured metadata
  for tooling.
- Track the four basic flow metrics: WIP, throughput, work item age, and cycle
  time.
- Improve the policy from observed bottlenecks instead of adding more roles.

