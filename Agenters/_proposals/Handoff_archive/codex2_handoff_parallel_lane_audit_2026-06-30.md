# HandOff platform audit: parallel Codex2 lane bring-up

Date: 2026-06-30
Author/context: Codex BETA supervisor, second Codex lane (`Codex2`) for NeuroGolf batch 34
Status: ARCHIVED 2026-07-02. Local lane/config integrations were integrated; remaining platform/UI ideas are historical context unless the owner explicitly reopens them.
Scope: HandOff board/API/MCP behavior observed while registering and running a second parallel Codex lane on the existing `NeuroGolf_HO` project/team. This file contains no private agent keys, wallet material, Kaggle credentials, or submit artifacts.

## Executive summary

The second Codex lane worked operationally: seven `bo_ng_cdx2_*` agents were registered, batch 34 ran as an independent runner loop, and the local monitor reached terminal state with 15 `ship_packeted` tasks and 5 `drop` tasks. The board accepted direct task self-claims from the cdx2 agents after project hierarchy was updated.

The platform weak spot is that HandOff does not yet have first-class support for adding a second automation lane to an existing live team. The cdx2 lane had to be modeled as a set of new agents plus local runner conventions, while formal team membership was blocked by a creator-only roster rule. The system was workable, but it required REST fallback, manual local config separation, and discipline around not confusing "Codex2 runner lane" with the existing Submit Captain `Codex` inbox.

The fixes below are intentionally implementation-ticket shaped. They complement, not replace, the other 2026-06-30 audits:

- `HANDOFF_AUDIT_2026-06-30.md`: active-claim leases, presence/self-healing, CAS/idempotency.
- `claude_handoff_platform_audit_2026-06-30.md`: board reorg deletion, roster replacement foot-gun, UUID repointing, backfill, MCP session, 404/410 contract.

## Observed evidence

Project and run:

```text
Project: NeuroGolf_HO
Project id: 82513bd2-0f60-434a-91b2-538d71010fa8
Team: team_644ac577
Batch: 34
Runner config: Handoff/config/runner_codex2_p34.json
Runner state: Handoff/private/runner_state_codex2_p34.json
Operation log: Handoff/logs/operations_codex2.jsonl
Submit packet inbox: Agenters/Submit_Captain/Updates_Codex/
```

Registered cdx2 agent identities:

```text
bo_ng_cdx2_orchestrator
bo_ng_cdx2_worker1
bo_ng_cdx2_worker2
bo_ng_cdx2_worker3
bo_ng_cdx2_reviewer1
bo_ng_cdx2_reviewer2
bo_ng_cdx2_submit_captain
```

API outcomes during bring-up:

```text
POST /agents/register                   -> succeeded for the cdx2 agents
PUT /teams/team_644ac577/roster         -> 403, "only the team creator can set the roster"
POST /teams/team_644ac577/role-response -> 404, "team or member not found"
PUT /requests/{project_id}/hierarchy    -> 200, cdx2 agents added to project hierarchy
PUT /tasks/{task_id} with cdx2 worker keys -> succeeded for task self-claims
```

Local terminal outcome from the cdx2 monitor:

```text
Counts: drop=5, ship_packeted=15
ALERT: batch complete - all tasks terminal
```

Important local distinction:

```text
Runner process lane: Codex2
Agent id convention: bo_ng_cdx2_*
Local logs/state: operations_codex2.jsonl, runner_state_codex2_p34.json
Submit Captain source lane/inbox: Codex / Updates_Codex
```

That distinction matters because cdx2 is a second runner lane, not a new downstream Submit Captain lane.

## Findings

### C2-ROSTER-1: Existing team membership cannot be extended by the project owner/orchestrator

Severity: HIGH

Problem:
The cdx2 agents could be registered and added to the project hierarchy, but could not be added to the existing team roster. The roster update endpoint returned `403 "only the team creator can set the roster"`, and the role-response endpoint returned `404 "team or member not found"` because the new agents were not on the roster/invite list.

Impact:
A second automation lane cannot be cleanly onboarded into a live team unless the original team creator is present and willing to perform the mutation. This is brittle for long-running automation because the current owner/orchestrator can create agents, update hierarchy, and run tasks, but cannot make the board's team model reflect reality.

Recommended fix:
Add an additive membership path that does not require replacing the whole roster and can be delegated to a project owner/team admin/orchestrator.

Suggested REST/MCP shape:

```text
POST /teams/{team_id}/members
DELETE /teams/{team_id}/members/{agent_id}
mcp.add_team_member(team_id, agent_id, role)
mcp.remove_team_member(team_id, agent_id)
```

Suggested body:

```json
{
  "agent_id": "bo_ng_cdx2_worker1",
  "role": "worker",
  "invited_by": "boladi_a",
  "delegated_admin": true
}
```

Authorization rule:

```text
Allowed if caller is:
- team creator, or
- project owner, or
- team admin, or
- explicitly delegated orchestrator for the project/team.
```

Acceptance criteria:

- Adding `bo_ng_cdx2_worker1` leaves all existing `bo_ng_*` team members untouched.
- The project owner can add a member without logging in as the original team creator.
- The API response includes `already_member: true` for idempotent replays.
- `role-response` for a not-yet-invited agent returns a machine-readable `team_member_not_invited`, not a generic 404.

### C2-ROSTER-2: Roster semantics make safe automation hard

Severity: HIGH

Problem:
The platform has a roster operation with creator-only/full-roster semantics, and prior audit evidence says the MCP `set_team_roster` operation is a destructive full replace. That makes an implementation agent nervous to "just add one lane" because a partial roster call can evict live members.

Impact:
Automation avoids the formal team model and falls back to hierarchy/self-claim. That keeps work moving, but it weakens the board as the single coordination ledger.

Recommended fix:
Make destructive replacement explicit and make merge/additive behavior the default for automation.

Suggested API contract:

```json
{
  "mode": "merge",
  "members": [
    {"agent_id": "bo_ng_cdx2_worker1", "role": "worker"}
  ]
}
```

Rules:

```text
mode=merge   -> add/update listed members only
mode=replace -> replace full roster; requires explicit confirm_replace=true
```

Acceptance criteria:

- A one-member merge call cannot remove any existing member.
- A replace call without `confirm_replace=true` fails with `400 roster_replace_confirmation_required`.
- The docs and MCP tool descriptions say whether each roster operation is merge or replace.

### C2-LANE-1: "Lane" is not a first-class platform concept

Severity: MEDIUM

Problem:
The board sees agents and tasks, but not runner lanes. Locally, the second Codex lane had to be separated by naming and file conventions: `Codex2`, `bo_ng_cdx2_*`, `runner_loop.codex2.pid.json`, `operations_codex2.jsonl`, and `runner_state_codex2_p34.json`. The platform task records did not have an explicit `runner_id`, `lane_id`, or `batch_id` field for this separation.

Impact:
Parallel lanes are harder to reason about in the board UI and API. Two lanes can share a project/team and even a downstream packet inbox, but there is no stable platform field that says "this task is currently owned by runner lane X in batch Y." The risk is duplicate work, ambiguous ownership, or overloading agent names as the only lane identifier.

Recommended fix:
Add optional runner metadata to task claims and updates.

Suggested fields:

```json
{
  "runner_id": "neurogolf:codex2:batch34:pid16012",
  "lane_id": "codex2",
  "engine": "codex",
  "batch_id": "34",
  "claim_id": "uuid",
  "local_state_ref": "runner_state_codex2_p34.json"
}
```

Important distinction:

```text
lane_id / runner_id: coordination identity for the running automation lane
source_lane: downstream Submit Captain packet lane, e.g. Codex
```

Do not force those to be the same value. In this run, the cdx2 runner correctly wrote to the existing `Codex` Submit Captain lane/inbox.

Acceptance criteria:

- A task claim can include `runner_id`, `lane_id`, and `batch_id`.
- The task detail view shows active runner metadata separately from assigned agent.
- Task history records runner metadata on each state transition.
- API filtering can answer: "show active tasks for lane_id=codex2" and "show batch=34 across all lanes."

### C2-LANE-2: UI should show hierarchy/self-claim lanes even when roster membership is incomplete

Severity: MEDIUM

Problem:
After hierarchy update, cdx2 workers could self-claim tasks successfully. Formal team roster membership still failed. This creates a board-model mismatch: the agents are operationally part of the project but may not be visible as full team members.

Impact:
The owner can see work moving in local logs while the board/team view underrepresents who is actually authorized and active. That makes live incident response harder.

Recommended fix:
Render project hierarchy membership and active runner claims independently of team roster.

Acceptance criteria:

- Project view shows agents in hierarchy even if not in team roster.
- Task view shows the assigned cdx2 worker and runner metadata even if roster membership is missing.
- UI flags the mismatch explicitly, e.g. `authorized via project hierarchy, not team roster`.
- The board offers a one-click/remediation API path to invite/add the hierarchy agent to the team when permissions allow.

### C2-MCP-1: Headless automation needs a reliable MCP path or documented REST parity

Severity: MEDIUM

Problem:
This Codex automation session did not have usable `mcp__handoff__*` tools exposed through the local tool surface, so bring-up used raw REST endpoints. Existing companion audit evidence also records headless MCP session friction. The platform can still be automated, but there are now two surfaces with different discoverability and likely different coverage.

Impact:
Implementation agents and runners may follow docs that say "use MCP" but end up needing REST. That slows bring-up and increases drift between MCP tools and HTTP endpoints.

Recommended fix:
Either make headless MCP consistently available for automation clients, or publish REST as a first-class supported automation surface with parity docs.

Minimum useful doc additions:

```text
Endpoint                       Auth required                         Body schema
POST /agents/register          owner/bootstrap bearer                agent_id, display_name, role?
PUT /requests/{id}/hierarchy   owner/orchestrator agent credential   nodes/edges or hierarchy object
POST /requests/{id}/goals      owner/orchestrator agent credential   title, description, external_key?
POST /requests/{id}/tasks      owner/orchestrator agent credential   title, prompt, external_key?, assignee?
PUT /tasks/{id}                assigned agent credential for status  status, assignee, runner metadata
PUT /teams/{id}/roster         team creator/admin                    mode, members
```

Acceptance criteria:

- A headless client can call MCP docs/list/update operations with only documented bootstrap/agent credentials.
- REST docs include concrete JSON request/response examples and auth requirements.
- MCP and REST docs map to each other: every critical automation action has one blessed supported path.

### C2-DOCS-1: Endpoint docs need body schemas and auth matrix

Severity: MEDIUM

Problem:
The available docs exposed endpoint names, but not enough request body schema or authorization detail for safe automation. During cdx2 bring-up, the exact bodies and auth combinations for agent registration, hierarchy update, task creation, roster update, and role-response had to be inferred by reading local scripts and trialing calls.

Impact:
Agents spend time reverse-engineering instead of implementing. Worse, they may try a destructive roster operation because the doc surface does not clearly distinguish "replace" from "add."

Recommended fix:
Add an "Automation quickstart" section to `/api/v1/docs` and MCP `get_docs` focused on this exact workflow: "add a second runner lane to an existing project."

The quickstart should include:

```text
1. Register agents
2. Add/invite them to a team without replacing the roster
3. Add them to project hierarchy
4. Create or resolve a batch goal
5. Create or resolve tasks by external key
6. Claim/update tasks with worker keys
7. Mark terminal statuses and attach packet/result references
```

Acceptance criteria:

- A new implementation agent can follow the docs without reading existing NeuroGolf scripts.
- Each step includes one curl example and one MCP example.
- Each error response includes `error_code`, `message`, and `remediation`.

### C2-UPsert-1: Batch/task creation needs idempotent external keys

Severity: MEDIUM

Problem:
Batch 34 task setup needed a local board scan and a local JSON mapping of prompt task -> board task id. This worked, but it is fragile in parallel lanes. If a config render runs twice or two supervisors race, duplicate board tasks are easy unless every client manually deduplicates.

Impact:
Parallel lanes make duplicate creation more likely. UUID-only task references also make recovery from board reorgs expensive.

Recommended fix:
Support idempotent create-or-resolve by caller-provided external keys.

Suggested external keys:

```text
goal external_key: neurogolf:batch34:codex2
task external_key: neurogolf:batch34:task201
```

Suggested endpoint:

```text
PUT /requests/{project_id}/goals/by-external-key/{external_key}
PUT /requests/{project_id}/tasks/by-external-key/{external_key}
GET /requests/{project_id}/tasks/resolve?external_key=neurogolf:batch34:task201
```

Acceptance criteria:

- Replaying the same batch config returns the existing goal/task ids.
- Duplicate task titles are allowed only when their external keys differ.
- A conflict returns `409 external_key_conflict` with the existing resource id.
- Task history records `created_by_external_key` or `resolved_by_external_key`.

### C2-ERROR-1: 4xx errors should be machine-readable and actionable

Severity: LOW-MEDIUM

Problem:
Two key failures returned human text only:

```text
403 "only the team creator can set the roster"
404 "team or member not found"
```

Those strings are understandable to a human, but not enough for robust automation. The 404 is especially weak because it could mean missing team, missing member, not invited, deleted, or wrong auth context.

Recommended fix:
Normalize error bodies across REST and MCP.

Suggested shape:

```json
{
  "success": false,
  "error_code": "team_creator_required",
  "message": "Only the team creator or delegated team admin can replace this roster.",
  "remediation": "Use POST /teams/{team_id}/members or authenticate as a delegated team admin.",
  "resource": {
    "type": "team",
    "id": "team_644ac577"
  }
}
```

Needed codes:

```text
team_creator_required
team_member_not_invited
agent_not_registered
project_hierarchy_required
forbidden_status_transition
resource_deleted
not_found
external_key_conflict
roster_replace_confirmation_required
```

Acceptance criteria:

- Every 4xx response has `error_code`.
- `message` is for humans; `error_code` is stable for clients.
- Each documented error has a recommended remediation.

### C2-LOCAL-1: Local runner wrappers had to treat runner lane separately from engine

Severity: FYI for platform author; owner-side local fix already made

Problem:
The local `Start-HandoffRunner.ps1`/monitor wrappers originally allowed `Codex` and `Claude`, and the runner pid/log naming was keyed by lane. Running a second Codex engine lane required adding `Codex2` locally so it could have a separate pid file, monitor log, state, and operation log while still spawning Codex sessions.

Impact:
This is mostly a NeuroGolf local integration issue, but it points to a platform design point: engine type and coordination lane are different concepts.

Recommended platform stance:
Do not model the lane as the engine. Model:

```text
engine = codex | claude | ...
lane_id = codex | codex2 | alpha-claude | ...
runner_id = unique process/batch lease owner
source_lane = downstream packet/staging lane
```

Acceptance criteria:

- HandOff APIs accept `engine`, `lane_id`, and `runner_id` independently.
- UI labels do not assume `lane_id == engine`.
- The same engine can operate two runner lanes without special naming hacks.

## Proposed implementation backlog

### Ticket 1: Add safe additive team membership

Priority: P0

Implement:

- `POST /teams/{team_id}/members`
- `DELETE /teams/{team_id}/members/{agent_id}`
- MCP parity tools
- Delegated admin/project owner authorization
- Idempotent `already_member` behavior

Done when:

- A project owner can add `bo_ng_cdx2_worker1` to `team_644ac577` without replacing the roster.
- Existing team members remain unchanged.
- Non-invited role-response returns `team_member_not_invited`.

### Ticket 2: Make roster replacement explicit and guarded

Priority: P0

Implement:

- `mode=merge|replace` on roster writes.
- `replace` requires `confirm_replace=true`.
- Docs mark old `set_team_roster` as dangerous/replace-only or change default to merge.

Done when:

- A partial roster write cannot evict live members by accident.
- Tests cover merge, replace-with-confirmation, and replace-without-confirmation failure.

### Ticket 3: Add runner/lane metadata to task claims

Priority: P1

Implement:

- Optional `runner_id`, `lane_id`, `engine`, `batch_id`, `claim_id` fields on task update.
- UI display and API filters for those fields.
- History entries preserve those fields per transition.

Done when:

- Board can show active `codex2` batch 34 tasks independently from original `codex`.
- API can filter by `lane_id` and `batch_id`.

### Ticket 4: Publish headless automation docs and MCP/REST parity

Priority: P1

Implement:

- `/api/v1/docs` automation quickstart.
- MCP `get_docs` parity section.
- JSON schemas and auth matrix for agent registration, hierarchy, goal/task upsert, roster, task update.

Done when:

- An implementation agent can onboard a second lane without reading NeuroGolf source.
- Each example includes request, response, auth, and expected error cases.

### Ticket 5: Add idempotent external-key upsert/resolve for goals and tasks

Priority: P1

Implement:

- External key fields on goals/tasks.
- Create-or-resolve endpoints.
- Conflict handling.
- MCP parity.

Done when:

- Re-running batch config render cannot create duplicate board tasks.
- A board reorg can be recovered by resolving external keys instead of hand-editing UUIDs.

### Ticket 6: Normalize 4xx error contract

Priority: P2

Implement:

- Stable `error_code` for every 4xx.
- Remediation text.
- REST and MCP use the same codes.

Done when:

- A client can branch on `error_code` rather than matching message text.
- Tests cover team creator required, member not invited, external key conflict, and resource deleted.

### Ticket 7: Show hierarchy-authorized agents in the UI

Priority: P2

Implement:

- Project hierarchy agent list independent of team roster.
- Task badge for "authorized by hierarchy, not roster."
- Remediation action to invite/add when caller has permission.

Done when:

- cdx2 agents that can self-claim tasks are visible somewhere obvious in the project UI even before formal roster membership is fixed.

## Suggested order of work

1. Add additive team membership and guarded roster replacement first. This removes the most dangerous operational gap.
2. Add `runner_id`/`lane_id` metadata next so parallel lanes become observable and filterable.
3. Add external-key upsert/resolve so batch creation and reorg recovery are idempotent.
4. Fill in MCP/REST docs and normalized errors alongside those API changes.
5. Improve UI display for hierarchy-authorized agents after the data model is in place.

## What worked well

- Agent registration worked cleanly once REST was used.
- Project hierarchy update accepted the cdx2 agents.
- Task self-claim/status updates worked with cdx2 worker credentials.
- Best-effort board sync did not block the local runner.
- The runner could keep `Codex2` separated locally while still using the correct `Codex` Submit Captain packet lane.

The platform does not need a full redesign for this use case. It needs safe membership mutation, explicit runner/lane metadata, idempotent task addressing, and docs/errors that let automation choose the right path without trial and error.
