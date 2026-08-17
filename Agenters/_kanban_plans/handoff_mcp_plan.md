# Plan 2 - Handoff/Hermes Kanban Orchestration

## Goal

Use the Hermes Kanban board as the durable orchestration kernel and treat
Handoff/MCP/subagent execution as worker effort attached to cards. This is the
recommended route for long-running, multi-role NeuroGolf work.

## Core Decision

Hermes Kanban should own lifecycle truth:

```text
triage -> todo -> ready -> running -> blocked | done | archived
```

Agents, CLI workers, and Handoff/MCP subagents execute work, but they do not
own the truth. Every handoff must land back on the card as a comment, block,
complete metadata, or follow-up card.

## Handoff Docs Deep Dive

The relevant Hermes docs are not describing one thing called "handoff"; they
describe three separate surfaces that need to be layered deliberately:

1. Hermes Kanban is the durable queue. It stores tasks, comments, runs, events,
   parent links, workspaces, and metadata in a per-board SQLite DB.
2. Hermes `delegate_task` / subagent delegation is a synchronous fork/join
   call. It gives a child a fresh context, restricted tools, and a final
   summary, but it is not durable. If the parent is interrupted, children are
   interrupted.
3. Worker lanes are the execution adapters. A lane is an assignee string plus a
   spawn mechanism plus a lifecycle terminator. A healthy lane must end each
   claim with exactly one terminal action: complete, block, or a dispatcher
   failure path.

What that means for NeuroGolf:

- Use Kanban for cross-agent work, auditability, restart survival, human
  interjection, review loops, and long-running tasks.
- Use short-lived subagents only inside a card when a running worker needs
  bounded parallel reasoning. Their output must be mirrored back to the card.
- Do not treat a final chat/subagent answer as a handoff. The durable handoff is
  the Kanban comment, run metadata, block reason, or created follow-up card.
- Do not make parent links do two jobs. In Hermes, a parent link is a dependency
  gate: the child sits in `todo` until the parent reaches `done`.
- Use `worktree` or `dir:<absolute-path>` workspaces for artifacts that must
  survive. Scratch workspaces are deleted when the task completes.
- Long workers must heartbeat. Hermes has claim TTL, crash detection,
  max-runtime timeout, stale detection, diagnostics, and run history, but these
  only help if workers use the lifecycle tools.

Relevant docs:

- Hermes Kanban overview:
  https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban
- Hermes subagent delegation:
  https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation
- Hermes worker lanes:
  https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban-worker-lanes
- Hermes kanban worker skill:
  https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/devops/kanban-worker/SKILL.md
- Hermes kanban orchestrator skill:
  https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/devops/kanban-orchestrator/SKILL.md

## Current Board State

Observed on 2026-06-11:

```text
hermes kanban boards list
  current board: default
  counts: archived=59, blocked=1, done=1018

hermes kanban stats
  ready=0, running=0, blocked=1, done=1018
```

The one blocked card is `t_1c63f1e9`, assigned to `neurogolf-worker-1`, for a
task110 clean restage packet. It has a structured `review-required handoff`
comment and has been blocked for about 411 hours. First action is to triage
that card before starting new work.

Local profile discovery on 2026-06-11 showed the NeuroGolf lane names already
exist:

```text
neurogolf-blocked-55
neurogolf-board-analyst
neurogolf-orchestrator
neurogolf-reviewer
neurogolf-reviewer-2
neurogolf-submit-captain
neurogolf-sync-manager
neurogolf-worker-1
neurogolf-worker-2
neurogolf-worker-3
neurogolf-worker-4
neurogolf-worker-5
neurogolf-worker-6
neurogolf-worker-7
neurogolf-worker-8
```

Profiles 5, 7, and 8 are configured as manual Claude-code style profiles in the
current local setup. The Hermes docs call external CLI lanes "not yet a paved
path", so these lanes should not be treated as unattended dispatcher lanes until
a wrapper maps their exit states back into Kanban comments, complete, or block.

## Board Choice

Two viable options:

- Continue on `default` for full historical continuity.
- Create a clean `neurogolf-live` board and leave `default` as history.

Recommended: continue on `default` until the stale task110 blocked card is
resolved, then decide whether to create `neurogolf-live`. If a new board is
created, every script must pass `--board neurogolf-live` or set
`HERMES_KANBAN_BOARD=neurogolf-live`; do not rely on whichever board is current
in a terminal.

## Can We Use Handoff/Hermes For This?

Yes, with one important boundary: Hermes Kanban can be the orchestration system
now; generic Handoff/MCP subagents should not be the orchestration system.

Usable immediately:

- Board state through `hermes kanban ... --json`.
- Profile lanes for orchestrator, reviewers, submit captain, sync manager,
  board analyst, blocked drain, and workers.
- Card creation with `--assignee`, `--parent`, `--workspace`, `--branch`,
  `--tenant`, `--priority`, `--idempotency-key`, `--max-runtime`,
  `--max-retries`, `--skill`, and `--goal`.
- Completion with `--summary` and `--metadata` JSON for downstream handoffs.
- Blocking with `review-required: ...` and a preceding structured comment.
- Diagnostics for stale blocked cards and stranded ready cards.
- Worktree/dir workspaces for preserving artifacts.

Needs setup or code before unattended use:

- The Hermes gateway or a controlled dispatch loop must be running. Without it,
  `ready` cards stay ready.
- `tools/neurogolf_kanban_python.py` needs explicit board support and richer
  wrappers for parent, metadata, workspace, branch, skill, max-runtime, and
  goal-mode options.
- Existing `.hermes/board/*.json` assumptions need to be moved behind artifact
  helpers. Board state now comes from the Hermes DB/CLI/tool layer.
- Worker 5/7/8 manual lanes need a wrapper or a documented human loop before
  they can be part of an autonomous dispatcher wave.
- Review routing should be explicit. A worker can block for review, but either
  a human/reviewer comments and unblocks the same card, or `auto_triage` creates
  a reviewer card and completes the worker handoff card. Do not create a child
  reviewer card with a blocked parent and leave the parent blocked.

## Handoff Contract

For every worker-style card:

1. Worker reads the card and only that card's scope.
2. Worker writes artifacts in the card workspace or task-local repo paths.
3. Worker posts a structured handoff comment before terminating.
4. If review is required, worker calls block with reason prefix
   `review-required: ...`.
5. Reviewer reads the handoff and either:
   - creates/links a captain card,
   - comments concrete rework and unblocks/reassigns,
   - completes the card as no-fix/stale/rejected,
   - creates a librarian lesson card when durable knowledge exists.
6. Captain is the only owner that probes/stages/submits.
7. Librarian is the only owner that edits `Onnx_Advanced/`.
8. Every terminal worker/reviewer/captain action records structured metadata,
   not just prose.

For NeuroGolf worker handoffs, preserve the existing schema direction in
`tools/schemas/worker_handoff.schema.json`:

```json
{
  "task_id": "task110",
  "candidate_id": "...",
  "worker_owner": "neurogolf-worker-1",
  "package_sha256": "...",
  "lineage_base": "...",
  "changed_package_entries": ["task110.onnx"],
  "scores": {
    "current_cost": 198764,
    "candidate_cost": 99810,
    "cost_delta": -98954,
    "point_delta": 0.6888498126067208
  },
  "expected_point_gain": 0.6888498126067208,
  "algorithm_family": "uint8chain",
  "tests_run": [],
  "risk_notes": []
}
```

Do not scrape prose when metadata can carry the same fact.

## Handoff MCP / Subagent Usage

This session exposes `multi_agent_v1` spawn/wait/close tools, not a separately
named `handoff` tool. Hermes docs distinguish short `delegate_task`-style calls
from durable Kanban: a delegate/subagent call is fork-join; Kanban is durable
state and handoff history.

Policy for this repo:

- Use MCP/subagents for bounded execution, not source-of-truth state.
- Spawn a subagent only after a Kanban card exists, unless the subagent is doing
  pre-card research for the human in the current turn.
- Pass all required context. Subagents start fresh; they do not know the card
  thread, prior attempts, local conventions, or user instructions unless the
  parent includes them.
- The prompt must include card id, workspace, owned paths, forbidden paths,
  acceptance checks, handoff JSON shape, and the exact terminal expectation.
- The subagent final answer is not enough. The parent/orchestrator must mirror
  the result to the Kanban card as a comment, block, complete, or created
  follow-up.
- Do not block the parent waiting on subagents unless the next critical-path
  action needs that result. Let cards carry state.
- Close completed subagents after their card result is captured.
- Never spawn two subagents with overlapping write scopes unless the card body
  explicitly says this is an intentional race.

Practical prompt footer for spawned agents:

```text
You are working card <CARD_ID>. You are not alone in the codebase. Do not
revert others' work. Own only the paths listed above. End by returning a JSON
handoff with: task_id, changed_files, artifact_paths, tests_run, result_status,
residual_risk, and next_owner. If the work needs review, say review_required.
```

When a parent receives that final answer, it must do one of these immediately:

- `hermes kanban comment <card> "review-required handoff:\n<json>"` then
  `hermes kanban block <card> "review-required: ..."`
- `hermes kanban complete <card> --summary "..." --metadata "{...}"`
- `hermes kanban create ... --idempotency-key ...` for a follow-up/review card,
  then complete or comment the source card with the created id.
- `hermes kanban block <card> "needs-human: ..."` if the subagent could not
  finish and human input is required.

Do not leave subagent results only in the parent chat transcript.

## Card Graph Conventions

Use Hermes parent links only for true dependencies:

```text
research A ----\
research B ----- synthesize ---- implement ---- review ---- captain
static audit --/
```

Do not use links as mere lineage labels unless the parent is already `done` or
will be completed as part of the same transition. If a worker card blocks with
`review-required`, there are two valid review patterns:

Pattern A - same-card review:

1. Worker comments structured handoff.
2. Worker blocks `review-required: ...`.
3. Reviewer comments `SHIP`, `KEEP_WORKING`, `DROP`, or `NEEDS_HUMAN`.
4. Operator/reviewer unblocks or completes the same card based on verdict.

Pattern B - review card handoff:

1. Worker comments structured handoff.
2. Worker blocks `review-required: ...`.
3. Auto-triage creates a reviewer card with idempotency key
   `review:<candidate_id>`.
4. Auto-triage links worker -> reviewer only if it also completes the worker
   card immediately, so the reviewer card can promote to `ready`.
5. Reviewer card emits structured verdict metadata.

Pattern B matches the current `tools/auto_triage_blocked.py` shape and is safer
for throughput because the worker lane is freed while review proceeds.

## Roles As Lanes

Use assignee strings as lanes:

- `neurogolf-worker-1` ... `neurogolf-worker-8`
- `neurogolf-reviewer`
- `neurogolf-reviewer-2`
- `neurogolf-submit-captain`
- `neurogolf-librarian` or current Librarian profile if configured
- `neurogolf-janitor`
- `neurogolf-board-analyst`
- `neurogolf-blocked-55`
- `neurogolf-orchestrator`
- `neurogolf-sync-manager`

Each lane has a contract, not just a model identity.

Lane contracts:

- `neurogolf-orchestrator`: decomposes owner goals into cards, discovers real
  profile names, never implements, uses parent links only for true dependencies,
  and completes its own planning card with the graph it created.
- `neurogolf-worker-*`: owns exactly one task/candidate scope, writes only its
  assigned paths/workspace, heartbeats during long operations, and ends with a
  canonical handoff.
- `neurogolf-reviewer` / `neurogolf-reviewer-2`: consumes worker handoff JSON,
  attacks proof quality, emits `SHIP_TO_CAPTAIN`, `KEEP_WORKING`, `DROP`, or
  `NEEDS_HUMAN`.
- `neurogolf-submit-captain`: only lane that mutates staged submission state,
  probes own-stack/team-stack impact, records submit outcomes, and creates
  sync-manager follow-ups.
- `neurogolf-sync-manager`: reconciles current-best manifests, submit credit,
  and ledger state after captain action.
- `neurogolf-librarian`: converts durable lessons into approved library/docs
  updates, never ad hoc edits from workers.
- `neurogolf-janitor`: reports and cleans scratch/artifact hygiene only after
  explicit card scope.
- `neurogolf-board-analyst` / `neurogolf-blocked-55`: audits board health,
  duplicate claims, stale blocked cards, and migration cleanup.

## WIP Limits

Initial limits:

- Workers: one active card per `neurogolf-worker-*` across ready/running/blocked.
- Reviewers: one active review card per reviewer profile.
- Submit Captain: one active captain card.
- Librarian: one active library integration card.
- Board analyst/blocked drain: one active cleanup card.
- Blocked cards older than 4 hours require triage; older than 24 hours require
  either a child follow-up owner or explicit archive/complete rationale.

These limits can be enforced by adapting:

- `tools/neurogolf_board_policy.py`
- `tools/neurogolf_dependency_serializer.py`
- `tools/auto_triage_blocked.py`
- `tools/update_current_state.py`

Hermes can also enforce some behavior through card options:

- Use `--max-runtime 2h` or a shorter cap for bounded worker cards.
- Use `--max-retries 1` for fragile/manual profiles so failures block instead
  of thrashing.
- Use `--goal` only for open-ended cards with explicit acceptance criteria.
- Use `--idempotency-key` everywhere automation creates cards.
- Use `--workspace worktree:<abs-or-project-path>` or `dir:<abs-path>` when
  artifacts must survive completion.

## Tooling Changes Needed

The biggest implementation task is adapter cleanup. Several current tools read
old `.hermes/board/*.json` files, but Hermes now exposes board state through
SQLite/CLI/tool calls.

Required changes:

1. Introduce a board adapter:
   - `list_cards()`
   - `show_card(id)`
   - `create_card(...)`
   - `comment_card(...)`
   - `complete_card(...)`
   - `block_card(...)`
   - `link_cards(parent, child)`
   - `reassign_card(id, assignee, reclaim=False)`
   - `reclaim_card(id, reason)`
   - `stats()`
2. Back it initially with `hermes kanban ... --json` via
   `tools/neurogolf_kanban_python.py`.
3. Add board selection to every call:
   - explicit `--board <slug>` for CLI,
   - or `HERMES_KANBAN_BOARD=<slug>` in wrapper env.
4. Support exact Hermes card creation options in wrappers:
   - parent links,
   - workspace,
   - branch,
   - tenant,
   - idempotency key,
   - max runtime,
   - max retries,
   - skills,
   - goal mode.
5. Support completion metadata:
   - `--summary`,
   - `--metadata` JSON,
   - legacy `--result`.
6. Move direct `.hermes/board/*.json` reads behind explicit artifact-store
   helpers. Board state and staged artifacts are different things.
7. Decide the artifact root:
   - existing top-level `staged_submits/`, or
   - Hermes workspace/artifact dirs, or
   - a new explicit `Agenters/00_Kanban/artifacts/` path.
8. Update `auto_triage_blocked.py` and `submit_gate.py` to parse comments from
   `hermes kanban show --json`, then emit child cards or comments instead of
   assuming local registry files exist.
9. Add a `tools/kanban_health.py` dry-run command:
   - prints WIP by lane,
   - finds duplicate task claims,
   - lists aged blocked cards,
   - finds `review-required` cards with no reviewer child,
   - finds positive expected deltas with no captain owner,
   - checks captain/librarian single-writer limits.
10. Add a `tools/kanban_card_factory.py` helper:
    - reads owner-approved task lists,
    - generates idempotency keys,
    - fills a standard worker/reviewer/captain body template,
    - dry-runs the graph before creating cards.
11. Add a manual-lane bridge for worker 5/7/8:
    - pull one ready card assigned to that manual lane,
    - print the exact Claude/Codex prompt and workspace,
    - accept/paste a returned JSON handoff,
    - write the Kanban comment/block/complete transition.

## Command Recipes

Create a worker card with preserved worktree artifacts:

```powershell
hermes kanban create "[worker task110] restage v16 clean candidate" `
  --assignee neurogolf-worker-1 `
  --workspace "worktree:<NOGIT_ROOT>\worktrees\ng-task110-v16" `
  --branch "wt/task110-v16" `
  --priority 80 `
  --max-runtime 2h `
  --max-retries 1 `
  --idempotency-key "neurogolf:worker:task110:restage-v16-clean" `
  --body "<standard worker body>" `
  --json
```

Post a review-required worker handoff:

```powershell
hermes kanban comment t_xxxxxxxx --author neurogolf-worker-1 `
  "review-required handoff:`n{...canonical JSON...}"

hermes kanban block t_xxxxxxxx `
  "review-required: task110 v16 clean candidate ready; +0.68 expected; needs reviewer"
```

Complete a terminal research/no-fix card:

```powershell
hermes kanban complete t_xxxxxxxx `
  --summary "no candidate shipped; ruled out uint8chain and bbox-cache variants" `
  --metadata "{""task_id"":""task110"",""result_status"":""no_candidate"",""tests_run"":[""visible parity""]}"
```

Create a reviewer card from a worker handoff:

```powershell
hermes kanban create "[review task110 candidate_id] from t_worker" `
  --assignee neurogolf-reviewer `
  --priority 75 `
  --idempotency-key "review:candidate_id" `
  --body "<handoff JSON plus review instructions>" `
  --json

hermes kanban link t_worker t_review
hermes kanban complete t_worker --summary "auto-triage handed off to t_review"
```

The link-before-complete order is acceptable only if completion happens in the
same automated transition. Otherwise create the reviewer card without a parent
dependency and use body/comment fields for lineage.

## Operating Loop

### 0. Reset Safely

1. Resolve or explicitly carry forward `t_1c63f1e9`.
2. Run `hermes kanban diagnostics --json`.
3. Run `hermes kanban stats`.
4. Run `hermes kanban dispatch --dry-run --json`.
5. Confirm Hermes gateway/dispatcher state before adding ready work.
6. Decide default vs `neurogolf-live` board.
7. Document the active board in this folder and in Agenters role docs.

### 1. Intake

Create cards only from an owner assignment, a prompter batch, or a deterministic
backlog foreman. Every created card needs:

- task id,
- type,
- assignee,
- source of truth for team-best cost,
- owned write paths,
- forbidden write paths,
- required handoff metadata,
- idempotency key.
- workspace kind.
- max runtime / retry policy.
- whether the card is terminal, review-required, or captain-bound.

Use idempotency keys for automation, e.g.:

```text
neurogolf:worker:task110:20260611:restage-v16-clean
```

### 2. Pull

Do not push multiple ready cards onto one worker. A worker pulls only when its
lane has zero active cards.

For Hermes-profile lanes, the dispatcher pulls. For manual lanes, a human or
bridge script pulls one card, launches the external worker, and writes the
result back to Hermes before pulling another.

### 3. Worker Result

Worker either completes terminal no-change/no-fix work or blocks
`review-required` with a handoff comment.

Worker handoff JSON should include at least:

- `task_id`
- `candidate_id` or explicit no-candidate status
- `worker_owner`
- `package_path`
- `package_sha256`
- `lineage_base`
- `changed_package_entries`
- `scores`
- `expected_point_gain`
- `algorithm_family`
- `retry_attempt_index`
- `tests_run`
- `risk_notes`
- `rejected_attempts`
- `artifact_paths`
- `next_owner`

### 4. Review

Reviewer cards are child cards or explicitly linked follow-ups. Reviewer output
must be one of:

- `SHIP_TO_CAPTAIN`
- `KEEP_WORKING`
- `DROP`
- `NEEDS_HUMAN`

Reviewer affinity remains a policy: rework returns to the same reviewer unless
the reviewer card is explicitly retired.

Reviewer metadata should include:

- `source_worker_card_id`
- `candidate_id`
- `verdict`
- `blocking_findings`
- `required_rework_prompt`
- `captain_lane` when shipping
- `confidence`
- `reviewer_slot`

If verdict is `KEEP_WORKING`, create a rework card assigned to the original
worker owner and include the reviewer attack log in the body. Do not simply
unblock the original worker card if it was already completed by auto-triage.

### 5. Captain

Captain cards own:

- package audit,
- single-entry diff proof,
- profile/static cost,
- own-stack probe,
- staging,
- submit result handling,
- sync-manager follow-up.

No other lane should mutate submission packages.

Captain metadata should include:

- `candidate_id`
- `source_review_card_id`
- `package_sha256`
- `static_cost`
- `profile_cost`
- `own_stack_result`
- `team_stack_result`
- `submit_decision`
- `submit_id` or `not_submitted_reason`
- `sync_manager_followup_card_id`

### 6. Librarian

Only durable lessons become librarian cards. The lesson card must include:

- task id,
- technique/trap name,
- before/after cost,
- evidence path,
- suggested `Onnx_Advanced` destination,
- reviewer/captain confidence.

## First Pilot

Pilot with low WIP:

1. Triage the existing task110 blocked card.
2. Run `hermes kanban dispatch --dry-run --json` and fix stranded lanes before
   creating new ready cards.
3. Create 3 worker cards from an owner-approved exact task list.
4. Use only Hermes-profile lanes for the first pilot unless a manual-lane bridge
   is already written.
5. Allow only 2 review cards and 1 captain card at once.
6. Require every worker to use the canonical handoff JSON.
7. Run `kanban_health.py` after each card transition.
8. Close the pilot only when every positive delta is terminal or actively owned.

## Implementation Phases

### Phase 1 - Control Plane Only

Goal: make the board truthful before trusting autonomous workers.

- Document active board slug.
- Patch `KanbanClient` with board, metadata, parent, and workspace support.
- Add `kanban_health.py` dry-run.
- Add standard card body templates.
- Run health checks against the current `default` board.
- Triage `t_1c63f1e9`.

### Phase 2 - Manual Pilot

Goal: use Hermes as the state system while humans/this session still launch
workers manually.

- Create worker cards with idempotency keys.
- Manually execute or supervise each worker.
- Post canonical handoff comments and block/complete cards.
- Let auto-triage create reviewer cards.
- Route one approved card to Submit Captain.

### Phase 3 - Hermes Profile Dispatch

Goal: let Hermes spawn the profiles that are already configured.

- Start gateway or use controlled dispatch ticks.
- Use `dispatch --dry-run` before each wave.
- Assign only known-good Hermes profiles.
- Keep worker 5/7/8 manual unless wrapped.
- Use max-runtime and max-retries on every created card.

### Phase 4 - External CLI Lane Bridge

Goal: make Codex/Claude CLI/manual profiles first-class lanes.

- Implement a small bridge that claims/prints/pastes/writes back one card at a
  time.
- Later, replace the paste step with a real spawn wrapper if desired.
- Wrapper must map success to complete, review-needed to comment+block, and
  failure to block with a concrete reason.

### Phase 5 - Dashboard/Reporting

Goal: replace scattered run CSVs with board-derived reporting.

- Generate `CURRENT_STATE.md` from `hermes kanban list/show --json`.
- Generate reviewer/captain queues from comments/run metadata.
- Keep CSV exports only as compatibility artifacts during migration.

## Why This Should Work Better

- The board, not chat context, stores state.
- Handoffs are durable comments/metadata, not ad hoc final messages.
- WIP limits make reviewer/captain bottlenecks visible.
- Blocked cards age visibly and can be drained deterministically.
- Linked cards replace buried run-folder state.
- Hermes already handles crashed workers, stale claims, task runs, comments,
  event logs, boards, and workspaces.

## Risks

- Current repo tools still mix old `.hermes/board` artifact paths with board
  state. That must be cleaned up before serious automation.
- The default Hermes board has a large history. A fresh board may be cleaner,
  but history links and old card ids will be less convenient.
- External CLI lanes are not a fully paved Hermes path. If Codex/Claude CLI
  workers are used outside Hermes profiles, they need a wrapper that maps exit
  states back to `kanban_complete` or `kanban_block`.
- Scratch workspaces are ephemeral on completion. Use `worktree` or `dir:<path>`
  for any task whose artifacts must survive.
- Parent links are dependency gates. Incorrectly linking blocked worker cards to
  reviewer cards can park reviews forever.
- Subagent delegation is not durable. If it is used without mirroring results
  back to Kanban, the exact failure mode of the current orchestration returns.

## Success Criteria

- `hermes kanban stats` shows active WIP within policy.
- `hermes kanban diagnostics --json` has no stale blocked or stranded ready
  cards before starting a new wave.
- Every `review-required` worker card has exactly one reviewer owner or a
  documented reason it is waiting for a human.
- Every reviewer-approved positive delta has exactly one captain owner.
- Captain and Librarian remain single-writer lanes.
- `CURRENT_STATE.md` can be generated from the board without manual edits.
- A worker/reviewer/captain can crash or lose context and the next actor can
  continue from the card history alone.
