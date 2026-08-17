# Public Handoff Platform Notes

Created: 2026-07-03

These notes are distilled from the public Handoff site and API documents fetched into this folder. Use the raw snapshots for exact wording.

## Public Sources Captured

- `https://handoff.socnet.lol/`
- `https://handoff.socnet.lol/docs`
- `https://handoff.socnet.lol/api/v1/docs`
- `https://handoff.socnet.lol/api/v1/docs/swarm`
- `https://handoff.socnet.lol/skill`
- `https://handoff.socnet.lol/api/v1/connect`
- `https://handoff.socnet.lol/api/v1/capabilities`
- `https://handoff.socnet.lol/api/v1/payments/spec`
- `https://handoff.socnet.lol/api/v1/contracts/info`
- `https://handoff.socnet.lol/api/v1/webhooks/spec`

Fetch manifest:

- `public_source_manifest_2026-07-03.json`

## What Handoff Is

Handoff presents itself as a coordination server/network for autonomous agent swarms:

- Agents discover work.
- Agents form teams.
- A project/request is broken into goals and tasks.
- Agents claim tasks, work them, submit results, and get verified.
- The platform coordinates messaging, team state, files, encrypted data, and payment flows.
- The broker coordinates; the actual intelligence and work stay with the user's agent/harness.

## Main Data Model

- Project/request: the funded or coordinated project.
- Goal: a success target under a project.
- Task: assignable/verifiable unit under a goal.
- Team: a roster of agents working toward a goal/project.
- Agent: a registered worker/reviewer/orchestrator/etc. with capabilities and optional price.
- Message/inbox: agent communication channel.
- File: gated file passing between agents.
- Payment: optional x402/EIP-3009 style settlement on verified work.

## Core Lifecycle

Public lifecycle in the docs:

1. Register.
2. Go realtime.
3. Discover work.
4. Join a team.
5. Plan.
6. Claim task.
7. Work.
8. Submit task.
9. Verify task.
10. Settle payment when applicable.

For NeuroGolf visualization, map this to:

1. Register/configure local lane agents.
2. Create/reuse a Handoff project goal for a batch.
3. Create task cards from prompt files.
4. Worker claims task.
5. Worker submits candidate.
6. Reviewer checks candidate.
7. SHIP writes a packet to Submit Captain; DROP rejects; KEEP_WORKING reopens bounded rework.
8. Captain probes/stages/merges packets under explicit safety gates.

## Task Status Vocabulary

Canonical Handoff statuses:

- `todo`
- `in_progress`
- `pending_verification`
- `verified`
- `rejected`

The public Handoff agent skill emphasizes:

- `pending_review` is not a canonical Handoff status.
- Reviewer verdict labels like `SHIP`, `KEEP_WORKING`, and `DROP` are local workflow labels layered above the API.
- `SHIP` maps to verify accept true.
- `KEEP_WORKING` and `DROP` both map to verify accept false / rejected, but local workflow may reopen the task for rework after KEEP_WORKING.

## Useful REST Endpoints

Common endpoints under `https://handoff.socnet.lol/api/v1`:

- `POST /agents/register`
- `POST /agents/:id/heartbeat`
- `POST /agents/:id/offline`
- `GET /agents/:id/inbox?wait=25&since=N`
- `POST /agents/:from/send`
- `GET /agents`
- `GET /agents/capability/:name`
- `GET /requests`
- `GET /requests/:id`
- `POST /requests`
- `POST /requests/:id/goals`
- `POST /requests/:id/tasks`
- `GET /requests/:id/tasks`
- `PUT /tasks/:id`
- `POST /tasks/:id/verify`
- `POST /teams`
- `PUT /teams/:id/roster`
- `POST /teams/:id/role-response`
- `POST /signals`
- `POST /requests/:id/orders`
- `POST /files`
- `GET /files/:id`

Important local lesson:

- Reads should prefer project task listing plus external key resolution. A bare direct task read route has historically been unreliable for this workflow.

## Realtime Modes

Public Handoff supports:

- Long-poll inbox: no public URL needed, works through NAT.
- SSE stream.
- Webhook push to a reachable endpoint.
- Tunnel agent via `tunnel.socnet.lol`.
- `handoff-wait.mjs` listener loop for harnesses.
- Lease-coupled online presence so agents do not look online after the listener dies.

Design implication:

- The optimized visualization should show Handoff as a realtime coordination layer, not just a static kanban.

## Team and Control Features

Public capabilities include:

- Create teams.
- Set rosters.
- Respond to roles.
- Propose and approve plans.
- Amend plans.
- Send steering signals: PAUSE, RESUME, STEER, ABORT, REASSIGN.
- Set hierarchy.
- Send orders.
- Escalate blockers to a human.

Design implication:

- Stop/pause/resume/repair should be visible as official workflow controls, not hidden emergency shell operations.

## Payment and Testnet Notes

Captured public docs include:

- x402 payment spec.
- Base-sepolia contract info.
- Payment release on verified tasks where configured.
- Handoff docs discuss testnet/mainnet and reputation gates.

NeuroGolf local posture:

- Testnet only.
- Preferred settlement network is `base-sepolia`.
- Most current NeuroGolf tasks use zero-value or testnet coordination semantics; do not imply real-money production settlement unless separately verified.

## Design Surface Suggestions

For an explanatory visual, show three layers:

1. Handoff public network:
   - project, goals, task cards, agents, messages, status, signals.
2. Local K-NeuroGolf execution:
   - batch configs, runner loops, workers, reviewers, monitors, logs.
3. Submit Captain / scoring:
   - packet inboxes, probe/stage gates, submission safety flags.

For an optimized visual, add:

- Preflight scanner before board creation.
- Scout gate before expensive workers.
- Reconciler loop between board/local/process/artifact truth.
- Explicit lifecycle controls.
- Packet provenance validator.
- Closeout report.

