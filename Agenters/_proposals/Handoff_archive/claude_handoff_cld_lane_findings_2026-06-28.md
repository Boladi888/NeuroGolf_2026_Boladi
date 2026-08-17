# HandOff critiques & suggestions — Claude cld-lane bring-up (2026-06-28)

**Author:** Claude (Opus 4.8) session, owner-directed.
**Status:** OPEN — kept active because **A1–A10 are HandOff-platform feedback for the friend** (not fixable in this repo). The local-integration items partly landed this session; see "Local status" below.

**Local status (2026-06-30):**
- **B2 (`worker_max_active:1` serializes)** — ✅ FIXED. The Codex runner now has a refilling concurrent scheduler (`worker_max_active:3`); see `claude_handoff_audit_actionable_2026-06-29.md` RUN-1.
- **B1 (one vendor-parameterized runner)** — ⏳ partial. The two runner *files* still exist, but `Start-HandoffRunner.ps1` / `Start-HandoffMonitor.ps1` are now unified behind `-Lane Codex|Claude` (one launch command per lane; lane-specific pid/logs). A full code merge is the remaining optional refactor.
- **B3 (captain↔board outcome link)** / **B4 (per-run captain inbox)** — open; cross-system / shared-structure changes, not done.
- **A1–A10** — HandOff broker/protocol feedback for the friend; unchanged (external).
**Context:** Owner asked me to register my own `bo_ng_cld_*` agents, build my own runner, and run `The_400_Tasks/Task Prompts/32` (20 tasks) through HandOff as a Claude lane parallel to Boladi's Codex lane (`bo_ng_*`). This doc records friction with **HandOff itself** (the broker/protocol at `handoff.socnet.lol`) and with the **local HandOff integration** (`Handoff/`), plus concrete suggestions. Evidence is from the live setup of team `team_8c28e5ff` (project `c093e9dd-…`) and observation of the prior `bo_ng` run (team `team_644ac577`).

Scope note: this is process/tooling feedback, not task lessons (those route to `Onnx_Advanced/` per `agent_feedback_rules.md` §8). Routing this to `_proposals` per owner instruction.

---

## A. HandOff platform (broker + protocol)

### A1. The inbox "standing-orders gate" blocks observability and is easy to get wedged on
`get_agent_inbox` (and the REST inbox) returns **ONLY** the unacknowledged standing-orders envelope until the agent calls `ack_standing_orders`; everything else is hidden unless you pass `bypass_gate`. In the `bo_ng` run, **all four** agents (`orchestrator/worker1/reviewer1/submit_captain`) showed `orders_acked: false` for the entire run — they operated purely via the local runner and never acked, so a naive inbox read surfaced nothing but the gate.
- **Impact:** anyone (human or agent) inspecting state via the documented inbox call sees only the gate, not the work. Looks "stuck" when it isn't.
- **Suggestion:** make the gate **non-suppressing for reads** — always return the inbox, but include an `unacked_standing_orders` flag/section. Reserve hard-gating for *write/claim* actions, not reads. At minimum surface `bypass_gate` in the MCP tool description (it currently only appears in the REST hint).

### A2. Duplicate standing-order delivery is noisy
The `bo_ng_worker1` standing-order envelope had `duplicate_count: 20` (re-delivered every ~10 min, 17:14→20:22Z). Same fingerprint (`orders_fp` unchanged) re-sent ~20×.
- **Suggestion:** dedupe by `orders_fp` — a single undelivered/​unacked copy with a `last_repeated_at`, not N envelopes. Saves inbox bloat and makes `duplicate_timestamps` meaningful as a heartbeat instead of spam.

### A3. Presence/heartbeat flapping — "online" is not honest
`bo_ng_worker1` was repeatedly nagged "you hold task110 (pending_verification) but read OFFLINE — reconnect" while it was, in fact, still actively producing the candidate (task110 reached verified at 20:28Z, after ~20 offline nags). Presence flapped because the heartbeat wasn't lease-coupled to the working process.
- **HandOff already ships the fix** (`HANDOFF_LEASE=1` + `handoff-wait.mjs` self-exit/beacon), but it's opt-in and undocumented in the local setup. 
- **Suggestion:** (a) default the coordinator to lease-coupled presence; (b) treat a `pending_verification` task whose assignee is "offline" but whose `updated_at` is advancing as **working**, not stranded — base the nag on *task staleness* (no status change in >N min), not raw presence. This matches the owner's global rule "watch artifacts, not processes."

### A4. `verified` (board) ≠ `shipped` (pipeline) — the most dangerous gap
HandOff marks a task `verified` when the reviewer says SHIP, but that says nothing about whether the downstream submit pipeline accepted it. In the `bo_ng` run, **task110 was `verified` on the board** ("reviewer SHIP; captain packet …") while the **actual captain packet on disk was rejected** (`…task110.rejected_no-gain-threshold.json` — the candidate beat the local 7851 by 3 but was not byte-equal to the submitted 7852 and failed the LB no-gain guardrail). A consumer trusting the board alone would believe task110 shipped; it did not.
- **Suggestion:** allow a verification to carry a **downstream outcome** field (e.g. `verify_task` accepts a `result_status: shipped|rejected|probe_pending` + reason), or model "verified-by-reviewer" and "accepted-downstream" as distinct states. Otherwise the board overstates success.

### A5. Status transitions require the assignee key; the owner bearer 403s — undocumented in MCP
Per our `Handoff/runner/README.md`, PUT `/tasks/{id}` status transitions **403 with the owner bootstrap token alone**; they need the assigned agent's `X-Agent-Key`. The MCP `update_task` encodes this (it wants `agent_key` for a self-claim) but the **error mode isn't discoverable** — a runner using only the bearer silently fails to transition. Cost us a detour until we read the README note.
- **Suggestion:** return a structured 403 body naming the missing credential ("status transition requires assignee X-Agent-Key or orchestrator authority"), and say so in the `update_task` tool description.

### A6. `set_team_roster` role enum is too narrow for real pipelines
MCP `set_team_roster` only accepts `orchestrator | builder | tester | tool_manager | messenger`. Our pipeline has **worker / reviewer / submit_captain** semantics; the `bo_ng` team (created via a different path) shows free-form roles (`worker`, `reviewer`, `submit_captain`, `owner_admin`), so the data model clearly supports them — only the MCP enum doesn't. We had to map worker→builder, reviewer→tester, captain→tool_manager and carry the real role in the `agent_id` name.
- **Suggestion:** allow free-form `role` strings (or a richer enum) in `set_team_roster`, matching what the REST/team object already stores.

### A7. Tier-0 spawn gate is not enforced — anyone with the account can mint agents
Registering 7 new `bo_ng_cld_*` agents needed **no `tier0_approval`** (the schema says it's only required "when the broker runs the Tier-0 action gate"). On this broker it's off, so any holder of the account token can spawn unlimited custodial-wallet agents. Fine for our testnet use, but worth a conscious default.
- **Suggestion:** document whether Tier-0 is on per-deployment, and consider a soft rate-limit on new-agent spawns per account.

### A8. Encryption path (`handoff-e2e/1`) is heavy enough that everyone turns it off
Every team we saw with `encrypt:true` is stuck `forming` with **0 members** (e.g. the four "Sponsor integration proof" teams). Every team that actually did work used `encrypt:false` (incl. ours and `bo_ng`). The X25519-unseal-team-key + AES-256-GCM envelope + `coord.report __enc__` flow is real work for an agent to implement, so agents skip it.
- **Suggestion:** ship a turnkey helper (in `handoff-wait.mjs` or a tiny lib) that does challenge-sign + team-key unseal + envelope encrypt/decrypt in one call. Adoption of E2E will stay ~0% until it's a one-liner.

### A9. Minor: project→goal→task nesting required before tasks exist
`add_node` under a *project* makes a *goal*; tasks must hang off a goal. Reasonable, but the first-time path ("create team → it opens a job/project → add a goal → add tasks") isn't spelled out; I inferred it. A `create_team` response hint ("add a goal node, then tasks under it") would save a step.

### A10. MCP-registered agents are born ORPHANED (no account owner); the claim recipe is undiscoverable
Agents registered via `register_agent` (MCP) come back with `owner_id: null` — **unowned**, unlike account-token-registered agents (`bo_ng_orchestrator` → `owner_id usr_fa78a7879f`). An unowned agent is "orphaned": the account can't manage it with the owner Bearer token, and the maker warns they can be reclaimed/GC'd. The claim path is not in the `register_agent` response, the MCP tool description, or `/connect` (which only mentions coordinator "adopt", a different concept); `GET /agents/:id/claim` and `/owner` 404. The working recipe, learned only from a 403 body, is **`POST /api/v1/agents/:id/claim`** with the account Bearer token **and** the agent's `X-Agent-Key` (else: *"this agent is key-protected — send its X-Agent-Key to claim it"*). Suggest: (a) `register_agent` over MCP should auto-bind owner to the calling account, or (b) the MCP `register_agent` result should return the claim recipe + a `claimed:false` flag. Also a `User-Agent` header is mandatory on the REST API (UA-less requests get a Cloudflare 1010 block that reads like an auth failure) — worth documenting.

---

## B. Local HandOff integration (`Handoff/`)

### B1. The runner was hard-wired to Codex; "make your own runner" was necessary
`Handoff/runner/handoff_runner.py` shells `codex exec` exclusively (`config["codex"].executable`, `model gpt-5.5`). To run a Claude lane I had to author `Handoff/runner/handoff_runner_claude.py` + `Handoff/config/runner_claude.json` + `Handoff/runner/prompts_claude/` (cld-named templates, lane `Claude`, Updates_Claude inbox). The two runners are 90% identical.
- **Suggestion:** refactor to one runner with a pluggable `engine` block (`{kind: codex|claude, executable, model, invoke_args}`) so a lane is pure config. I kept the schema parallel so this merge is mechanical. The only real engine delta is the invoke + output parse:
  - Codex: `codex exec [-C repo] -m <model> -c reasoning_effort -c approval_policy -s sandbox --json -o final.md -` (session id mined from stdout).
  - Claude: `claude -p --output-format json --model <m> --effort <e> --dangerously-skip-permissions --add-dir <d> [--resume <sid>]` (single JSON result with `result` + `session_id`).

### B2. `worker_max_active: 1` serializes the whole board
The Codex runner config sets `worker_max_active: 1` and `step()` does one decision per invocation, so 20 deep tasks run **sequentially** at up to a 14400s timeout each — impractical for a real batch. (I drove this session's grind with a parallel workflow instead; the runner stands as the autonomous artifact.)
- **Suggestion:** support `worker_max_active > 1` with per-task working isolation. Tasks are already file-isolated under `Task_by_Task/taskNNN/`, so concurrency is safe without worktrees as long as workers honor the task-local write rule.

### B3. Captain packets and board verification are not linked
When the runner writes a captain packet and marks the board `verified`, nothing later stamps the board with the packet's *probe/stage* outcome (see A4). The packet gets renamed `…rejected_no-gain…` on disk but the board stays `verified`.
- **Suggestion:** have `captain_ctl.py stage/reject` (it already does run-folder writeback per the 2026-06-20 proposal) also PUT the downstream result back to the HandOff task, closing the loop.

### B4. Pre-existing Updates_Claude packets will auto-trigger the captain
Dry-running my runner with `--role all` found **10 pre-existing packets** already in `Agenters/Submit_Captain/Updates_Claude/`; the captain step fired immediately (held only by `--allow-kaggle-submit`). A fresh lane runner shouldn't act on another session's queued packets.
- **Suggestion:** scope the captain inbox per-run (a run-id subfolder) or filter packets by lane+run, so a new runner doesn't adopt stale queue items.

---

## C. What worked well
- Agent registration + custodial wallet issuance: instant, clean response with the `agent_key`.
- `create_team` → `set_team_roster` → `respond_role` → `set_hierarchy` → `add_node` lifecycle worked first-try; **team auto-activated** on the 2nd accept (`activated:true`), unlike the `bo_ng` team which is stuck `forming` (orchestrator never transitioned it — so the activation path works; the gap is when no one accepts).
- `get_hierarchy` / `list_tasks` / `get_team` give a clean board read for an external observer.
- The `bo_ng` reference (runner + prompts + packets) was a faithful template; cloning the structure for the Claude lane was straightforward.

---

## D. Concrete suggestion summary (priority order)
1. **Decouple board `verified` from downstream `shipped`** (A4/B3) — highest-value correctness fix.
2. **Lease-couple presence + nag on task staleness, not raw offline** (A3).
3. **Don't suppress inbox reads behind the standing-orders gate** (A1); dedupe order re-delivery (A2).
4. **One vendor-parameterized runner** (B1) + real worker concurrency (B2).
5. **Structured 403 + docs for status-transition auth** (A5); free-form roster roles (A6).
6. **Turnkey E2E-crypto helper** (A8) if encryption is meant to be used at all.
