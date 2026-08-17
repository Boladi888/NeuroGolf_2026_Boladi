# HandOff platform audit — the board-reorg incident + new findings (2026-06-30)

**Author:** Claude (Opus 4.8) session, owner-directed. Ran as the **ALPHA supervisor for the NeuroGolf Claude lane** — drove the HandOff runner for folder-32 (17 tasks) and the cross-lane Submit-Captain staging, live, for ~3h on 2026-06-30.
**Status:** ARCHIVED 2026-07-02. Local stale-id, launch, monitor, and packet-validation fixes were integrated; remaining platform deletion/backfill/MCP ideas are historical context unless the owner explicitly reopens them.
**Scope:** HandOff-the-platform (broker/protocol at `handoff.socnet.lol` + its MCP surface). A short local-integration note is clearly fenced at the end (owner's code, not the friend's).

**Companion docs (do not re-argue here):**
- `Handoff_archive/claude_handoff_cld_lane_findings_2026-06-28.md` — **A1–A10** (inbox gate, presence flap, verified≠shipped, status-auth, roster role-enum, Tier-0, E2E, goal nesting, orphaned agents).
- `Handoff_archive/claude_handoff_audit_actionable_2026-06-29.md` — **SKILL-1…10** (the skill package) + RUN/MON (mostly fixed locally).

This audit adds **6 findings that those don't cover**, all surfaced by a live event the prior audits predate: **a project/team was hard-deleted off the board mid-run and the lanes were consolidated onto one shared project.** Where today's run merely *re-confirms* a prior finding, it's listed once in §3 with fresh evidence — not re-argued.

---

## 1. TL;DR — new findings (priority order)

| ID | Sev | One-liner | Fix effort |
|---|---|---|---|
| **P-DEL** | **HIGH** | Hard-deleting a project/team silently orphans every dependent client — bare `404 "Task not found"`, no tombstone, no move-pointer, no event. ~70 min of silent sync failures. | M |
| **P-ROSTER** | **HIGH** | `set_team_roster` is a destructive **full replace** with no additive add/remove-member op. Adding one agent can wipe a live roster (owner had to explicitly warn against it). | S |
| **P-KEYS** | MED | Clients hardcode project/team/goal/**task UUIDs** that all change on reorg; node labels (`[cld task279]`) exist but there's no resolve-by-label, so every reorg is a manual UUID repoint. | M |
| **P-BACKFILL** | MED | No reconcile/backfill after a sync outage — status transitions during the dead window are lost; only post-recovery deltas sync, leaving a permanent hole in board history. | M |
| **P-MCP-SESSION** | MED | MCP server is unusable from a headless/automation client (`"No valid session — send an initialize request first"`), forcing automation onto the raw REST API and bifurcating the surface. | M |
| **P-410** | LOW-MED | Deletion is indistinguishable from a typo: a PUT to a deleted resource returns the same `404` as a never-existed id. (Folds into P-DEL; listed for the error-contract work.) | S |

The two HIGHs are the headline: **HandOff has no graceful story for *deleting* or *moving* a resource that external automation references.** Everything else this session worked (12 probe-verified candidates staged across both lanes, +2.06 LB pts).

---

## 2. The incident (what grounds these findings)

The Claude lane's own HandOff project `NeuroGolf_HO_CLD` / team `team_8c28e5ff` was **deleted** off the board (intentional consolidation). The runner's best-effort `--sync-handoff` kept PUTting status to the now-deleted task ids:

```
operations_claude.jsonl (lane sync log):
05:34:00  handoff_sync_failed   task279  HTTP 404  {"success":false,"error":"Task not found"}
05:50:19  handoff_sync_failed   task279  HTTP 404   ...
   ... ~70 minutes, every poll, silent (best-effort; runner kept producing) ...
06:44:03  handoff_sync_failed   task191  HTTP 404
06:45:34  handoff_sync_complete task191  -> pending_verification     ← after repoint
```

**Recovery (all owner-side, manual):** recreate the 17 tasks on the shared `NeuroGolf_HO` (`82513bd2-…`, team `team_644ac577`) under a new goal *"Batch 32 rewrite prompts (Claude lane)"* (`3aff87eb-…`) as `[cld taskNNN]` nodes → rewrite the runner config (project/team/goal + **all 17** `handoff_task_id`s) → change the runner to treat `handoff_task_id` as **config-authoritative** (it had been reading the stale value from runner state). Only *then* did syncs recover — and only for transitions happening *after* the repoint.

During the dead window the runner produced real state the board never saw: `task279` reached `ship_packeted`, `task191` ran a full `KEEP_WORKING → rework` cycle. Those transitions are permanently absent from the board.

---

## 3. New findings (detailed & actionable)

### P-DEL — Hard deletion cascade-orphans dependent clients with no graceful signal  `[HIGH]`
**Evidence:** §2 — ~70 min of `404 "Task not found"` after the parent project/team was deleted; the dependent runner had no way to learn *why* (deleted? typo? auth?) or *where the work moved*.
**Impact:** any automation or human reading the board for these tasks sees nothing; a consumer trusting the board believes the lane is dead. Silent, and only the *client's* local log shows it (see P-BACKFILL, §4 visibility).
**Root cause:** deletion is a hard drop; ids vanish and every reference 404s with no tombstone or forwarding.
**Fix (implementation-ready):**
1. **Tombstone on delete.** Keep a lightweight record for deleted project/team/goal/task ids: `{id, type, deleted_at, deleted_by, moved_to?: <id|null>}`.
2. **Return `410 Gone`, not `404`,** for a tombstoned id, with body `{"error_code":"resource_deleted","type":"task","deleted_at":"…","moved_to":"<new_id|null>"}`. Reserve `404` for ids that genuinely never existed.
3. **Emit an event** `resource.deleted{type,id,moved_to}` (and `resource.moved`) over the existing webhook channel (`register_webhook` is already in the tool surface) so dependent clients can auto-repoint instead of silently 404-looping.
**Acceptance:** a PUT/GET to a deleted task returns `410` with `error_code=resource_deleted` and a `moved_to` when consolidated; a registered webhook receives `resource.deleted`. A client can branch on `error_code` to re-resolve rather than alert-or-spin.

### P-ROSTER — `set_team_roster` is a destructive full-replace; no additive member op  `[HIGH]`
**Evidence:** owner's standing warning this session, verbatim: *"Do not run `set_team_roster` to add them — it replaces the roster and would reset the live Codex members."* The MCP surface confirms it: there is **`set_team_roster`** but **no `add_team_member` / `remove_team_member`**. (Distinct from **A6**, which critiqued the role *enum*; this is about the *replace* semantics / data-loss footgun.)
**Impact:** the only documented way to add an agent to a team is to re-send the entire roster; a partial call silently evicts every member you didn't re-list. On a shared, live team that's a coordination-wiping foot-gun — exactly why the cross-lane agents were kept *out* of the formal roster and made to self-claim instead.
**Root cause:** mutation modeled only as whole-roster PUT.
**Fix:**
1. Add idempotent **`add_team_member(team_id, agent_id, role)`** and **`remove_team_member(team_id, agent_id)`** (MCP + REST). Additive; no effect on other members.
2. Make `set_team_roster` require an explicit **`mode:"replace"|"merge"`** (default `merge`), so a naive call can't evict.
**Acceptance:** adding a member leaves existing members untouched; replacing requires `mode:"replace"`. Existing single-call setups keep working under `merge`.

### P-KEYS — Reorg requires manual repoint of every UUID; labels exist but aren't resolvable  `[MED]`
**Evidence:** consolidation forced rewriting `project_id` + `team_id` + `goal_id` + **all 17 `handoff_task_id`s** into the config, plus a runner code change to stop trusting the stale stored ids. The new nodes carry human labels (`[cld task279]`) but there's no "resolve task by (project, goal, label)" call, so clients pin raw UUIDs that don't survive a reorg. (Extends **A9**, which only notes the goal-nesting path.)
**Impact:** every board reorganization is an O(tasks) manual edit across every dependent client + a code change; high-friction and error-prone (a missed id = a silent 404 per P-DEL).
**Fix:**
1. Support a **stable external key** on task/goal nodes (e.g. accept and index the `cld task279` label) and a **resolve endpoint** `GET /tasks/resolve?project=…&goal=…&external_key=cld task279 → {task_id}` (+ MCP `resolve_task`).
2. Let `update_task` accept `external_key` in lieu of the UUID so a client never hardcodes a UUID that a reorg invalidates.
**Acceptance:** a client drives a task by `(project, goal, external_key)` and survives a project/goal move without config edits.

### P-BACKFILL — No reconcile after a sync outage; missed transitions are lost forever  `[MED]`
**Evidence:** during the §2 dead window the runner advanced `task279→ship_packeted` and ran `task191`'s rework; post-repoint, only the *next* transition synced. The board's history of those tasks has a hole; the terminal state was only luck (the next poll happened to push current status).
**Impact:** board state is "last successful PUT," not "current truth." After any outage (delete, network, 5xx) the board silently diverges from reality until/unless a later transition happens to overwrite it. Compounds **A4** (verified≠shipped): the board can be both *stale* and *over-optimistic*.
**Fix:**
1. Make status PUT **idempotent "set current state"** (already close) and give clients a **one-shot reconcile**: `PUT /tasks/{id}/state {status, result_status?, updated_at}` that a client replays for all its tasks on recovery.
2. Optionally a bulk **`POST /tasks/reconcile {project, [{external_key,status,result_status}]}`** so a recovering runner re-asserts its whole lane in one call.
**Acceptance:** after a simulated outage, a single reconcile call brings the board to the client's current per-task truth (status **and** downstream `result_status`, per A4/B3).

### P-MCP-SESSION — MCP unusable headless; forces raw-REST automation  `[MED]`
**Evidence:** calling `mcp__handoff__get_docs` this session returned `{"code":-32000,"message":"No valid session — send an initialize request first"}`. The runner therefore never uses MCP at all — it hits raw `PUT /api/v1/tasks/{id}` with `Authorization: Bearer <bootstrap>` + `X-Agent-Id`/`X-Agent-Key`. (Related to **A10**'s UA/Cloudflare-1010 note, but distinct: this is the MCP *session handshake* being unavailable to non-interactive clients.)
**Impact:** automation can't use the richer MCP surface (`list_tasks`, `verify_task`, `add_node`, …) and is stuck on the thin REST path; the two surfaces drift, and skill docs that say "use MCP" (see SKILL-2) don't work for a headless runner.
**Fix:** support a **non-interactive auth → session** path: an API-key/bootstrap-token bearer that mints an MCP session without the interactive `initialize` round-trip (or document the exact headless `initialize` handshake). Then publish a REST↔MCP parity table so automation has one blessed path.
**Acceptance:** a headless client authenticates with the bootstrap token and successfully calls `get_docs`/`list_tasks` over MCP.

### P-410 — Deletion vs not-found vs unauthorized are indistinguishable  `[LOW-MED]`
**Evidence:** the deleted-task PUT returned the same `404 {"error":"Task not found"}` a typo would. **A5** already asked for a structured `403` on the auth path; this extends that contract to deletion.
**Impact:** a client can't tell "I have a stale id (repoint)" from "I have a bug (wrong id)" from "I lack the agent key," so it can't choose between auto-recover and alert.
**Fix:** machine-readable `error_code` on every 4xx — `not_found` (404) / `resource_deleted` (410, +`moved_to`) / `forbidden_status_transition` (403, +required credential, per A5). Document in the `update_task` tool description.
**Acceptance:** a client branches purely on `error_code` for repoint / claim / alert.

---

## 4. Confirmations of prior findings (fresh 2026-06-30 evidence — not re-argued)

| Prior | Confirmed this session |
|---|---|
| **A4 / B3** verified≠shipped, no downstream link | Today I staged 12 candidates + **rejected** the stale `task279` (no-gain) and ran `captain_ctl` ledger writeback locally — none of those downstream outcomes (`staged`/`rejected`) are reflected on the board. The verified↔shipped gap is live. |
| **B4** pre-existing packets auto-trigger the captain | Hit exactly this at launch: 10 stale `Updates_Claude/` packets made the runner's captain step fire every poll. **New angle (local):** it doesn't just *fire the captain* — in `handoff_runner_claude.py` the captain branch in `step()` **short-circuits and starves the worker stage entirely** while a packet backlog exists (returns non-"captain idle"), so the lane produced **zero** workers until I raised `captain_batch_size`. See local note below. |
| **A5** status-transition auth | Auth itself worked (the failures were `404 deleted`, not `403`), so A5's model is sound; today reinforces the *need for structured error bodies* it asked for (→ P-410). |

---

## 5. Local-integration note (owner's runner — **not** the friend's platform; FYI only)

Surfaced today, recorded so it isn't lost; the implementing agent for the *platform* can skip it.

- **Captain-batch gate starves the worker stage.** `handoff_runner_claude.py` `step()` evaluates reviewer → captain → worker. When `Updates_<lane>/` holds ≥ `captain_batch_size` un-probed packets, `run_captain()` returns a non-idle string and `step()` returns there, so **no worker ever runs** while a backlog exists — even though, in an alpha-driven lane, the runner's captain step is a no-op (`--allow-kaggle-submit` is off; the alpha drains out-of-band). Worked around by setting `captain_batch_size: 9999` in `runner_claude_p32.json` (documented inline). Proper fix: decouple the worker stage from the captain gate, or skip the captain branch when `--allow-kaggle-submit` is off. (Distinct from the now-fixed RUN-7 "cld lane is serial.")

---

## 6. What worked well (keep)

- **Best-effort sync is genuinely resilient** — the runner produced through a 70-min board outage with zero workload impact (404s never blocked work). The resilience is the reason the gap was survivable; pair it with P-DEL/P-BACKFILL so it's also *visible* and *recoverable*.
- **Config-authoritative repoint** recovered cleanly on the very next poll (`handoff_sync_complete` at 06:45:34) with no restart — the right design for id changes.
- **Agent-key consent / self-claim** let the `bo_ng_cld_*` agents drive tasks on a shared project without formal roster membership — a clean way to dodge the P-ROSTER footgun until additive member ops exist.
- **Bearer + `X-Agent-Id`/`X-Agent-Key` RBAC** is clear and behaved exactly as the README documents.

---

## 7. Suggested order of attack

1. **P-ROSTER** (additive member ops) + **P-410/P-DEL error codes** — small, stops data-loss + ambiguous failures.
2. **P-DEL** tombstones + `resource.deleted` webhook — makes deletion observable & auto-recoverable.
3. **P-KEYS** resolve-by-label + **P-BACKFILL** reconcile — makes reorgs and outages non-events for clients.
4. **P-MCP-SESSION** headless auth — unifies the surface (also unblocks SKILL-2's "use MCP" guidance).
5. Then the still-open prior items: **A4/B3** (verified≠shipped) and **SKILL-1/2/3** (the skill can't be followed end-to-end without them).
