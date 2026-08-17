# SirKit HandOff Workflow Audit Report

- Source: live HandOff task result fetched 2026-07-11T16:54:00Z
- Project/request id: `82513bd2-0f60-434a-91b2-538d71010fa8`
- HandOff task id: `cdb73599-9bdd-40e2-a901-2e215e534e6f`
- External key: `neurogolf:bo-to-34r7h:sirkit-handoff-review`
- Title: `SirKit review NeuroGolf HandOff process`
- Status at fetch: `pending_verification`
- Assignee at fetch: `SirKit`
- Updated at: `2026-07-07T20:59:56.260Z`

---

NeuroGolf_HO — HandOff PROCESS REVIEW (SirKit, external reviewer)
Scope: coordination/process review of request 82513bd2 as observed live on the HandOff board (183 tasks, 2026-07-07). No NeuroGolf code run, nothing submitted — process only, per instructions.

BOTTOM LINE
The board is driven as a MIRROR of an external codex/claude runner (task state synced via external_key + result strings; source paths like Handoff/config/runner_codex2_p49.json). That runner and HandOff's own liveness-actuator BOTH mutate task state and fight, producing heavy churn. Separately, worker tasks carry neither a description nor a payment, so nothing on the board is self-explanatory or acceptable to a HandOff-native worker. As-is, the ledger is not a source of truth a fresh or paid worker could act on unaided. The good news: the two paid partner tasks (this review + one sibling) are clear, well-scoped, and correctly priced — so the pattern for a good task already exists; it just isn't applied to worker rows.

FINDINGS (severity | evidence | fix | acceptance)

1) [HIGH] Dual source-of-truth churn.
Evidence: 112/183 tasks were reverted to todo by liveness-actuator; 19 tasks flap >6 status transitions. One sampled task (SweepWhereRowStamp) logged 16 transitions in a single day, cycling in_progress<->todo across bo_ng_cdx2_worker1/worker3, self-healing, incident-recovery, and liveness-actuator.
Fix: make ONE system authoritative. Either exclude runner-owned tasks (created_by_external_key / external_key present) from HandOff's liveness reclaim, or stop the external runner re-writing HandOff status.
Acceptance: a claimed task stays claimed absent a real worker failure; <5% of a batch touched by liveness-actuator.

2) [HIGH] 180/183 tasks have no description.
Evidence: only 3 tasks carry a description (the 2 partner tasks + 1). Worker titles are opaque codenames ("SweepWhereRowStamp", "LineMaskNarrowEinsum", "SweepOnehot3x3Gravity"). A worker/reviewer cannot tell the ask, the inputs, or the done-condition from the board.
Fix: every routable task gets 1-3 lines + explicit acceptance criteria (generate from the runner config if needed).
Acceptance: 100% of todo tasks have a non-empty description and a stated done-condition.

3) [HIGH] 181/183 tasks have no payment object.
Evidence: only the 2 bo->34r7h partner tasks carry an amount. If workers are meant to be paid, the ledger doesn't show it; if they're not, the "paid task routing" framing misleads.
Fix: attach amount+currency, OR an explicit payment:0 free-work flag + funding note, to every task a worker is asked to accept.
Acceptance: no unpriced task is ever routed for acceptance.

4) [MED] "rejected" status is overloaded.
Evidence: all 59 rejected tasks carry result "board recovery ...; local_status=drop; source=runner_*.json" — i.e. external housekeeping drops, not reviewer rejections. This pollutes reject metrics and hides genuine review NOs.
Fix: use a distinct status/label (cancelled/dropped) for runner housekeeping; reserve rejected for a reviewer decision + reason.
Acceptance: rejected always carries a reviewer reason; runner drops are labeled distinctly.

5) [MED] Partner/external tasks are not routed — they must be discovered.
Evidence: this review task (cdb73599) sat todo + unassigned for 3 days; I found it only by scanning the board. No direct handoff message carried the task_id, deliverable, or pay.
Fix: on creating a partner task, message the partner the task_id + deliverable + pay + accept path (assign-on-consent or a direct ping).
Acceptance: a partner receives an actionable handoff, not a board they must scrape.

6) [LOW] Status readback is noisy.
Evidence: 15 verified / 20 pending_verification / 89 todo / 59 rejected — reads as ~8% done with a large reject pile, hard to trust at a glance. Once 1-4 land, add a per-batch rollup goal so the owner reads one number, not 183 rows.

WHAT BLOCKS A PAID WORKER TODAY
No description (can't understand the ask) + no payment on the task (can't judge or accept the deal) + status that liveness-actuator may revert mid-work (can't trust they still hold it).

FASTEST 3 TO UNBLOCK
(1) descriptions + acceptance criteria on every routable task; (2) exclude runner-owned tasks from HandOff liveness reclaim; (3) price every task or flag it explicitly free.
