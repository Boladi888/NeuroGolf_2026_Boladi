from argparse import Namespace
from datetime import datetime, timezone

from tools.neurogolf_backlog_autoforeman import (
    POOL_A_COST_FLOOR,
    body_for_task,
    candidate_cost_floor_defer_reason,
    candidate_defer_reason,
    candidate_has_cost_floor_exception,
    next_retry_attempt_index,
    plan_backlog,
    priority_for_candidate,
    task_attempts_exhausted_this_pass,
)
from tools.neurogolf_blocked_mechanical_cleanup import classify_blocked_card, plan_blocked_cleanup
from tools.neurogolf_dependency_serializer import plan_serialization
from tools.neurogolf_review_packet_builder import card_needs_review, plan_review_packet
from tools.neurogolf_submit_trigger import plan_submit_trigger, plan_waiting_submit_cleanup
from tools.neurogolf_sync_current_best import effective_main_local_dir, effective_orchestrator_current_best


# ----- Pool A cost-floor enforcement (added 2026-05-17) ---------------------


def test_cost_floor_hard_defers_sub_floor_candidate_without_exception():
    """Per Bovath high-EV: tasks with current_cost < 75,000 must be deferred
    unless a reviewer-exception marker is present. Default backlog candidates
    should never get assigned to a Pool A worker on the basis of cost alone.
    """
    candidate = {"task": "task007", "current_cost": 9231}
    reason = candidate_cost_floor_defer_reason(candidate)
    assert reason
    assert "9231" in reason
    assert str(POOL_A_COST_FLOOR) in reason
    # Hard-defer path: candidate_defer_reason should refuse even with
    # allow_soft=True (allow_soft bypasses soft defers, not hard ones).
    assert candidate_defer_reason(candidate, allow_soft=True)


def test_cost_floor_allows_at_or_above_floor():
    for cost in (POOL_A_COST_FLOOR, POOL_A_COST_FLOOR + 1, 500_000):
        candidate = {"task": "task200", "current_cost": cost}
        assert candidate_cost_floor_defer_reason(candidate) == ""


def test_cost_floor_skipped_when_no_cost_data():
    """Missing cost data should not be a hard defer — let other gates handle it."""
    candidate = {"task": "task300"}
    assert candidate_cost_floor_defer_reason(candidate) == ""
    candidate2 = {"task": "task301", "current_cost": ""}
    assert candidate_cost_floor_defer_reason(candidate2) == ""
    candidate3 = {"task": "task302", "current_cost": "n/a"}
    assert candidate_cost_floor_defer_reason(candidate3) == ""


def test_cost_floor_structured_exception_bypasses_check():
    """A truthy reviewer_exception / cost_floor_exception field allows sub-floor."""
    cand_bool = {
        "task": "task007",
        "current_cost": 9231,
        "cost_floor_exception": True,
    }
    assert candidate_cost_floor_defer_reason(cand_bool) == ""
    assert candidate_has_cost_floor_exception(cand_bool) == "cost_floor_exception=true"

    cand_text = {
        "task": "task007",
        "current_cost": 9231,
        "reviewer_exception": "task007 sits at the 1-pt floor on Kaggle; scout sees a binary-neighborhood LUT route",
    }
    assert candidate_cost_floor_defer_reason(cand_text) == ""
    assert "binary-neighborhood" in candidate_has_cost_floor_exception(cand_text)


def test_cost_floor_freeform_text_exception_bypasses_check():
    """The workflow doc's exact phrase 'high-upside structural rationale' bypasses the floor."""
    cand = {
        "task": "task150",
        "current_cost": 4674,
        "next_action": "shrink (orchestrator flagged high-upside structural rationale: cost-1 trivial)",
    }
    assert candidate_cost_floor_defer_reason(cand) == ""


def test_cost_floor_does_not_apply_to_specialist_or_review_paths():
    """Specialist / review cards have no current_cost field; floor must not fire."""
    candidate = {
        "task": "task007",
        "status": "queued_review_candidate",
        "next_action": "reviewer to verify lineage",
    }
    # No current_cost → no floor defer (other gates handle queued review status).
    assert candidate_cost_floor_defer_reason(candidate) == ""


def test_dependency_serializer_unassigns_future_cards_without_touching_context_holder():
    cards = [
        {
            "id": "t_context",
            "title": "worker context",
            "assignee": "neurogolf-worker-2",
            "status": "blocked",
            "priority": 90,
            "created_at": 1,
        },
        {
            "id": "t_future",
            "title": "queued next",
            "assignee": "neurogolf-worker-2",
            "status": "todo",
            "priority": 30,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == [
        {
            "op": "serialize_future_card",
            "worker": "neurogolf-worker-2",
            "head_id": "t_context",
            "task_id": "t_future",
            "status": "todo",
            "title": "queued next",
        }
    ]
    assert report["unresolved"] == []


def test_dependency_serializer_reports_extra_blocked_instead_of_unassigning_it():
    cards = [
        {
            "id": "t_a",
            "title": "context a",
            "assignee": "neurogolf-worker-4",
            "status": "blocked",
            "priority": 80,
            "created_at": 1,
        },
        {
            "id": "t_b",
            "title": "context b",
            "assignee": "neurogolf-worker-4",
            "status": "blocked",
            "priority": 70,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == []
    assert report["unresolved"][0]["task_id"] == "t_b"
    assert "reviewer/blocked cleanup" in report["unresolved"][0]["reason"]


def test_dependency_serializer_reclaims_running_return_card_behind_active_head():
    cards = [
        {
            "id": "t_context",
            "title": "worker context",
            "assignee": "neurogolf-worker-5",
            "status": "running",
            "priority": 90,
            "created_at": 1,
        },
        {
            "id": "t_return",
            "title": "[optimization-return] task137 stronger evidence",
            "assignee": "neurogolf-worker-5",
            "status": "running",
            "priority": 84,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == [
        {
            "op": "serialize_running_return_card",
            "worker": "neurogolf-worker-5",
            "head_id": "t_context",
            "task_id": "t_return",
            "status": "running",
            "title": "[optimization-return] task137 stronger evidence",
        }
    ]
    assert report["unresolved"] == []


def test_dependency_serializer_assigns_unassigned_return_card_to_idle_origin_worker():
    cards = [
        {
            "id": "t_return",
            "title": "[repair] task300 returned for stronger evidence",
            "body": "Return this to neurogolf-worker-6 when the lane is free.",
            "assignee": "",
            "status": "ready",
            "priority": 85,
            "created_at": 1,
        }
    ]

    report = plan_serialization(cards)

    assert report["actions"] == [
        {
            "op": "assign_return_card",
            "worker": "neurogolf-worker-6",
            "task_id": "t_return",
            "status": "ready",
            "title": "[repair] task300 returned for stronger evidence",
        }
    ]
    assert report["return_waiting"] == []


def test_dependency_serializer_parses_compact_worker_hint_on_return_card():
    cards = [
        {
            "id": "t_return",
            "title": "[optimization-return] task017 worker6 cursor shrink needs stronger v2",
            "body": "Return to the same origin worker.",
            "assignee": "",
            "status": "ready",
            "priority": 84,
            "created_at": 1,
        }
    ]

    report = plan_serialization(cards)

    assert report["actions"][0]["op"] == "assign_return_card"
    assert report["actions"][0]["worker"] == "neurogolf-worker-6"


def test_dependency_serializer_waits_to_assign_return_card_when_origin_worker_busy():
    cards = [
        {
            "id": "t_context",
            "title": "worker context",
            "assignee": "neurogolf-worker-6",
            "status": "blocked",
            "priority": 90,
            "created_at": 1,
        },
        {
            "id": "t_return",
            "title": "[optimization-return] task306 stronger candidate needed",
            "body": "worker owner `neurogolf-worker-6`",
            "assignee": "",
            "status": "ready",
            "priority": 85,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == []
    assert report["return_waiting"][0]["task_id"] == "t_return"
    assert report["return_waiting"][0]["worker"] == "neurogolf-worker-6"


def test_dependency_serializer_routes_return_by_source_worker_card_id():
    cards = [
        {
            "id": "t_7c1743e3",
            "title": "active worker1 context",
            "assignee": "neurogolf-worker-1",
            "status": "blocked",
            "priority": 80,
            "created_at": 1,
        },
        {
            "id": "t_cb4eb45b",
            "title": "[worker task202 neurogolf-worker-1 current-live]",
            "assignee": "neurogolf-worker-1",
            "status": "done",
            "priority": 58,
            "created_at": 2,
        },
        {
            "id": "t_return",
            "title": "[optimization-return] task202 active-extent bypass is stale",
            "body": "Candidate reviewed from source worker card t_cb4eb45b.",
            "assignee": "",
            "status": "ready",
            "priority": 68,
            "created_at": 3,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == []
    assert report["return_waiting"][0]["task_id"] == "t_return"
    assert report["return_waiting"][0]["worker"] == "neurogolf-worker-1"


def test_dependency_serializer_routes_return_by_generic_source_card_id():
    cards = [
        {
            "id": "t_cb4eb45b",
            "title": "[worker task202 neurogolf-worker-1 current-live]",
            "assignee": "neurogolf-worker-1",
            "status": "done",
            "priority": 58,
            "created_at": 1,
        },
        {
            "id": "t_return",
            "title": "[optimization-return] task202 active-extent bypass is stale",
            "body": "Reviewed source card: t_cb4eb45b.",
            "assignee": "",
            "status": "ready",
            "priority": 68,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"][0]["op"] == "assign_return_card"
    assert report["actions"][0]["worker"] == "neurogolf-worker-1"


def test_dependency_serializer_routes_return_through_review_packet_source_lines():
    cards = [
        {
            "id": "t_11111111",
            "title": "[review packet] blocked local-delta candidates",
            "body": "\n".join(
                [
                    "Source cards:",
                    "- t_a1111111 | neurogolf-worker-1 | tasks=task092 | [worker task092 neurogolf-worker-1 current-live]",
                    "- t_b2222222 | neurogolf-worker-6 | tasks=task084 | [worker task084 neurogolf-worker-6 current-live]",
                ]
            ),
            "assignee": "neurogolf-reviewer",
            "status": "done",
            "priority": 85,
            "created_at": 1,
        },
        {
            "id": "t_return",
            "title": "[optimization-return] task084 claimed exact-public shrink failed reviewer replay",
            "body": "Review outcome from t_11111111: return for repair/debug.",
            "assignee": "",
            "status": "ready",
            "priority": 80,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"][0]["op"] == "assign_return_card"
    assert report["actions"][0]["worker"] == "neurogolf-worker-6"


def test_dependency_serializer_parks_future_return_card_behind_lane_head_without_unassigning():
    """Bug fix 2026-05-17:

    Submit-captain creates a `[repair taskNNN] ...` card after a public-negative
    Kaggle result, assigned to the original worker. Previously the serializer's
    `is_return_card` only matched literal `[repair]` prefixes, so a card titled
    `[repair task034] ...` was treated as a generic extra and unassigned via
    `serialize_future_card` — destroying the worker's context handoff.

    With the regex-based prefix check + the new `park_future_return_card_assigned`
    op, the repair card is linked behind the lane head as a child but its
    `assignee` is preserved. When the head completes, the worker picks up
    the repair with their per-profile memory and worktree state intact.
    """
    cards = [
        {
            "id": "t_head",
            "title": "[worker task050 neurogolf-worker-6 current-live] cursor backlog task",
            "assignee": "neurogolf-worker-6",
            "status": "running",
            "priority": 60,
            "created_at": 1,
        },
        {
            "id": "t_repair",
            "title": "[repair task034] public-negative fixed-9x9 crop invariant after task034_v4",
            "body": "Original worker card: t_82528e59 (neurogolf-worker-6). Failed +1.288 expected -> -13.42 actual.",
            "assignee": "neurogolf-worker-6",
            "status": "todo",
            "priority": 60,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    park_actions = [a for a in report["actions"] if a.get("op") == "park_future_return_card_assigned"]
    assert len(park_actions) == 1, f"expected one park action, got: {report['actions']}"
    park = park_actions[0]
    assert park["worker"] == "neurogolf-worker-6"
    assert park["task_id"] == "t_repair"
    assert park["head_id"] == "t_head"
    # CRITICAL: no `serialize_future_card` (which would unassign).
    assert not any(a.get("op") == "serialize_future_card" and a.get("task_id") == "t_repair" for a in report["actions"])


def test_dependency_serializer_does_not_misroute_via_multi_worker_review_packet():
    """Bug fix 2026-05-18:

    A reviewer's `[review packet]` card often bundles multiple worker
    handoffs in one packet, listing them as structured lines like
        - t_XXX | neurogolf-worker-N | tasks=taskNNN | ...
    When an opt-return card for one of those candidates references the
    review packet via its body, `worker_from_source_card()` was doing a
    body-scan for `neurogolf-worker-N` patterns. The scan picked the FIRST
    worker mentioned (often a different task's worker), causing the dep
    serializer to think the opt-return belongs to the wrong worker — and
    repeatedly unassign it as a "misassigned return."

    Observed pattern: task004 v12 opt-return assigned to worker-8 kept
    getting unassigned every minute because the review packet's first
    source line was for task092 by worker-1. 244 ping-pong comments
    before this fix.

    Fix: when the source is a reviewer profile (any reviewer, not just
    `neurogolf-reviewer`), skip the body-scan and let
    `worker_from_review_source_lines` parse the structured source lines
    with proper task-overlap filtering.
    """
    cards = [
        # Review packet bundling task004 (worker-8) + task092 (worker-1) candidates.
        # Card IDs must match the kanban's t_[0-9a-f]{8} hex format so the
        # serializer's regex parses them as references.
        {
            "id": "t_aaaaaaaa",
            "title": "[review packet] blocked local-delta candidates abc123",
            "body": (
                "PYTHON REVIEW PACKET\n\n"
                "- t_92529292 | neurogolf-worker-1 | tasks=task092 | [worker task092 neurogolf-worker-1 current-live]\n"
                "  evidence: t_92529292 worker-1 build for task092\n"
                "- t_04040404 | neurogolf-worker-8 | tasks=task004 | [worker task004 neurogolf-worker-8 solver-hunt]\n"
                "  evidence: t_04040404 worker-8 build for task004\n"
            ),
            "assignee": "neurogolf-reviewer-2",
            "status": "done",
            "created_at": 1,
        },
        # Opt-return that references the review packet via its body. Card
        # is for task004, so the deterministic origin worker MUST be worker-8.
        {
            "id": "t_bbbbbbbb",
            "title": "[optimization-return] task004 v9 shrink fails competition decode",
            "body": (
                "Return from review packet t_aaaaaaaa.\n"
                "Candidate under review: task004_v9_droppedbgconcat_t_04040404.\n"
                "Decision: do not queue / do not rebuild this lane unchanged.\n"
            ),
            "assignee": "neurogolf-worker-8",
            "status": "todo",
            "created_at": 2,
        },
    ]
    from tools.neurogolf_dependency_serializer import intended_return_worker_from_context
    cards_by_id = {c["id"]: c for c in cards}
    result = intended_return_worker_from_context(cards[1], cards_by_id)
    # Must NOT pick worker-1 (first mentioned in review packet) just because
    # of body-scan order. Must respect task-overlap and pick worker-8.
    assert result == "neurogolf-worker-8", f"expected worker-8 (task004 origin), got {result!r}"


def test_dependency_serializer_allows_pool_b_worker_to_claim_backlog():
    """As of 2026-05-18: Pool B workers (5/7/8) are now in
    ``_BACKLOG_BANK_ALLOWED_WORKERS``. With Hermes-paid LLM seats off, the
    manual /loop sessions are the only active candidate producers, so
    they need to be able to cold-start BACKLOG bank cards without the
    misclaim guard stripping the assignee mid-flight.

    Historical context (see git log): pre-2026-05-18 the misclaim guard
    released Pool B claims on BACKLOG because the Pool A foreman was
    expected to convert them via create_worker_from_bank. That foreman
    path is dormant now. To revive the strict Pool-A-only behavior, drop
    POOL_B_SOLVER_HUNT_WORKERS from the allowed set in
    tools/neurogolf_dependency_serializer.py.
    """
    cards = [
        {
            "id": "t_pool_b_backlog",
            "title": "[BACKLOG single task] task233 cursor backlog current-live",
            "assignee": "neurogolf-worker-8",  # Pool B — now allowed
            "status": "running",
            "priority": 60,
            "created_at": 1,
        },
    ]

    report = plan_serialization(cards)
    release_actions = [a for a in report["actions"] if a.get("op") == "unassign_backlog_misclaim"]
    assert release_actions == [], (
        f"Pool B worker on BACKLOG must NOT be released under the current "
        f"consolidated policy, got: {release_actions}"
    )


def test_dependency_serializer_keeps_backlog_assigned_to_pool_a_worker():
    """A BACKLOG card legitimately assigned to a Pool A worker (1-3) by
    the autoforeman's `create_worker_from_bank` flow must NOT be released."""
    cards = [
        {
            "id": "t_legit",
            "title": "[BACKLOG single task] task100 cursor backlog current-live",
            "assignee": "neurogolf-worker-2",  # Pool A
            "status": "ready",
            "priority": 60,
            "created_at": 1,
        },
    ]
    report = plan_serialization(cards)
    assert not any(a.get("op") == "unassign_backlog_misclaim" for a in report["actions"])


def test_dependency_serializer_protects_blocked_backlog_regardless_of_body():
    """The reliable misclaim-vs-real-handoff signal is status. A BACKLOG
    card that's already `blocked` means the worker has finished steps
    2+3 of the handoff sequence (post JSON comment + run `kanban block`).
    By the time the dep serializer would see it, the candidate is real.

    Body-text scanning can't help because `hermes kanban list --json`
    doesn't expose comments and the BACKLOG body is the stock template
    with no worker_owner field. Status is the only signal that survives
    the snapshot.

    Real case: t_30560b2a (BACKLOG task233) had `worker_owner` only in
    comments. After being re-assigned to worker-7, the dep serializer
    re-released it because the body+text checks missed the handoff.
    Status-based guard catches it.
    """
    cards = [
        {
            "id": "t_30560b2a",
            "title": "[BACKLOG single task] task233 cursor backlog current-live",
            "body": "UNASSIGNED BACKLOG CARD - do not dispatch directly",  # stock body
            "assignee": "neurogolf-worker-7",  # Pool B
            "status": "blocked",  # worker has finished blocking
            "priority": 60,
            "created_at": 1,
        },
    ]
    report = plan_serialization(cards)
    release_ops = [a for a in report["actions"] if a.get("op") == "unassign_backlog_misclaim"]
    assert release_ops == [], (
        f"blocked BACKLOG card must not be released (handoff complete), got: {release_ops}"
    )


def test_dependency_serializer_allows_pool_b_backlog_in_running_status():
    """Companion to ``allows_pool_b_worker_to_claim_backlog``: Pool B is
    allowed regardless of whether the card is ``running`` (worker mid-
    spawn) or ``blocked`` (handoff complete). The 2026-05-18 policy
    change made the assignee-based check authoritative; status no longer
    distinguishes the release case for Pool B."""
    cards = [
        {
            "id": "t_pool_b_running",
            "title": "[BACKLOG single task] task300 cursor backlog current-live",
            "body": "UNASSIGNED BACKLOG CARD - do not dispatch directly",
            "assignee": "neurogolf-worker-8",  # Pool B
            "status": "running",
            "priority": 60,
            "created_at": 1,
        },
    ]
    report = plan_serialization(cards)
    release_ops = [a for a in report["actions"] if a.get("op") == "unassign_backlog_misclaim"]
    assert release_ops == [], (
        f"Pool B worker on BACKLOG (running) must NOT be released, got: {release_ops}"
    )


def test_dependency_serializer_protects_card_with_review_required_summary():
    """Bug fix 2026-05-18 (followup):

    Worker-7 cold-starts onto a BACKLOG, builds a candidate, posts the
    JSON handoff as a COMMENT, and runs `hermes kanban block --reason
    "review-required: ..."`. Comments aren't in `hermes kanban list --json`
    so the misclaim guard's worker_owner regex misses them — but the
    `review-required:` reason bubbles up to the card's `latest_summary`
    field which IS in the snapshot. Match that pattern too.

    Real case: t_30560b2a (BACKLOG task233) had the JSON handoff only in
    comments; `worker_owner` was not in body. The previous fix re-removed
    the assignee after a manual restore. The new latest_summary check
    catches it.
    """
    cards = [
        {
            "id": "t_summary_only",
            "title": "[BACKLOG single task] task233 cursor backlog current-live",
            # The body is the stock BACKLOG body — no worker_owner JSON here.
            "body": "UNASSIGNED BACKLOG CARD - do not dispatch directly",
            # But latest_summary HAS the review-required marker that workers
            # leave when they block the card after building a candidate.
            "latest_summary": "review-required: task233 equal-fold candidate, 266/266 verify_against_live",
            "assignee": "neurogolf-worker-7",  # Pool B — would normally trigger misclaim
            "status": "blocked",
            "priority": 60,
            "created_at": 1,
        },
    ]
    report = plan_serialization(cards)
    release_ops = [a for a in report["actions"] if a.get("op") == "unassign_backlog_misclaim"]
    assert release_ops == [], (
        f"card with review-required latest_summary must not be unassigned, but got: {release_ops}"
    )


def test_dependency_serializer_keeps_backlog_with_worker_handoff_intact():
    """Bug fix 2026-05-18:

    When a Pool B worker's /loop cold-starts onto a BACKLOG card, the
    misclaim guard correctly releases it BEFORE the worker has done
    anything. But if the worker has already posted a review-required
    JSON handoff (`worker_owner: neurogolf-worker-N`), the candidate is
    legitimate work in flight — releasing the card would orphan it.

    Observed regression: t_dc8111c3 (BACKLOG task264) — worker-7's /loop
    built a +0.05-floor candidate with 265/265 verify_against_live, but
    the misclaim guard unassigned the card mid-block. Card ended up stuck
    `blocked` with no assignee.

    Fix: misclaim guard now checks for a `worker_owner` JSON pattern in
    the card body and backs off if found.
    """
    cards = [
        {
            "id": "t_handoff_done",
            "title": "[BACKLOG single task] task264 cursor backlog current-live",
            # Worker posted real candidate JSON before the misclaim guard could fire.
            "body": (
                "UNASSIGNED BACKLOG CARD ...\n"
                '{"task_id":"task264","candidate_id":"task264_v2_t_dc8111c3",'
                '"worker_owner":"neurogolf-worker-7","scores":{"point_delta":0.045}}'
            ),
            "assignee": "neurogolf-worker-7",  # Pool B — would normally trigger misclaim
            "status": "blocked",
            "priority": 60,
            "created_at": 1,
        },
        {
            "id": "t_no_handoff",
            "title": "[BACKLOG single task] task100 cursor backlog current-live",
            "body": "UNASSIGNED BACKLOG CARD - no worker handoff yet",
            "assignee": "neurogolf-worker-8",  # Pool B
            "status": "running",
            "priority": 60,
            "created_at": 2,
        },
    ]
    report = plan_serialization(cards)
    release_ops = [a for a in report["actions"] if a.get("op") == "unassign_backlog_misclaim"]
    released_ids = {a["task_id"] for a in release_ops}
    # Card with handoff is PROTECTED — must not be released. The handoff
    # check is defensive cover for the case where a future config change
    # re-strips Pool B from the allowed-claimer set.
    assert "t_handoff_done" not in released_ids
    # As of 2026-05-18 a Pool B worker on a no-handoff BACKLOG is ALSO
    # protected because Pool B is now in _BACKLOG_BANK_ALLOWED_WORKERS.
    # Pre-2026-05-18 this assertion was `in released_ids`; the flip
    # follows the consolidation of all candidate production through the
    # manual /loop sessions.
    assert "t_no_handoff" not in released_ids


def test_dependency_serializer_keeps_backlog_assigned_to_draining_worker():
    """A BACKLOG card mid-flight on a draining Pool A worker (4 or 6)
    should be left alone — they can finish what they're holding."""
    cards = [
        {
            "id": "t_draining",
            "title": "[BACKLOG single task] task100 cursor backlog current-live",
            "assignee": "neurogolf-worker-4",  # draining Pool A
            "status": "running",
            "priority": 60,
            "created_at": 1,
        },
    ]
    report = plan_serialization(cards)
    assert not any(a.get("op") == "unassign_backlog_misclaim" for a in report["actions"])


def test_dependency_serializer_park_apply_is_idempotent_when_already_linked():
    """Bug fix 2026-05-18:

    The `park_future_return_card_assigned` apply path was running
    `client.link(head, task)` + `client.comment(...)` every minute even
    when the parent-child edge already existed. Hermes silently dedupes
    the link at the data level but still records a `linked` event AND
    accepts the new comment, so cards accumulated hundreds of duplicate
    park comments (462 observed on t_e254b1fa before the fix).

    The fix: read `client.show(task_id).parents` first; if head_id is
    already in the list, skip link + comment entirely.
    """
    from tools.neurogolf_dependency_serializer import apply_serialization_plan

    class FakeClient:
        def __init__(self, existing_parents):
            self._existing_parents = list(existing_parents)
            self.calls = []

        def show(self, task_id):
            self.calls.append(("show", task_id))
            return {"id": task_id, "parents": list(self._existing_parents)}

        def link(self, parent_id, child_id):
            self.calls.append(("link", parent_id, child_id))

        def comment(self, task_id, body):
            self.calls.append(("comment", task_id, body[:30]))

        def assign(self, task_id, assignee):
            self.calls.append(("assign", task_id, assignee))

    # Case 1: parent NOT yet in the card's parents → link + comment fire.
    fresh_client = FakeClient(existing_parents=[])
    plan = {
        "actions": [
            {
                "op": "park_future_return_card_assigned",
                "worker": "neurogolf-worker-5",
                "head_id": "t_head",
                "task_id": "t_child",
                "status": "todo",
                "title": "[optimization-return] taskNNN",
            }
        ]
    }
    apply_serialization_plan(fresh_client, plan)
    ops = [c[0] for c in fresh_client.calls]
    assert "link" in ops
    assert "comment" in ops

    # Case 2: parent ALREADY in the card's parents → no link, no comment.
    idempotent_client = FakeClient(existing_parents=["t_head"])
    apply_serialization_plan(idempotent_client, plan)
    ops = [c[0] for c in idempotent_client.calls]
    assert "link" not in ops, f"link should be skipped when already-parent; got calls={idempotent_client.calls}"
    assert "comment" not in ops, f"comment should be skipped when already-parent; got calls={idempotent_client.calls}"
    # show() is the only call that should happen — the idempotency probe.
    assert ops == ["show"]


def test_dependency_serializer_parks_pool_b_solver_hunt_card_without_unassigning():
    """Bug fix 2026-05-17:

    When a Pool B worker's `/loop` session cold-starts onto a BACKLOG bank
    card because their Pool B card is in transient blocked (dispatcher
    closed-port whack-a-mole), the dep serializer used to demote the Pool B
    card and strip its assignee — destroying any in-flight candidate work
    the worker had already posted as a JSON handoff comment.

    Pool B `[worker taskNNN <worker> solver-hunt] …` cards carry the same
    kind of accumulated context as repair cards (worktree, candidate ID,
    JSON handoff). They now go through `park_future_return_card_assigned`
    instead of `serialize_future_card`, preserving the assignee.
    """
    cards = [
        {
            "id": "t_cold_start",
            "title": "[worker task285 neurogolf-worker-7 current-live] cursor backlog task",
            "assignee": "neurogolf-worker-7",
            "status": "running",
            "priority": 60,
            "created_at": 1,
        },
        {
            "id": "t_pool_b",
            "title": "[worker task000 neurogolf-worker-7 solver-hunt] new-solver candidate hunt",
            "body": "task_id: task000\npool: B (new-solver hunt on unsolved task)",
            "assignee": "neurogolf-worker-7",
            "status": "ready",
            "priority": 55,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    park_actions = [a for a in report["actions"] if a.get("op") == "park_future_return_card_assigned"]
    assert len(park_actions) == 1, f"expected one park action, got: {report['actions']}"
    park = park_actions[0]
    assert park["worker"] == "neurogolf-worker-7"
    assert park["task_id"] == "t_pool_b"
    assert park["head_id"] == "t_cold_start"
    # CRITICAL: no `serialize_future_card` (which would unassign).
    assert not any(a.get("op") == "serialize_future_card" and a.get("task_id") == "t_pool_b" for a in report["actions"])


def test_dependency_serializer_promotes_return_card_to_lane_head_among_futures():
    """When BOTH worker cards are in future status (ready/todo), the repair
    card becomes the lane head and the generic backlog parks behind it.
    Otherwise the worker would burn iterations on a fresh backlog target
    instead of repairing the failed submit first."""
    cards = [
        {
            "id": "t_backlog",
            "title": "[worker task060 neurogolf-worker-4 current-live] cursor backlog task",
            "assignee": "neurogolf-worker-4",
            "status": "ready",
            "priority": 60,
            "created_at": 1,  # older — would have won lane head pre-fix
        },
        {
            "id": "t_repair",
            "title": "[repair task397] public-negative active10 constant_pad fp32 hidden-bad",
            "body": "Original worker card: t_xxx (neurogolf-worker-4). Submit returned 5727 vs expected +1.87.",
            "assignee": "neurogolf-worker-4",
            "status": "ready",
            "priority": 60,
            "created_at": 2,  # newer; would have parked pre-fix
        },
    ]
    report = plan_serialization(cards)

    # Repair becomes lane head; the backlog card is serialized behind it.
    serialize_actions = [a for a in report["actions"] if a.get("op") == "serialize_future_card"]
    assert len(serialize_actions) == 1, f"expected backlog to be serialized, got: {report['actions']}"
    assert serialize_actions[0]["task_id"] == "t_backlog"
    assert serialize_actions[0]["head_id"] == "t_repair"


def test_dependency_serializer_unassigns_return_running_on_wrong_worker():
    cards = [
        {
            "id": "t_22222222",
            "title": "[worker task106 neurogolf-worker-2 current-live]",
            "assignee": "neurogolf-worker-2",
            "status": "done",
            "priority": 58,
            "created_at": 1,
        },
        {
            "id": "t_return",
            "title": "[optimization-return] task106 idx899 rewrite failed reviewer replay",
            "body": "Candidate reviewed from source card t_22222222.",
            "assignee": "neurogolf-worker-1",
            "status": "running",
            "priority": 80,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"][0]["op"] == "unassign_misassigned_worker_return"
    assert report["actions"][0]["worker"] == "neurogolf-worker-1"
    assert report["actions"][0]["target_worker"] == "neurogolf-worker-2"


def test_dependency_serializer_recreates_scratch_return_card_in_source_workspace():
    cards = [
        {
            "id": "t_cb4eb45b",
            "title": "[worker task202 neurogolf-worker-1 current-live]",
            "assignee": "neurogolf-worker-1",
            "status": "done",
            "workspace_path": "C:\\worktrees\\t_cb4eb45b",
            "priority": 58,
            "created_at": 1,
        },
        {
            "id": "t_return",
            "title": "[optimization-return] task202 active-extent bypass is stale",
            "body": "Candidate reviewed from source worker card t_cb4eb45b.",
            "assignee": "neurogolf-worker-1",
            "status": "running",
            "workspace_kind": "scratch",
            "priority": 68,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"][0]["op"] == "replace_scratch_return_card"
    assert report["actions"][0]["workspace"] == "dir:C:\\worktrees\\t_cb4eb45b"
    assert report["actions"][0]["worker"] == "neurogolf-worker-1"


def test_dependency_serializer_normalizes_submit_alias_and_parks_extra_submit_cards():
    cards = [
        {
            "id": "t_submit_a",
            "title": "[submit task063] approved candidate",
            "assignee": "submit-captain",
            "status": "ready",
            "priority": 75,
            "created_at": 1,
        },
        {
            "id": "t_submit_b",
            "title": "[submit task070] approved candidate",
            "assignee": "submit-captain",
            "status": "ready",
            "priority": 65,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == [
        {
            "op": "normalize_specialist_assignee",
            "worker": "neurogolf-submit-captain",
            "old_assignee": "submit-captain",
            "task_id": "t_submit_a",
            "status": "ready",
            "title": "[submit task063] approved candidate",
        },
        {
            "op": "serialize_specialist_future_card",
            "worker": "neurogolf-submit-captain",
            "old_assignee": "submit-captain",
            "head_id": "t_submit_a",
            "task_id": "t_submit_b",
            "status": "ready",
            "title": "[submit task070] approved candidate",
        },
    ]


def test_dependency_serializer_serializes_duplicate_reviewer_todo_cards():
    cards = [
        {
            "id": "t_review_a",
            "title": "[review task382] candidate",
            "assignee": "neurogolf-reviewer",
            "status": "todo",
            "priority": 85,
            "created_at": 1,
        },
        {
            "id": "t_review_b",
            "title": "[review task071] candidate",
            "assignee": "neurogolf-reviewer",
            "status": "todo",
            "priority": 80,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == [
        {
            "op": "serialize_specialist_future_card",
            "worker": "neurogolf-reviewer",
            "old_assignee": "neurogolf-reviewer",
            "head_id": "t_review_a",
            "task_id": "t_review_b",
            "status": "todo",
            "title": "[review task071] candidate",
        }
    ]


def test_dependency_serializer_assigns_unassigned_specialist_continuation_when_lane_free():
    cards = [
        {
            "id": "t_submit_next",
            "title": "[submit task070] approved candidate",
            "assignee": "",
            "status": "todo",
            "priority": 65,
            "created_at": 1,
        },
        {
            "id": "t_review_next",
            "title": "[review] task071 candidate",
            "body": "Decide whether this should queue for submit-captain after review.",
            "assignee": "",
            "status": "todo",
            "priority": 80,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == [
        {
            "op": "assign_specialist_continuation",
            "worker": "neurogolf-reviewer",
            "task_id": "t_review_next",
            "status": "todo",
            "title": "[review] task071 candidate",
        },
        {
            "op": "assign_specialist_continuation",
            "worker": "neurogolf-submit-captain",
            "task_id": "t_submit_next",
            "status": "todo",
            "title": "[submit task070] approved candidate",
        },
    ]


def test_dependency_serializer_does_not_route_backlog_cards_to_reviewers():
    """A backlog worker card whose body has the stock `next_action:
    Review candidate queue ...` line must NOT be assigned to a reviewer
    profile. Worker-routing happens later via the backlog autoforeman.

    Real-world failure case: t_b8abe4d8 ([BACKLOG single task] task275
    cursor backlog current-live) was misrouted to neurogolf-reviewer-2
    because `"review candidate" in card_text` matched the body line.
    """
    cards = [
        {
            "id": "t_b8abe4d8",
            "title": "[BACKLOG single task] task275 cursor backlog current-live",
            "body": (
                "UNASSIGNED BACKLOG CARD - do not dispatch directly; "
                "Python will convert this into one worker card.\n"
                "task_id: task275\n"
                "next_action: Review candidate queue before changing this task."
            ),
            "assignee": "",
            "status": "ready",
            "priority": 60,
            "created_at": 1,
        }
    ]

    report = plan_serialization(cards)

    assert report["actions"] == []
    assert report["unresolved"] == []


# ---- Orphan ready-card fallback to running gateway (2026-05-19) -----------


def test_dependency_serializer_routes_orphan_ready_card_to_running_gateway():
    """When a ready+unassigned card has no parent-lineage worker hint and
    no recognizable specialist title, fall back to assigning it to a
    running gateway profile (the user-stated policy: auto-assign to
    available workers, including default if it's the only one up)."""
    cards = [
        {
            "id": "t_orphan",
            "title": "task117 return crop15 fails parity",  # no [bracket prefix]
            "body": "Generic body without worker_owner hint.",
            "assignee": "",
            "status": "ready",
            "priority": 60,
            "created_at": 1,
        }
    ]
    report = plan_serialization(cards, running_gateways=["default"])
    assert len(report["actions"]) == 1
    action = report["actions"][0]
    assert action["op"] == "assign_orphan_to_running_gateway"
    assert action["task_id"] == "t_orphan"
    assert action["worker"] == "default"


def test_dependency_serializer_prefers_worker_profile_over_default():
    """When multiple gateways are running, prefer a worker profile over
    default — workers 1-6 are configured for ONNX-building, default is
    a generic chat agent. Only fall through to default when no workers
    are up."""
    cards = [
        {
            "id": "t_orphan",
            "title": "task100 misc work",
            "body": "x",
            "assignee": "",
            "status": "ready",
            "priority": 60,
            "created_at": 1,
        }
    ]
    report = plan_serialization(
        cards,
        running_gateways=["default", "neurogolf-worker-3"],
    )
    assert len(report["actions"]) == 1
    assert report["actions"][0]["worker"] == "neurogolf-worker-3"


def test_dependency_serializer_skips_orphan_fallback_when_no_gateways_up():
    """If no gateways are running (e.g. the probe failed), the orphan
    stays unassigned — the fallback is silent, not destructive."""
    cards = [
        {
            "id": "t_orphan",
            "title": "task100 misc work",
            "body": "x",
            "assignee": "",
            "status": "ready",
            "priority": 60,
            "created_at": 1,
        }
    ]
    report = plan_serialization(cards, running_gateways=[])
    assert report["actions"] == []


def test_dependency_serializer_orphan_fallback_does_not_double_up_lane():
    """If a worker is already reserved for another card in this run, the
    orphan picker doesn't dump a second card on the same worker. With
    only default available AND default already reserved by another action,
    fall back to no-action."""
    cards = [
        {
            # First card: worker-3 already has an active lane (running card)
            "id": "t_busy",
            "title": "[worker task050 neurogolf-worker-3 current-live] active",
            "body": "x",
            "assignee": "neurogolf-worker-3",
            "status": "running",
            "priority": 80,
            "created_at": 1,
        },
        {
            "id": "t_orphan",
            "title": "task100 misc",
            "body": "x",
            "assignee": "",
            "status": "ready",
            "priority": 60,
            "created_at": 2,
        },
    ]
    # worker-3 is the only worker profile up but it's already reserved
    report = plan_serialization(cards, running_gateways=["neurogolf-worker-3"])
    # Orphan can't claim worker-3 (already has a lane); no default available
    actions_for_orphan = [a for a in report["actions"] if a.get("task_id") == "t_orphan"]
    assert actions_for_orphan == []


def test_dependency_serializer_orphan_fallback_skips_structured_titles():
    """[worker ...] and [BACKLOG ...] and [repair ...] cards have their
    own routing in earlier branches; the orphan fallback must not catch
    them (would double-route)."""
    structured = [
        ("[worker task100 neurogolf-worker-1] x", "ready", ""),
        ("[BACKLOG single task] task200 cursor", "ready", ""),
        ("[review task300 cand_v1] x", "ready", ""),
        ("[submit task400 cand_v1] approved", "ready", ""),
        ("[repair task500] failed", "ready", ""),
    ]
    cards = [
        {
            "id": f"t_{i}",
            "title": title,
            "body": "x",
            "assignee": "",
            "status": status,
            "priority": 60,
            "created_at": i + 1,
        }
        for i, (title, status, _) in enumerate(structured)
    ]
    report = plan_serialization(cards, running_gateways=["default"])
    orphan_actions = [a for a in report["actions"] if a.get("op") == "assign_orphan_to_running_gateway"]
    assert orphan_actions == []


def test_dependency_serializer_prefers_running_worker_over_manual_pool_b():
    """When BOTH a running gateway worker AND a manual Pool B worker are
    in the pool, prefer the running one — it's immediately dispatchable;
    Pool B depends on the Claude Code /loop tick."""
    cards = [
        {
            "id": "t_orphan",
            "title": "task100 misc",
            "body": "x",
            "assignee": "",
            "status": "ready",
            "priority": 60,
            "created_at": 1,
        }
    ]
    # Pool order per available_profile_pool: running first, then manual
    pool = ["default", "neurogolf-worker-3", "neurogolf-worker-7", "neurogolf-worker-8"]
    report = plan_serialization(cards, running_gateways=pool)
    assert len(report["actions"]) == 1
    assert report["actions"][0]["worker"] == "neurogolf-worker-3"


def test_dependency_serializer_falls_back_to_manual_worker_before_default():
    """When NO running gateway worker is available but manual Pool B
    workers ARE in the pool, route to a manual worker BEFORE default."""
    cards = [
        {
            "id": "t_orphan",
            "title": "task100 misc",
            "body": "x",
            "assignee": "",
            "status": "ready",
            "priority": 60,
            "created_at": 1,
        }
    ]
    # Only default has a running gateway; manual Pool B is configured.
    pool = ["default", "neurogolf-worker-7", "neurogolf-worker-8"]
    report = plan_serialization(cards, running_gateways=pool)
    assert len(report["actions"]) == 1
    # worker-7 wins over default because it's a neurogolf-worker-* profile.
    assert report["actions"][0]["worker"] == "neurogolf-worker-7"


def test_dependency_serializer_only_picks_default_when_no_worker_anywhere():
    """If neither running nor manual pool has a worker-* profile, fall
    through to default as the true last resort."""
    cards = [
        {
            "id": "t_orphan",
            "title": "task100 misc",
            "body": "x",
            "assignee": "",
            "status": "ready",
            "priority": 60,
            "created_at": 1,
        }
    ]
    pool = ["default"]
    report = plan_serialization(cards, running_gateways=pool)
    assert len(report["actions"]) == 1
    assert report["actions"][0]["worker"] == "default"


def test_dependency_serializer_orphan_fallback_skips_blocked_cards():
    """The fallback only routes READY cards; blocked stays blocked
    (auto_triage handles those)."""
    cards = [
        {
            "id": "t_blocked",
            "title": "task100 random",
            "body": "x",
            "assignee": "",
            "status": "blocked",
            "priority": 60,
            "created_at": 1,
        }
    ]
    report = plan_serialization(cards, running_gateways=["default"])
    assert report["actions"] == []


# ---- Manual Pool B profile inclusion in the available pool --------------


def test_manual_claude_code_profiles_parses_profile_list_output(monkeypatch):
    """Parse the real shape of `hermes profile list` output. Lines for
    none-claude-code-manual profiles should yield the profile name."""
    from types import SimpleNamespace
    import tools.neurogolf_dependency_serializer as ds

    fake_output = (
        " Profile          Model                        Gateway      Alias        Distribution\n"
        " ───────────────    ───────────────────────────    ───────────    ───────────    ────────────────────\n"
        " ◆default         gpt-5.5                      running      —            —\n"
        "  neurogolf-worker-1 gpt-5.5                      stopped      —            —\n"
        "  neurogolf-worker-7 none-claude-code-manual      stopped      —            —\n"
        "  neurogolf-worker-8 none-claude-code-manual      stopped      neurogolf-worker-8 —\n"
    )
    monkeypatch.setattr(
        ds.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=fake_output, stderr=""),
    )
    result = ds.manual_claude_code_profiles()
    assert result == ["neurogolf-worker-7", "neurogolf-worker-8"]


def test_manual_claude_code_profiles_returns_empty_on_subprocess_fail(monkeypatch):
    from types import SimpleNamespace
    import tools.neurogolf_dependency_serializer as ds
    monkeypatch.setattr(
        ds.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="error"),
    )
    assert ds.manual_claude_code_profiles() == []


def test_available_profile_pool_combines_running_and_manual(monkeypatch):
    """Running gateways come FIRST, manual profiles appended (de-duped)."""
    import tools.neurogolf_dependency_serializer as ds
    monkeypatch.setattr(ds, "running_gateway_profiles", lambda: ["default", "neurogolf-worker-3"])
    monkeypatch.setattr(
        ds, "manual_claude_code_profiles",
        lambda: ["neurogolf-worker-7", "neurogolf-worker-8"],
    )
    pool = ds.available_profile_pool()
    assert pool == ["default", "neurogolf-worker-3", "neurogolf-worker-7", "neurogolf-worker-8"]


def test_available_profile_pool_dedupes_when_manual_profile_also_running(monkeypatch):
    """Edge: if a profile appears in BOTH running and manual lists (e.g.
    operator bound a gateway to a manual profile), keep it only once
    in the running position."""
    import tools.neurogolf_dependency_serializer as ds
    monkeypatch.setattr(
        ds, "running_gateway_profiles",
        lambda: ["default", "neurogolf-worker-7"],
    )
    monkeypatch.setattr(
        ds, "manual_claude_code_profiles",
        lambda: ["neurogolf-worker-7", "neurogolf-worker-8"],
    )
    pool = ds.available_profile_pool()
    # worker-7 appears once, before worker-8
    assert pool.count("neurogolf-worker-7") == 1
    assert pool.index("neurogolf-worker-7") < pool.index("neurogolf-worker-8")


def test_dependency_serializer_keeps_return_cards_out_of_specialist_lanes():
    cards = [
        {
            "id": "t_return",
            "title": "[optimization-return] task202 active-extent bypass is stale",
            "body": "Return from review packet. Decide whether to queue for submit-captain only after more worker evidence.",
            "assignee": "neurogolf-reviewer",
            "status": "todo",
            "priority": 68,
            "created_at": 1,
        }
    ]

    report = plan_serialization(cards)

    assert report["actions"] == [
        {
            "op": "unassign_misassigned_specialist_return",
            "worker": "neurogolf-reviewer",
            "target_worker": "",
            "task_id": "t_return",
            "status": "todo",
            "title": "[optimization-return] task202 active-extent bypass is stale",
        }
    ]


def test_dependency_serializer_reclaims_running_return_from_specialist_lane():
    cards = [
        {
            "id": "t_return",
            "title": "[optimization-return] task202 active-extent bypass is stale",
            "body": "Return from review packet. origin worker neurogolf-worker-1",
            "assignee": "neurogolf-reviewer",
            "status": "running",
            "priority": 68,
            "created_at": 1,
        }
    ]

    report = plan_serialization(cards)

    assert report["actions"][0]["op"] == "unassign_misassigned_specialist_return"
    assert report["actions"][0]["status"] == "running"
    assert report["actions"][0]["target_worker"] == "neurogolf-worker-1"


def test_dependency_serializer_does_not_promote_unassigned_return_card_to_submit_specialist():
    cards = [
        {
            "id": "t_return",
            "title": "[optimization-return] task071 improve local evidence",
            "body": "Return from review packet. Later submit-captain may see it if a stronger package is made.",
            "assignee": "",
            "status": "ready",
            "priority": 68,
            "created_at": 1,
        }
    ]

    report = plan_serialization(cards)

    assert report["actions"] == []
    assert report["unresolved"][0]["task_id"] == "t_return"
    assert "no deterministic target worker" in report["unresolved"][0]["reason"]


def test_dependency_serializer_waits_on_unassigned_specialist_when_lane_busy():
    cards = [
        {
            "id": "t_submit_running",
            "title": "submit-captain drain",
            "assignee": "neurogolf-submit-captain",
            "status": "running",
            "priority": 90,
            "created_at": 1,
        },
        {
            "id": "t_submit_next",
            "title": "[submit task070] approved candidate",
            "assignee": "",
            "status": "todo",
            "priority": 65,
            "created_at": 2,
        },
    ]

    report = plan_serialization(cards)

    assert report["actions"] == []


def test_review_packet_builder_batches_review_required_worker_blockers():
    cards = [
        {
            "id": "t_src",
            "title": "task201 local delta",
            "body": "review-required candidate produced local delta package_sha=ABC",
            "assignee": "neurogolf-worker-2",
            "status": "blocked",
            "priority": 80,
            "created_at": 1,
        }
    ]

    assert card_needs_review(cards[0]) is True
    report = plan_review_packet(cards)

    assert report["actions"][0]["op"] == "create_reviewer_packet"
    assert report["actions"][0]["source_cards"] == ["t_src"]
    assert "Do not submit to Kaggle" in report["actions"][0]["body"]


def test_review_packet_builder_uses_second_reviewer_when_primary_busy():
    cards = [
        {
            "id": "t_reviewer",
            "title": "active review",
            "assignee": "neurogolf-reviewer",
            "status": "running",
            "priority": 80,
            "created_at": 1,
        },
        {
            "id": "t_src",
            "title": "task201 local delta",
            "body": "review-required candidate produced",
            "assignee": "neurogolf-worker-2",
            "status": "blocked",
            "priority": 80,
            "created_at": 2,
        },
    ]

    report = plan_review_packet(cards)

    assert report["actions"][0]["op"] == "create_reviewer_packet"
    assert report["actions"][0]["assignee"] == "neurogolf-reviewer-2"
    assert report["reviewer_load"]["neurogolf-reviewer"] == ["t_reviewer"]


def test_review_packet_builder_waits_when_all_reviewer_lanes_busy():
    cards = [
        {
            "id": "t_reviewer_1",
            "title": "active review",
            "assignee": "neurogolf-reviewer",
            "status": "running",
            "priority": 80,
            "created_at": 1,
        },
        {
            "id": "t_reviewer_2",
            "title": "active review 2",
            "assignee": "neurogolf-reviewer-2",
            "status": "running",
            "priority": 80,
            "created_at": 2,
        },
        {
            "id": "t_src",
            "title": "task201 local delta",
            "body": "review-required candidate produced",
            "assignee": "neurogolf-worker-2",
            "status": "blocked",
            "priority": 80,
            "created_at": 3,
        },
    ]

    report = plan_review_packet(cards)

    assert report["actions"] == []
    assert report["skip_reason"] == "all reviewer lanes busy"


def test_blocked_cleanup_completes_obvious_superseded_and_delegates_review():
    cards = [
        {
            "id": "t_old",
            "title": "old blocker",
            "body": "superseded by t_new",
            "assignee": "neurogolf-worker-1",
            "status": "blocked",
            "priority": 20,
            "created_at": 1,
        },
        {
            "id": "t_review",
            "title": "task330 candidate",
            "body": "review-required candidate produced local delta",
            "assignee": "neurogolf-worker-6",
            "status": "blocked",
            "priority": 80,
            "created_at": 2,
        },
    ]

    assert classify_blocked_card(cards[0])["class"] == "complete"
    report = plan_blocked_cleanup(cards, best_synced=True)

    ops = [action["op"] for action in report["actions"]]
    assert ops == ["complete", "create_reviewer_packet"]
    assert report["manual"] == []


def test_blocked_cleanup_uses_second_reviewer_when_primary_active():
    cards = [
        {
            "id": "t_reviewer",
            "title": "active review",
            "assignee": "neurogolf-reviewer",
            "status": "running",
            "priority": 90,
            "created_at": 1,
        },
        {
            "id": "t_review",
            "title": "task330 candidate",
            "body": "review-required candidate produced local delta",
            "assignee": "neurogolf-worker-6",
            "status": "blocked",
            "priority": 80,
            "created_at": 2,
        },
    ]

    report = plan_blocked_cleanup(cards, best_synced=True)

    assert report["actions"][0]["op"] == "create_reviewer_packet"
    assert report["actions"][0]["assignee"] == "neurogolf-reviewer-2"
    assert report["reviewer_active"] == ["t_reviewer"]
    assert report["waiting_review_sources"] == []


def test_blocked_cleanup_releases_parent_gated_reviewer_child():
    cards = [
        {
            "id": "t_source",
            "title": "task382 candidate",
            "body": "review-required candidate produced local delta",
            "assignee": "neurogolf-worker-4",
            "status": "blocked",
            "priority": 80,
            "created_at": 1,
            "children": ["t_reviewer"],
        },
        {
            "id": "t_reviewer",
            "title": "[review task382] candidate",
            "assignee": "neurogolf-reviewer",
            "status": "todo",
            "priority": 85,
            "created_at": 2,
        },
    ]

    report = plan_blocked_cleanup(cards, best_synced=True)

    assert report["actions"] == [
        {
            "op": "release_parent_gated_reviewer",
            "task_id": "t_source",
            "reviewer_task_id": "t_reviewer",
            "title": "task382 candidate",
            "reason": "reviewer child is todo but parent-gated behind the blocked source it must review",
        }
    ]


def test_blocked_cleanup_completes_terminal_no_submit_worker_handoff_with_reviewer_active():
    cards = [
        {
            "id": "t_reviewer",
            "title": "active review",
            "assignee": "neurogolf-reviewer",
            "status": "running",
            "priority": 90,
            "created_at": 1,
        },
        {
            "id": "t_negative",
            "title": "task030 sweep",
            "body": (
                "review-required handoff: NO_SUBMIT_CANDIDATE; "
                "best_replacement_cost: null; no positive cost delta"
            ),
            "assignee": "neurogolf-worker-8",
            "status": "blocked",
            "priority": 80,
            "created_at": 2,
        },
    ]

    report = plan_blocked_cleanup(cards, best_synced=True)

    assert report["actions"] == [
        {
            "op": "complete",
            "task_id": "t_negative",
            "title": "task030 sweep",
            "reason": "worker handoff is terminal no-submit/no-positive-delta; no reviewer packet needed",
        }
    ]
    assert report["waiting_review_sources"] == []


def test_blocked_cleanup_completes_reviewer_base_sha_blocker_when_registry_matches(monkeypatch):
    monkeypatch.setattr(
        "tools.neurogolf_blocked_mechanical_cleanup.summarize_manifest",
        lambda: {"package_sha256": "ABC123"},
    )
    monkeypatch.setattr(
        "tools.neurogolf_blocked_mechanical_cleanup.read_json",
        lambda _path, _fallback: {"entry": {"base_package_sha256": "ABC123"}},
    )
    card = {
        "id": "t_reviewer",
        "title": "[review packet] blocked local-delta candidates",
        "body": "candidate_registry/submit_queue updated; needs eyes on the base-SHA anchoring note. verified_base_package_sha256 ABC123",
        "assignee": "neurogolf-reviewer",
        "status": "blocked",
        "priority": 85,
        "created_at": 1,
    }

    report = plan_blocked_cleanup([card], best_synced=True)

    assert report["actions"][0]["op"] == "complete"
    assert "base-SHA blocker resolved" in report["actions"][0]["reason"]


def test_blocked_cleanup_closes_completed_reviewer_handoff_and_opens_next_packet():
    cards = [
        {
            "id": "t_reviewer",
            "title": "[review packet] blocked local-delta candidates",
            "body": "PYTHON REVIEW PACKET",
            "latest_summary": (
                "review-required handoff: approved_candidates queued into submit_queue; "
                "return_tasks_created ['t_return']; changed_files ['submit_queue.json']"
            ),
            "assignee": "neurogolf-reviewer",
            "status": "blocked",
            "priority": 85,
            "created_at": 1,
        },
        {
            "id": "t_new_source",
            "title": "task300 candidate",
            "body": "review-required candidate produced local delta",
            "assignee": "neurogolf-worker-6",
            "status": "blocked",
            "priority": 80,
            "created_at": 2,
        },
    ]

    report = plan_blocked_cleanup(cards, best_synced=True)

    ops = [action["op"] for action in report["actions"]]
    assert ops == ["complete", "create_reviewer_packet"]
    assert report["actions"][0]["task_id"] == "t_reviewer"
    assert report["actions"][1]["source_cards"] == ["t_new_source"]


def test_submit_trigger_respects_post_refresh_hold_and_safety():
    snapshot = {
        "submit_queue": {"submit_ready_count": 2, "submit_ready_key": "ng-submit-captain-abc", "submit_ready_preview": []},
        "submit_mode": {"mode": "post-refresh-awaiting-human-cap"},
        "queue_audit": {"ok": True},
        "current_best": {"score": 5736.01, "candidate_id": "best", "manifest_sha_matches_package": True},
        "board": {"specialist_active": {"neurogolf-submit-captain": []}},
    }

    report = plan_submit_trigger(snapshot)

    assert report["actions"] == []
    assert "post-refresh auto-submit disabled or hourly cap unset" in report["skip_reasons"]


def test_submit_trigger_allows_post_refresh_from_runtime_config():
    snapshot = {
        "submit_queue": {
            "submit_ready_count": 2,
            "submit_ready_key": "ng-submit-captain-abc",
            "submit_ready_preview": [],
            "known_public_best": {"is_ahead_of_synced": False},
        },
        "submit_mode": {"mode": "post-refresh-awaiting-human-cap"},
        "queue_audit": {"ok": True},
        "current_best": {"score": 5736.01, "candidate_id": "best", "manifest_sha_matches_package": True},
        "board": {"specialist_active": {"neurogolf-submit-captain": []}},
    }

    report = plan_submit_trigger(
        snapshot,
        submit_config={"post_refresh_auto_submit_enabled": True, "max_submits_per_hour": 6},
    )

    assert report["actions"][0]["op"] == "create_submit_captain"
    assert report["submit_config"]["max_submits_per_hour"] == 6
    assert "Do not sleep or hold this card" in report["actions"][0]["body"]


def test_submit_trigger_enforces_post_refresh_hourly_cap():
    snapshot = {
        "submit_queue": {
            "submit_ready_count": 2,
            "submit_ready_key": "ng-submit-captain-abc",
            "submit_ready_preview": [],
            "known_public_best": {"is_ahead_of_synced": False},
        },
        "submit_mode": {"mode": "post-refresh-awaiting-human-cap"},
        "queue_audit": {"ok": True},
        "current_best": {"score": 5736.01, "candidate_id": "best", "manifest_sha_matches_package": True},
        "board": {"specialist_active": {"neurogolf-submit-captain": []}},
    }

    report = plan_submit_trigger(
        snapshot,
        submit_config={"post_refresh_auto_submit_enabled": True, "max_submits_per_hour": 6},
        credit_state={
            "policy": {"daily_submit_cap": 100},
            "credits": {"available": 5, "daily_used": 3, "last_submit_at": "2026-05-17T01:00:00Z"},
        },
        now=datetime(2026, 5, 17, 1, 5, tzinfo=timezone.utc),
    )

    assert report["actions"] == []
    assert any(reason.startswith("post-refresh rate limit waiting") for reason in report["skip_reasons"])


def test_submit_trigger_creates_when_ready_and_allowed():
    snapshot = {
        "submit_queue": {
            "submit_ready_count": 2,
            "submit_ready_key": "ng-submit-captain-abc",
            "submit_ready_preview": [{"task_id": "task001", "candidate_id": "cand1", "status": "ready"}],
            "known_public_best": {"is_ahead_of_synced": False},
        },
        "submit_mode": {"mode": "pre-refresh-burst"},
        "queue_audit": {"ok": True},
        "current_best": {"score": 5736.01, "candidate_id": "best", "manifest_sha_matches_package": True},
        "board": {"specialist_active": {"neurogolf-submit-captain": []}},
    }

    report = plan_submit_trigger(snapshot)

    assert report["actions"][0]["op"] == "create_submit_captain"
    # Idempotency key now includes a 15-minute freshness bucket so cron retries
    # against an unchanged ready-queue can still produce a fresh card after the
    # previous submit-captain card completes. Bug: prior to the bucket, Hermes'
    # idempotency dedup matched done cards and stalled the submit pipeline.
    assert report["actions"][0]["idempotency_key"].startswith("ng-submit-captain-abc-")
    assert "Do not sleep or hold this card" in report["actions"][0]["body"]


def test_submit_trigger_closes_submit_captain_waiting_on_rate_gate():
    detail = {
        "task": {
            "id": "t_submit",
            "assignee": "neurogolf-submit-captain",
            "status": "running",
            "title": "submit-captain drain batch",
        },
        "events": [
            {
                "kind": "heartbeat",
                "payload": {
                    "note": "First 3-stack submitted and synced; waiting for 10-minute spacing before draining remaining candidate."
                },
            }
        ],
        "runs": [{"worker_pid": 1234}],
    }

    action = plan_waiting_submit_cleanup(detail)

    assert action["op"] == "complete_waiting_submit_captain"
    assert action["task_id"] == "t_submit"
    assert action["worker_pid"] == 1234


def test_submit_trigger_closes_silent_stale_submit_captain():
    detail = {
        "task": {
            "id": "t_submit",
            "assignee": "neurogolf-submit-captain",
            "status": "running",
            "title": "submit-captain drain batch",
            "started_at": 100,
        },
        "events": [{"kind": "spawned", "created_at": 100, "payload": {"pid": 1234}}],
        "runs": [{"worker_pid": 1234}],
    }

    assert plan_waiting_submit_cleanup(detail, now=100 + (24 * 60), stale_after_seconds=25 * 60) is None
    action = plan_waiting_submit_cleanup(detail, now=100 + (25 * 60), stale_after_seconds=25 * 60)

    assert action["op"] == "complete_waiting_submit_captain"
    assert "silent" in action["reason"]


def test_submit_trigger_closes_submit_captain_after_synced_public_win():
    detail = {
        "task": {
            "id": "t_submit",
            "assignee": "neurogolf-submit-captain",
            "status": "running",
            "title": "[submit] task069 attempt14 restore final 30x30 pad",
            "body": "Queue only task069 reviewed local package.",
            "started_at": 1779027756,
        },
        "events": [],
        "runs": [{"worker_pid": 1234}],
    }
    snapshot = {
        "current_best": {
            "changed_entries": ["task069.onnx"],
            "last_winning_submit": {"synced_at": "2026-05-17T14:28:00Z"},
        }
    }

    action = plan_waiting_submit_cleanup(detail, snapshot=snapshot)

    assert action["op"] == "complete_waiting_submit_captain"
    assert "deterministic best sync promoted" in action["reason"]


def test_sync_current_best_ignores_legacy_mirrors_by_default():
    args = Namespace(
        orchestrator_current_best="C:\\legacy\\current_best\\submission.zip",
        main_local_dir="C:\\legacy",
        sync_legacy_mirrors=False,
    )

    assert effective_orchestrator_current_best(args) is None
    assert effective_main_local_dir(args) is None


def test_backlog_candidate_defer_policy_keeps_queue_and_canonical_out_of_worker_bank():
    assert candidate_defer_reason(
        {
            "task": "task147",
            "status": "queued_candidate_member",
            "next_action": "Review candidate queue before changing this task.",
        }
    )
    assert candidate_defer_reason(
        {
            "task": "task146",
            "status": "in_canonical_best",
            "next_action": "Keep current task146 v2 exact payload.",
        }
    )
    assert candidate_defer_reason(
        {
            "task": "task167",
            "current_cost": 817,
            "status": "no_current_replacement_record",
            "next_action": "Leave unchanged unless new Alpha/Beta work targets this task.",
        }
    )
    assert candidate_defer_reason(
        {
            "task": "task382",
            "current_cost": 330517,
            "status": "big-delta",
            "next_action": "Exclude from current score ladders; do not work later unless a new source or organic proof appears.",
        }
    )
    assert candidate_defer_reason(
        {
            "task": "task306",
            "current_cost": 192047,
            "status": "validated_pending_integration",
            "next_action": "Integrate task306_v3 onto current task177_v8 canonical package in a stack/rebase pass.",
        }
    )
    assert candidate_defer_reason(
        {
            "task": "task380",
            "current_cost": 1089,
            "status": "avoid_or_retest",
            "next_action": "Retest only as an isolated hidden-safe submission if a new source appears.",
        }
    )
    assert not candidate_defer_reason(
        {
            "task": "task170",
            "current_cost": 82130,
            "status": "big-delta",
            "next_action": "Leave unchanged unless new Alpha/Beta work targets this task.",
        }
    )


def test_backlog_candidate_defer_policy_skips_no_action_current_best_rows():
    assert candidate_defer_reason(
        {
            "task": "task070",
            "current_cost": 90178,
            "status": "landed_current_best",
            "next_action": "No action; current canonical payload is task070_cumsum_min17_bbox_recolor.",
        }
    )
    assert candidate_defer_reason(
        {
            "task": "task009",
            "current_cost": 115441,
            "status": "organic_strict_clean_hidden_bad",
            "next_action": "Do not stack task009 unless fresh isolated evidence overturns this exact result.",
        }
    )


def test_backlog_soft_defer_can_be_loosened_for_idle_workers():
    candidate = {
        "task": "task370",
        "current_cost": 261275,
        "current_points": 12.52,
        "status": "big-delta",
        "next_action": "Keep organic 20crop task370 in lime10 base; do not use old external task370 payloads.",
    }

    assert candidate_defer_reason(candidate)
    assert not candidate_defer_reason(candidate, allow_soft=True)


def test_blocked_cleanup_recovers_reviewer_iteration_budget_payload(tmp_path):
    package = tmp_path / "submission.zip"
    package.write_bytes(b"zip")
    cards = [
        {
            "id": "t_review",
            "title": "[review packet] blocked local-delta candidates abc",
            "body": "PYTHON REVIEW PACKET",
            "assignee": "neurogolf-reviewer",
            "status": "blocked",
            "priority": 85,
            "latest_summary": "\n".join(
                [
                    "Iteration budget exhausted (60/60)",
                    "Bottom line",
                    "- Candidate task000_v3_minparams appears valid and queue-worthy.",
                    "Recommended queue payload fields",
                    "- candidate_id: task000_v3_minparams_on_5aa1390a_20260517T155146Z",
                    "- task_id: task000",
                    "- worker_owner: neurogolf-worker-2",
                    "- origin_kanban_task_id: t_9e393d31",
                    "- lineage_base_candidate_id: current-best",
                    "- verified_base_package_sha256: 5AA1390",
                    "- current_best_manifest_package_sha256: CA3C88",
                    f"- package_path: {package}",
                    "- package_sha256: B0193927BA656A62185A1B584AAA8420133AE91A2751B17B698E649FDDCB8A64",
                    "- changed_task_ids: [\"task000\"]",
                    "- changed_entries: [\"task000.onnx\"]",
                    "- expected_point_gain: 0.011741004529659094",
                    "- reviewer_disposition: approved-for-submit-captain",
                    "- risk_flags: [\"base-manifest-sha-mismatch-noted\"]",
                ]
            ),
        }
    ]

    report = plan_blocked_cleanup(cards, best_synced=True)

    assert report["actions"][0]["op"] == "recover_reviewer_queue_payload"
    candidate = report["actions"][0]["candidate"]
    assert candidate["candidate_id"] == "task000_v3_minparams_on_5aa1390a_20260517T155146Z"
    assert candidate["submit_ready"] is True
    assert candidate["approval_packet"]["preflight_history"]["status"] == "reviewer-exception"


def test_backlog_priority_prefers_hard_high_cost_over_tiny_high_points():
    hard = {"task": "task153", "current_cost": 215987, "current_points": 12.7, "status": "big-delta", "next_action": "try shrink"}
    tiny = {"task": "task164", "current_cost": 30, "current_points": 21.6, "status": "", "next_action": "try rewrite"}

    assert priority_for_candidate(hard) > priority_for_candidate(tiny)


def test_backlog_autoforeman_converts_existing_bank_to_worker_card(monkeypatch):
    # POOL_A_SHRINK_WORKERS is empty in live policy (2026-05-18). Patch it
    # to include worker-1 so this test exercises the bank-to-worker
    # conversion logic regardless of live policy.
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.POOL_A_SHRINK_WORKERS",
        ("neurogolf-worker-1",),
    )
    cards = [
        {
            "id": "t_bank",
            "title": "[BACKLOG single task] task119 cursor backlog current-live",
            "body": "UNASSIGNED BACKLOG CARD - do not dispatch directly",
            "assignee": "",
            "status": "todo",
            "priority": 62,
            "created_at": 1,
        }
    ]

    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [],
            "next_task_id": "task119",
            "pass_count": 0,
            "window_end_task_id": "task119",
        },
    )

    report = plan_backlog(cards, docs={}, target_bank=1, create_limit=2, assign_limit=1)

    assert report["actions"][0]["op"] == "create_worker_from_bank"
    assert report["actions"][0]["task_id"] == "task119"
    assert report["actions"][0]["assignee"] == "neurogolf-worker-1"
    assert "UNASSIGNED BACKLOG CARD" not in report["actions"][0]["body"]


def test_backlog_autoforeman_reserves_idle_worker_for_return_card(monkeypatch):
    # Patch POOL_A_SHRINK_WORKERS to include workers 1+2 so the test can
    # exercise the "reserve worker-1 for the return, give worker-2 the
    # cursor task" path. Live policy has Pool A empty (2026-05-18); this
    # test verifies the reservation logic itself.
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.POOL_A_SHRINK_WORKERS",
        ("neurogolf-worker-1", "neurogolf-worker-2"),
    )
    cards = [
        {
            "id": "t_return",
            "title": "[repair] task300 returned for stronger evidence",
            "body": "Return to neurogolf-worker-1 when free.",
            "assignee": "",
            "status": "ready",
            "priority": 85,
            "created_at": 1,
        }
    ]
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [
                {"task": "task119", "current_cost": 80000, "current_points": 12.0, "status": "", "next_action": "try rewrite"}
            ],
            "next_task_id": "task119",
            "pass_count": 0,
            "window_end_task_id": "task119",
        },
    )

    report = plan_backlog(cards, docs={}, target_bank=1, create_limit=1, assign_limit=1)

    assert report["return_reserved_workers"] == ["neurogolf-worker-1"]
    # With Pool A = (worker-1, worker-2), worker-1 is reserved for the
    # return card → worker-2 picks up the cursor task as create_worker_card.
    assert report["actions"][0]["op"] == "create_worker_card"
    assert report["actions"][0]["assignee"] == "neurogolf-worker-2"


def test_backlog_autoforeman_closes_deferred_bank_before_assignment(monkeypatch):
    cards = [
        {
            "id": "t_bank",
            "title": "[BACKLOG single task] task147 cursor backlog current-live",
            "body": "UNASSIGNED BACKLOG CARD\ntask_id: task147\nstatus: queued_candidate_member\nnext_action: Review candidate queue before changing this task.",
            "assignee": "",
            "status": "todo",
            "priority": 62,
            "created_at": 1,
        }
    ]

    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [],
            "next_task_id": "task147",
            "pass_count": 0,
            "window_end_task_id": "task147",
        },
    )

    report = plan_backlog(cards, docs={}, target_bank=1, create_limit=2, assign_limit=1)

    assert report["actions"][0]["op"] == "complete_deferred_bank_card"
    assert report["actions"][0]["task_id"] == "task147"


def test_backlog_autoforeman_creates_bank_in_numeric_order(monkeypatch):
    cards = []
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [
                {"task": "task119", "current_cost": 80000, "current_points": 12.0, "status": "", "next_action": "try rewrite"},
                # Numeric-ordering test: task120 needs to clear the Pool A
                # cost floor to be a valid backlog target. The test exercises
                # ORDER (task119 then task120), not the floor policy.
                {"task": "task120", "current_cost": 90000, "current_points": 5.0, "status": "", "next_action": ""},
            ],
            "next_task_id": "task119",
            "pass_count": 0,
            "window_end_task_id": "task120",
        },
    )

    report = plan_backlog(cards, docs={}, target_bank=2, create_limit=2, assign_limit=0)

    assert [action["task_id"] for action in report["actions"]] == ["task119", "task120"]
    assert [action["op"] for action in report["actions"]] == ["create_bank_card", "create_bank_card"]
    assert report["cursor_advance_from"] == "task120"
    assert report["actions"][0]["idempotency_key"] == "ng-backlog-task119-pass0-fromtask119"


def test_backlog_autoforeman_assigns_soft_deferred_rows_when_idle(monkeypatch):
    # POOL_A_SHRINK_WORKERS is empty in live policy (2026-05-18). Patch it
    # to include worker-1 so this test exercises the soft-defer assignment
    # path regardless of live policy.
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.POOL_A_SHRINK_WORKERS",
        ("neurogolf-worker-1",),
    )
    cards = []
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [
                {
                    "task": "task370",
                    "current_cost": 261275,
                    "current_points": 12.52,
                    "status": "big-delta",
                    "next_action": "Keep organic 20crop task370 in lime10 base; do not use old external task370 payloads.",
                }
            ],
            "next_task_id": "task370",
            "pass_count": 0,
            "window_end_task_id": "task370",
        },
    )

    report = plan_backlog(cards, docs={}, target_bank=0, create_limit=1, assign_limit=1)

    assert report["actions"][0]["op"] == "create_worker_card"
    assert report["actions"][0]["task_id"] == "task370"
    assert "Loosened backlog note" in report["actions"][0]["body"]


def test_backlog_autoforeman_worker_idempotency_key_is_cursor_scoped(monkeypatch):
    # POOL_A_SHRINK_WORKERS is empty in live policy. Patch to test
    # idempotency-key scoping under the worker-assignment path.
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.POOL_A_SHRINK_WORKERS",
        ("neurogolf-worker-1",),
    )
    cards = []
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [
                {"task": "task228", "current_cost": 90000, "current_points": 14.374, "status": "", "next_action": "try shrink"}
            ],
            "next_task_id": "task225",
            "pass_count": 0,
            "window_end_task_id": "task228",
        },
    )

    report = plan_backlog(cards, docs={}, target_bank=0, create_limit=1, assign_limit=1)

    assert report["actions"][0]["op"] == "create_worker_card"
    assert report["actions"][0]["idempotency_key"] == "ng-worker-neurogolf-worker-1-task228-pass0-fromtask225"


def test_backlog_autoforeman_skips_recently_completed_task(monkeypatch):
    cards = [
        {
            "id": "t_done",
            "title": "[worker task228 neurogolf-worker-7 current-live] cursor backlog task",
            "body": "task_id: task228",
            "assignee": "neurogolf-worker-7",
            "status": "done",
            "completed_at": 1_000_000,
        }
    ]
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [
                {"task": "task228", "current_cost": 90000, "current_points": 14.374, "status": "", "next_action": "try shrink"}
            ],
            "next_task_id": "task228",
            "pass_count": 0,
            "window_end_task_id": "task228",
        },
    )
    monkeypatch.setattr("tools.neurogolf_backlog_autoforeman.time.time", lambda: 1_000_100)

    report = plan_backlog(cards, docs={}, target_bank=0, create_limit=1, assign_limit=1)

    assert report["actions"] == []
    assert report["skipped"][0]["reason"] == "defer recently completed worker sweep until task_state/docs catch up"


def test_backlog_apply_ignores_inactive_idempotency_result(monkeypatch):
    from tools import neurogolf_backlog_autoforeman as backlog

    advances = []
    monkeypatch.setattr(backlog, "advance_cursor", lambda task_id, reason: advances.append((task_id, reason)))

    class FakeClient:
        dry_run = False

        def __init__(self):
            self.actions = []

        def create(self, title, **kwargs):
            self.actions.append({"op": "create", "title": title, **kwargs, "task_id": "t_old"})
            return "t_old"

        def show(self, task_id):
            return {"task": {"id": task_id, "status": "done", "assignee": "neurogolf-worker-7"}}

        def comment(self, *_args, **_kwargs):
            raise AssertionError("inactive worker result must not consume the source bank card")

        def complete(self, *_args, **_kwargs):
            raise AssertionError("inactive worker result must not consume the source bank card")

    client = FakeClient()
    plan = {
        "actions": [
            {
                "op": "create_worker_from_bank",
                "source_backlog_id": "t_bank",
                "task_id": "task228",
                "assignee": "neurogolf-worker-7",
                "title": "[worker task228 neurogolf-worker-7 current-live] cursor backlog task",
                "body": "task_id: task228",
                "priority": 49,
                "idempotency_key": "ng-worker-neurogolf-worker-7-task228-pass0",
            }
        ],
        "skipped": [],
        "skipped_count": 0,
        "cursor_advance_from": "task228",
    }

    backlog.apply_backlog_plan(client, plan)

    assert any(action["op"] == "stale_create_result" for action in client.actions)
    assert advances == []


def test_backlog_autoforeman_does_not_advance_cursor_when_bank_is_full(monkeypatch):
    cards = [
        {
            "id": f"t_bank_{index}",
            "title": f"[BACKLOG single task] task{index:03d} cursor backlog current-live",
            "body": f"UNASSIGNED BACKLOG CARD\ntask_id: task{index:03d}\nstatus:\nnext_action: try shrink",
            "assignee": "",
            "status": "todo",
            "priority": 50,
            "created_at": index,
        }
        for index in range(24)
    ]
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [
                {"task": "task146", "current_cost": 80000, "current_points": 12.0, "status": "", "next_action": "try rewrite"},
                {"task": "task147", "current_cost": 90000, "current_points": 12.0, "status": "", "next_action": "try rewrite"},
            ],
            "next_task_id": "task146",
            "pass_count": 0,
            "window_end_task_id": "task147",
        },
    )

    report = plan_backlog(cards, docs={}, target_bank=24, create_limit=4, assign_limit=0)

    assert report["actions"] == []
    assert report["cursor_advance_from"] == ""


def test_backlog_autoforeman_advances_only_to_last_created_when_bank_fills(monkeypatch):
    cards = [
        {
            "id": f"t_bank_{index}",
            "title": f"[BACKLOG single task] task{index:03d} cursor backlog current-live",
            "body": f"UNASSIGNED BACKLOG CARD\ntask_id: task{index:03d}\nstatus:\nnext_action: try shrink",
            "assignee": "",
            "status": "todo",
            "priority": 50,
            "created_at": index,
        }
        for index in range(23)
    ]
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [
                {"task": "task146", "current_cost": 80000, "current_points": 12.0, "status": "", "next_action": "try rewrite"},
                {"task": "task147", "current_cost": 90000, "current_points": 12.0, "status": "", "next_action": "try rewrite"},
            ],
            "next_task_id": "task146",
            "pass_count": 0,
            "window_end_task_id": "task147",
        },
    )

    report = plan_backlog(cards, docs={}, target_bank=24, create_limit=4, assign_limit=0)

    assert [action["task_id"] for action in report["actions"]] == ["task146"]
    assert report["cursor_advance_from"] == "task146"


# ----- submit-gate redesign: pass-aware retry counter ----------------------


def _outcome_doc_with_attempts(task_id, attempts, families, last_pass, summary=""):
    """Helper: build a task_outcome_doc snippet with the new submit_gate fields."""
    entry = {
        "attempts_this_pass": attempts,
        "attempted_families": families,
        "last_attempt_pass": last_pass,
    }
    if summary:
        entry["last_attempt_summary"] = summary
    return {"tasks": {task_id: entry}}


class TestTaskAttemptsExhaustedThisPass:
    def test_returns_empty_when_no_doc(self):
        assert task_attempts_exhausted_this_pass("task100", None, current_pass=1) == ""

    def test_returns_empty_when_task_missing(self):
        doc = {"tasks": {}}
        assert task_attempts_exhausted_this_pass("task100", doc, current_pass=1) == ""

    def test_returns_empty_when_attempts_below_threshold(self):
        doc = _outcome_doc_with_attempts("task100", attempts=2, families=["a", "b"], last_pass=1)
        assert task_attempts_exhausted_this_pass("task100", doc, current_pass=1) == ""

    def test_returns_defer_reason_when_attempts_at_threshold(self):
        doc = _outcome_doc_with_attempts(
            "task100", attempts=3, families=["a", "b", "c"], last_pass=1,
            summary="tested_non_improving_this_pass - all 3 below floor",
        )
        result = task_attempts_exhausted_this_pass("task100", doc, current_pass=1)
        assert result != ""
        assert "attempts_this_pass=3" in result
        assert "tested_non_improving_this_pass" in result

    def test_returns_empty_when_last_pass_differs_from_current(self):
        """Counter only applies in its own pass; when pass_count advances the
        task becomes eligible again."""
        doc = _outcome_doc_with_attempts("task100", attempts=3, families=["a", "b", "c"], last_pass=1)
        assert task_attempts_exhausted_this_pass("task100", doc, current_pass=2) == ""

    def test_tolerates_string_counter_values(self):
        doc = {"tasks": {"task100": {"attempts_this_pass": "3", "last_attempt_pass": "1"}}}
        result = task_attempts_exhausted_this_pass("task100", doc, current_pass=1)
        assert "attempts_this_pass=3" in result

    def test_custom_max_attempts(self):
        doc = _outcome_doc_with_attempts("task100", attempts=4, families=["a"]*4, last_pass=1)
        assert task_attempts_exhausted_this_pass("task100", doc, current_pass=1, max_attempts=5) == ""
        result = task_attempts_exhausted_this_pass("task100", doc, current_pass=1, max_attempts=4)
        assert "attempts_this_pass=4" in result


class TestNextRetryAttemptIndex:
    def test_returns_one_when_no_doc(self):
        assert next_retry_attempt_index("task100", None, current_pass=1) == 1

    def test_returns_one_when_task_missing(self):
        assert next_retry_attempt_index("task100", {"tasks": {}}, current_pass=1) == 1

    def test_returns_one_when_pass_advanced(self):
        doc = _outcome_doc_with_attempts("task100", attempts=2, families=["a", "b"], last_pass=1)
        assert next_retry_attempt_index("task100", doc, current_pass=2) == 1

    def test_increments_from_recorded_count(self):
        doc = _outcome_doc_with_attempts("task100", attempts=2, families=["a", "b"], last_pass=5)
        assert next_retry_attempt_index("task100", doc, current_pass=5) == 3


def test_backlog_autoforeman_defers_task_when_attempts_exhausted_this_pass(monkeypatch):
    """If submit_gate already burned all 3 attempts this pass, the autoforeman
    must defer the task rather than dispatching another worker.

    Real-world failure mode this prevents: submit_gate gives up on a task
    after 3 failed attempts, but the autoforeman immediately re-dispatches it
    on the same pass, burning more cycles on something we just concluded
    cannot be improved this round.
    """
    cards = [
        {
            "id": "t_bank",
            "title": "[BACKLOG single task] task100 cursor backlog current-live",
            "body": "UNASSIGNED BACKLOG CARD\ntask_id: task100\nstatus:\nnext_action: try rewrite",
            "assignee": "",
            "status": "todo",
            "priority": 50,
            "created_at": 1,
        }
    ]
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [],
            "next_task_id": "task100",
            "pass_count": 5,
            "window_end_task_id": "task100",
        },
    )

    docs = {
        "task_outcome_doc": _outcome_doc_with_attempts(
            "task100", attempts=3, families=["where-select", "reduce-broadcast", "conv-stamp"],
            last_pass=5,
            summary="tested_non_improving_this_pass - 3 attempts all below floor",
        ),
    }

    report = plan_backlog(cards, docs=docs, target_bank=1, create_limit=2, assign_limit=1)

    deferred_actions = [a for a in report["actions"] if a["op"] == "complete_deferred_bank_card"]
    assert len(deferred_actions) == 1
    assert deferred_actions[0]["task_id"] == "task100"
    assert "attempts_this_pass=3" in deferred_actions[0]["reason"]


def test_backlog_autoforeman_does_not_defer_when_pass_advanced(monkeypatch):
    """Same task with 3 attempts recorded on pass 4 should NOT defer when
    we are now on pass 5 - the counter only applies to its own pass."""
    cards = [
        {
            "id": "t_bank",
            "title": "[BACKLOG single task] task100 cursor backlog current-live",
            "body": "UNASSIGNED BACKLOG CARD\ntask_id: task100\nstatus:\nnext_action: try rewrite",
            "assignee": "",
            "status": "todo",
            "priority": 50,
            "created_at": 1,
        }
    ]
    monkeypatch.setattr(
        "tools.neurogolf_backlog_autoforeman.summarize_task_progress",
        lambda _cards: {
            "numeric_preview": [],
            "next_task_id": "task100",
            "pass_count": 5,
            "window_end_task_id": "task100",
        },
    )

    docs = {
        "task_outcome_doc": _outcome_doc_with_attempts(
            "task100", attempts=3, families=["a", "b", "c"], last_pass=4,
        ),
    }

    report = plan_backlog(cards, docs=docs, target_bank=1, create_limit=2, assign_limit=1)

    deferred_with_attempts_reason = [
        a for a in report["actions"]
        if a.get("op") == "complete_deferred_bank_card"
        and "attempts_this_pass" in (a.get("reason") or "")
    ]
    assert deferred_with_attempts_reason == []


class TestBodyForTaskRetryFields:
    def test_includes_retry_attempt_index_when_provided(self):
        body = body_for_task(task_id="task100", retry_attempt_index=2)
        assert "retry_attempt_index: 2" in body

    def test_includes_attempted_families_when_provided(self):
        body = body_for_task(
            task_id="task100",
            retry_attempt_index=2,
            attempted_families=["where-select", "reduce-broadcast"],
        )
        assert "do_not_use_families:" in body
        assert "where-select" in body
        assert "reduce-broadcast" in body

    def test_omits_attempted_families_when_empty(self):
        body = body_for_task(task_id="task100", retry_attempt_index=1, attempted_families=[])
        assert "do_not_use_families" not in body

    def test_omits_retry_attempt_index_field_line_when_none(self):
        """The field-style line `retry_attempt_index: N` should be absent when
        not provided. The phrase still appears in the worker instructions, but
        not as a structured field row."""
        body = body_for_task(task_id="task100")
        assert "retry_attempt_index: " not in body
        # The instructions text mentions the field name for guidance — that's fine
        assert "Copy `retry_attempt_index`" in body

    def test_worker_instructions_mention_algorithm_family(self):
        body = body_for_task(task_id="task100")
        assert "algorithm_family" in body
