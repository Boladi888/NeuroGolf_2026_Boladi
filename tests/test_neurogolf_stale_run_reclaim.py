"""Unit tests for tools/neurogolf_stale_run_reclaim deterministic logic.

The Hermes-CLI integration (run_hermes, hermes_list_running, hermes_show_json,
apply_reassignment) is left to integration coverage. Here we test:
  - active-run detection
  - latest-alive-timestamp extraction from show --json shape
  - reassignment target routing for worker vs reviewer cards
  - load-balancing across /loop targets, including pending reassignments
  - plan_reassignments end-to-end with an injected show_fetcher
"""
from __future__ import annotations

import pytest

from tools.neurogolf_stale_run_reclaim import (
    AGENT_ALIVE_EVENT_KINDS,
    DEFAULT_STALE_SECONDS,
    DISPATCHER_PROFILES,
    LOOP_WORKER_TARGETS,
    REVIEWER_PEERS,
    find_active_run,
    latest_alive_timestamp,
    pick_loop_worker,
    plan_reassignments,
    reassignment_target,
)


# ----- find_active_run -----------------------------------------------------


def test_find_active_run_returns_run_without_ended_at():
    runs = [
        {"id": 1, "started_at": 100, "ended_at": 200},  # finished
        {"id": 2, "started_at": 300, "ended_at": None}, # active
    ]
    assert find_active_run(runs)["id"] == 2


def test_find_active_run_treats_ended_at_zero_as_in_flight():
    """Some Hermes versions emit 0 instead of null for unfinished runs."""
    runs = [{"id": 7, "started_at": 100, "ended_at": 0}]
    assert find_active_run(runs)["id"] == 7


def test_find_active_run_picks_latest_when_multiple_in_flight():
    """Defensive: shouldn't happen, but if two runs claim to be active take
    the one that started most recently."""
    runs = [
        {"id": 1, "started_at": 100, "ended_at": None},
        {"id": 2, "started_at": 500, "ended_at": None},
    ]
    assert find_active_run(runs)["id"] == 2


def test_find_active_run_returns_none_when_no_runs():
    assert find_active_run([]) is None
    assert find_active_run([{"id": 1, "ended_at": 50}]) is None
    assert find_active_run(None) is None


# ----- latest_alive_timestamp ---------------------------------------------


def _show(active_run_id: int, active_started_at: int, events: list[dict]) -> dict:
    return {
        "runs": [
            {"id": active_run_id, "started_at": active_started_at, "ended_at": None},
            {"id": active_run_id - 1, "started_at": active_started_at - 1000, "ended_at": active_started_at - 100},
        ],
        "events": events,
    }


def test_latest_alive_timestamp_picks_most_recent_agent_event_on_active_run():
    data = _show(2, 1000, [
        {"kind": "spawned", "run_id": 2, "created_at": 1000},
        {"kind": "heartbeat", "run_id": 2, "created_at": 1300},
        {"kind": "heartbeat", "run_id": 2, "created_at": 1700},
    ])
    assert latest_alive_timestamp(data) == 1700


def test_latest_alive_timestamp_ignores_events_from_finished_runs():
    """A heartbeat on a previous (finished) run must NOT count as recent
    activity for the active run — that would mask a silent active run."""
    data = _show(2, 1000, [
        {"kind": "heartbeat", "run_id": 1, "created_at": 9999},  # finished run!
        {"kind": "spawned", "run_id": 2, "created_at": 1000},
    ])
    assert latest_alive_timestamp(data) == 1000


def test_latest_alive_timestamp_falls_back_to_active_run_started_at():
    """When the active run has no alive-events yet (just spawned and silent),
    treat run.started_at as the floor."""
    data = _show(5, 1000, [
        {"kind": "completed", "run_id": 4, "created_at": 950},
        {"kind": "promoted", "run_id": 5, "created_at": 995},  # not an alive event
    ])
    assert latest_alive_timestamp(data) == 1000


def test_latest_alive_timestamp_returns_none_when_no_active_run():
    data = {
        "runs": [{"id": 1, "started_at": 100, "ended_at": 200}],
        "events": [],
    }
    assert latest_alive_timestamp(data) is None


def test_latest_alive_timestamp_handles_malformed_event_timestamps():
    data = _show(2, 1000, [
        {"kind": "heartbeat", "run_id": 2, "created_at": "not-a-number"},
        {"kind": "heartbeat", "run_id": 2, "created_at": None},
        {"kind": "heartbeat", "run_id": 2, "created_at": 1500},
    ])
    assert latest_alive_timestamp(data) == 1500


def test_latest_alive_timestamp_accepts_all_documented_alive_kinds():
    """Each kind in AGENT_ALIVE_EVENT_KINDS should count."""
    for kind in AGENT_ALIVE_EVENT_KINDS:
        data = _show(2, 1000, [{"kind": kind, "run_id": 2, "created_at": 1234}])
        assert latest_alive_timestamp(data) == 1234, f"kind {kind!r} should count as alive"


# ----- reassignment_target / pick_loop_worker -----------------------------


def test_reviewer_card_routes_to_peer_reviewer():
    card = {"id": "t_x", "assignee": "neurogolf-reviewer"}
    assert reassignment_target(card, []) == "neurogolf-reviewer-2"
    card2 = {"id": "t_y", "assignee": "neurogolf-reviewer-2"}
    assert reassignment_target(card2, []) == "neurogolf-reviewer"


def test_worker_card_routes_to_loop_worker_target():
    card = {"id": "t_x", "assignee": "neurogolf-worker-2"}
    target = reassignment_target(card, [])
    assert target in LOOP_WORKER_TARGETS


def test_pool_b_worker_card_is_not_reassigned():
    """A card already on a /loop worker isn't a dispatcher problem."""
    for pool_b in LOOP_WORKER_TARGETS:
        card = {"id": "t_x", "assignee": pool_b}
        assert reassignment_target(card, []) is None


def test_pick_loop_worker_balances_load():
    """Worker with fewer current cards wins."""
    running = [
        {"id": "t_a", "assignee": "neurogolf-worker-5"},
        {"id": "t_b", "assignee": "neurogolf-worker-5"},
        {"id": "t_c", "assignee": "neurogolf-worker-7"},
    ]
    # worker-8 has 0, worker-7 has 1, worker-5 has 2 → worker-8 wins.
    target = pick_loop_worker(running, targets=("neurogolf-worker-5", "neurogolf-worker-7", "neurogolf-worker-8"))
    assert target == "neurogolf-worker-8"


def test_pick_loop_worker_accounts_for_pending_reassignments():
    """Two stuck cards in one watchdog pass must not both go to the same
    least-loaded worker."""
    running = [{"id": "t_a", "assignee": "neurogolf-worker-5"}]
    targets = ("neurogolf-worker-5", "neurogolf-worker-7", "neurogolf-worker-8")
    first = pick_loop_worker(running, targets=targets)
    # Pending tracks the first reassignment landing on `first`.
    second = pick_loop_worker(running, pending_reassignments={"t_stuck1": first}, targets=targets)
    assert first != second, "second stuck card must go to a different /loop worker"


# ----- plan_reassignments (end-to-end with injected fetcher) -------------


def _fake_fetcher(by_card_id: dict[str, dict]):
    return lambda cid: by_card_id.get(cid)


def test_plan_emits_reassign_for_stale_worker_card():
    now = 10_000
    running = [
        {"id": "t_stuck", "assignee": "neurogolf-worker-2", "started_at": 1_000},
    ]
    fetcher = _fake_fetcher({
        "t_stuck": _show(
            7, 1_000,
            [{"kind": "heartbeat", "run_id": 7, "created_at": 1_500}],
        ),
    })
    actions = plan_reassignments(
        running, stale_seconds=DEFAULT_STALE_SECONDS, now_ts=now, show_fetcher=fetcher,
    )
    assert len(actions) == 1
    assert actions[0]["verdict"] == "reassign"
    assert actions[0]["target"] in LOOP_WORKER_TARGETS
    assert actions[0]["age_s"] == 8_500  # now 10000 - heartbeat 1500


def test_plan_skips_card_with_recent_heartbeat():
    now = 10_000
    running = [{"id": "t_healthy", "assignee": "neurogolf-worker-2", "started_at": 1_000}]
    fetcher = _fake_fetcher({
        "t_healthy": _show(
            7, 1_000,
            [{"kind": "heartbeat", "run_id": 7, "created_at": now - 30}],
        ),
    })
    actions = plan_reassignments(
        running, stale_seconds=DEFAULT_STALE_SECONDS, now_ts=now, show_fetcher=fetcher,
    )
    assert actions == []


def test_plan_skips_card_assigned_to_loop_worker_even_if_stale():
    """Pool B (loop) cards aren't reassigned by this watchdog regardless of
    age — the user's /loop session manages those itself."""
    now = 10_000
    running = [{"id": "t_loop", "assignee": "neurogolf-worker-7", "started_at": 1_000}]
    fetcher = _fake_fetcher({
        "t_loop": _show(
            9, 1_000, [{"kind": "heartbeat", "run_id": 9, "created_at": 1_000}],
        ),
    })
    actions = plan_reassignments(
        running, stale_seconds=DEFAULT_STALE_SECONDS, now_ts=now, show_fetcher=fetcher,
    )
    assert actions == []


def test_plan_routes_stale_reviewer_card_to_peer():
    now = 10_000
    running = [{"id": "t_review", "assignee": "neurogolf-reviewer", "started_at": 1_000}]
    fetcher = _fake_fetcher({
        "t_review": _show(
            3, 1_000,
            [{"kind": "spawned", "run_id": 3, "created_at": 1_000}],
        ),
    })
    actions = plan_reassignments(
        running, stale_seconds=DEFAULT_STALE_SECONDS, now_ts=now, show_fetcher=fetcher,
    )
    assert len(actions) == 1
    assert actions[0]["target"] == "neurogolf-reviewer-2"


def test_plan_balances_two_stuck_worker_cards_across_loop_workers():
    """Two stuck cards in a single pass must land on different /loop workers."""
    now = 10_000
    running = [
        {"id": "t_stuck_a", "assignee": "neurogolf-worker-2", "started_at": 1_000},
        {"id": "t_stuck_b", "assignee": "neurogolf-worker-3", "started_at": 1_000},
    ]
    fetcher = _fake_fetcher({
        "t_stuck_a": _show(7, 1_000, [{"kind": "spawned", "run_id": 7, "created_at": 1_000}]),
        "t_stuck_b": _show(8, 1_000, [{"kind": "spawned", "run_id": 8, "created_at": 1_000}]),
    })
    actions = plan_reassignments(
        running, stale_seconds=DEFAULT_STALE_SECONDS, now_ts=now, show_fetcher=fetcher,
    )
    reassigns = [a for a in actions if a["verdict"] == "reassign"]
    assert len(reassigns) == 2
    targets = {a["target"] for a in reassigns}
    assert len(targets) == 2, f"both stuck cards should land on different /loop workers, got {targets}"


def test_plan_skips_card_with_no_active_run():
    """If show --json reports no in-flight run, the card is between states
    and we leave it alone (the dispatcher will reclaim or fail it)."""
    now = 10_000
    running = [{"id": "t_between", "assignee": "neurogolf-worker-2", "started_at": 1_000}]
    fetcher = _fake_fetcher({
        "t_between": {
            "runs": [{"id": 4, "started_at": 100, "ended_at": 200}],
            "events": [],
        },
    })
    actions = plan_reassignments(
        running, stale_seconds=DEFAULT_STALE_SECONDS, now_ts=now, show_fetcher=fetcher,
    )
    assert actions == []


def test_plan_skips_card_when_show_fetch_returns_none():
    """Hermes show returning a non-card (network error, missing card) must
    not crash the watchdog or generate a phantom reassign."""
    now = 10_000
    running = [{"id": "t_missing", "assignee": "neurogolf-worker-2", "started_at": 1_000}]
    actions = plan_reassignments(
        running, stale_seconds=DEFAULT_STALE_SECONDS, now_ts=now,
        show_fetcher=lambda cid: None,
    )
    assert actions == []


def test_default_stale_threshold_is_8_minutes():
    assert DEFAULT_STALE_SECONDS == 8 * 60


def test_dispatcher_profiles_include_workers_1_through_6_and_both_reviewers():
    """Tripwire: if the user adds a pool A worker later (e.g. flips POOL_A_SHRINK_WORKERS
    back to non-empty), the watchdog must keep covering it."""
    expected_subset = {
        "neurogolf-worker-1", "neurogolf-worker-2", "neurogolf-worker-3",
        "neurogolf-worker-4", "neurogolf-worker-6",
        "neurogolf-reviewer", "neurogolf-reviewer-2",
    }
    actual = set(DISPATCHER_PROFILES)
    missing = expected_subset - actual
    assert not missing, f"DISPATCHER_PROFILES missing: {missing}"


def test_pool_b_workers_are_never_dispatcher_profiles():
    """Sanity: the watchdog must NEVER reassign cards away from /loop workers."""
    overlap = set(LOOP_WORKER_TARGETS) & set(DISPATCHER_PROFILES)
    assert overlap == set(), f"loop workers in DISPATCHER_PROFILES: {overlap}"
