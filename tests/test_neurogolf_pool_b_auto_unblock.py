"""Tests for the Pool B auto-unblock cron.

The cron looks at `hermes kanban diagnostics --json`, picks blocked Pool B
cards whose only error is a dispatcher closed-port crash, and unblocks them.
These tests verify the planning logic without touching the live Hermes board.
"""

import subprocess

import pytest

from tools import neurogolf_pool_b_auto_unblock as auto_unblock


@pytest.fixture(autouse=True)
def _no_live_hermes(monkeypatch):
    """Hold the module docstring's promise: never reach the `hermes` CLI.

    `plan_unblocks()` calls `card_show()`, which shells out to `hermes kanban show`.
    On a machine with the CLI installed that silently spawned a process; on one
    without it (CI) the tests failed outright. Both call sites already treat a
    non-zero return as "no data", so stubbing at that seam leaves the behaviour
    under test unchanged.
    """

    def _stub(args: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=["hermes", "kanban", *args], returncode=1, stdout=b"", stderr=b""
        )

    monkeypatch.setattr(auto_unblock, "_run_hermes", _stub)


def _diag(
    *,
    task_id: str,
    assignee: str,
    status: str = "blocked",
    kind: str = "repeated_crashes",
    last_error: str = "pid 12345 not alive",
    outcome: str | None = None,
) -> dict:
    data: dict = {"consecutive_crashes": 2, "last_error": last_error}
    if kind == "repeated_failures":
        data = {"consecutive_failures": 3, "last_error": last_error}
        if outcome is None:
            outcome = "crashed"
    if outcome is not None:
        data["most_recent_outcome"] = outcome
    return {
        "task_id": task_id,
        "title": f"[worker task000 {assignee} solver-hunt] new-solver candidate hunt",
        "status": status,
        "assignee": assignee,
        "diagnostics": [
            {
                "kind": kind,
                "severity": "error",
                "title": f"Agent crashed: {last_error}",
                "detail": last_error,
                "data": data,
            }
        ],
    }


def test_is_pool_b_dispatch_crash_matches_closed_port_signature():
    entry = _diag(task_id="t_aaa", assignee="neurogolf-worker-7")
    assert auto_unblock.is_pool_b_dispatch_crash(entry) is True


def test_is_pool_b_dispatch_crash_matches_repeated_failures_after_first_cycle():
    """After one auto-unblock cycle Hermes emits `repeated_failures` instead
    of `repeated_crashes`. Both shapes mean the same thing for Pool B
    closed-port workers, so we accept either."""
    entry = _diag(
        task_id="t_aaa",
        assignee="neurogolf-worker-7",
        kind="repeated_failures",
        outcome="crashed",
    )
    assert auto_unblock.is_pool_b_dispatch_crash(entry) is True


def test_is_pool_b_dispatch_crash_rejects_repeated_failures_with_real_outcome():
    """If the most_recent_outcome is something other than 'crashed' (e.g.
    a worker process actually started and then exited with an error), this
    is NOT the closed-port signature and needs human eyes — leave it alone."""
    entry = _diag(
        task_id="t_aaa",
        assignee="neurogolf-worker-7",
        kind="repeated_failures",
        outcome="failed",
    )
    assert auto_unblock.is_pool_b_dispatch_crash(entry) is False


def test_is_pool_b_dispatch_crash_rejects_non_pool_b_worker():
    entry = _diag(task_id="t_aaa", assignee="neurogolf-worker-1")
    assert auto_unblock.is_pool_b_dispatch_crash(entry) is False


def test_is_pool_b_dispatch_crash_rejects_non_blocked_card():
    entry = _diag(task_id="t_aaa", assignee="neurogolf-worker-7", status="ready")
    assert auto_unblock.is_pool_b_dispatch_crash(entry) is False


def test_is_pool_b_dispatch_crash_rejects_non_crash_diagnostic():
    entry = _diag(task_id="t_aaa", assignee="neurogolf-worker-7", kind="human-eyes")
    assert auto_unblock.is_pool_b_dispatch_crash(entry) is False


def test_is_pool_b_dispatch_crash_rejects_unrelated_error_message():
    entry = _diag(
        task_id="t_aaa",
        assignee="neurogolf-worker-7",
        last_error="connection refused after 30s",
    )
    # We only auto-unblock the closed-port crash signature; real crashes need
    # human eyes.
    assert auto_unblock.is_pool_b_dispatch_crash(entry) is False


def test_plan_unblocks_picks_pool_b_only(monkeypatch):
    diags = [
        _diag(task_id="t_pool_a", assignee="neurogolf-worker-1"),
        _diag(task_id="t_pool_b_7", assignee="neurogolf-worker-7"),
        _diag(task_id="t_pool_b_8", assignee="neurogolf-worker-8"),
    ]
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 0)
    plan = auto_unblock.plan_unblocks(diags=diags)
    chosen = {a["task_id"] for a in plan["actions"]}
    assert chosen == {"t_pool_b_7", "t_pool_b_8"}


def test_plan_unblocks_caps_at_max_unblocks(monkeypatch):
    diags = [_diag(task_id="t_loop", assignee="neurogolf-worker-7")]
    # Already auto-unblocked 8 times -> cap reached.
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 8)
    plan = auto_unblock.plan_unblocks(diags=diags, max_unblocks=8)
    assert plan["actions"] == []
    assert plan["skipped"]
    assert "prior auto-unblock count 8" in plan["skipped"][0]["reason"]


def test_plan_unblocks_allows_at_one_below_cap(monkeypatch):
    diags = [_diag(task_id="t_aaa", assignee="neurogolf-worker-7")]
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 7)
    plan = auto_unblock.plan_unblocks(diags=diags, max_unblocks=8)
    assert len(plan["actions"]) == 1
    assert plan["actions"][0]["prior_count"] == 7
    # The marker comment should describe this as attempt 8/8.
    assert "8/8" in plan["actions"][0]["comment"]


def test_plan_unblocks_emits_marker_comment(monkeypatch):
    diags = [_diag(task_id="t_aaa", assignee="neurogolf-worker-7")]
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 0)
    plan = auto_unblock.plan_unblocks(diags=diags)
    comment = plan["actions"][0]["comment"]
    assert auto_unblock.COMMENT_MARKER in comment
    assert "attempt 1/" in comment


def test_plan_unblocks_skips_card_with_review_required_block(monkeypatch):
    """Workers post `BLOCKED: review-required: ...` comments and run
    `hermes kanban block --reason "review-required: ..."` once they have a
    real candidate. The dispatcher's stale crash diagnostic is still on the
    card, but the block is INTENTIONAL — the auto-unblock cron must skip,
    otherwise the dep serializer demotes the card before auto-triage gets
    to create a review card.
    """
    diags = [_diag(task_id="t_handoff", assignee="neurogolf-worker-7")]
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 0)
    monkeypatch.setattr(
        auto_unblock,
        "card_show",
        lambda _tid: "review-required: task000 drop-shadow v2, cost 9174",
    )
    plan = auto_unblock.plan_unblocks(diags=diags)
    assert plan["actions"] == []
    assert plan["skipped"]
    assert "review-required" in plan["skipped"][0]["reason"]


def test_plan_unblocks_skips_card_with_worker_owner_json_handoff(monkeypatch):
    """If the comment trail contains the JSON handoff schema's
    `worker_owner` field, a worker has already deposited a real candidate.
    Auto-triage / route-repair own the next step — auto-unblock must not
    intervene."""
    diags = [_diag(task_id="t_handoff", assignee="neurogolf-worker-8")]
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 0)
    monkeypatch.setattr(
        auto_unblock,
        "card_show",
        lambda _tid: '{"task_id":"task004","candidate_id":"task004_v1","worker_owner":"neurogolf-worker-8","scores":{...}}',
    )
    plan = auto_unblock.plan_unblocks(diags=diags)
    assert plan["actions"] == []
    assert plan["skipped"]


def test_plan_unblocks_unblocks_when_no_review_required_signal(monkeypatch):
    """Normal closed-port crash with no handoff signals — auto-unblock proceeds."""
    diags = [_diag(task_id="t_normal", assignee="neurogolf-worker-7")]
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 0)
    monkeypatch.setattr(auto_unblock, "card_show", lambda _tid: "Body has no handoff markers, just standard pool-b card")
    plan = auto_unblock.plan_unblocks(diags=diags)
    assert len(plan["actions"]) == 1
    assert plan["skipped"] == []


def test_diagnostic_cooldown_satisfied_when_last_crash_is_old():
    """Bug fix 2026-05-18: worker-5 was offline for hours, cron hit the
    8/8 cap, then when worker-5 came back the card stayed blocked. New
    cooldown logic: if dispatcher has been quiet > 30 min, reset the cap.
    """
    import time as _time
    now = _time.time()
    entry = {
        "diagnostics": [
            {"kind": "repeated_failures", "last_seen_at": now - (35 * 60)},  # 35 min ago
        ],
    }
    assert auto_unblock.diagnostic_cooldown_satisfied(entry, now_ts=now) is True


def test_diagnostic_cooldown_not_satisfied_when_active_loop():
    """While the dispatcher is actively crashing the card, the cap MUST
    still apply — that's the whole point of the cap."""
    import time as _time
    now = _time.time()
    entry = {
        "diagnostics": [
            {"kind": "repeated_failures", "last_seen_at": now - 60},  # 1 min ago
        ],
    }
    assert auto_unblock.diagnostic_cooldown_satisfied(entry, now_ts=now) is False


def test_diagnostic_cooldown_defaults_to_false_with_no_timestamps():
    """Legacy diagnostic data without last_seen_at: don't assume cooldown
    is satisfied. Keep conservative cap behavior."""
    entry = {"diagnostics": [{"kind": "repeated_failures"}]}
    assert auto_unblock.diagnostic_cooldown_satisfied(entry) is False


def test_plan_unblocks_resets_cap_when_dispatcher_dormant(monkeypatch):
    """End-to-end: card at 8/8 cap WITH dormant dispatcher → unblock fires
    again (cooldown reset). This is the worker-5 offline-then-online case."""
    import time as _time
    now = _time.time()
    diags = [_diag(task_id="t_dormant", assignee="neurogolf-worker-5")]
    diags[0]["diagnostics"][0]["last_seen_at"] = now - (35 * 60)  # stale
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 8)
    monkeypatch.setattr(auto_unblock, "card_show", lambda _tid: "no handoff in body")
    plan = auto_unblock.plan_unblocks(diags=diags, max_unblocks=8)
    assert len(plan["actions"]) == 1
    assert plan["actions"][0]["cooldown_reset"] is True
    assert "cooldown-reset" in plan["actions"][0]["comment"]


def test_plan_unblocks_keeps_cap_when_dispatcher_still_active(monkeypatch):
    """Card at 8/8 cap with RECENT crash → still skipped. The cap is doing
    its job; cool-down hasn't kicked in yet."""
    import time as _time
    now = _time.time()
    diags = [_diag(task_id="t_active_loop", assignee="neurogolf-worker-7")]
    diags[0]["diagnostics"][0]["last_seen_at"] = now - 30  # recent crash
    monkeypatch.setattr(auto_unblock, "auto_unblock_count", lambda _tid: 8)
    monkeypatch.setattr(auto_unblock, "card_show", lambda _tid: "no handoff")
    plan = auto_unblock.plan_unblocks(diags=diags, max_unblocks=8)
    assert plan["actions"] == []
    assert "dispatcher still active" in plan["skipped"][0]["reason"]


def test_pid_pattern_matches_real_signatures():
    # The dispatcher's reported error wording (sampled from a real diagnostic).
    samples = [
        "pid 38720 not alive",
        "pid 29188 not alive",
        "pid 1 not alive",
        "Pid 99999 NOT ALIVE",  # case insensitive
    ]
    for s in samples:
        assert auto_unblock.PID_NOT_ALIVE_RE.search(s) is not None

    non_matches = [
        "Connection refused",
        "API key invalid",
        "exit code 137",
    ]
    for s in non_matches:
        assert auto_unblock.PID_NOT_ALIVE_RE.search(s) is None
