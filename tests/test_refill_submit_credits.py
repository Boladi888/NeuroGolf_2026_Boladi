"""Tests for the submit-credit refill cron, especially the UTC-midnight
daily_used reset added 2026-05-18 (the previous implementation never
reset daily_used, so the counter drifted away from Kaggle's actual
100/day count).
"""
from __future__ import annotations

from datetime import datetime, timezone

from tools.refill_submit_credits import compute_daily_reset, compute_refill


def _state(**overrides):
    base = {
        "created_at": "2026-05-15T12:00:00Z",
        "policy": {
            "submit_rate_limit_minutes": 10,
            "daily_submit_cap": 100,
        },
        "credits": {
            "available": 5,
            "reserved": 0,
            "daily_used": 50,
            "last_refill_at": "2026-05-18T03:00:00Z",
            "last_submit_at": "2026-05-18T03:00:00Z",
            "last_daily_reset_at": "2026-05-18T00:00:00Z",
        },
    }
    base.update(overrides)
    return base


def test_compute_daily_reset_does_not_reset_within_same_utc_day():
    state = _state()
    state["credits"]["last_daily_reset_at"] = "2026-05-18T00:00:00Z"
    now = datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc)
    plan = compute_daily_reset(state, now)
    assert plan["reset"] is False
    assert plan["new_daily_used"] == 50


def test_compute_daily_reset_zeros_when_utc_day_rolls_over():
    """The trigger: today's UTC date > last_daily_reset_at's UTC date.
    On a 2026-05-18 -> 2026-05-19 boundary, daily_used must reset to 0
    so the 100/day cap doesn't perpetually count yesterday's submits."""
    state = _state()
    state["credits"]["daily_used"] = 73
    state["credits"]["last_daily_reset_at"] = "2026-05-18T00:00:00Z"
    now = datetime(2026, 5, 19, 0, 5, tzinfo=timezone.utc)
    plan = compute_daily_reset(state, now)
    assert plan["reset"] is True
    assert plan["new_daily_used"] == 0
    assert plan["prior_daily_used"] == 73
    assert plan["new_last_daily_reset_at"].startswith("2026-05-19")
    assert "rolled" in plan["reason"]


def test_compute_daily_reset_seeds_anchor_when_missing():
    """Legacy state files don't have last_daily_reset_at. Seed it from
    `now` without zeroing daily_used (we don't know what's already been
    submitted today)."""
    state = _state()
    state["credits"].pop("last_daily_reset_at", None)
    state["credits"]["daily_used"] = 9
    state.pop("created_at", None)
    now = datetime(2026, 5, 18, 5, 0, tzinfo=timezone.utc)
    plan = compute_daily_reset(state, now)
    assert plan["reset"] is True
    assert plan["new_daily_used"] == 9  # preserved, not zeroed
    assert plan["new_last_daily_reset_at"].startswith("2026-05-18")


def test_compute_daily_reset_uses_created_at_fallback():
    """If last_daily_reset_at is missing but created_at is in a past day,
    we should reset (the legacy state was for a different day)."""
    state = _state()
    state["created_at"] = "2026-05-15T12:00:00Z"
    state["credits"].pop("last_daily_reset_at", None)
    state["credits"]["daily_used"] = 73
    now = datetime(2026, 5, 18, 5, 0, tzinfo=timezone.utc)
    plan = compute_daily_reset(state, now)
    assert plan["reset"] is True
    assert plan["new_daily_used"] == 0  # zeroed because anchor is older
    assert "rolled" in plan["reason"]


def test_compute_refill_applies_daily_reset_before_computing_headroom():
    """End-to-end: when the daily reset zeros daily_used, the refill's
    daily_headroom calculation should use the FRESH 0, not yesterday's
    stale count. Otherwise the cap stops refilling prematurely."""
    state = _state()
    state["credits"]["daily_used"] = 99  # yesterday almost at cap
    state["credits"]["available"] = 0
    state["credits"]["last_daily_reset_at"] = "2026-05-18T00:00:00Z"
    state["credits"]["last_refill_at"] = "2026-05-18T23:00:00Z"  # 2h before midnight
    now = datetime(2026, 5, 19, 1, 0, tzinfo=timezone.utc)  # 2h post-reset
    plan = compute_refill(state, now)
    # Daily reset fired → fresh day → headroom = 100, not 1.
    assert plan["daily_used"] == 0
    assert plan["daily_reset"]["reset"] is True
    # 2 hours / 10min = 12 tokens. Headroom > 12, so we get all of them.
    assert plan["tokens_added"] == 12
    assert plan["available_after"] == 12


def test_compute_refill_no_reset_within_same_day():
    state = _state()
    state["credits"]["daily_used"] = 9
    state["credits"]["available"] = 20
    state["credits"]["last_daily_reset_at"] = "2026-05-18T00:00:00Z"
    state["credits"]["last_refill_at"] = "2026-05-18T04:00:00Z"
    now = datetime(2026, 5, 18, 5, 0, tzinfo=timezone.utc)
    plan = compute_refill(state, now)
    assert plan["daily_used"] == 9  # preserved
    assert plan["daily_reset"]["reset"] is False
    # 60 min / 10 = 6 tokens
    assert plan["tokens_added"] == 6
