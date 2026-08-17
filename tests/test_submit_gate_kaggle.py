"""Tests for tools/submit_gate_kaggle.py: budget gates, lineage check, half-fail.

These tests never call the real Kaggle CLI — every external interaction is
injected as a callable. Same pattern as the mechanical-layer tests.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.auto_triage_blocked import Handoff
from tools.submit_gate_kaggle import (
    KaggleSubmitRecord,
    SubmitOutcome,
    _count_submits_today,
    _extract_score_from_lineage,
    _read_current_best_score,
    _record_successful_submit,
    load_budget_state,
    load_lineage_state,
    submit_with_safety,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_runtime_config(path: Path, *, enabled: bool = True, max_per_hour: int = 20) -> None:
    path.write_text(
        json.dumps({
            "schema": "neurogolf.submit_runtime_config.v1",
            "post_refresh_auto_submit_enabled": enabled,
            "max_submits_per_hour": max_per_hour,
        }),
        encoding="utf-8",
    )


def _write_credit_state(
    path: Path,
    *,
    daily_cap: int = 100,
    rate_limit_minutes: int = 10,
    last_submit_at: str | None = None,
    spent_total: int = 0,
    daily_used: int = 0,
) -> None:
    state = {
        "schema": "neurogolf.submit_credit_state.v1",
        "policy": {
            "daily_submit_cap": daily_cap,
            "submit_rate_limit_minutes": rate_limit_minutes,
        },
        "credits": {
            "spent_total": spent_total,
            "daily_used": daily_used,
            "last_submit_at": last_submit_at,
        },
    }
    path.write_text(json.dumps(state), encoding="utf-8")


def _write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "number", "fruit", "kind", "kaggle_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _handoff(*, lineage_base: str = "current-best stub_on_5751_13_2026") -> Handoff:
    return Handoff(
        raw_text="{}",
        parsed={},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.5,
        expected_point_gain=0.5,
        package_sha="a" * 64,
        lineage_base=lineage_base,
        changed_entries=["task100.onnx"],
    )


# ---------------------------------------------------------------------------
# _count_submits_today (ledger helper)
# ---------------------------------------------------------------------------


def test_count_submits_today_counts_today_only(tmp_path):
    ledger = tmp_path / "submissions_ledger.csv"
    _write_ledger(ledger, [
        {"date": "2026-05-18", "number": "1", "fruit": "apple", "kind": "x", "kaggle_score": "5.0"},
        {"date": "2026-05-18", "number": "2", "fruit": "banana", "kind": "x", "kaggle_score": "5.0"},
        {"date": "2026-05-17", "number": "3", "fruit": "cherry", "kind": "x", "kaggle_score": "5.0"},
    ])
    assert _count_submits_today(ledger, "2026-05-18") == 2
    assert _count_submits_today(ledger, "2026-05-17") == 1
    assert _count_submits_today(ledger, "2026-05-19") == 0


def test_count_submits_today_missing_ledger_returns_zero(tmp_path):
    """Defensive default: a brand-new install has no ledger yet."""
    assert _count_submits_today(tmp_path / "nope.csv", "2026-05-18") == 0


def test_count_submits_today_handles_corrupt_ledger(tmp_path):
    """Returns 0 rather than crashing the cron tick."""
    ledger = tmp_path / "broken.csv"
    ledger.write_bytes(b"this is not, csv,\x00\x00\xff garbage")
    # Should not raise
    result = _count_submits_today(ledger, "2026-05-18")
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# load_budget_state
# ---------------------------------------------------------------------------


def test_load_budget_state_returns_remaining_capacity(tmp_path):
    runtime = tmp_path / "rt.json"
    credit = tmp_path / "credit.json"
    ledger = tmp_path / "ledger.csv"
    _write_runtime_config(runtime, enabled=True, max_per_hour=20)
    _write_credit_state(credit, daily_cap=100, rate_limit_minutes=10, last_submit_at=None)
    _write_ledger(ledger, [
        {"date": "2026-05-18", "number": str(i), "fruit": "x", "kind": "x", "kaggle_score": "1"}
        for i in range(30)
    ])
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    budget = load_budget_state(
        runtime_config_path=runtime,
        credit_state_path=credit,
        ledger_path=ledger,
        now=now,
    )
    assert budget.daily_remaining == 70  # 100 - 30
    assert budget.seconds_until_next_slot == 0.0  # never submitted
    assert budget.can_submit_now is True


def test_load_budget_state_exhausted_daily_cap(tmp_path):
    runtime = tmp_path / "rt.json"
    credit = tmp_path / "credit.json"
    ledger = tmp_path / "ledger.csv"
    _write_runtime_config(runtime, enabled=True, max_per_hour=20)
    _write_credit_state(credit, daily_cap=50)
    _write_ledger(ledger, [
        {"date": "2026-05-18", "number": str(i), "fruit": "x", "kind": "x", "kaggle_score": "1"}
        for i in range(50)
    ])
    now = datetime(2026, 5, 18, 23, 0, 0, tzinfo=timezone.utc)
    budget = load_budget_state(
        runtime_config_path=runtime,
        credit_state_path=credit,
        ledger_path=ledger,
        now=now,
    )
    assert budget.daily_remaining == 0
    assert budget.can_submit_now is False


def test_load_budget_state_within_rate_limit(tmp_path):
    """Rate-limit gate: ``last_submit_at`` was 3 minutes ago, rate limit is 10 min."""
    runtime = tmp_path / "rt.json"
    credit = tmp_path / "credit.json"
    ledger = tmp_path / "ledger.csv"
    _write_runtime_config(runtime, enabled=True, max_per_hour=20)
    _write_credit_state(
        credit,
        daily_cap=100,
        rate_limit_minutes=10,
        last_submit_at="2026-05-18T11:57:00Z",
    )
    _write_ledger(ledger, [])
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    budget = load_budget_state(
        runtime_config_path=runtime,
        credit_state_path=credit,
        ledger_path=ledger,
        now=now,
    )
    assert budget.daily_remaining == 100
    assert budget.seconds_until_next_slot == 420.0  # 10 min - 3 min = 7 min = 420 s
    assert budget.can_submit_now is False


def test_load_budget_state_auto_submit_disabled_blocks_all(tmp_path):
    """If the operator has disabled auto-submit in runtime config, even a clean
    candidate must not submit. Treat budget as exhausted."""
    runtime = tmp_path / "rt.json"
    credit = tmp_path / "credit.json"
    ledger = tmp_path / "ledger.csv"
    _write_runtime_config(runtime, enabled=False, max_per_hour=20)
    _write_credit_state(credit)
    _write_ledger(ledger, [])
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    budget = load_budget_state(
        runtime_config_path=runtime,
        credit_state_path=credit,
        ledger_path=ledger,
        now=now,
    )
    assert budget.can_submit_now is False
    assert budget.daily_remaining == 0


def test_load_budget_state_zero_per_hour_blocks_all(tmp_path):
    runtime = tmp_path / "rt.json"
    credit = tmp_path / "credit.json"
    ledger = tmp_path / "ledger.csv"
    _write_runtime_config(runtime, enabled=True, max_per_hour=0)
    _write_credit_state(credit)
    _write_ledger(ledger, [])
    budget = load_budget_state(
        runtime_config_path=runtime,
        credit_state_path=credit,
        ledger_path=ledger,
        now=datetime(2026, 5, 18, 12, tzinfo=timezone.utc),
    )
    assert budget.can_submit_now is False


def test_load_budget_state_handles_missing_config_files(tmp_path):
    """A fresh install without these files yet must not crash submit_gate."""
    budget = load_budget_state(
        runtime_config_path=tmp_path / "missing_rt.json",
        credit_state_path=tmp_path / "missing_credit.json",
        ledger_path=tmp_path / "missing_ledger.csv",
        now=datetime(2026, 5, 18, 12, tzinfo=timezone.utc),
    )
    # With no runtime config, post_refresh_auto_submit defaults to False -> blocked
    assert budget.can_submit_now is False


# ---------------------------------------------------------------------------
# Lineage extraction
# ---------------------------------------------------------------------------


class TestExtractScoreFromLineage:
    def test_extracts_score_from_canonical_form(self):
        # task141_v4_fp16_pos_t_6856eb96_on_5743_97_20260517T2202Z
        result = _extract_score_from_lineage("current-best task141_v4_fp16_pos_t_6856eb96_on_5743_97_20260517T2202Z")
        assert result == 5743.97

    def test_extracts_score_with_single_decimal_digit(self):
        result = _extract_score_from_lineage("task019_slice12_compute12_pad30_fp16_on_5739_7_20260517T160659Z")
        assert result == 5739.7

    def test_extracts_score_at_end_of_string(self):
        """A lineage string that ends exactly at the score (no trailing date
        or underscore) must still parse. Earlier regex required a trailing
        underscore and silently failed on this real-world shape."""
        assert _extract_score_from_lineage("task141_v4_on_5743_97") == 5743.97
        assert _extract_score_from_lineage("task019_on_5739_7") == 5739.7

    def test_returns_none_when_no_score_segment(self):
        assert _extract_score_from_lineage("current-best task141") is None
        assert _extract_score_from_lineage("") is None

    def test_returns_none_for_malformed_score(self):
        assert _extract_score_from_lineage("foo_on_notanumber_2026Z") is None


def test_read_current_best_score_production_shape(tmp_path):
    """The shape that's actually on disk in submissions/current_best_manifest.json
    (verified 2026-05-20). The NESTED ``current_best.score`` is the source
    of truth because ``tools/neurogolf_sync_current_best.py::update_v2_manifest``
    only writes to the nested block — the top-level ``score`` is v1-legacy
    and goes STALE after a v2 promote (megastack153 incident 2026-05-20:
    the top-level was 5835.3 while the actual canonical was 5861.43)."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "current_best": {"candidate_id": "x", "score": 5805.19},
        "score": 5756.12,  # STALE v1 leftover — must NOT win
        "candidate_id": "y",
        "task_id": "task100",
        "lineage_base_score": 5755.8,
    }), encoding="utf-8")
    # NESTED current_best.score wins (5805.19), not the stale top-level (5756.12)
    assert _read_current_best_score(path) == 5805.19


def test_read_current_best_score_falls_back_to_current_best_dot_score(tmp_path):
    """If only the nested current_best.score is present, use it."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "current_best": {"score": 5805.19, "candidate_id": "x"},
    }), encoding="utf-8")
    assert _read_current_best_score(path) == 5805.19


def test_read_current_best_score_legacy_public_score_form(tmp_path):
    """Backward-compat: an older manifest with public_score still parses."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "current_best": {"public_score": 5751.13},
    }), encoding="utf-8")
    assert _read_current_best_score(path) == 5751.13


def test_read_current_best_score_flat_legacy_public_score(tmp_path):
    """Even older manifest with public_score at root."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"public_score": 5800.0}), encoding="utf-8")
    assert _read_current_best_score(path) == 5800.0


def test_read_current_best_score_returns_none_for_missing(tmp_path):
    assert _read_current_best_score(tmp_path / "nope.json") is None


def test_read_current_best_score_string_score_coerced(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"score": "5751.13"}), encoding="utf-8")
    assert _read_current_best_score(path) == 5751.13


def test_read_current_best_score_megastack153_regression(tmp_path):
    """Exact replay of the 2026-05-20 megastack153 incident: after a manual
    promote, the v2 sync wrote nested current_best=5861.43 but left the
    top-level score=5835.3 stale. submit_gate must read 5861.43 (NEW
    canonical) so its lineage-stale gate fires correctly on candidates
    that were built against 5835.30."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": "neurogolf.candidate_registry.v2-draft",
        "current_best": {
            "candidate_id": "megastack153_on_5835_30_20260520T1700Z",
            "score": 5861.43,
            "package_sha256": "CB7779655379A7293F0E63FE22346499FADC1E8CA279B5937A232E76AAA1A7E0",
        },
        "score": 5835.3,  # STALE v1 leftover from before the megastack153 promote
        "candidate_id": "megastack93_on_5833_74_20260520T033437Z",  # also stale
    }), encoding="utf-8")
    assert _read_current_best_score(path) == 5861.43


def test_load_lineage_state_detects_stale_handoff(tmp_path):
    """Worker built against 5743.97; current-best is now 5751.13 -> stale."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"current_best": {"public_score": 5751.13}}), encoding="utf-8")
    handoff = _handoff(lineage_base="current-best task141_v4_on_5743_97_2026")
    lineage = load_lineage_state(handoff, manifest_path=manifest)
    assert lineage.candidate_base_score == 5743.97
    assert lineage.current_best_score == 5751.13
    assert lineage.is_stale is True


def test_load_lineage_state_matching_not_stale(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"current_best": {"public_score": 5751.13}}), encoding="utf-8")
    handoff = _handoff(lineage_base="current-best task141_v4_on_5751_13_2026")
    lineage = load_lineage_state(handoff, manifest_path=manifest)
    assert lineage.is_stale is False


def test_load_lineage_state_unknown_does_not_block(tmp_path):
    """Missing lineage_base or missing manifest -> None on both sides ->
    is_stale=False (don't block on missing data; surface in audit log)."""
    handoff = _handoff(lineage_base="not-parseable-lineage")
    lineage = load_lineage_state(handoff, manifest_path=tmp_path / "missing.json")
    assert lineage.is_stale is False


# ---------------------------------------------------------------------------
# submit_with_safety
# ---------------------------------------------------------------------------


def test_submit_with_safety_ok_updates_credit_state(tmp_path):
    credit = tmp_path / "credit.json"
    _write_credit_state(credit, spent_total=10, daily_used=3)
    record = KaggleSubmitRecord(submitted_at="2026-05-18T12:00:00Z", public_score=None, kaggle_ref="123")
    submit_fn = lambda h, p: SubmitOutcome(status="ok", record=record)
    handoff = _handoff()
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    outcome = submit_with_safety(
        handoff,
        candidate_zip_path=tmp_path / "fake.zip",
        submit_fn=submit_fn,
        credit_state_path=credit,
        now=now,
    )
    assert outcome.status == "ok"
    state = json.loads(credit.read_text(encoding="utf-8"))
    assert state["credits"]["spent_total"] == 11
    assert state["credits"]["daily_used"] == 4
    assert state["credits"]["last_submit_at"] == "2026-05-18T12:00:00Z"


def test_submit_with_safety_uncertain_recovered_by_verifier(tmp_path):
    """submit_fn raises but verify_after_fn confirms the submit landed."""
    credit = tmp_path / "credit.json"
    _write_credit_state(credit, spent_total=10)

    def submit_fn(h, p):
        raise RuntimeError("network blip during upload")

    record = KaggleSubmitRecord(submitted_at="2026-05-18T12:00:00Z", public_score=None, kaggle_ref="124")
    verify_fn = lambda h: record  # Kaggle confirms the submit DID land

    outcome = submit_with_safety(
        _handoff(),
        candidate_zip_path=tmp_path / "fake.zip",
        submit_fn=submit_fn,
        verify_after_fn=verify_fn,
        credit_state_path=credit,
        now=datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert outcome.status == "ok"  # downgraded from uncertain
    # Credit state was updated despite the half-fail
    state = json.loads(credit.read_text(encoding="utf-8"))
    assert state["credits"]["spent_total"] == 11


def test_submit_with_safety_uncertain_unverified_notifies(tmp_path):
    """submit_fn raises, verify_after_fn says no record -> alert operator."""
    credit = tmp_path / "credit.json"
    _write_credit_state(credit, spent_total=10)

    def submit_fn(h, p):
        return SubmitOutcome(status="uncertain", error="post-upload timeout")

    verify_fn = lambda h: None  # No record found

    notifications = []
    notifier_fn = lambda severity, msg: notifications.append((severity, msg))

    outcome = submit_with_safety(
        _handoff(),
        candidate_zip_path=tmp_path / "fake.zip",
        submit_fn=submit_fn,
        verify_after_fn=verify_fn,
        notifier_fn=notifier_fn,
        credit_state_path=credit,
        now=datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert outcome.status == "uncertain"
    assert len(notifications) == 1
    severity, msg = notifications[0]
    assert severity == "warn"
    assert "uncertain" in msg.lower()
    # Credit state UNCHANGED on uncertain outcome — we don't know if submit landed
    state = json.loads(credit.read_text(encoding="utf-8"))
    assert state["credits"]["spent_total"] == 10


def test_submit_with_safety_uncertain_no_verifier_stays_uncertain(tmp_path):
    """No verifier wired up -> can't recover from uncertain, return as-is."""
    credit = tmp_path / "credit.json"
    _write_credit_state(credit, spent_total=10)
    submit_fn = lambda h, p: SubmitOutcome(status="uncertain", error="x")

    outcome = submit_with_safety(
        _handoff(),
        candidate_zip_path=tmp_path / "fake.zip",
        submit_fn=submit_fn,
        verify_after_fn=None,
        credit_state_path=credit,
    )
    assert outcome.status == "uncertain"
    state = json.loads(credit.read_text(encoding="utf-8"))
    assert state["credits"]["spent_total"] == 10  # no mutation


def test_submit_with_safety_failed_no_state_mutation(tmp_path):
    """submit_fn returns failed -> no half-fail recovery, no state update,
    safe to retry on next cron tick."""
    credit = tmp_path / "credit.json"
    _write_credit_state(credit, spent_total=10)
    submit_fn = lambda h, p: SubmitOutcome(status="failed", error="kaggle CLI not installed")
    notifier = MagicMock()

    outcome = submit_with_safety(
        _handoff(),
        candidate_zip_path=tmp_path / "fake.zip",
        submit_fn=submit_fn,
        notifier_fn=notifier,
        credit_state_path=credit,
    )
    assert outcome.status == "failed"
    state = json.loads(credit.read_text(encoding="utf-8"))
    assert state["credits"]["spent_total"] == 10
    notifier.assert_not_called()


def test_submit_with_safety_submit_fn_raising_becomes_uncertain(tmp_path):
    """If submit_fn raises an unhandled exception (not just returns a failed
    status), treat as uncertain — we don't know if the upload landed."""
    credit = tmp_path / "credit.json"
    _write_credit_state(credit)
    def submit_fn(h, p):
        raise OSError("connection reset mid-upload")

    outcome = submit_with_safety(
        _handoff(),
        candidate_zip_path=tmp_path / "fake.zip",
        submit_fn=submit_fn,
        verify_after_fn=lambda h: None,  # verifier says it didn't land
        credit_state_path=credit,
    )
    assert outcome.status == "uncertain"
    assert "connection reset" in outcome.error


def test_submit_with_safety_verifier_raising_keeps_uncertain(tmp_path):
    """If the verifier itself raises, don't upgrade to ok — stay uncertain."""
    credit = tmp_path / "credit.json"
    _write_credit_state(credit)
    submit_fn = lambda h, p: SubmitOutcome(status="uncertain", error="x")

    def verifier(h):
        raise RuntimeError("kaggle API rate limited")

    outcome = submit_with_safety(
        _handoff(),
        candidate_zip_path=tmp_path / "fake.zip",
        submit_fn=submit_fn,
        verify_after_fn=verifier,
        credit_state_path=credit,
    )
    assert outcome.status == "uncertain"
    assert "verifier raised" in outcome.error


# ---------------------------------------------------------------------------
# _record_successful_submit (atomic credit-state writer)
# ---------------------------------------------------------------------------


def test_record_successful_submit_creates_file_when_missing(tmp_path):
    """A fresh install: no credit state file yet. First submit creates it."""
    credit = tmp_path / "fresh.json"
    _record_successful_submit(credit, datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc))
    assert credit.exists()
    state = json.loads(credit.read_text(encoding="utf-8"))
    assert state["credits"]["spent_total"] == 1
    assert state["credits"]["daily_used"] == 1
    assert state["credits"]["last_submit_at"] == "2026-05-18T12:00:00Z"


def test_record_successful_submit_increments_counters(tmp_path):
    credit = tmp_path / "credit.json"
    _write_credit_state(credit, spent_total=42, daily_used=7)
    _record_successful_submit(credit, datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc))
    state = json.loads(credit.read_text(encoding="utf-8"))
    assert state["credits"]["spent_total"] == 43
    assert state["credits"]["daily_used"] == 8


def test_record_successful_submit_writes_atomically(tmp_path):
    """A SIGKILL mid-write must never corrupt the credit state file."""
    credit = tmp_path / "credit.json"
    _write_credit_state(credit, spent_total=42, daily_used=7)
    initial_bytes = credit.read_bytes()

    from unittest.mock import patch
    with patch("tools.task_outcome_registry_io.os.replace", side_effect=OSError("kernel panic")):
        with pytest.raises(OSError):
            _record_successful_submit(credit, datetime(2026, 5, 18, 12, tzinfo=timezone.utc))

    # File on disk unchanged because the .tmp was never replaced
    assert credit.read_bytes() == initial_bytes
