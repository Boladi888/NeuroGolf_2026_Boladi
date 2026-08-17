"""Tests for tools/submit_gate_runner.py.

Focus on the orchestration / wiring layer:
- ControlLock acquire/release behavior
- Audit log writes (apply + dry-run modes)
- process_card mutation semantics (dry-run vs apply)
- Verdict counting / summary output
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.auto_triage_blocked import Handoff
from tools.submit_gate import PassState, Submit, VerifyResult
from tools.submit_gate_kaggle import KaggleSubmitRecord, SubmitOutcome
from tools.submit_gate_runner import (
    AuditRow,
    ControlLock,
    KanbanWriter,
    _count_verdicts,
    build_retry_card_spec,
    process_card,
    read_current_pass,
    write_audit_log,
)


class _RecordingKanbanWriter:
    """Test double that records what would have been mutated, without IO."""

    def __init__(self):
        self.created: list[dict] = []
        self.completed: list[dict] = []

    def create_retry(self, spec):
        self.created.append(dict(spec))
        return {"id": f"t_new_{len(self.created)}", "ok": True, "error": ""}

    def complete_blocked(self, *, card_id, result):
        self.completed.append({"card_id": card_id, "result": result})
        return {"ok": True, "error": ""}


# ---------------------------------------------------------------------------
# ControlLock
# ---------------------------------------------------------------------------


def test_control_lock_acquires_when_path_free(tmp_path):
    lock = ControlLock(path=tmp_path / "test.lock")
    with lock as acquired:
        assert acquired is True
        assert (tmp_path / "test.lock").exists()
    assert not (tmp_path / "test.lock").exists()


def test_control_lock_blocks_concurrent_acquire(tmp_path):
    """Second instance can't acquire while first holds it."""
    path = tmp_path / "concurrent.lock"
    first = ControlLock(path=path, wait_seconds=0.1, retry_seconds=0.05)
    second = ControlLock(path=path, wait_seconds=0.1, retry_seconds=0.05)
    with first as first_acquired:
        assert first_acquired is True
        with second as second_acquired:
            assert second_acquired is False


def test_control_lock_takes_stale_lock(tmp_path):
    """A lock file older than stale_seconds is reclaimed (orphaned by a
    crashed previous run)."""
    path = tmp_path / "stale.lock"
    path.write_text("pid=99999 started=0\n", encoding="utf-8")
    # Make the file look old
    very_old = time.time() - 3600  # 1 hour ago
    os.utime(path, (very_old, very_old))

    lock = ControlLock(path=path, stale_seconds=60, wait_seconds=1.0)
    with lock as acquired:
        assert acquired is True


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def _row(**overrides) -> AuditRow:
    defaults = {
        "card_id": "t_abcd1234",
        "task_id": "task100",
        "attempts_before": 0,
        "predicted_gain": 0.5,
        "verdict_kind": "Submit",
        "verdict_reason": "submitted, predicted +0.5000",
        "submit_invoked": True,
        "handoff_summary": {"candidate_id": "task100_v1"},
    }
    defaults.update(overrides)
    return AuditRow(**defaults)


def test_audit_log_written_in_dry_run_mode(tmp_path):
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    rows = [_row(submit_invoked=False)]
    log_path = write_audit_log(rows, out_dir=tmp_path, now=now, dry_run=True)
    text = log_path.read_text(encoding="utf-8")
    assert "_dryrun" in log_path.name
    assert "DRY-RUN" in text
    assert "Cards processed: 1" in text
    assert "t_abcd1234" in text


def test_audit_log_apply_mode_filename_has_no_suffix(tmp_path):
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    log_path = write_audit_log([_row()], out_dir=tmp_path, now=now, dry_run=False)
    assert log_path.name == "submit_gate_20260518T120000Z.md"
    text = log_path.read_text(encoding="utf-8")
    assert "APPLY" in text


def test_audit_log_includes_handoff_json_extract(tmp_path):
    """Per the plan: full handoff parsed-extract appended for forensics."""
    rows = [_row(handoff_summary={"candidate_id": "x", "package_sha": "abc...", "expected_point_gain": 0.42})]
    log_path = write_audit_log(rows, out_dir=tmp_path, now=datetime(2026, 5, 18, tzinfo=timezone.utc))
    text = log_path.read_text(encoding="utf-8")
    assert "Handoff extracts" in text
    assert "0.42" in text
    assert "candidate_id" in text


def test_audit_log_renders_none_predicted_gain_cleanly(tmp_path):
    """No-candidate rows have predicted_gain=None — render as '(none)' not 'None'."""
    rows = [_row(predicted_gain=None, verdict_kind="Retry", submit_invoked=False)]
    log_path = write_audit_log(rows, out_dir=tmp_path, now=datetime(2026, 5, 18, tzinfo=timezone.utc))
    text = log_path.read_text(encoding="utf-8")
    assert "(none)" in text


def test_audit_log_per_card_verdict_visible_in_table(tmp_path):
    rows = [
        _row(card_id="t_aaa", verdict_kind="Submit", submit_invoked=True),
        _row(card_id="t_bbb", verdict_kind="Retry", submit_invoked=False, predicted_gain=0.1),
        _row(card_id="t_ccc", verdict_kind="GiveUpForPass", submit_invoked=False),
    ]
    log_path = write_audit_log(rows, out_dir=tmp_path, now=datetime(2026, 5, 18, tzinfo=timezone.utc))
    text = log_path.read_text(encoding="utf-8")
    assert "t_aaa" in text
    assert "t_bbb" in text
    assert "t_ccc" in text
    assert "Submit" in text
    assert "Retry" in text
    assert "GiveUpForPass" in text


# ---------------------------------------------------------------------------
# read_current_pass
# ---------------------------------------------------------------------------


def test_read_current_pass_reads_real_task_backlog_cursor_format(tmp_path):
    """Production guard: the path constant in submit_gate_runner must point
    at the file that actually has pass_count. We mistakenly pointed it at
    state.json initially, which has many fields but no pass_count, so the
    counter was always 0. The real source is task_backlog_cursor.json
    written by tools/neurogolf_backlog_candidates.py::advance_cursor."""
    path = tmp_path / "task_backlog_cursor.json"
    path.write_text(json.dumps({
        "schema": "neurogolf.task_backlog_cursor.v1",
        "next_task_id": "task366",
        "pass_count": 7,
        "last_advanced_from": "task365",
    }), encoding="utf-8")
    assert read_current_pass(path) == 7


def test_read_current_pass_missing_file_returns_zero(tmp_path):
    assert read_current_pass(tmp_path / "nope.json") == 0


def test_read_current_pass_extracts_int(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"pass_count": 5}), encoding="utf-8")
    assert read_current_pass(path) == 5


def test_read_current_pass_corrupt_returns_zero(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    assert read_current_pass(path) == 0


def test_read_current_pass_missing_field_returns_zero(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"other": "stuff"}), encoding="utf-8")
    assert read_current_pass(path) == 0


# ---------------------------------------------------------------------------
# _count_verdicts
# ---------------------------------------------------------------------------


def test_count_verdicts_aggregates_by_kind():
    rows = [
        _row(verdict_kind="Submit"),
        _row(verdict_kind="Submit"),
        _row(verdict_kind="Retry"),
        _row(verdict_kind="GiveUpForPass"),
    ]
    counts = _count_verdicts(rows)
    assert counts == {"Submit": 2, "Retry": 1, "GiveUpForPass": 1}


def test_count_verdicts_empty():
    assert _count_verdicts([]) == {}


# ---------------------------------------------------------------------------
# process_card — orchestration with stubs
# ---------------------------------------------------------------------------


def test_process_card_dry_run_does_not_mutate_registry(monkeypatch, tmp_path):
    """In dry-run mode, classify happens but no registry writes occur."""
    fake_handoff = Handoff(
        raw_text="{}",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.5,
        expected_point_gain=0.5,
        package_sha="a" * 64,
        algorithm_family="where-select",
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)

    registry = {"tasks": {}}
    card = {"id": "t_aaa"}
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")  # exists so mechanical doesn't immediately fail

    # Stub mechanical to pass
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    row = process_card(
        card,
        registry=registry,
        current_pass=1,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        dry_run=True,
    )

    # Submit verdict expected (first attempt, clean candidate, no budget gate)
    assert row.verdict_kind == "Submit"
    assert row.submit_invoked is False  # dry-run; no submit_fn invoked
    # Registry must be unchanged
    assert registry == {"tasks": {}}


def test_process_card_apply_mode_mutates_registry_on_retry(monkeypatch, tmp_path):
    fake_handoff = Handoff(
        raw_text="{}",
        # No cost reduction → Retry under the cost-reduction gate
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 10000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.0,
        expected_point_gain=0.0,
        package_sha="a" * 64,
        algorithm_family="where-select",
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    registry = {"tasks": {}}
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    row = process_card(
        {"id": "t_aaa"},
        registry=registry,
        current_pass=2,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        dry_run=False,
    )

    assert row.verdict_kind == "Retry"
    assert row.submit_invoked is False
    # Registry was mutated: attempts_this_pass incremented, family recorded
    assert registry["tasks"]["task100"]["attempts_this_pass"] == 1
    assert registry["tasks"]["task100"]["attempted_families"] == ["where-select"]
    assert registry["tasks"]["task100"]["last_attempt_pass"] == 2
    # Known attempts list grew
    assert len(registry["tasks"]["task100"]["known_attempts"]) == 1


def test_process_card_apply_mode_records_give_up_summary(monkeypatch, tmp_path):
    """3rd attempt with no cost reduction -> GiveUpForPass writes last_attempt_summary."""
    fake_handoff = Handoff(
        raw_text="{}",
        # No cost reduction on the 3rd attempt → GiveUpForPass under the new gate
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 10000}},
        candidate_id="task100_v3",
        task_id="task100",
        point_delta=0.0,
        expected_point_gain=0.0,
        package_sha="a" * 64,
        algorithm_family="conv-stamp",
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 2,  # this is the 3rd
                "attempted_families": ["where-select", "reduce-broadcast"],
                "last_attempt_pass": 2,
            }
        }
    }
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    row = process_card(
        {"id": "t_aaa"},
        registry=registry,
        current_pass=2,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        dry_run=False,
    )

    assert row.verdict_kind == "GiveUpForPass"
    entry = registry["tasks"]["task100"]
    assert entry["attempts_this_pass"] == 3
    assert "conv-stamp" in entry["attempted_families"]
    assert "tested_non_improving_this_pass" in entry["last_attempt_summary"]


def test_process_card_unparseable_handoff_does_not_crash(monkeypatch, tmp_path):
    """Bad JSON in the handoff comment -> graceful AuditRow with no
    verdict_kind that triggers IO."""
    bad_handoff = Handoff(
        raw_text="{not json",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id="",
        task_id="",
        parse_error="no JSON object found",
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: bad_handoff)

    registry = {"tasks": {}}
    row = process_card(
        {"id": "t_aaa"},
        registry=registry,
        current_pass=1,
        candidate_zip_resolver=lambda c, h: None,
        current_best_zip_path=None,
        dry_run=False,
    )
    assert row.verdict_kind == "UnparseableHandoff"
    assert "could not parse handoff" in row.verdict_reason
    # Registry untouched
    assert registry == {"tasks": {}}


def test_process_card_zip_unresolvable_emits_mechanical_fail(monkeypatch, tmp_path):
    fake_handoff = Handoff(
        raw_text="{}",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.5,
        expected_point_gain=0.5,
        package_sha="a" * 64,
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)

    row = process_card(
        {"id": "t_aaa"},
        registry={"tasks": {}},
        current_pass=1,
        candidate_zip_resolver=lambda c, h: None,  # can't find the zip
        current_best_zip_path=None,
        dry_run=True,
    )
    # Verify result was failed → Retry verdict
    assert row.verdict_kind == "Retry"
    assert "could not be resolved" in row.verdict_reason


def test_process_card_submit_invokes_submit_fn_in_apply_mode(monkeypatch, tmp_path):
    """When verdict is Submit and we're in apply mode, submit_fn IS called."""
    fake_handoff = Handoff(
        raw_text="{}",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.5,
        expected_point_gain=0.5,
        package_sha="a" * 64,
        algorithm_family="where-select",
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    submit_calls = []

    def submit_fn(handoff, zip_path):
        submit_calls.append((handoff.candidate_id, zip_path))
        return SubmitOutcome(
            status="ok",
            record=KaggleSubmitRecord(submitted_at="x", public_score=None, kaggle_ref="r"),
        )

    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    registry = {"tasks": {}}
    row = process_card(
        {"id": "t_aaa"},
        registry=registry,
        current_pass=2,  # not first attempt, so no forced-retry
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        submit_fn=submit_fn,
        dry_run=False,
    )

    # Submit was called
    assert len(submit_calls) == 1
    assert submit_calls[0][0] == "task100_v1"
    assert row.submit_invoked is True
    # Registry mutation occurred
    assert registry["tasks"]["task100"]["attempts_this_pass"] == 1


# ---------------------------------------------------------------------------
# build_retry_card_spec
# ---------------------------------------------------------------------------


def _retry_handoff(task_id="task100", candidate_id="task100_v2", family="where-select") -> Handoff:
    return Handoff(
        raw_text="{}",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id=candidate_id,
        task_id=task_id,
        algorithm_family=family,
    )


def test_build_retry_card_spec_title_matches_route_repair_patterns():
    """The retry card title MUST match tools/route_repair_cards.py
    REPAIR_TITLE_PATTERNS so that script routes it back to the original
    worker via parent-link fallback. Without this, retry cards sit
    unassigned forever."""
    import re
    spec = build_retry_card_spec(
        original_card={"id": "t_aaa"},
        handoff=_retry_handoff(),
        verdict_reason="x",
        directive="d",
        retry_attempt_index=2,
        attempted_families=[],
    )
    # Mirror the actual patterns from route_repair_cards.py:56-60
    repair_patterns = [
        re.compile(r"^\[repair\s+task\d+\]", re.IGNORECASE),
        re.compile(r"^repair\s+task\d+", re.IGNORECASE),
    ]
    assert any(p.search(spec["title"]) for p in repair_patterns), \
        f"retry title {spec['title']!r} does not match route_repair patterns"


def test_build_retry_card_spec_chain_key_includes_attempt_index():
    """Multiple retries on the same source card must produce distinct chain_keys."""
    h = _retry_handoff()
    spec1 = build_retry_card_spec(
        original_card={"id": "t_aaa"},
        handoff=h,
        verdict_reason="below floor",
        directive="try a new family",
        retry_attempt_index=2,
        attempted_families=["where-select"],
    )
    spec2 = build_retry_card_spec(
        original_card={"id": "t_aaa"},
        handoff=h,
        verdict_reason="below floor",
        directive="try a new family",
        retry_attempt_index=3,
        attempted_families=["where-select", "reduce-broadcast"],
    )
    assert spec1["chain_key"] != spec2["chain_key"]
    assert "attempt2" in spec1["chain_key"]
    assert "attempt3" in spec2["chain_key"]


def test_build_retry_card_spec_parent_link_present():
    """route_repair_cards reads parent assignment to route the retry back to
    the original worker. Without parent_id, that routing breaks."""
    spec = build_retry_card_spec(
        original_card={"id": "t_aaa"},
        handoff=_retry_handoff(),
        verdict_reason="x",
        directive="d",
        retry_attempt_index=2,
        attempted_families=[],
    )
    assert spec["parent_id"] == "t_aaa"


def test_build_retry_card_spec_no_assignee_lets_router_decide():
    """submit_gate should NOT pre-assign — leaving it unassigned lets
    route_repair_cards.py route by parent on the next dep-serializer tick."""
    spec = build_retry_card_spec(
        original_card={"id": "t_aaa"},
        handoff=_retry_handoff(),
        verdict_reason="x",
        directive="d",
        retry_attempt_index=2,
        attempted_families=[],
    )
    assert "assignee" not in spec


def test_build_retry_card_spec_body_contains_blacklist():
    spec = build_retry_card_spec(
        original_card={"id": "t_aaa"},
        handoff=_retry_handoff(),
        verdict_reason="predicted gain below floor",
        directive="try a different family",
        retry_attempt_index=2,
        attempted_families=["where-select", "reduce-broadcast"],
    )
    body = spec["body"]
    assert "where-select" in body
    assert "reduce-broadcast" in body
    assert "retry_attempt_index: 2" in body
    assert "do_not_use_families" in body
    assert "source_worker_task_id: t_aaa" in body
    assert "predicted gain below floor" in body
    assert "try a different family" in body


def test_build_retry_card_spec_first_retry_shows_no_blacklist():
    """attempt 1 has nothing to forbid yet."""
    spec = build_retry_card_spec(
        original_card={"id": "t_aaa"},
        handoff=_retry_handoff(),
        verdict_reason="x",
        directive="d",
        retry_attempt_index=1,
        attempted_families=[],
    )
    assert "(none yet)" in spec["body"]


def test_process_card_retry_creates_kanban_card_and_completes_original(monkeypatch, tmp_path):
    """A Retry verdict in apply mode MUST create a successor card and
    complete the blocked original — otherwise the lane freezes.

    This catches the production-readiness gap where process_card mutated
    only the registry but never moved the kanban state."""
    fake_handoff = Handoff(
        raw_text="{}",
        # No cost reduction → Retry verdict
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 10000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.0,
        expected_point_gain=0.0,
        package_sha="a" * 64,
        algorithm_family="where-select",
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    writer = _RecordingKanbanWriter()
    registry = {"tasks": {}}
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    row = process_card(
        {"id": "t_orig"},
        registry=registry,
        current_pass=1,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        kanban_writer=writer,
        dry_run=False,
    )

    assert row.verdict_kind == "Retry"
    assert len(writer.created) == 1
    assert writer.created[0]["parent_id"] == "t_orig"
    assert len(writer.completed) == 1
    assert writer.completed[0]["card_id"] == "t_orig"
    assert "retry" in writer.completed[0]["result"].lower()


def test_process_card_dry_run_does_not_mutate_kanban(monkeypatch, tmp_path):
    """Dry-run must NOT call create_retry or complete_blocked even when the
    verdict would be Retry/GiveUp/Submit. Shadow mode depends on this."""
    fake_handoff = Handoff(
        raw_text="{}",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.10,
        expected_point_gain=0.10,
        package_sha="a" * 64,
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    writer = _RecordingKanbanWriter()
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    process_card(
        {"id": "t_orig"},
        registry={"tasks": {}},
        current_pass=1,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        kanban_writer=writer,  # writer present...
        dry_run=True,  # ...but dry-run suppresses mutations
    )

    assert writer.created == []
    assert writer.completed == []


def test_process_card_give_up_completes_original_no_retry_card(monkeypatch, tmp_path):
    """3-strike give-up: complete the blocked original, do NOT create a
    new worker card (autoforeman will skip the task next pass)."""
    fake_handoff = Handoff(
        raw_text="{}",
        # No cost reduction on 3rd attempt → GiveUpForPass
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 10000}},
        candidate_id="task100_v3",
        task_id="task100",
        point_delta=0.0,
        expected_point_gain=0.0,
        package_sha="a" * 64,
        algorithm_family="conv-stamp",
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    writer = _RecordingKanbanWriter()
    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 2,
                "attempted_families": ["where-select", "reduce-broadcast"],
                "last_attempt_pass": 1,
            }
        }
    }
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    row = process_card(
        {"id": "t_orig"},
        registry=registry,
        current_pass=1,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        kanban_writer=writer,
        dry_run=False,
    )

    assert row.verdict_kind == "GiveUpForPass"
    # NO new card created (no retry directive on give-up)
    assert writer.created == []
    # But the blocked original IS completed
    assert len(writer.completed) == 1
    assert writer.completed[0]["card_id"] == "t_orig"
    assert "give-up" in writer.completed[0]["result"].lower()


def test_process_card_submit_with_forced_retry_creates_both(monkeypatch, tmp_path):
    """First attempt + good candidate -> Submit(also_retry=True): MUST do
    BOTH the submit AND create a retry card for forced exploration."""
    fake_handoff = Handoff(
        raw_text="{}",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.5,
        expected_point_gain=0.5,
        package_sha="a" * 64,
        algorithm_family="where-select",
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    submit_fn = lambda h, p: SubmitOutcome(
        status="ok",
        record=KaggleSubmitRecord(submitted_at="x", public_score=None, kaggle_ref="r"),
    )

    writer = _RecordingKanbanWriter()
    registry = {"tasks": {}}
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    row = process_card(
        {"id": "t_orig"},
        registry=registry,
        current_pass=1,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        kanban_writer=writer,
        submit_fn=submit_fn,
        dry_run=False,
    )

    assert row.verdict_kind == "Submit"
    assert row.submit_invoked is True
    # Both side effects fired
    assert len(writer.created) == 1
    assert len(writer.completed) == 1
    assert writer.created[0]["parent_id"] == "t_orig"
    assert "forced exploration" in writer.created[0]["body"].lower() or "submitted" in writer.completed[0]["result"].lower()


def test_process_card_stale_lineage_creates_rebase_card(monkeypatch, tmp_path):
    """Stale lineage routes back to worker for rebase. Should create the
    new card (no attempt consumed) and complete the blocked original."""
    from tools.submit_gate import LineageState
    fake_handoff = Handoff(
        raw_text="{}",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.5,
        expected_point_gain=0.5,
        package_sha="a" * 64,
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    writer = _RecordingKanbanWriter()
    registry = {"tasks": {}}
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    row = process_card(
        {"id": "t_orig"},
        registry=registry,
        current_pass=1,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        kanban_writer=writer,
        lineage_loader=lambda h: LineageState(candidate_base_score=5743.97, current_best_score=5751.13),
        dry_run=False,
    )

    assert row.verdict_kind == "DeferredForStaleLineage"
    # Created a rebase card; original is completed
    assert len(writer.created) == 1
    assert "rebase" in writer.completed[0]["result"].lower()
    # Attempts NOT consumed — this is a coordination issue, not a worker bug
    assert registry["tasks"] == {} or registry["tasks"].get("task100", {}).get("attempts_this_pass", 0) == 0


def test_process_card_deferred_for_budget_does_not_mutate_kanban(monkeypatch, tmp_path):
    """Budget exhausted -> no kanban mutation. Card stays blocked for next tick."""
    from tools.submit_gate import BudgetState
    fake_handoff = Handoff(
        raw_text="{}",
        parsed={"scores": {"current_cost": 10000, "candidate_cost": 8000}},
        candidate_id="task100_v1",
        task_id="task100",
        point_delta=0.5,
        expected_point_gain=0.5,
        package_sha="a" * 64,
    )
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: fake_handoff)
    monkeypatch.setattr(
        "tools.submit_gate_runner.run_mechanical_checks",
        lambda inp, **kw: VerifyResult.passed(),
    )

    writer = _RecordingKanbanWriter()
    zip_path = tmp_path / "fake.zip"
    zip_path.write_bytes(b"x")

    row = process_card(
        {"id": "t_orig"},
        registry={"tasks": {}},
        current_pass=1,
        candidate_zip_resolver=lambda c, h: zip_path,
        current_best_zip_path=None,
        kanban_writer=writer,
        budget_loader=lambda: BudgetState(daily_remaining=0, seconds_until_next_slot=3600),
        dry_run=False,
    )

    assert row.verdict_kind == "DeferredForBudget"
    assert writer.created == []
    assert writer.completed == []


def test_build_retry_card_spec_title_under_200_chars():
    """Hermes kanban title limit. Truncation is applied so very long
    candidate_ids don't break card creation."""
    long_handoff = _retry_handoff(candidate_id="x" * 250)
    spec = build_retry_card_spec(
        original_card={"id": "t_aaa"},
        handoff=long_handoff,
        verdict_reason="r",
        directive="d",
        retry_attempt_index=2,
        attempted_families=[],
    )
    assert len(spec["title"]) <= 200
