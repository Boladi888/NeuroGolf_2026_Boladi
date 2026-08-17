"""Tests for the submit-stack picker.

The stack picker is the biggest LLM-time saver in the new pipeline —
submit-captain used to spend 11 minutes choosing a stack from scratch
every card. The picker writes a JSON recommendation so the agent can
read-and-verify in ~30 seconds. Tests focus on:

  1. Stack-compatibility: greedy pick respects disjoint changed_task_ids.
  2. EV ordering: top-EV-first iteration.
  3. Stack size cap honored.
  4. Hidden-bad / failed-SHA filtering (handled by cache, picker just
     uses ``safe_to_submit``).
  5. Package validation: existence + zip format + entry count + changed
     entries present.
  6. Lineage consistency reporting.
  7. Output schema stability so submit-captain knows what to read.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tools.neurogolf_stack_picker import (
    SCHEMA_VERSION,
    build_recommendation,
    pick_stack,
    validate_package_path,
)


def _cand(
    cid,
    changed=("task001",),
    ev=1.0,
    package_path="",
    lineage_base="current-best",
    safe=True,
    package_sha256="",
):
    return {
        "candidate_id": cid,
        "task_id": changed[0] if changed else "",
        "queue_id": f"submit-{cid}",
        "status": "ready",
        "ev": ev,
        "changed_task_ids": list(changed),
        "package_sha256": package_sha256,
        "package_path": package_path,
        "lineage_base_candidate_id": lineage_base,
        "lineage_base_score": 5750.0,
        "reviewer_disposition": "approved-for-submit-captain",
        "approval_mode": "auto-rate-limited",
        "worker_owner": "neurogolf-worker-8",
        "return_worker": "neurogolf-worker-8",
        "risk_flags": [],
        "hidden_bad_task_hit": [],
        "failed_sha_hit": False,
        "safe_to_submit": safe,
    }


def _cache(*ready, current_best=None):
    return {
        "schema": "neurogolf.submit_picker_cache.v1",
        "computed_at": "2026-05-18T05:00:00Z",
        "current_best": current_best or {"score": 5755.05, "candidate_id": "task013_v_w8a"},
        "hidden_bad_task_ids": [],
        "failed_public_negative_shas": [],
        "ready_candidates_by_ev": list(ready),
        "ready_count": len(ready),
        "safe_ready_count": sum(1 for r in ready if r.get("safe_to_submit")),
        "blocked_ready_count": sum(1 for r in ready if not r.get("safe_to_submit")),
        "top_safe_ev": max((r["ev"] for r in ready if r.get("safe_to_submit")), default=0.0),
        "total_safe_ev": sum(r["ev"] for r in ready if r.get("safe_to_submit")),
        "task_outcome_registry_updated_at": None,
    }


@pytest.fixture
def good_zip(tmp_path: Path) -> Path:
    """A submission.zip with exactly 401 task ONNX entries (task000..task400),
    matching the canonical competition submission shape."""
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(401):
            z.writestr(f"task{i:03d}.onnx", b"\x00" * 100)
    return path


# --- pick_stack ---------------------------------------------------------------


def test_pick_stack_picks_top_ev_first():
    # Input in arbitrary order — picker sorts by EV descending defensively.
    cands = [
        _cand("a", ev=0.5, changed=("task001",)),
        _cand("b", ev=1.5, changed=("task002",)),
        _cand("c", ev=1.0, changed=("task003",)),
    ]
    selected, _ = pick_stack(cands)
    assert [c["candidate_id"] for c in selected] == ["b", "c", "a"]  # EV: 1.5, 1.0, 0.5


def test_pick_stack_excludes_overlapping_changed_tasks():
    cands = [
        _cand("first", ev=2.0, changed=("task004",)),
        _cand("conflict_with_first", ev=1.0, changed=("task004",)),  # same task → skipped
        _cand("compatible", ev=0.5, changed=("task008",)),
    ]
    selected, skipped = pick_stack(cands)
    selected_ids = [c["candidate_id"] for c in selected]
    assert "first" in selected_ids
    assert "compatible" in selected_ids
    assert "conflict_with_first" not in selected_ids
    assert any(s["candidate_id"] == "conflict_with_first" for s in skipped)
    assert "overlap" in skipped[0]["skip_reason"]


def test_pick_stack_respects_max_stack_size():
    cands = [_cand(f"c{i}", ev=1.0, changed=(f"task{i:03d}",)) for i in range(10)]
    selected, _ = pick_stack(cands, max_stack_size=3)
    assert len(selected) == 3


def test_pick_stack_skips_empty_changed_task_ids():
    cands = [
        _cand("normal", ev=1.0, changed=("task001",)),
        _cand("legacy_empty", ev=2.0, changed=()),  # higher EV but no changed_tasks → skipped
    ]
    selected, skipped = pick_stack(cands)
    assert [c["candidate_id"] for c in selected] == ["normal"]
    assert any(s["candidate_id"] == "legacy_empty" for s in skipped)


def test_pick_stack_refuses_multi_lineage_by_default():
    """Bug fix 2026-05-18:

    The picker's previous greedy algorithm built cross-lineage stacks
    whenever the task_id sets were disjoint, then flagged
    lineage_consistent=False as an advisory warning. The captain submitted
    one anyway and lost -26.45 LB. New default: pick the highest-EV
    same-lineage group. Multi-lineage stacks only with explicit override.
    """
    cands = [
        _cand("a_5750", ev=2.0, changed=("task004",), lineage_base="base_5750"),
        _cand("b_5750", ev=1.5, changed=("task008",), lineage_base="base_5750"),
        _cand("c_5755", ev=3.0, changed=("task011",), lineage_base="base_5755"),
        _cand("d_5755", ev=0.1, changed=("task022",), lineage_base="base_5755"),
    ]
    # base_5755 group: c (3.0) + d (0.1) = 3.1 EV
    # base_5750 group: a (2.0) + b (1.5) = 3.5 EV — wins
    selected, skipped = pick_stack(cands)
    ids = [c["candidate_id"] for c in selected]
    assert ids == ["a_5750", "b_5750"]
    # All selected share base_5750
    bases = {c.get("lineage_base_candidate_id") for c in selected}
    assert bases == {"base_5750"}
    # Cross-lineage candidates are explicitly skipped with the rationale
    skip_reasons = [s.get("skip_reason", "") for s in skipped]
    assert any("different lineage_base" in r for r in skip_reasons)


def test_pick_stack_picks_highest_ev_lineage_group():
    """When two groups exist, the picker chooses the one with the
    highest combined EV — even if the individual top candidate is in
    a smaller group."""
    cands = [
        _cand("big_a", ev=5.0, changed=("task001",), lineage_base="lonely"),
        _cand("med_a", ev=2.0, changed=("task002",), lineage_base="shared"),
        _cand("med_b", ev=2.0, changed=("task003",), lineage_base="shared"),
        _cand("med_c", ev=2.0, changed=("task004",), lineage_base="shared"),
    ]
    # lonely: 5.0 (1 candidate); shared: 2.0+2.0+2.0 = 6.0 (3 candidates)
    selected, _ = pick_stack(cands, max_stack_size=3)
    ids = [c["candidate_id"] for c in selected]
    assert sorted(ids) == ["med_a", "med_b", "med_c"]


def test_pick_stack_allow_multi_lineage_flag_restores_legacy_behavior():
    """Opt-out for callers who really want the old cross-lineage greedy
    pick (e.g., emergency batched submits when single-lineage groups are
    empty)."""
    cands = [
        _cand("hi", ev=3.0, changed=("task001",), lineage_base="base_a"),
        _cand("lo", ev=0.1, changed=("task002",), lineage_base="base_b"),
    ]
    selected, _ = pick_stack(cands, require_single_lineage=False)
    ids = [c["candidate_id"] for c in selected]
    # Without single-lineage requirement: both selected, top EV first.
    assert ids == ["hi", "lo"]


def test_pick_stack_with_multi_task_candidates_respects_full_set():
    """A 2-task candidate must block ANY other candidate touching either task."""
    cands = [
        _cand("multi", ev=1.5, changed=("task010", "task011")),
        _cand("c10", ev=1.0, changed=("task010",)),  # overlaps multi
        _cand("c11", ev=0.9, changed=("task011",)),  # overlaps multi
        _cand("c20", ev=0.8, changed=("task020",)),  # ok
    ]
    selected, _ = pick_stack(cands)
    ids = [c["candidate_id"] for c in selected]
    assert "multi" in ids
    assert "c20" in ids
    assert "c10" not in ids
    assert "c11" not in ids


# --- validate_package_path ---------------------------------------------------


def test_validate_package_path_returns_ok_for_good_zip(good_zip: Path):
    result = validate_package_path(str(good_zip), ["task004"])
    assert result["ok"] is True
    assert result["exists"] is True
    assert result["is_zip"] is True
    assert result["entry_count"] == 401
    assert result["missing_changed_entries"] == []


def test_validate_package_path_flags_missing_file(tmp_path: Path):
    result = validate_package_path(str(tmp_path / "ghost.zip"), ["task004"])
    assert result["ok"] is False
    assert result["exists"] is False
    assert "does not exist" in result["messages"][0]


def test_validate_package_path_flags_wrong_entry_count(tmp_path: Path):
    path = tmp_path / "small.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(10):
            z.writestr(f"task{i:03d}.onnx", b"\x00")
    result = validate_package_path(str(path), ["task004"])
    assert result["ok"] is False
    assert result["entry_count"] == 10
    assert any("entry count" in m for m in result["messages"])


def test_validate_package_path_flags_missing_changed_entries(good_zip: Path):
    # task500 isn't in the zip (we only put 0..400). The package validation
    # should flag it as a missing changed entry.
    result = validate_package_path(str(good_zip), ["task500"])
    assert result["ok"] is False
    assert result["missing_changed_entries"] == ["task500.onnx"]


def test_validate_package_path_flags_bad_zip(tmp_path: Path):
    bad = tmp_path / "not_a_zip.zip"
    bad.write_bytes(b"random garbage")
    result = validate_package_path(str(bad), ["task004"])
    assert result["ok"] is False
    assert result["is_zip"] is False


# --- build_recommendation: end-to-end ---------------------------------------


def test_build_recommendation_with_clean_stack(good_zip: Path):
    cache = _cache(
        _cand("c11", ev=1.04, changed=("task011",), package_path=str(good_zip), lineage_base="cb_v1"),
        _cand("c22", ev=1.65, changed=("task022",), package_path=str(good_zip), lineage_base="cb_v1"),
        _cand("c04", ev=0.17, changed=("task004",), package_path=str(good_zip), lineage_base="cb_v1"),
    )
    rec = build_recommendation(cache, max_stack_size=3)
    assert rec["schema"] == SCHEMA_VERSION
    r = rec["recommendation"]
    # Sorted by EV descending: c22 > c11 > c04
    assert r["stack"] == ["c22", "c11", "c04"]
    assert r["stack_size"] == 3
    assert r["expected_combined_gain"] == pytest.approx(2.86, abs=0.001)
    assert r["lineage_consistent"] is True
    assert r["package_validation_pass"] is True


def test_build_recommendation_skips_unsafe_candidates(good_zip: Path):
    cache = _cache(
        _cand("hidden_bad", ev=5.0, changed=("task004",), package_path=str(good_zip), safe=False),
        _cand("clean", ev=1.0, changed=("task005",), package_path=str(good_zip)),
    )
    rec = build_recommendation(cache)
    assert rec["recommendation"]["stack"] == ["clean"]
    assert len(rec["blocked_for_hidden_bad_or_failed_sha"]) == 1


def test_build_recommendation_default_picks_single_lineage_only(good_zip: Path):
    """With require_single_lineage=True (new default), the recommendation
    contains a single-lineage stack — even if a higher cross-lineage stack
    is theoretically possible. The from_b lineage is skipped because
    from_a has higher per-group EV."""
    cache = _cache(
        _cand("from_a", ev=1.0, changed=("task011",), package_path=str(good_zip), lineage_base="base_a"),
        _cand("from_b", ev=0.5, changed=("task022",), package_path=str(good_zip), lineage_base="base_b"),
    )
    rec = build_recommendation(cache)
    # Single-lineage default keeps only the winning group.
    assert rec["recommendation"]["lineage_consistent"] is True
    assert rec["recommendation"]["stack"] == ["from_a"]


def test_build_recommendation_with_multi_lineage_override_keeps_cross_stack(good_zip: Path):
    """The override flag restores the legacy behavior so callers can still
    build cross-lineage stacks when they accept the rebase risk."""
    cache = _cache(
        _cand("from_a", ev=1.0, changed=("task011",), package_path=str(good_zip), lineage_base="base_a"),
        _cand("from_b", ev=0.5, changed=("task022",), package_path=str(good_zip), lineage_base="base_b"),
    )
    rec = build_recommendation(cache, require_single_lineage=False)
    # Both candidates make the stack.
    assert sorted(rec["recommendation"]["stack"]) == ["from_a", "from_b"]
    # lineage_consistent reflects reality: False.
    assert rec["recommendation"]["lineage_consistent"] is False
    assert "different lineage_base" in rec["recommendation"]["rationale"]


def test_build_recommendation_handles_empty_safe_set():
    cache = _cache(
        _cand("only_unsafe", ev=5.0, changed=("task001",), safe=False),
    )
    rec = build_recommendation(cache, validate_packages=False)
    r = rec["recommendation"]
    assert r["stack"] == []
    assert r["stack_size"] == 0
    assert r["expected_combined_gain"] == 0
    assert "No safe candidates" in r["rationale"]


def test_build_recommendation_skips_package_validation_when_disabled():
    cache = _cache(_cand("a", ev=1.0, changed=("task011",), package_path="/missing/path.zip"))
    rec = build_recommendation(cache, validate_packages=False)
    # validate_packages=False → package_validation_pass should be True
    # because no validation was attempted.
    assert rec["recommendation"]["package_validation_pass"] is True
    assert rec["package_validations"] == {}


def test_build_recommendation_falls_back_to_evidence_in_rationale(good_zip: Path):
    cache = _cache(
        _cand("good", ev=1.0, changed=("task011",), package_path=str(good_zip)),
    )
    # Add some blocked entries.
    cache["ready_candidates_by_ev"].append(
        _cand("blocked", ev=2.0, changed=("task004",), package_path=str(good_zip), safe=False)
    )
    cache["blocked_ready_count"] = 1
    rec = build_recommendation(cache)
    assert "1 ready candidate(s) excluded" in rec["recommendation"]["rationale"]


def test_build_recommendation_schema_is_stable():
    """Submit-captain reads the recommendation by these keys. Lock them down."""
    cache = _cache(_cand("a", ev=1.0, changed=("task011",), package_path=""))
    rec = build_recommendation(cache, validate_packages=False)
    required = {
        "schema", "computed_at", "based_on_cache_at", "current_best",
        "recommendation", "selected_candidates", "package_validations",
        "blocked_for_hidden_bad_or_failed_sha", "skipped_due_to_stack_overlap",
        "max_stack_size",
    }
    assert required <= set(rec.keys())
    rec_required = {
        "stack", "stack_size", "expected_combined_gain", "changed_task_ids",
        "lineage_bases", "lineage_consistent", "package_validation_pass", "rationale",
    }
    assert rec_required <= set(rec["recommendation"].keys())
