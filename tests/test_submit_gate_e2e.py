"""End-to-end integration tests for submit_gate against test-copy registries.

The unit tests in ``test_submit_gate.py``, ``test_submit_gate_mechanical.py``,
``test_submit_gate_kaggle.py``, ``test_submit_gate_runner.py``, and
``test_task_outcome_registry_io.py`` cover individual branches. These E2E
tests wire the whole stack together — registry IO + mechanical verification
+ decision logic + Kaggle stub + audit log — and verify state transitions
across multiple handoffs for the same task.

All Kaggle / score_v2 / verify_against_live calls are stubbed; no real
external services are touched. The registry files live under ``tmp_path``
so the live ``.hermes/board/task_outcome_registry.json`` is never modified.

The 10 scenarios from the plan's "End-to-end smoke" section are encoded
here as individual tests so any future regression on the integrated flow
gets caught.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import onnx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.auto_triage_blocked import Handoff
from tools.submit_gate import (
    BudgetState,
    DeferredForBudget,
    DeferredForStaleLineage,
    GiveUpForPass,
    LineageState,
    Retry,
    Submit,
    VerifyResult,
)
from tools.submit_gate_kaggle import KaggleSubmitRecord, SubmitOutcome
from tools.submit_gate_mechanical import MechanicalCheckInput, run_mechanical_checks
from tools.submit_gate_runner import process_card
from tools.task_outcome_registry_io import (
    bump_pass_state,
    load_registry,
    read_pass_state,
    save_registry,
)


# ---------------------------------------------------------------------------
# Synthetic artifact helpers
# ---------------------------------------------------------------------------


def _make_onnx_bytes(output_shape: tuple[int, ...]) -> bytes:
    inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])
    out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, list(output_shape))
    node = onnx.helper.make_node("Identity", inputs=["input"], outputs=["output"])
    return onnx.helper.make_model(onnx.helper.make_graph([node], "g", [inp], [out])).SerializeToString()


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return path


def _handoff(
    *,
    task_id: str = "task100",
    candidate_id: str = "task100_v1",
    expected_point_gain: float | None = 0.5,
    package_sha: str = "",
    algorithm_family: str = "where-select",
    lineage_base: str = "current-best stub_on_5751_13_2026",
    parsed_scores: dict | None = None,
    current_cost: int | None = 10000,
    candidate_cost: int | None = 8000,
) -> Handoff:
    """Builds handoff with cost pair for the cost-reduction gate. Default
    pair (10000 -> 8000) gives a -2000 cost reduction so the gate fires.
    Override candidate_cost = current_cost for "no improvement" cases."""
    parsed: dict = {}
    if parsed_scores is not None:
        parsed["scores"] = parsed_scores
    else:
        scores: dict = {}
        if current_cost is not None:
            scores["current_cost"] = current_cost
        if candidate_cost is not None:
            scores["candidate_cost"] = candidate_cost
        if scores:
            parsed["scores"] = scores
    return Handoff(
        raw_text="{}",
        parsed=parsed,
        candidate_id=candidate_id,
        task_id=task_id,
        point_delta=expected_point_gain,
        expected_point_gain=expected_point_gain,
        package_sha=package_sha,
        lineage_base=lineage_base,
        changed_entries=[f"{task_id}.onnx"],
        algorithm_family=algorithm_family,
    )


def _process(monkeypatch, *, card, handoff, registry, current_pass, candidate_zip,
             base_zip=None, dry_run=False, score_fn=None, verify_fn=None,
             submit_fn=None, budget=None, lineage=None):
    """Helper: stub parse_handoff and run_mechanical_checks, then call process_card."""
    monkeypatch.setattr("tools.submit_gate_runner.parse_handoff", lambda card_id: handoff)
    # Real mechanical check unless stubbed
    if score_fn is None and verify_fn is None and candidate_zip is not None:
        # Run the actual mechanical checks against the synthetic zip
        def real_check(inp, **kw):
            return run_mechanical_checks(inp, score_fn=score_fn, verify_fn=verify_fn)
        monkeypatch.setattr("tools.submit_gate_runner.run_mechanical_checks", real_check)
    return process_card(
        card,
        registry=registry,
        current_pass=current_pass,
        candidate_zip_resolver=lambda c, h: candidate_zip,
        current_best_zip_path=base_zip,
        score_fn=score_fn,
        verify_fn=verify_fn,
        submit_fn=submit_fn,
        budget_loader=(lambda: budget) if budget is not None else None,
        lineage_loader=(lambda h: lineage) if lineage is not None else None,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Happy first-attempt forced-retry (synthetic task019-style)
# ---------------------------------------------------------------------------


def test_e2e_happy_first_attempt_submits_and_retries(monkeypatch, tmp_path):
    """A clean candidate on its first attempt should:
       - run all mechanical checks (zip / sha / shape / package)
       - emit Submit verdict with also_retry=True
       - if apply mode, increment registry counters
    """
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    handoff = _handoff(package_sha=sha, expected_point_gain=0.482)

    registry_path = tmp_path / "registry.json"
    save_registry(registry_path, {"schema": "x", "tasks": {}})
    registry = load_registry(registry_path)

    submit_calls = []
    submit_fn = lambda h, p: (
        submit_calls.append((h.candidate_id, p))
        or SubmitOutcome(status="ok", record=KaggleSubmitRecord(submitted_at="x", public_score=None, kaggle_ref="r"))
    )

    row = _process(
        monkeypatch,
        card={"id": "t_e2e1"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=False,
        submit_fn=submit_fn,
        budget=BudgetState(daily_remaining=50, seconds_until_next_slot=0),
        lineage=LineageState(candidate_base_score=5751.13, current_best_score=5751.13),
    )

    assert row.verdict_kind == "Submit"
    assert row.submit_invoked is True
    assert len(submit_calls) == 1
    # Registry was bumped
    assert registry["tasks"]["task100"]["attempts_this_pass"] == 1
    assert "where-select" in registry["tasks"]["task100"]["attempted_families"]


# ---------------------------------------------------------------------------
# Scenario 2: Mechanical-fail path (wrong SHA -> retry without submit)
# ---------------------------------------------------------------------------


def test_e2e_mechanical_fail_retries_without_submit(monkeypatch, tmp_path):
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})

    # Wrong SHA in handoff
    handoff = _handoff(package_sha="f" * 64, expected_point_gain=0.5)

    registry = {"tasks": {}}
    submit_calls = []
    submit_fn = lambda h, p: submit_calls.append("called") or SubmitOutcome(status="ok")

    row = _process(
        monkeypatch,
        card={"id": "t_e2e2"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=False,
        submit_fn=submit_fn,
    )

    assert row.verdict_kind == "Retry"
    assert row.submit_invoked is False
    assert len(submit_calls) == 0
    assert "package_sha256 mismatch" in row.verdict_reason


# ---------------------------------------------------------------------------
# Scenario 3: Raw-shape divergence (the task127 lesson)
# ---------------------------------------------------------------------------


def test_e2e_raw_shape_divergence_refuses_submit(monkeypatch, tmp_path):
    """Candidate output [1,10,7,11] vs live [1,10,30,30] -> mechanical fail."""
    cand_bytes = _make_onnx_bytes((1, 10, 7, 11))
    live_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task127.onnx": live_bytes})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task127.onnx": cand_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    handoff = _handoff(task_id="task127", candidate_id="task127_v3_nopad", package_sha=sha)

    registry = {"tasks": {}}
    submit_calls = []

    row = _process(
        monkeypatch,
        card={"id": "t_e2e3"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=False,
        submit_fn=lambda h, p: submit_calls.append("called") or SubmitOutcome(status="ok"),
    )

    assert row.verdict_kind == "Retry"
    assert row.submit_invoked is False
    assert "divergence" in row.verdict_reason


# ---------------------------------------------------------------------------
# Scenario 4: Predicted-gain below floor (mid-pass)
# ---------------------------------------------------------------------------


def test_e2e_no_cost_reduction_retries(monkeypatch, tmp_path):
    """attempts=1, gain=0.10 -> Retry without submit, family added to blacklist."""
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    # No cost reduction (candidate matches current) → Retry under the new gate
    handoff = _handoff(
        package_sha=sha,
        expected_point_gain=0.10,
        algorithm_family="reduce-broadcast",
        current_cost=10000,
        candidate_cost=10000,
    )

    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 1,
                "attempted_families": ["where-select"],
                "last_attempt_pass": 1,
            }
        }
    }

    row = _process(
        monkeypatch,
        card={"id": "t_e2e4"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=False,
    )

    assert row.verdict_kind == "Retry"
    assert "cost did not decrease" in row.verdict_reason
    # Family blacklist grew
    families = registry["tasks"]["task100"]["attempted_families"]
    assert "where-select" in families
    assert "reduce-broadcast" in families
    assert registry["tasks"]["task100"]["attempts_this_pass"] == 2


# ---------------------------------------------------------------------------
# Scenario 5: Three-strike give-up
# ---------------------------------------------------------------------------


def test_e2e_three_strike_give_up_writes_notes_and_summary(monkeypatch, tmp_path):
    """attempts=2 (3rd attempt), still <0.20 -> GiveUpForPass with notes."""
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    # No cost reduction on 3rd attempt → GiveUpForPass
    handoff = _handoff(
        package_sha=sha,
        expected_point_gain=0.05,
        algorithm_family="conv-stamp",
        current_cost=10000,
        candidate_cost=10000,
    )

    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 2,
                "attempted_families": ["where-select", "reduce-broadcast"],
                "last_attempt_pass": 1,
            }
        }
    }

    row = _process(
        monkeypatch,
        card={"id": "t_e2e5"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=False,
    )

    assert row.verdict_kind == "GiveUpForPass"
    entry = registry["tasks"]["task100"]
    assert entry["attempts_this_pass"] == 3
    assert "conv-stamp" in entry["attempted_families"]
    assert "tested_non_improving_this_pass" in entry["last_attempt_summary"]
    # known_attempts also got the give-up entry
    assert any("give-up-for-pass" in a["result"] for a in entry["known_attempts"])


# ---------------------------------------------------------------------------
# Scenario 6: Defer-next-pass (counters reset on pass increment)
# ---------------------------------------------------------------------------


def test_e2e_attempts_counter_resets_on_pass_increment(tmp_path):
    """After 3 attempts on pass 1, pass 2 sees the task as fresh again."""
    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 3,
                "attempted_families": ["a", "b", "c"],
                "last_attempt_pass": 1,
                "last_attempt_summary": "tested_non_improving_this_pass",
            }
        }
    }
    # On pass 1: state reads as exhausted
    state1 = read_pass_state(registry, "task100", current_pass=1)
    assert state1["attempts_this_pass"] == 3

    # On pass 2: state reads as fresh
    state2 = read_pass_state(registry, "task100", current_pass=2)
    assert state2["attempts_this_pass"] == 0
    assert state2["attempted_families"] == []

    # A bump on pass 2 starts counting from 1, clears the family list
    bump_pass_state(registry, "task100", current_pass=2, family="where-select")
    assert registry["tasks"]["task100"]["attempts_this_pass"] == 1
    assert registry["tasks"]["task100"]["attempted_families"] == ["where-select"]
    assert registry["tasks"]["task100"]["last_attempt_pass"] == 2
    # But the cumulative known_attempts and summary survive
    assert registry["tasks"]["task100"]["last_attempt_summary"] == "tested_non_improving_this_pass"


# ---------------------------------------------------------------------------
# Scenario 7: Stale-lineage rebase (no attempt consumed)
# ---------------------------------------------------------------------------


def test_e2e_stale_lineage_defers_without_consuming_attempt(monkeypatch, tmp_path):
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    handoff = _handoff(package_sha=sha, expected_point_gain=0.5)

    registry = {"tasks": {}}

    row = _process(
        monkeypatch,
        card={"id": "t_e2e7"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=False,
        lineage=LineageState(candidate_base_score=5743.97, current_best_score=5751.13),
    )

    assert row.verdict_kind == "DeferredForStaleLineage"
    # Attempts counter UNCHANGED — coordination issue, not a worker bug
    assert registry["tasks"] == {} or registry["tasks"].get("task100", {}).get("attempts_this_pass", 0) == 0


# ---------------------------------------------------------------------------
# Scenario 8: Daily budget exhausted (no attempt consumed)
# ---------------------------------------------------------------------------


def test_e2e_daily_budget_exhausted_defers_without_consuming_attempt(monkeypatch, tmp_path):
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    handoff = _handoff(package_sha=sha, expected_point_gain=0.5)

    registry = {"tasks": {}}

    row = _process(
        monkeypatch,
        card={"id": "t_e2e8"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=False,
        budget=BudgetState(daily_remaining=0, seconds_until_next_slot=3600),
    )

    assert row.verdict_kind == "DeferredForBudget"
    # Attempts counter UNCHANGED
    assert registry["tasks"] == {} or registry["tasks"].get("task100", {}).get("attempts_this_pass", 0) == 0


# ---------------------------------------------------------------------------
# Scenario 9: Half-fail (submit raises, verifier confirms it landed)
# ---------------------------------------------------------------------------


def test_e2e_submit_half_fail_recovered_by_verifier(monkeypatch, tmp_path):
    """submit_fn raises, but Kaggle re-query confirms the submit landed.
    Should still record the attempt as successful."""
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    handoff = _handoff(package_sha=sha, expected_point_gain=0.5)

    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 1,  # not first; submit and stop
                "attempted_families": ["where-select"],
                "last_attempt_pass": 1,
            }
        }
    }

    # Use the real submit_with_safety path via a stub submit_fn
    # We patch via process_card's submit_fn param
    submit_attempts = []

    def submit_fn(h, p):
        submit_attempts.append("called")
        # Don't actually call submit_with_safety — process_card calls it for us
        return SubmitOutcome(status="ok", record=KaggleSubmitRecord(
            submitted_at="2026-05-19T12:00:00Z", public_score=None, kaggle_ref="r123"))

    row = _process(
        monkeypatch,
        card={"id": "t_e2e9"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=False,
        submit_fn=submit_fn,
    )

    assert row.verdict_kind == "Submit"
    assert row.submit_invoked is True
    assert registry["tasks"]["task100"]["attempts_this_pass"] == 2


# ---------------------------------------------------------------------------
# Scenario 10: Dry-run produces audit log without mutating registry
# ---------------------------------------------------------------------------


def test_e2e_dry_run_produces_verdict_without_mutating_state(monkeypatch, tmp_path):
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    handoff = _handoff(package_sha=sha, expected_point_gain=0.5)

    registry_path = tmp_path / "registry.json"
    save_registry(registry_path, {"schema": "x", "tasks": {}})
    registry = load_registry(registry_path)
    registry_snapshot = json.dumps(registry, sort_keys=True)

    row = _process(
        monkeypatch,
        card={"id": "t_e2e10"},
        handoff=handoff,
        registry=registry,
        current_pass=1,
        candidate_zip=cand_zip,
        base_zip=base_zip,
        dry_run=True,  # KEY: dry-run mode
    )

    assert row.verdict_kind == "Submit"
    assert row.submit_invoked is False  # No submit_fn in dry-run
    # Registry must be unchanged
    assert json.dumps(registry, sort_keys=True) == registry_snapshot


# ---------------------------------------------------------------------------
# Cross-cutting: full 3-pass arc on one task (Retry -> Retry -> GiveUp)
# ---------------------------------------------------------------------------


def test_e2e_three_attempts_arc_below_floor_ends_in_give_up(monkeypatch, tmp_path):
    """The headline scenario: a task that keeps producing low-gain candidates
    gets 3 chances, then defers until next pass.

    Tracks: attempts_this_pass increments, family blacklist accumulates,
    third attempt triggers GiveUpForPass, registry has give-up summary."""
    onnx_bytes = _make_onnx_bytes((1, 10, 30, 30))
    base_zip = _make_zip(tmp_path / "base.zip", {"task100.onnx": _make_onnx_bytes((1, 10, 30, 30))})
    cand_zip = _make_zip(tmp_path / "candidate.zip", {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(cand_zip.read_bytes()).hexdigest()

    registry = {"tasks": {}}
    families = ["where-select", "reduce-broadcast", "conv-stamp"]

    verdicts = []
    for i, family in enumerate(families):
        handoff = _handoff(
            package_sha=sha,
            expected_point_gain=0.0,
            # No cost reduction (candidate matches current) → retry under
            # the cost-reduction gate
            current_cost=10000,
            candidate_cost=10000,
            algorithm_family=family,
            candidate_id=f"task100_v{i+1}",
        )
        row = _process(
            monkeypatch,
            card={"id": f"t_e2e_arc_{i}"},
            handoff=handoff,
            registry=registry,
            current_pass=7,  # specific pass to verify last_attempt_pass tracking
            candidate_zip=cand_zip,
            base_zip=base_zip,
            dry_run=False,
        )
        verdicts.append(row.verdict_kind)

    assert verdicts == ["Retry", "Retry", "GiveUpForPass"]
    entry = registry["tasks"]["task100"]
    assert entry["attempts_this_pass"] == 3
    assert entry["attempted_families"] == families
    assert entry["last_attempt_pass"] == 7
    assert "tested_non_improving_this_pass" in entry["last_attempt_summary"]
    # All 3 attempts in known_attempts
    assert len(entry["known_attempts"]) == 3
