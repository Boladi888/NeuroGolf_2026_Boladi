"""Tests for tools/submit_gate.py::classify_handoff (pure decision logic).

Every branch in the decision tree gets exercised here without any kanban /
Kaggle / registry side effects. The IO-bearing layers (mechanical checks,
Kaggle submit, atomic registry writes) are tested in their own test files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    PassState,
    Retry,
    Submit,
    VerifyResult,
    classify_handoff,
    normalized_family_for_handoff,
)


# ---------------------------------------------------------------------------
# Test fixtures / builders
# ---------------------------------------------------------------------------


def _handoff(
    *,
    task_id: str = "task100",
    candidate_id: str = "task100_v1",
    expected_point_gain: float | None = 0.5,
    algorithm_family: str = "where-select",
    package_sha: str = "a" * 64,
    changed_entries: list[str] | None = None,
    current_cost: int | None = 10000,
    candidate_cost: int | None = 8000,
) -> Handoff:
    """Build a Handoff with the fields submit_gate actually reads.

    Default cost pair gives a ``-2000`` cost reduction so the gate fires.
    Tests that want the gate to NOT fire set ``candidate_cost >= current_cost``.
    Tests that need cost data missing pass ``current_cost=None`` (etc).
    """
    parsed: dict[str, Any] = {}
    if current_cost is not None or candidate_cost is not None:
        parsed["scores"] = {}
        if current_cost is not None:
            parsed["scores"]["current_cost"] = current_cost
        if candidate_cost is not None:
            parsed["scores"]["candidate_cost"] = candidate_cost
    return Handoff(
        raw_text="{}",
        parsed=parsed,
        candidate_id=candidate_id,
        task_id=task_id,
        point_delta=expected_point_gain,
        expected_point_gain=expected_point_gain,
        package_sha=package_sha,
        lineage_base="current-best stub",
        changed_entries=changed_entries or [f"{task_id}.onnx"],
        algorithm_family=algorithm_family,
    )


def _pass_state(*, attempts: int = 0, families: list[str] | None = None, pass_num: int = 1) -> PassState:
    return PassState(
        attempts_this_pass=attempts,
        attempted_families=families or [],
        current_pass=pass_num,
    )


# ---------------------------------------------------------------------------
# Happy path: clean candidate, first attempt → Submit + Retry
# ---------------------------------------------------------------------------


def test_first_attempt_clean_candidate_submits_and_retries():
    """The forced-exploration rule: first attempt this pass with a clean
    candidate should both submit AND queue a retry with a new-family directive."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Submit)
    assert verdict.also_retry is True
    assert verdict.consumes_attempt is True
    assert "0.5000" in verdict.reason
    assert "forced exploration" in verdict.reason.lower()
    assert verdict.retry_directive  # non-empty directive text


def test_second_attempt_clean_candidate_submits_without_retry():
    """attempts_this_pass=1 means this is the 2nd attempt — submit and stop."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1, families=["where-select"]),
    )
    assert isinstance(verdict, Submit)
    assert verdict.also_retry is False
    assert verdict.consumes_attempt is True


def test_third_attempt_clean_candidate_submits_without_retry():
    """attempts_this_pass=2 means this is the 3rd (last) attempt — submit and stop.
    Even though it's the last attempt, this branch doesn't trigger GiveUpForPass
    because the candidate is GOOD; we just stop trying more for this pass."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=2, families=["where-select", "reduce-broadcast"]),
    )
    assert isinstance(verdict, Submit)
    assert verdict.also_retry is False


# ---------------------------------------------------------------------------
# No candidate produced (iteration-budget / no-fix / missing fields)
# ---------------------------------------------------------------------------


def test_no_candidate_skips_submit_and_retries():
    """Worker handoff with no candidate_id — count as attempt, no submit."""
    verdict = classify_handoff(
        _handoff(candidate_id="", expected_point_gain=None),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "no candidate" in verdict.reason
    assert verdict.consumes_attempt is True
    assert verdict.directive  # has a directive for the worker


def test_no_candidate_id_but_gain_present_still_retries():
    """Defensive: workers might leave candidate_id blank while reporting a
    predicted gain (sloppy partial handoff). Treat as no-candidate."""
    verdict = classify_handoff(
        _handoff(candidate_id="", expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "no candidate" in verdict.reason


def test_iteration_budget_exhausted_skips_submit_and_retries():
    """Workers that hit max_turns before producing artifacts: typically have
    no candidate_id AND no predicted gain. The no-candidate path catches it."""
    verdict = classify_handoff(
        _handoff(candidate_id="", expected_point_gain=None),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1),
    )
    assert isinstance(verdict, Retry)


# ---------------------------------------------------------------------------
# Mechanical verification failures
# ---------------------------------------------------------------------------


def test_mechanical_fail_skips_submit_and_retries():
    """SHA mismatch, score_v2 doesn't reproduce, etc. — worker bug, retry."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.failed("package_sha256 mismatch"),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "mechanical-fail" in verdict.reason
    assert "package_sha256 mismatch" in verdict.reason
    assert "Fix the safety violation" in verdict.directive


def test_mechanical_fail_multiple_reasons_joined():
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.failed("SHA mismatch", "score_v2 disagreement", "verify_against_live 100/267"),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "SHA mismatch" in verdict.reason
    assert "score_v2 disagreement" in verdict.reason
    assert "verify_against_live 100/267" in verdict.reason


def test_mechanical_fail_with_empty_failure_list_does_not_crash():
    """Edge case: verify reports ok=False but no specific failures. Don't
    crash; treat as a generic mechanical fail."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult(ok=False, failures=[]),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "mechanical-fail" in verdict.reason


# ---------------------------------------------------------------------------
# Cost-reduction gate (replaces the legacy predicted-gain floor 2026-05-19)
# ---------------------------------------------------------------------------


def test_no_cost_reduction_skips_submit_and_retries():
    """candidate_cost == current_cost → no improvement, retry without submit."""
    verdict = classify_handoff(
        _handoff(current_cost=10000, candidate_cost=10000, expected_point_gain=0.0),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "cost did not decrease" in verdict.reason
    assert "delta=+0" in verdict.reason


def test_tiny_cost_reduction_submits():
    """Even a 1-unit cost reduction is worth submitting under the new gate."""
    verdict = classify_handoff(
        _handoff(current_cost=10000, candidate_cost=9999, expected_point_gain=0.0001),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1),
    )
    assert isinstance(verdict, Submit)


def test_zero_predicted_gain_but_cost_reduced_still_submits():
    """The headline new behavior: predicted_gain reported as 0 (or tiny)
    but candidate_cost < current_cost → SUBMIT. Old gate would have retried."""
    verdict = classify_handoff(
        _handoff(current_cost=10000, candidate_cost=9990, expected_point_gain=0.0),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1, families=["where-select"]),
    )
    assert isinstance(verdict, Submit)


def test_cost_regression_skips_submit():
    """candidate_cost > current_cost (worse candidate) → retry, no submit."""
    verdict = classify_handoff(
        _handoff(current_cost=10000, candidate_cost=11000, expected_point_gain=-0.3),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "cost did not decrease" in verdict.reason
    assert "delta=+1000" in verdict.reason


def test_missing_current_cost_treated_as_no_reduction():
    """Defensive: handoff missing current_cost → retry (can't compute delta)."""
    verdict = classify_handoff(
        _handoff(current_cost=None, candidate_cost=8000, expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1),
    )
    # The handoff lacks current_cost so we can't compute the cost delta.
    # The decision tree's "no candidate produced" check fires first since
    # cost data is partial. Either way: not a Submit.
    assert not isinstance(verdict, Submit)


def test_missing_candidate_cost_treated_as_no_reduction():
    verdict = classify_handoff(
        _handoff(current_cost=10000, candidate_cost=None, expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1),
    )
    assert not isinstance(verdict, Submit)


# ---------------------------------------------------------------------------
# Three-strike give-up paths
# ---------------------------------------------------------------------------


def test_third_attempt_with_no_candidate_writes_notes_and_gives_up():
    verdict = classify_handoff(
        _handoff(candidate_id="", expected_point_gain=None),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=2, families=["a", "b"]),
    )
    assert isinstance(verdict, GiveUpForPass)
    assert "3 attempts exhausted" in verdict.reason
    assert "tested_non_improving_this_pass" in verdict.notes_summary
    assert verdict.last_attempt_reason == "worker produced no candidate"


def test_third_attempt_no_cost_reduction_writes_notes_and_gives_up():
    """3rd attempt, candidate matches current cost (no improvement) → GiveUp."""
    verdict = classify_handoff(
        _handoff(current_cost=10000, candidate_cost=10000, expected_point_gain=0.0),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=2, families=["a", "b"]),
    )
    assert isinstance(verdict, GiveUpForPass)
    assert "cost did not decrease" in verdict.last_attempt_reason


def test_third_attempt_mechanical_fail_writes_notes_and_gives_up():
    """Extra coverage: the GiveUpForPass path also fires when the 3rd attempt
    has a mechanical issue. Same terminal verdict regardless of which check
    failed."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.failed("SHA mismatch"),
        pass_state=_pass_state(attempts=2, families=["a", "b"]),
    )
    assert isinstance(verdict, GiveUpForPass)
    assert "mechanical-fail" in verdict.last_attempt_reason


def test_custom_max_attempts_respected():
    """Some test scenarios may want a 5-attempt budget; submit_gate respects it."""
    verdict = classify_handoff(
        _handoff(current_cost=10000, candidate_cost=10000, expected_point_gain=0.0),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=2, families=["a", "b"]),
        max_attempts_per_pass=5,
    )
    assert isinstance(verdict, Retry)  # not GiveUp, because budget is 5


# ---------------------------------------------------------------------------
# Lineage staleness (no attempt consumed)
# ---------------------------------------------------------------------------


def test_stale_lineage_defers_without_consuming_attempt():
    """The worker started building when current-best was 5743.97, but it
    advanced to 5751.13 mid-build. Route back with new-base directive,
    DON'T consume an attempt — this is a coordination issue, not a worker bug."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
        lineage=LineageState(candidate_base_score=5743.97, current_best_score=5751.13),
    )
    assert isinstance(verdict, DeferredForStaleLineage)
    assert verdict.consumes_attempt is False
    assert "stale" in verdict.reason.lower()
    assert verdict.new_base_score == 5751.13
    assert "rebase" in verdict.directive.lower()


def test_matching_lineage_proceeds_to_submit():
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1),
        lineage=LineageState(candidate_base_score=5751.13, current_best_score=5751.13),
    )
    assert isinstance(verdict, Submit)


def test_unknown_lineage_proceeds_to_submit():
    """If we don't know the candidate's base score, don't block on it —
    log it and proceed. (Defensive: a missing lineage_base in the handoff
    shouldn't break the whole pipeline.)"""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1),
        lineage=LineageState(candidate_base_score=None, current_best_score=5751.13),
    )
    assert isinstance(verdict, Submit)


# ---------------------------------------------------------------------------
# Daily budget exhaustion (no attempt consumed)
# ---------------------------------------------------------------------------


def test_daily_budget_exhausted_defers_without_consuming_attempt():
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
        budget=BudgetState(daily_remaining=0, seconds_until_next_slot=3600),
    )
    assert isinstance(verdict, DeferredForBudget)
    assert verdict.consumes_attempt is False


def test_budget_exhausted_but_would_not_have_submitted_anyway_retries():
    """Worker handed off a no-cost-reduction candidate that would have been
    Retry anyway. Don't mask it as DeferredForBudget — actually emit the
    Retry so the worker fixes it. Budget gate only fires when we WOULD
    have submitted."""
    verdict = classify_handoff(
        _handoff(current_cost=10000, candidate_cost=10000, expected_point_gain=0.0),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
        budget=BudgetState(daily_remaining=0, seconds_until_next_slot=3600),
    )
    # Should be Retry (no cost reduction), NOT DeferredForBudget
    assert isinstance(verdict, Retry)
    assert "cost did not decrease" in verdict.reason


def test_budget_available_proceeds_to_submit():
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=1),
        budget=BudgetState(daily_remaining=3, seconds_until_next_slot=0),
    )
    assert isinstance(verdict, Submit)


# ---------------------------------------------------------------------------
# normalized_family_for_handoff helper
# ---------------------------------------------------------------------------


def test_normalized_family_for_handoff_passes_through_normalize():
    handoff = _handoff(algorithm_family="Where_Select")
    assert normalized_family_for_handoff(handoff) == "where-select"


def test_normalized_family_for_handoff_empty_to_unspecified():
    handoff = _handoff(algorithm_family="")
    assert normalized_family_for_handoff(handoff) == "unspecified"


# ---------------------------------------------------------------------------
# Decision ordering / precedence (regressions if branches reorder)
# ---------------------------------------------------------------------------


def test_budget_check_runs_before_lineage_check():
    """If both budget exhausted AND lineage stale, budget wins — we shouldn't
    even bother routing back with rebase if we couldn't submit anyway."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
        budget=BudgetState(daily_remaining=0, seconds_until_next_slot=3600),
        lineage=LineageState(candidate_base_score=5743.97, current_best_score=5751.13),
    )
    assert isinstance(verdict, DeferredForBudget)


def test_lineage_check_runs_before_no_candidate_check():
    """A stale-lineage handoff with no candidate is still routed as a
    coordination issue first. Without this, we'd consume an attempt for
    something that's not the worker's fault."""
    verdict = classify_handoff(
        _handoff(candidate_id="", expected_point_gain=None),
        verify=VerifyResult.passed(),
        pass_state=_pass_state(attempts=0),
        lineage=LineageState(candidate_base_score=5743.97, current_best_score=5751.13),
    )
    assert isinstance(verdict, DeferredForStaleLineage)


def test_no_candidate_check_runs_before_mechanical_check():
    """Mechanical check is irrelevant when there's no candidate. Verifying a
    nonexistent zip would crash the mechanical layer; ensure we short-circuit."""
    verdict = classify_handoff(
        _handoff(candidate_id="", expected_point_gain=None),
        verify=VerifyResult.failed("would crash if called"),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "no candidate" in verdict.reason
    assert "mechanical" not in verdict.reason


def test_mechanical_check_runs_before_gain_check():
    """A mechanically-broken candidate with a good predicted gain should
    still get the mechanical-fail verdict — don't submit a broken package
    just because the worker claimed +0.5."""
    verdict = classify_handoff(
        _handoff(expected_point_gain=0.5),
        verify=VerifyResult.failed("SHA mismatch"),
        pass_state=_pass_state(attempts=0),
    )
    assert isinstance(verdict, Retry)
    assert "mechanical-fail" in verdict.reason
