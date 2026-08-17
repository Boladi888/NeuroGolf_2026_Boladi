"""Python-driven submit + retry gate for NeuroGolf worker handoffs.

Replaces the LLM submit-captain + LLM reviewer in the routine path. Reads
``review-required`` handoff JSON from blocked kanban cards, performs mechanical
verification, decides submit-vs-retry-vs-give-up, and (when this module's side-
effect entry point is wired up in subsequent tasks) submits to Kaggle directly
or routes the card back to the worker with a "try a different algorithm family"
directive.

This file contains the **pure decision logic** (Task 5 of the plan). Side
effects — running mechanical checks (Task 6), submitting to Kaggle (Task 7),
and atomic registry writes (Task 8) — live in other functions / modules and
are wired into the runner at the end. ``classify_handoff`` itself is a pure
function so the test suite can exercise every branch without mocks for IO.

See plan at ``<CLAUDE_HOME>/plans/lets-go-overview-for-async-fiddle.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.algorithm_family_directive import build_directive, normalize_family
try:
    from tools.auto_triage_blocked import Handoff
except ImportError:  # auto_triage_blocked is private orchestration, withheld from the shared build
    class Handoff:  # used only as a type annotation here; this stand-in keeps the module importable
        """Stand-in for the private orchestration Handoff (shared build)."""

# Public knobs — tweakable by CLI flags later, but the defaults match the
# plan's confirmed design choices.
# DEFAULT_GAIN_FLOOR retained for backward-compat (some callers still pass it)
# but the active gate is cost-based as of 2026-05-19. See classify_handoff.
DEFAULT_GAIN_FLOOR = 0.20
DEFAULT_MAX_ATTEMPTS_PER_PASS = 3


def extract_cost_pair(handoff: Handoff) -> tuple[int | None, int | None]:
    """Pull (current_cost, candidate_cost) from a handoff.

    Handles the common shapes workers produce:
        - ``scores.current_cost`` + ``scores.candidate_cost``
        - ``baseline.cost`` + ``candidate.cost``
        - top-level ``current_cost`` + ``candidate_cost``
        - ``score_current_live.cost`` + ``score_candidate.cost``

    Returns (None, None) when costs can't be extracted — callers treat that
    as "no candidate produced".
    """
    parsed = handoff.parsed if isinstance(handoff.parsed, dict) else {}

    # Try the canonical shape first.
    scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else None
    if scores:
        cur = _coerce_int(scores.get("current_cost"))
        cand = _coerce_int(scores.get("candidate_cost"))
        if cur is not None and cand is not None:
            return cur, cand

    # baseline / candidate nested
    base_obj = parsed.get("baseline") if isinstance(parsed.get("baseline"), dict) else None
    cand_obj = parsed.get("candidate") if isinstance(parsed.get("candidate"), dict) else None
    if base_obj and cand_obj:
        cur = _coerce_int(base_obj.get("cost"))
        cand = _coerce_int(cand_obj.get("cost"))
        if cur is not None and cand is not None:
            return cur, cand

    # score_current_live / score_candidate
    base_sl = parsed.get("score_current_live") if isinstance(parsed.get("score_current_live"), dict) else None
    cand_sc = parsed.get("score_candidate") if isinstance(parsed.get("score_candidate"), dict) else None
    if base_sl and cand_sc:
        cur = _coerce_int(base_sl.get("cost"))
        cand = _coerce_int(cand_sc.get("cost"))
        if cur is not None and cand is not None:
            return cur, cand

    # Top-level current_cost / candidate_cost (worker-5 task373 shape)
    cur = _coerce_int(parsed.get("current_cost"))
    cand = _coerce_int(parsed.get("candidate_cost"))
    if cur is not None and cand is not None:
        return cur, cand

    return None, None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


# ============================================================================
# Inputs to the pure decision function
# ============================================================================


@dataclass
class VerifyResult:
    """Outcome of mechanical worker-side verification (Task 6 layer).

    ``failures`` is a list of human-readable strings so they can be echoed into
    the retry directive verbatim — the worker should be able to read what went
    wrong without translating an enum.
    """

    ok: bool
    failures: list[str] = field(default_factory=list)

    @classmethod
    def passed(cls) -> "VerifyResult":
        return cls(ok=True)

    @classmethod
    def failed(cls, *reasons: str) -> "VerifyResult":
        return cls(ok=False, failures=list(reasons))


@dataclass
class PassState:
    """Per-task state for the CURRENT autoforeman pass.

    Read via ``tools.task_outcome_registry_io.read_pass_state``. The fields
    here are exactly the registry's per-pass counters after the
    "is this a new pass?" reset has been applied — the caller hands us already-
    fresh state when the pass advanced.
    """

    attempts_this_pass: int
    attempted_families: list[str]
    current_pass: int


@dataclass
class BudgetState:
    """Daily Kaggle submit budget and rate-limit state.

    ``daily_remaining`` is the count of submits still available today against
    the ``daily_max_submits`` policy. ``seconds_until_next_slot`` is the
    token-bucket wait (0 if a slot is available right now).
    """

    daily_remaining: int
    seconds_until_next_slot: float

    @property
    def can_submit_now(self) -> bool:
        return self.daily_remaining > 0 and self.seconds_until_next_slot <= 0


@dataclass
class LineageState:
    """Comparison between the candidate's lineage_base and current-best."""

    candidate_base_score: float | None
    current_best_score: float | None

    @property
    def is_stale(self) -> bool:
        """True when the candidate was built against a stale current-best.

        ``None`` on either side means "unknown" — submit_gate treats unknown
        as non-stale (don't block on missing data; the operator will see it
        in the audit log).
        """
        if self.candidate_base_score is None or self.current_best_score is None:
            return False
        return self.candidate_base_score != self.current_best_score


# ============================================================================
# Verdict types (tagged union via subclasses)
# ============================================================================


@dataclass
class _BaseVerdict:
    """Common fields for every verdict. ``reason`` is a one-line summary that
    goes into the audit log and the card-completion result message."""

    reason: str


@dataclass
class Submit(_BaseVerdict):
    """Caller should run the Kaggle submit.

    ``also_retry`` is the forced-exploration rule: on the first attempt this
    pass we always submit AND retry, so subsequent passes get at least 2
    independent cracks at every task. When ``also_retry`` is True, the
    ``retry_directive`` text must be embedded in the new worker card body.

    ``consumes_attempt`` is always True for Submit verdicts — even if the
    submit ends up being a forced-exploration retry, the attempt is consumed
    in the per-pass counter.
    """

    also_retry: bool = False
    retry_directive: str = ""
    consumes_attempt: bool = True


@dataclass
class Retry(_BaseVerdict):
    """No submit, but worker should try again with a new algorithm family.

    Consumes an attempt. The ``directive`` text is built by
    ``algorithm_family_directive.build_directive`` and goes into the new
    worker card's body.
    """

    directive: str
    consumes_attempt: bool = True


@dataclass
class GiveUpForPass(_BaseVerdict):
    """3 attempts exhausted (the configured ``max_attempts_per_pass``).

    Caller should complete the card with the give-up note, write a permanent
    record to ``known_attempts``, and the autoforeman will skip this task
    until ``pass_count`` increments. No reviewer-LLM escalation in the
    routine path — this is the terminal verdict.
    """

    notes_summary: str
    last_attempt_reason: str


@dataclass
class DeferredForBudget(_BaseVerdict):
    """Daily Kaggle submit budget exhausted. Card stays blocked.

    Does NOT consume an attempt — the worker did their job, we just can't
    submit right now. Next cron tick after midnight UTC will retry.
    """

    consumes_attempt: bool = False


@dataclass
class DeferredForRateLimit(_BaseVerdict):
    """Token bucket says wait. Card stays blocked, next cron tick retries."""

    seconds_to_wait: float
    consumes_attempt: bool = False


@dataclass
class DeferredForStaleLineage(_BaseVerdict):
    """Candidate's lineage_base_score doesn't match current-best.

    Worker built against a stale base; routing back with a "rebase to current
    best" directive lets them re-verify and re-submit cleanly. Crucially,
    this does NOT consume an attempt — it's a coordination problem, not a
    worker bug.
    """

    new_base_score: float
    directive: str
    consumes_attempt: bool = False


# ============================================================================
# Pure decision function
# ============================================================================


def classify_handoff(
    handoff: Handoff,
    *,
    verify: VerifyResult,
    pass_state: PassState,
    budget: BudgetState | None = None,
    lineage: LineageState | None = None,
    gain_floor: float = DEFAULT_GAIN_FLOOR,
    max_attempts_per_pass: int = DEFAULT_MAX_ATTEMPTS_PER_PASS,
) -> _BaseVerdict:
    """Decide what should happen for one ``review-required`` blocked card.

    Pure function — no kanban mutation, no Kaggle submit, no registry write.
    The caller interprets the returned verdict and performs IO.

    Decision order (matches the plan's pseudo-code):

    1. **Daily budget exhausted** → ``DeferredForBudget`` (no attempt consumed).
       Checked first so we don't waste analysis on a doomed-now attempt.
    2. **Stale lineage** (candidate built against a base that's behind
       current-best) → ``DeferredForStaleLineage`` (no attempt consumed).
       Catches the "worker started building 30 min ago; current-best advanced
       in the meantime" race.
    3. **No candidate produced** (missing ``candidate_id`` OR null predicted
       gain) → ``Retry`` (no submit), or ``GiveUpForPass`` if it's the third.
    4. **Mechanical verification failed** → ``Retry`` (no submit), or
       ``GiveUpForPass`` if third. Worker's bug — fix or try different family.
    5. **Predicted gain below floor** → ``Retry`` (no submit), or
       ``GiveUpForPass`` if third. The 0.2 threshold guards Kaggle budget.
    6. **First attempt this pass** with everything passing → ``Submit`` with
       ``also_retry=True``. Forced exploration — get at least 2 cracks at
       every task even when attempt 1 looks great.
    7. **Subsequent attempt** with everything passing → ``Submit`` with
       ``also_retry=False``. Task is done for this pass.
    """
    families = pass_state.attempted_families
    attempt_index = pass_state.attempts_this_pass + 1  # the attempt we're about to consume
    is_last_attempt = attempt_index >= max_attempts_per_pass

    # 1. Budget gate first — cheap and global. Don't waste analysis cycles if
    #    we can't submit anyway. Returns even if subsequent checks would have
    #    rejected, because we don't want to consume an attempt for something
    #    we couldn't have submitted regardless.
    if budget is not None and not budget.can_submit_now and _would_submit(
        handoff, verify, pass_state, gain_floor
    ):
        return DeferredForBudget(
            reason=(
                f"daily Kaggle submit budget exhausted "
                f"({budget.daily_remaining} remaining, {budget.seconds_until_next_slot:.0f}s to next slot)"
            ),
        )

    # 2. Stale lineage — also doesn't consume an attempt. The worker did their
    #    job against a base that has since moved; routing back with the new
    #    base is fair.
    if lineage is not None and lineage.is_stale:
        new_base = lineage.current_best_score if lineage.current_best_score is not None else 0.0
        return DeferredForStaleLineage(
            reason=(
                f"candidate lineage_base ({lineage.candidate_base_score}) is stale; "
                f"current-best is now {lineage.current_best_score}"
            ),
            new_base_score=new_base,
            directive=(
                "Rebase against the new current-best and re-run the local verifier. "
                "Your candidate may still be valid — but we won't submit until the "
                "lineage matches what Kaggle has scored."
            ),
            consumes_attempt=False,
        )

    # 3. No candidate produced — iteration-budget exhausted, no-fix marker,
    #    or worker explicitly handed off without a candidate_id.
    if _no_candidate_produced(handoff):
        return _retry_or_give_up(
            reason="worker produced no candidate",
            families_tried=families,
            attempt_index=attempt_index,
            is_last_attempt=is_last_attempt,
        )

    # 4. Mechanical verification — SHA, package structure, score_v2 reproduce,
    #    verify_against_live, raw-shape parity. Failures are worker bugs.
    if not verify.ok:
        failures_str = "; ".join(verify.failures) if verify.failures else "unknown failure"
        return _retry_or_give_up(
            reason=f"mechanical-fail: {failures_str}",
            families_tried=families,
            attempt_index=attempt_index,
            is_last_attempt=is_last_attempt,
            directive_override=(
                f"Fix the safety violation before resubmitting: {failures_str}"
            ),
        )

    # 5. Cost-reduction gate (replaces the old predicted-gain floor, 2026-05-19).
    #    Any cost reduction is worth submitting; predicted_gain is just derived
    #    from cost via 25 - log(memory+params), so a tiny cost win → tiny point
    #    gain that the old 0.2 floor would have thrown away.
    current_cost, candidate_cost = extract_cost_pair(handoff)
    predicted_gain = handoff.expected_point_gain  # kept for audit-log readability
    if current_cost is None or candidate_cost is None or candidate_cost >= current_cost:
        if current_cost is None or candidate_cost is None:
            reason = (
                f"cost data missing from handoff "
                f"(current={current_cost}, candidate={candidate_cost})"
            )
        else:
            delta = candidate_cost - current_cost
            reason = (
                f"cost did not decrease "
                f"(candidate={candidate_cost}, current={current_cost}, delta={delta:+d})"
            )
        return _retry_or_give_up(
            reason=reason,
            families_tried=families,
            attempt_index=attempt_index,
            is_last_attempt=is_last_attempt,
        )

    cost_delta = candidate_cost - current_cost  # negative = improvement
    gain_str = f"{predicted_gain:+.4f}" if predicted_gain is not None else "+0.0000"

    # 6. First attempt this pass: SUBMIT and RETRY (forced exploration).
    if pass_state.attempts_this_pass == 0:
        return Submit(
            reason=(
                f"first attempt this pass — forced exploration after "
                f"submitting cost {cost_delta:+d} (predicted {gain_str})"
            ),
            also_retry=True,
            retry_directive=build_directive(
                families,
                retry_attempt_index=attempt_index + 1,  # next attempt's index
                reason=(
                    "forced exploration: submit_gate always runs at least 2 "
                    "attempts per task per pass, even when attempt 1 looks good. "
                    f"Submitted cost {cost_delta:+d}; now find a different "
                    "algorithm family that may compose with it."
                ),
            ),
            consumes_attempt=True,
        )

    # 7. Normal path: cost went down, not first attempt — submit and stop.
    return Submit(
        reason=f"submitted, cost {cost_delta:+d} (predicted {gain_str})",
        also_retry=False,
        consumes_attempt=True,
    )


# ============================================================================
# Internal helpers
# ============================================================================


def _no_candidate_produced(handoff: Handoff) -> bool:
    """True when the worker didn't actually produce a candidate to submit.

    Covers: missing/blank ``candidate_id``, or no extractable cost data
    (which signals iter-budget-exhausted / no-fix handoffs where the worker
    never finished a measurable candidate).
    """
    if not handoff.candidate_id:
        return True
    current_cost, candidate_cost = extract_cost_pair(handoff)
    if current_cost is None and candidate_cost is None:
        # No cost info AND no expected_point_gain → truly no candidate.
        # If gain is present but cost isn't, we still treat as no-candidate
        # because the cost-reduction gate can't run.
        if handoff.expected_point_gain is None and handoff.point_delta is None:
            return True
    return False


def _would_submit(
    handoff: Handoff,
    verify: VerifyResult,
    pass_state: PassState,
    gain_floor: float,  # legacy param, kept for signature compat; not used
) -> bool:
    """Quick check: would the decision tree reach a Submit verdict?

    Used by the budget gate to avoid emitting ``DeferredForBudget`` when the
    decision would have been Retry / GiveUp anyway (no submit attempted means
    no budget consumed; no reason to defer).

    Cost-reduction gate (2026-05-19): submit happens iff there IS a candidate
    AND mechanical checks pass AND ``candidate_cost < current_cost``. The
    ``gain_floor`` parameter is preserved for signature compatibility but no
    longer participates in the decision.
    """
    if _no_candidate_produced(handoff):
        return False
    if not verify.ok:
        return False
    current_cost, candidate_cost = extract_cost_pair(handoff)
    if current_cost is None or candidate_cost is None:
        return False
    return candidate_cost < current_cost


def _retry_or_give_up(
    *,
    reason: str,
    families_tried: list[str],
    attempt_index: int,
    is_last_attempt: bool,
    directive_override: str | None = None,
) -> _BaseVerdict:
    """Shared exit for the no-candidate / mechanical-fail / below-floor paths.

    Centralizes the "3 attempts → give up" rule so every non-submit branch
    handles the terminal case the same way. The autoforeman's defer logic
    uses the ``notes_summary`` written for GiveUpForPass to skip this task
    until the next pass.
    """
    if is_last_attempt:
        return GiveUpForPass(
            reason=f"3 attempts exhausted this pass: {reason}",
            notes_summary=f"tested_non_improving_this_pass — {reason}",
            last_attempt_reason=reason,
        )
    directive = directive_override or build_directive(
        families_tried,
        retry_attempt_index=attempt_index + 1,
        reason=reason,
    )
    return Retry(reason=reason, directive=directive, consumes_attempt=True)


def normalized_family_for_handoff(handoff: Handoff) -> str:
    """Canonical form of the worker's reported algorithm_family.

    Convenience for callers — collapses the empty / missing / inconsistent
    cases into one canonical string via the directive module's normalizer.
    """
    return normalize_family(handoff.algorithm_family)
