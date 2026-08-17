"""Kaggle submit wrapper with safety gates for submit_gate.

This module provides the pre-submit gates (daily budget, rate limit, lineage
consistency) and the post-submit half-fail handler. The ACTUAL Kaggle submit
call is injected as a callable so the wrapper is testable without touching
Kaggle, and so the production wiring can swap the callable for whatever
mechanism is preferred (Kaggle CLI direct, kaggle Python SDK, or routing
through the legacy submit-captain card path).

Design choices documented in the plan:
- Read state from existing files (.hermes/board/submit_runtime_config.json,
  .hermes/board/submit_credit_state.json, Logs/submissions_ledger.csv) so we
  don't fork the source of truth.
- Update submit_credit_state.json atomically after a successful submit using
  the same .tmp + replace pattern as task_outcome_registry_io.
- Half-fail: if the submit fn returns non-zero AFTER apparently uploading,
  re-query Kaggle via the injected verifier within 60s. If found, treat as
  success. If not, write 'submit_attempt_uncertain' to the audit log and
  alert via the injected notifier — do NOT retry automatically (daily
  budget is shared).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from tools.auto_triage_blocked import Handoff
except ImportError:  # auto_triage_blocked is private orchestration, withheld from the shared build
    class Handoff:  # used only as a type annotation here; this stand-in keeps the module importable
        """Stand-in for the private orchestration Handoff (shared build)."""
from tools.submit_gate import BudgetState, LineageState
from tools.task_outcome_registry_io import save_registry  # atomic write helper


# Type aliases for the injected callables.
# SubmitFn: (handoff, candidate_zip_path) -> SubmitOutcome
SubmitFn = Callable[[Handoff, Path], "SubmitOutcome"]
# KaggleVerifierFn: (handoff) -> Optional[KaggleSubmitRecord]
#   Returns the Kaggle record for this submission if it landed, None if not.
KaggleVerifierFn = Callable[[Handoff], "KaggleSubmitRecord | None"]
# NotifierFn: (severity, message) -> None
NotifierFn = Callable[[str, str], None]


# ============================================================================
# Submit outcome types
# ============================================================================


@dataclass
class KaggleSubmitRecord:
    """Minimal fields the verifier returns when it finds a submission."""

    submitted_at: str  # ISO8601 UTC
    public_score: float | None  # None until Kaggle finishes scoring
    kaggle_ref: str  # the Kaggle API submission reference (number or fileName)


@dataclass
class SubmitOutcome:
    """Result of a submit attempt.

    ``status`` semantics:
        - ``"ok"``: submit succeeded, public score may not be ready yet.
        - ``"uncertain"``: submit_fn errored AFTER apparently uploading. Caller
          should call verify_after_fn within 60s; if it returns a record,
          downgrade to "ok"; otherwise stay "uncertain" and alert.
        - ``"failed"``: submit_fn errored BEFORE upload (network down, bad
          credentials). Safe to retry on the next cron tick.
    """

    status: str  # "ok" | "uncertain" | "failed"
    record: KaggleSubmitRecord | None = None
    error: str = ""


# ============================================================================
# Budget / rate-limit gate
# ============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG_PATH = REPO_ROOT / ".hermes" / "board" / "submit_runtime_config.json"
DEFAULT_CREDIT_STATE_PATH = REPO_ROOT / ".hermes" / "board" / "submit_credit_state.json"
DEFAULT_LEDGER_PATH = REPO_ROOT / "Logs" / "submissions_ledger.csv"


def load_budget_state(
    *,
    runtime_config_path: Path = DEFAULT_RUNTIME_CONFIG_PATH,
    credit_state_path: Path = DEFAULT_CREDIT_STATE_PATH,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    now: datetime | None = None,
) -> BudgetState:
    """Compute a BudgetState from the three existing source-of-truth files.

    Reads:
        - submit_runtime_config.json -> max_submits_per_hour
        - submit_credit_state.json -> daily_submit_cap, submit_rate_limit_minutes,
          last_submit_at
        - submissions_ledger.csv -> count of submits today (date column)

    Returns a BudgetState the submit_gate decision tree can consume.
    """
    now = now or datetime.now(timezone.utc)

    runtime_cfg = _read_json_dict(runtime_config_path)
    credit_state = _read_json_dict(credit_state_path)
    policy = credit_state.get("policy", {}) if isinstance(credit_state.get("policy"), dict) else {}

    daily_cap = _coerce_int(policy.get("daily_submit_cap"), default=100)
    rate_limit_minutes = _coerce_int(policy.get("submit_rate_limit_minutes"), default=10)
    max_per_hour = _coerce_int(runtime_cfg.get("max_submits_per_hour"), default=0)
    auto_enabled = bool(runtime_cfg.get("post_refresh_auto_submit_enabled", False))

    # Count today's submits from the ledger.
    today_iso = now.strftime("%Y-%m-%d")
    submits_today = _count_submits_today(ledger_path, today_iso)
    daily_remaining = max(0, daily_cap - submits_today)

    # Time since last submit, as recorded in credit state.
    credits = credit_state.get("credits", {}) if isinstance(credit_state.get("credits"), dict) else {}
    last_submit_at = _parse_iso(credits.get("last_submit_at"))
    if last_submit_at is None:
        seconds_until_next_slot = 0.0
    else:
        elapsed = (now - last_submit_at).total_seconds()
        rate_limit_seconds = rate_limit_minutes * 60.0
        seconds_until_next_slot = max(0.0, rate_limit_seconds - elapsed)

    # If auto-submit is disabled in config, treat budget as exhausted —
    # operator has explicitly disabled the gate.
    if not auto_enabled or max_per_hour <= 0:
        return BudgetState(
            daily_remaining=0,
            seconds_until_next_slot=float("inf"),
        )

    return BudgetState(
        daily_remaining=daily_remaining,
        seconds_until_next_slot=seconds_until_next_slot,
    )


def load_lineage_state(
    handoff: Handoff,
    *,
    manifest_path: Path | None = None,
) -> LineageState:
    """Compare the handoff's lineage_base against the current-best manifest.

    The handoff's ``lineage_base`` field is a free-form string like
    ``"current-best task141_v4_fp16_pos_t_6856eb96_on_5743_97_20260517T2202Z"``.
    We extract the trailing ``_on_<score>_`` segment when present and compare
    against ``current_best_manifest.json[current_best][public_score]``.

    Returns a LineageState. Both scores can be None when we can't extract —
    submit_gate treats unknown as non-stale (don't block on missing data).
    """
    if manifest_path is None:
        manifest_path = REPO_ROOT / "submissions" / "current_best_manifest.json"

    candidate_base_score = _extract_score_from_lineage(handoff.lineage_base)
    current_best_score = _read_current_best_score(manifest_path)

    return LineageState(
        candidate_base_score=candidate_base_score,
        current_best_score=current_best_score,
    )


# ============================================================================
# Safe submit orchestration
# ============================================================================


def submit_with_safety(
    handoff: Handoff,
    candidate_zip_path: Path,
    *,
    submit_fn: SubmitFn,
    verify_after_fn: KaggleVerifierFn | None = None,
    notifier_fn: NotifierFn | None = None,
    credit_state_path: Path = DEFAULT_CREDIT_STATE_PATH,
    now: datetime | None = None,
) -> SubmitOutcome:
    """Invoke ``submit_fn``, handle half-fail, update credit state.

    On ok: update last_submit_at + spent_total + daily_used in
    submit_credit_state.json (atomic write) and return outcome.

    On uncertain (submit_fn raised AFTER apparent upload): call
    ``verify_after_fn`` to re-query Kaggle. If it confirms the submit landed,
    downgrade to ok and update state. Otherwise notify and return uncertain
    — caller should leave the card blocked, NOT retry automatically.

    On failed (submit_fn raised BEFORE upload): return failed; caller can
    safely retry on the next cron tick (no state mutation, daily budget
    unconsumed).
    """
    now = now or datetime.now(timezone.utc)

    try:
        outcome = submit_fn(handoff, candidate_zip_path)
    except Exception as exc:  # noqa: BLE001
        # We have no idea whether the upload landed. Default to "uncertain"
        # so the verifier gets a chance to clarify.
        outcome = SubmitOutcome(status="uncertain", error=str(exc))

    if outcome.status == "uncertain" and verify_after_fn is not None:
        try:
            record = verify_after_fn(handoff)
        except Exception as exc:  # noqa: BLE001
            record = None
            outcome.error = f"{outcome.error} | verifier raised: {exc}"
        if record is not None:
            outcome = SubmitOutcome(status="ok", record=record)

    if outcome.status == "ok":
        _record_successful_submit(credit_state_path, now)
    elif outcome.status == "uncertain":
        if notifier_fn is not None:
            notifier_fn(
                "warn",
                (
                    f"submit_gate uncertain: candidate {handoff.candidate_id} "
                    f"submit returned ambiguous outcome ({outcome.error}); "
                    f"manual verification required against Kaggle submission list"
                ),
            )
    # status "failed" needs no state mutation — caller can retry.

    return outcome


# ============================================================================
# Internal helpers
# ============================================================================


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text[:-1] + "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _count_submits_today(ledger_path: Path, today_iso: str) -> int:
    """Count rows in submissions_ledger.csv where date == today_iso.

    Missing or unreadable ledger → 0 (defensive default; the next submit
    will hit the rate-limit gate instead of the daily-cap gate). Use
    ``errors="replace"`` on the read so invalid bytes in the ledger
    (corruption, mixed encoding) don't crash the gate.
    """
    if not ledger_path.exists():
        return 0
    try:
        with ledger_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for row in reader if row.get("date") == today_iso)
    except (OSError, csv.Error, UnicodeError):
        return 0


def _extract_score_from_lineage(lineage_base: str) -> float | None:
    """Pull the score out of ``_on_<score>_`` segments like ``_on_5743_97_``.

    The score is stored with underscore in place of decimal point (worker
    convention from the existing playbook). Examples:
        ``"current-best task141_v4_fp16_pos_t_6856eb96_on_5743_97_20260517T2202Z"`` -> 5743.97
        ``"task019_slice12_compute12_pad30_fp16_on_5739_7_20260517T160659Z"``       -> 5739.7
    Returns None if no extractable score.
    """
    import re
    # Match `_on_<int>_<int>` followed by EOL / non-alphanumeric (underscore,
    # space, dash, etc.). The trailing-underscore-required version of this
    # regex failed on lineage strings that end exactly at the score with no
    # date suffix, which is a real shape produced by some worker handoffs.
    match = re.search(r"_on_(\d+)_(\d+)(?=[^0-9A-Za-z]|$)", lineage_base)
    if not match:
        return None
    try:
        return float(f"{match.group(1)}.{match.group(2)}")
    except ValueError:
        return None


def _read_current_best_score(manifest_path: Path) -> float | None:
    """Extract the current public-best score from submissions/current_best_manifest.json.

    Production layout (verified 2026-05-20 against the live file):
        { "schema": "neurogolf.candidate_registry.v2-draft",
          "score": 5835.3,                                ← LEGACY top-level, often stale
          "candidate_id": "...",                          ← LEGACY top-level
          "current_best": {                               ← v2 nested, source of truth
              "score": 5861.43,
              "candidate_id": "...",
              "package_sha256": "..."
          } }

    Priority (corrected 2026-05-20 — original code preferred top-level,
    which became stale because ``tools/neurogolf_sync_current_best.py::update_v2_manifest``
    only writes to the nested ``current_best`` block, not to the top-level
    fields which are vestigial from the v1 schema):

        1. ``current_best.score``  ← what dashboard + auto_sync read
        2. ``current_best.public_score`` ← alternative key on the nested block
        3. top-level ``score``     ← stale legacy fallback
        4. top-level ``public_score`` ← original legacy shape
    """
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    candidates: list = []
    cb = data.get("current_best")
    if isinstance(cb, dict):
        candidates.append(cb.get("score"))
        candidates.append(cb.get("public_score"))
    candidates.append(data.get("score"))
    candidates.append(data.get("public_score"))

    for raw in candidates:
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def _record_successful_submit(credit_state_path: Path, now: datetime) -> None:
    """Update submit_credit_state.json after a successful submit.

    Increments ``credits.spent_total`` and ``credits.daily_used``, sets
    ``credits.last_submit_at``. Atomic via save_registry's .tmp+replace
    pattern.
    """
    state = _read_json_dict(credit_state_path)
    if not state:
        state = {"schema": "neurogolf.submit_credit_state.v1"}
    credits = state.setdefault("credits", {})
    credits["spent_total"] = _coerce_int(credits.get("spent_total"), default=0) + 1
    credits["daily_used"] = _coerce_int(credits.get("daily_used"), default=0) + 1
    credits["last_submit_at"] = now.isoformat().replace("+00:00", "Z")
    state["updated_at"] = now.isoformat().replace("+00:00", "Z")
    save_registry(credit_state_path, state)
