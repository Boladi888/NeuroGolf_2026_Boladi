#!/usr/bin/env python3
"""Deterministic drain of the Hermes 'blocked' column.

Workers self-block with a structured `review-required` handoff JSON when they
finish a candidate. With 8 workers each capped at one active card, the column
saturates fast and the whole sprint stalls until a reviewer agent processes
each one. Most blocked cards do not actually need LLM judgment: a candidate
that predicts < 0.05 LB gain is already below the standalone-submit floor and
gets banked; a follow-up card whose block reason names "still not enough" or
"do not resend" is a no-fix; a clean >= 0.05 gain candidate just needs a review
card created so the reviewer profile picks it up.

This script applies that policy deterministically. Cards that actually need
human/LLM judgment (high gain + high risk markers) are left alone.

Run:
  .venv/Scripts/python tools/auto_triage_blocked.py                # dry-run
  .venv/Scripts/python tools/auto_triage_blocked.py --apply        # mutate

Output is a per-card verdict table + a written summary in
`Logs/auto_triage_<timestamp>.md`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def cost_to_points(cost: int | float | None) -> float | None:
    """NeuroGolf points formula: max(1.0, 25 - log(memory + params))."""
    if cost is None or cost <= 0:
        return None
    return max(1.0, 25.0 - math.log(float(cost)))


# Key-name patterns the schema-agnostic walker looks for in handoffs. Workers
# spawned by different LLMs produce subtly different JSON structures — instead
# of enumerating every shape, we walk the tree and grab any field that matches
# these patterns. Order matters: the first match wins.
_POINT_DELTA_KEY_PATTERNS = (
    # Workers occasionally name the gain key with a `_vs_<context>` suffix
    # (e.g. `expected_point_gain_vs_task127_live`) to disambiguate when a
    # handoff carries multiple baselines. Allow the same `(_vs.*)?` tail
    # `point_delta` already accepts so the schema-agnostic walker doesn't
    # silently miss the gain and fall back to the candidate_id-only path.
    re.compile(r"^expected[_-]?point[_-]?gain(_vs.*)?$", re.IGNORECASE),
    re.compile(r"^expected[_-]?lb[_-]?point[_-]?gain(_vs.*)?$", re.IGNORECASE),
    re.compile(r"^point[_-]?gain(_vs.*)?$", re.IGNORECASE),
    re.compile(r"^point[_-]?delta(_vs.*)?$", re.IGNORECASE),
    re.compile(r"^points[_-]?delta(_vs.*)?$", re.IGNORECASE),
    re.compile(r"^delta[_-]?points?$", re.IGNORECASE),
    re.compile(r"^predicted[_-]?lb[_-]?delta(_vs.*)?$", re.IGNORECASE),
    re.compile(r"^predicted[_-]?delta(_vs.*)?$", re.IGNORECASE),
)
_BASELINE_COST_KEY_PATTERNS = (
    re.compile(r"(^|[_-])base(_|line)?[_-]?cost$", re.IGNORECASE),
    re.compile(r"(^|[_-])current(_|live)?[_-]?cost$", re.IGNORECASE),
    re.compile(r"^canonical(_current)?.*cost$", re.IGNORECASE),
)
_CANDIDATE_COST_KEY_PATTERNS = (
    re.compile(r"(^|[_-])candidate[_-]?cost$", re.IGNORECASE),
    re.compile(r"(^|[_-])new[_-]?cost$", re.IGNORECASE),
    re.compile(r"(^|[_-])best[_-]?cost$", re.IGNORECASE),
    re.compile(r"(^|[_-])successor[_-]?cost$", re.IGNORECASE),
)
_BASELINE_NESTED_KEY_PATTERNS = (  # objects expected to have a .cost child
    re.compile(r"^base$", re.IGNORECASE),
    re.compile(r"^baseline$", re.IGNORECASE),
    re.compile(r"^current$", re.IGNORECASE),
    re.compile(r"^current[_-]live$", re.IGNORECASE),
    re.compile(r"^score[_-]?current(_live)?$", re.IGNORECASE),
)
_CANDIDATE_NESTED_KEY_PATTERNS = (
    re.compile(r"^candidate$", re.IGNORECASE),
    re.compile(r"^score[_-]?candidate$", re.IGNORECASE),
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_number(value: Any) -> float | None:
    """Convert int/float/numeric-string to float. Returns None on failure.

    Legacy candidate_registry entries sometimes store ``expected_point_gain``
    as a string like ``"1.5"`` or ``"+0.05"``. Without coercion the walker
    skips these even though they carry valid signal.
    """
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _walk(node: Any, path: str = ""):
    """Generator yielding (key_path, key, value) for every leaf + named child."""
    if isinstance(node, dict):
        for k, v in node.items():
            new_path = f"{path}.{k}" if path else str(k)
            yield new_path, str(k), v
            yield from _walk(v, new_path)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            new_path = f"{path}[{i}]"
            yield from _walk(v, new_path)


def _matches(key: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.search(key) for p in patterns)


def _extract_first_cost_from_obj(obj: Any) -> float | None:
    """If `obj` is a dict, return its `cost`/`memory+params` numeric, else None."""
    if not isinstance(obj, dict):
        return None
    if _is_number(obj.get("cost")):
        return float(obj["cost"])
    mem = obj.get("memory")
    params = obj.get("params")
    if _is_number(mem) and _is_number(params):
        return float(mem) + float(params)
    if _is_number(mem):
        return float(mem)
    return None


def extract_point_delta(parsed: Any) -> tuple[float | None, str]:
    """Schema-agnostic point-delta extractor.

    Returns (point_delta, source_description). Tries in order:
    1. Any numeric leaf whose key matches one of the point-delta patterns.
    2. Pair (baseline_cost, candidate_cost) found anywhere in the tree.
    3. None.
    """
    # 1. Direct point-delta-named field. Accept numeric strings as well as
    # int/float — legacy registry entries sometimes serialize as "1.5".
    for path, key, value in _walk(parsed):
        if _matches(key, _POINT_DELTA_KEY_PATTERNS):
            coerced = _coerce_number(value)
            if coerced is not None:
                return coerced, f"direct@{path}"
        # Some workers nest the gain in a parent object whose name matches.
        # Operator precedence note: must group ("point" or "delta") AND ("cost" not in).
        # A bare key like "candidate_cost_delta" should NOT match — that's a cost
        # delta, not a points delta.
        if isinstance(value, dict) and key.lower() in ("expected_gain", "expected", "delta", "gain"):
            for sub_k, sub_v in value.items():
                sub_kl = sub_k.lower()
                has_point_or_delta = ("point" in sub_kl) or ("delta" in sub_kl)
                is_cost_field = "cost" in sub_kl
                if has_point_or_delta and not is_cost_field:
                    coerced = _coerce_number(sub_v)
                    if coerced is not None:
                        return coerced, f"nested@{path}.{sub_k}"

    # 2. Find a baseline_cost + candidate_cost pair.
    baseline_cost: float | None = None
    candidate_cost: float | None = None
    baseline_src = candidate_src = ""
    for path, key, value in _walk(parsed):
        if baseline_cost is None and _is_number(value) and _matches(key, _BASELINE_COST_KEY_PATTERNS):
            baseline_cost = float(value)
            baseline_src = path
        if candidate_cost is None and _is_number(value) and _matches(key, _CANDIDATE_COST_KEY_PATTERNS):
            candidate_cost = float(value)
            candidate_src = path
        if baseline_cost is None and isinstance(value, dict) and _matches(key, _BASELINE_NESTED_KEY_PATTERNS):
            cost = _extract_first_cost_from_obj(value)
            if cost is not None:
                baseline_cost = cost
                baseline_src = path
        if candidate_cost is None and isinstance(value, dict) and _matches(key, _CANDIDATE_NESTED_KEY_PATTERNS):
            cost = _extract_first_cost_from_obj(value)
            if cost is not None:
                candidate_cost = cost
                candidate_src = path
    if baseline_cost is not None and candidate_cost is not None:
        bp = cost_to_points(baseline_cost)
        cp = cost_to_points(candidate_cost)
        if bp is not None and cp is not None:
            return cp - bp, f"computed:base@{baseline_src}({baseline_cost})->cand@{candidate_src}({candidate_cost})"

    return None, "no-signal"

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_REGISTRY = REPO_ROOT / ".hermes" / "board" / "candidate_registry.json"

# --- Policy thresholds (mirror NEUROGOLF_KANBAN_WORKFLOW.md "Bovath high-EV") ---
STANDALONE_SUBMIT_FLOOR = 0.05  # LB; below this, bank-for-integration

ACTIVE_BLOCKED_STATUS = "blocked"
WORKER_PROFILE_PATTERN = re.compile(r"^neurogolf-worker-\d+$")
REVIEWER_PROFILE_PATTERN = re.compile(r"^neurogolf-reviewer(-\d+)?$")

# Block-reason substrings that indicate the candidate is NOT eligible for
# review or submit and the worker (or last reviewer) said so explicitly.
# These match the LATEST block reason / risk_notes only — NEVER the card title
# (titles can carry the previous reviewer's framing on opt-return cards).
# Note: "do not submit directly" is intentionally NOT here — it's a bank
# signal, handled by the sub-floor branch.
NO_FIX_MARKERS = (
    "still not enough",
    "still needs hidden-safe proof",
    "not a hidden-safety upgrade",
    "no singleton/all-zero repair preserves",
    "do not resend this exact candidate unchanged",
    "no-fix",
    "no hidden-safe upgrade",
    # Worker analyzed the task and concluded no improvement is possible.
    # The block reason is analytical-only; there's no candidate to review.
    # Close the card so the worker frees up for a new target.
    "no candidate built",
    "no candidate produced",
    "no improvement possible",
    "no viable optimization",
)

# Risk-note markers that elevate a candidate above auto-bank/auto-escalate
# and force human review. Match conservatively.
HIGH_RISK_MARKERS = (
    "hidden-bad",
    "public-flat",
    "public flat",
    "stale lineage",
    "stale-lineage",
    "decoder assumption",
    "package order",
    "package-order",
    "may write extra",
    "off-contract",
)

JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
HANDOFF_HEADER_PATTERN = re.compile(r"review[- ]required handoff:?\s*\n?\s*```?(?:json)?\s*\n?", re.IGNORECASE)
TASK_ID_PATTERN = re.compile(r"task\s*0*(\d+)", re.IGNORECASE)

# Summary-only fallback patterns. These fire only when the worker posted a
# block-reason summary string but no JSON envelope. We extract enough fields
# to safely escalate-to-review; the review card body explicitly flags the
# handoff as reconstructed so the reviewer verifies independently.
SUMMARY_TASK_ID_PATTERN = re.compile(r"task\s*0*(\d{1,4})", re.IGNORECASE)
SUMMARY_PTS_PATTERN = re.compile(r"(?:pts\s+|\+)?([+-]?\d+\.\d+)\s*pts|pts\s+([+-]?\d+\.\d+)", re.IGNORECASE)
SUMMARY_COST_ARROW_PATTERN = re.compile(r"cost\s+(\d+)\s*(?:->|→|=>)\s*(\d+)", re.IGNORECASE)
SUMMARY_COST_VS_PATTERN = re.compile(r"cost\s*\(\s*(\d+)\s+vs\s+(\d+)\s*\)", re.IGNORECASE)
SUMMARY_STAGED_PATTERN = re.compile(r"[Ss]taged:\s*([^\s,]+\.onnx)")
RELEASED_ASSIGNEE_PATTERN = re.compile(r"assigned to (neurogolf-worker-\d+)", re.IGNORECASE)


def extract_json_object(text: str, start: int = 0) -> str:
    """Pull a balanced JSON object starting at the first `{` >= `start`.

    Handles nested braces and strings with escaped quotes. Returns empty
    string if no balanced object is found.
    """
    n = len(text)
    i = text.find("{", start)
    if i == -1:
        return ""
    depth = 0
    in_str = False
    escape = False
    j = i
    while j < n:
        ch = text[j]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
        j += 1
    return ""


@dataclass
class Handoff:
    """Parsed review-required handoff structure from a worker comment."""

    raw_text: str
    parsed: dict[str, Any] = field(default_factory=dict)
    candidate_id: str = ""
    task_id: str = ""
    point_delta: float | None = None
    cost_delta: int | None = None
    expected_point_gain: float | None = None
    package_sha: str = ""
    lineage_base: str = ""
    changed_entries: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    # Added by the submit-gate redesign: workers self-report the algorithm family
    # they used so submit_gate.py can build a per-task "don't repeat these
    # families" blacklist for retry directives. Empty when the worker handoff
    # pre-dates the playbook update.
    algorithm_family: str = ""
    retry_attempt_index: int | None = None
    parse_error: str = ""


@dataclass
class Verdict:
    """Decision for one blocked card."""

    card_id: str
    assignee: str
    title: str
    block_reason: str
    handoff: Handoff
    # bank | no-fix | escalate-to-review | manual | awaiting-session | retry-give-up
    action: str
    reason: str  # one-line human-readable
    new_review_card: dict[str, Any] | None = None
    new_registry_entry: dict[str, Any] | None = None
    gave_up: bool = False


_POOL_B_PROFILES_CACHE: set[str] | None = None


def _is_pool_b_profile(assignee: str) -> bool:
    """True if ``assignee`` is a Pool B manually-driven Claude Code profile.

    Pool B = profiles whose model is ``none-claude-code-manual`` (workers
    5/7/8 today). They have no Hermes gateway daemon by design — the
    operator drives them via a Claude Code ``/loop`` window. Cards on
    these profiles WILL accumulate dispatcher-give-up events from
    instant subprocess crashes, but unassigning them would orphan the
    manual lane — the playbook's documented manual pickup path expects
    the assignee to remain intact.

    Reuses ``tools.neurogolf_dependency_serializer.manual_claude_code_profiles``
    so the source of truth (``hermes profile list`` model column) is
    shared with the orphan-fallback picker. Result cached in
    ``_POOL_B_PROFILES_CACHE`` to avoid re-probing per card on the same
    auto_triage run; cron restarts re-read.
    """
    if not assignee:
        return False
    global _POOL_B_PROFILES_CACHE
    if _POOL_B_PROFILES_CACHE is None:
        try:
            from tools.neurogolf_dependency_serializer import manual_claude_code_profiles
            _POOL_B_PROFILES_CACHE = set(manual_claude_code_profiles())
        except Exception:
            # If the helper fails (subprocess error, import issue), default
            # to empty set — auto_triage falls back to old behavior. Safer
            # than guessing which profiles are manual.
            _POOL_B_PROFILES_CACHE = set()
    return assignee in _POOL_B_PROFILES_CACHE


def _apply_result_status(result: dict[str, Any]) -> str:
    """Pick the ``ok`` / ``failed`` summary for an apply result dict.

    Different apply_* functions return different rc keys depending on what
    hermes operations they invoke. A result is "ok" iff every present rc
    key equals 0. The default rc to inspect is ``complete_rc`` (used by
    bank / no-fix / escalate / abandon paths). The stale-unassign path
    instead reports ``comment_rc`` + ``assign_rc`` + ``unblock_rc``; all
    must succeed for the operation to be considered ok.
    """
    rc_keys = [k for k in result if k.endswith("_rc")]
    if not rc_keys:
        return "failed"
    return "ok" if all(result.get(k) == 0 for k in rc_keys) else "failed"


def _load_task_outcome_doc() -> dict[str, Any] | None:
    """Read .hermes/board/task_outcome_registry.json — None on missing/corrupt.

    Used by main() to pass the doc into ``apply_stale_unassign`` so the
    broaden-scope check has the per-task attempts_this_pass data. Never
    raises — defensive against the registry not existing yet.
    """
    path = REPO_ROOT / ".hermes" / "board" / "task_outcome_registry.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_current_pass_count() -> int:
    """Read pass_count from .hermes/board/task_backlog_cursor.json. 0 on missing."""
    path = REPO_ROOT / ".hermes" / "board" / "task_backlog_cursor.json"
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    val = data.get("pass_count")
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    return 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_hermes(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Force-UTF-8 hermes invoke. Card bodies often contain emoji/extended
    characters that crash the Windows cp1252 default if `text=True` is set.
    """
    return subprocess.run(
        ["hermes", "kanban", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )


def _stdout_utf8(proc: subprocess.CompletedProcess[bytes]) -> str:
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def _stderr_utf8(proc: subprocess.CompletedProcess[bytes]) -> str:
    return (proc.stderr or b"").decode("utf-8", errors="replace")


def _card_has_review_required(card: dict[str, Any]) -> bool:
    """True if the card's title or latest_summary mentions 'review-required'.

    Used by the --skip-review-required Phase 3 flag to filter out cards that
    tools/submit_gate.py now owns. Defensive: any field we'd normally consider
    as a 'latest summary' source is scanned.
    """
    haystacks = [
        card.get("title") or "",
        card.get("latest_summary") or "",
        card.get("body") or "",
    ]
    for text in haystacks:
        if isinstance(text, str) and "review-required" in text.lower():
            return True
    return False


def hermes_list_blocked() -> list[dict[str, Any]]:
    proc = run_hermes(["list", "--status", ACTIVE_BLOCKED_STATUS, "--json"])
    if proc.returncode != 0:
        return []
    try:
        return [c for c in json.loads(_stdout_utf8(proc)) if isinstance(c, dict)]
    except json.JSONDecodeError:
        return []


def hermes_show(card_id: str) -> str:
    proc = run_hermes(["show", card_id])
    return _stdout_utf8(proc) if proc.returncode == 0 else ""


def reconstruct_handoff_from_summary(
    card: dict[str, Any],
    summary: str,
    card_show: str,
) -> Handoff | None:
    """Synthesize a Handoff from a summary-only block reason.

    Workers occasionally block with a free-text summary instead of the
    structured JSON envelope. If the summary is parseable (task_id + cost
    pair + staged ONNX), recover enough to escalate-to-review. The reviewer
    must verify independently — risk_notes flags the reconstruction.

    Returns None if the summary lacks the minimum signal (task_id + either
    a cost pair or a staged ONNX path).
    """
    if not summary:
        return None
    task_match = SUMMARY_TASK_ID_PATTERN.search(summary)
    if not task_match:
        return None
    task_id = f"task{int(task_match.group(1)):03d}"

    pts: float | None = None
    pts_match = SUMMARY_PTS_PATTERN.search(summary)
    if pts_match:
        for grp in pts_match.groups():
            if grp:
                try:
                    pts = float(grp)
                    break
                except ValueError:
                    continue

    current_cost: int | None = None
    candidate_cost: int | None = None
    arrow_match = SUMMARY_COST_ARROW_PATTERN.search(summary)
    if arrow_match:
        current_cost = int(arrow_match.group(1))
        candidate_cost = int(arrow_match.group(2))
    else:
        vs_match = SUMMARY_COST_VS_PATTERN.search(summary)
        if vs_match:
            candidate_cost = int(vs_match.group(1))
            current_cost = int(vs_match.group(2))

    staged_path = ""
    staged_match = SUMMARY_STAGED_PATTERN.search(summary)
    if staged_match:
        staged_path = staged_match.group(1).strip()

    # Require at least a cost pair OR a staged ONNX so we don't escalate
    # cards with truly empty signal.
    if current_cost is None and not staged_path:
        return None

    package_sha = ""
    if staged_path:
        candidate_path = (REPO_ROOT / staged_path.replace("\\", "/")).resolve()
        try:
            if candidate_path.is_file() and REPO_ROOT in candidate_path.parents:
                package_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest().upper()
        except OSError:
            package_sha = ""

    worker_owner = card.get("assignee") or ""
    if not worker_owner:
        released = RELEASED_ASSIGNEE_PATTERN.search(card_show)
        if released:
            worker_owner = released.group(1)

    cand_id = f"{task_id}_reconstructed_{card.get('id') or 'unknown'}"

    parsed = {
        "task_id": task_id,
        "candidate_id": cand_id,
        "worker_owner": worker_owner,
        "package_sha256": package_sha,
        "lineage_base": "current-best",
        "changed_package_entries": [f"{task_id}.onnx"],
        "scores": {
            "current_cost": current_cost,
            "candidate_cost": candidate_cost,
            "cost_delta": (candidate_cost - current_cost)
                if (current_cost is not None and candidate_cost is not None) else None,
            "point_delta": pts,
        },
        "expected_point_gain": pts,
        "staged_onnx_path": staged_path,
        "_reconstructed_from": "block_reason_summary",
        "_reconstructed_by": "tools/auto_triage_blocked.py",
        "_reconstruction_summary": summary[:500],
    }

    return Handoff(
        raw_text=json.dumps(parsed, indent=2),
        parsed=parsed,
        candidate_id=cand_id,
        task_id=task_id,
        point_delta=pts,
        cost_delta=parsed["scores"]["cost_delta"],
        expected_point_gain=pts,
        package_sha=package_sha,
        lineage_base="current-best",
        changed_entries=[f"{task_id}.onnx"],
        tests_run=[],
        risk_notes=[
            "RECONSTRUCTED HANDOFF: worker posted summary-only block reason without JSON envelope.",
            "Reviewer must verify staged ONNX, recompute costs, and re-run verify_against_live + adversarial suite before promoting.",
        ],
    )


def parse_handoff(
    card_id: str,
    *,
    card: dict[str, Any] | None = None,
    block_reason: str = "",
    card_show: str | None = None,
) -> Handoff:
    raw = card_show if card_show is not None else hermes_show(card_id)
    if not raw:
        return Handoff(raw_text="", parse_error="hermes show returned empty")
    # Pull comment region only (Events/Runs section adds noise).
    comments_section = raw
    cutoff = raw.find("\nEvents (")
    if cutoff != -1:
        comments_section = raw[:cutoff]
    # Anchor the search at the LATEST "review-required handoff:" line so we
    # don't pick up an older comment's JSON when an opt-return card has stacked
    # comments. Each card has multiple comments in chronological order — the
    # last `review-required handoff:` marks the fresh worker submission.
    header_matches = list(HANDOFF_HEADER_PATTERN.finditer(comments_section))
    if header_matches:
        search_start = header_matches[-1].end()
    else:
        search_start = 0
    payload_str = ""
    # Prefer fenced ```json``` block if present after the header.
    fence_match = JSON_BLOCK_PATTERN.search(comments_section, search_start)
    if fence_match:
        payload_str = fence_match.group(1)
    else:
        # Fall back to balanced-brace extractor.
        payload_str = extract_json_object(comments_section, search_start)
    if not payload_str:
        # Summary-only fallback: worker posted a free-text block reason
        # instead of the structured envelope. Try to recover task_id + costs
        # + staged ONNX from the summary so the candidate isn't lost.
        if card is not None and block_reason:
            fallback = reconstruct_handoff_from_summary(card, block_reason, raw)
            if fallback is not None:
                return fallback
        return Handoff(raw_text=comments_section[:500], parse_error="no JSON object found in latest handoff comment")
    try:
        parsed = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        return Handoff(raw_text=payload_str[:500], parse_error=f"JSON parse failed: {exc.msg}")
    # Schema-agnostic: walk the parsed tree, find ANY field that looks like
    # a point delta or any baseline/candidate cost pair. Workers spawned by
    # different LLMs use subtly different JSON shapes (e.g. `scores.point_delta`,
    # `score_delta.delta_points`, `expected_gain.point_gain`, `candidate.points_delta_vs_*`,
    # `score_current_live.cost` + `score_candidate.cost`, top-level
    # `base_cost`/`best_cost`...).
    point_delta, _delta_source = extract_point_delta(parsed)
    cost_delta = None
    for path, key, value in _walk(parsed):
        if cost_delta is None and _is_number(value) and re.search(r"(^|[_-])(cost[_-]?delta|delta[_-]?cost)$", key, re.IGNORECASE):
            cost_delta = int(value)
            break
    expected = point_delta  # extract_point_delta already prioritized direct expected_* keys
    candidate_id = parsed.get("candidate_id") or parsed.get("candidate") or ""
    task_id = parsed.get("task_id") or parsed.get("task") or ""
    task_match = TASK_ID_PATTERN.search(task_id if isinstance(task_id, str) else "")
    if task_match:
        task_id = f"task{int(task_match.group(1)):03d}"
    package_sha = parsed.get("package_sha256") or parsed.get("model_sha256") or ""
    return Handoff(
        raw_text=payload_str,
        parsed=parsed,
        candidate_id=str(candidate_id),
        task_id=str(task_id),
        point_delta=float(point_delta) if isinstance(point_delta, (int, float)) else None,
        cost_delta=int(cost_delta) if isinstance(cost_delta, (int, float)) else None,
        expected_point_gain=float(expected) if isinstance(expected, (int, float)) else None,
        package_sha=str(package_sha),
        lineage_base=str(parsed.get("lineage_base") or parsed.get("lineage_base_candidate_id") or ""),
        changed_entries=list(parsed.get("changed_package_entries") or parsed.get("changed_entries") or []),
        tests_run=list(parsed.get("tests_run") or []),
        risk_notes=list(parsed.get("risk_notes") or parsed.get("risk") or []),
        algorithm_family=str(parsed.get("algorithm_family") or ""),
        retry_attempt_index=(
            int(parsed["retry_attempt_index"])
            if isinstance(parsed.get("retry_attempt_index"), (int, float))
            else None
        ),
    )


def extract_block_reason(card_show: str) -> str:
    match = re.search(r"blocked\s+\{'reason':\s+'([^']+)'\}", card_show)
    if match:
        return match.group(1)
    summary_match = re.search(r"Latest summary:\s*\n(review-required:[^\n]+)", card_show)
    return summary_match.group(1) if summary_match else ""


GAVE_UP_PATTERN = re.compile(r"gave_up\s+\{'failures':", re.IGNORECASE)
AGENT_CRASH_PATTERN = re.compile(r"Agent crashed\s+\d+x", re.IGNORECASE)
DISPATCHER_GIVE_UP_REASON_PATTERN = re.compile(r"\bpid\s+\d+\s+not alive\b", re.IGNORECASE)
EVENT_LINE_PATTERN = re.compile(r"^\s*\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+(?:\[run \d+\]\s+)?(\w+)", re.MULTILINE)
SUCCESSFUL_RECOVERY_EVENTS = {"unblocked", "completed", "heartbeat"}


def is_session_broken(card_show: str, *, now_ts: float | None = None, stale_seconds: int = 3600) -> bool:
    """True when a dispatcher-give-up card looks like its worker session is
    genuinely dead (not just transient).

    Heuristics (any of these is enough — they're confirmatory of each other):
        - Last event in the log is older than ``stale_seconds`` (default 1hr).
        - All spawned worker pids in the last 3 spawn/crash pairs lived
          < 30 seconds before crashing.

    Reads the event log from ``hermes kanban show`` output. Times are parsed
    from event-line prefixes like ``[2026-05-18 18:44]``.

    Used by the Piece 1 stale-unassign branch in ``classify``. Tests inject
    ``now_ts`` to make the time-based check deterministic.
    """
    import time
    events_start = card_show.find("Events (")
    if events_start == -1:
        return False
    events_section = card_show[events_start:]
    runs_start = events_section.find("\nRuns (")
    if runs_start != -1:
        events_section = events_section[:runs_start]

    # Extract event timestamps + kinds.
    event_pattern = re.compile(
        r"^\s*\[(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?\]\s+(?:\[run \d+\]\s+)?(\w+)",
        re.MULTILINE,
    )
    spawn_crash_pairs: list[tuple[float, float]] = []  # (spawn_ts, crash_ts)
    pending_spawn: float | None = None
    latest_ts: float | None = None
    import datetime as _dt
    for match in event_pattern.finditer(events_section):
        year, mo, day, hh, mm, ss, kind = match.groups()
        try:
            ts = _dt.datetime(
                int(year), int(mo), int(day), int(hh), int(mm), int(ss or 0),
                tzinfo=_dt.timezone.utc,
            ).timestamp()
        except ValueError:
            continue
        latest_ts = ts
        if kind.lower() == "spawned":
            pending_spawn = ts
        elif kind.lower() == "crashed" and pending_spawn is not None:
            spawn_crash_pairs.append((pending_spawn, ts))
            pending_spawn = None

    if latest_ts is None:
        return False

    now = now_ts if now_ts is not None else time.time()

    # Stale by wall clock?
    if (now - latest_ts) > stale_seconds:
        return True

    # All recent (last 3) spawn-crash pairs lived <30s? Pattern of immediate-
    # crash on every spawn → session definitely broken.
    if len(spawn_crash_pairs) >= 3:
        recent = spawn_crash_pairs[-3:]
        if all((crash - spawn) < 30 for spawn, crash in recent):
            return True

    return False


def is_dispatcher_give_up(card_show: str) -> bool:
    """True when the card is CURRENTLY blocked because Hermes dispatcher gave up.

    Symptom: no review-required handoff was posted; the events log shows
    `gave_up {'failures': N, ...}` after one or more `crashed {...}` events,
    and the latest summary mentions "Agent crashed Nx: pid X not alive".
    Root cause is almost always: no Claude Code session is bound to the
    profile, so the spawned worker process dies immediately.

    Critical: a card that hit gave_up earlier, was then unblocked and
    successfully respawned (e.g. after `tools/reset_stale_crash_counters.py`
    applied), must NOT be flagged. The latest event in the log must still be
    the give_up — any later `unblocked`, `heartbeat`, or `completed` event
    clears the diagnosis.
    """
    if "review-required" in card_show.lower():
        return False
    # Slice to the Events section only — earlier "Latest summary:" may
    # quote a stale gave_up reason even after recovery.
    events_start = card_show.find("Events (")
    if events_start == -1:
        return False
    events_section = card_show[events_start:]
    runs_start = events_section.find("\nRuns (")
    if runs_start != -1:
        events_section = events_section[:runs_start]
    # Walk events; track the LAST event kind seen.
    last_kind = ""
    for match in EVENT_LINE_PATTERN.finditer(events_section):
        last_kind = match.group(1).lower()
    if not last_kind:
        # Fall back to the legacy any-occurrence check.
        if GAVE_UP_PATTERN.search(card_show):
            return True
        if AGENT_CRASH_PATTERN.search(card_show) and DISPATCHER_GIVE_UP_REASON_PATTERN.search(card_show):
            return True
        return False
    return last_kind == "gave_up"


def has_marker(text: str, markers: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(m in lower for m in markers)


ITERATION_BUDGET_MARKER = "iteration budget exhausted"


def is_iteration_budget_exhausted(block_reason: str) -> bool:
    return ITERATION_BUDGET_MARKER in (block_reason or "").lower()


def classify(
    card: dict[str, Any],
    handoff: Handoff,
    block_reason: str,
    *,
    card_show_text: str = "",
    gave_up_action: str = "leave",
) -> Verdict:
    cid = card.get("id") or "?"
    assignee = card.get("assignee") or ""
    title = card.get("title") or ""

    # 0a. Iteration-budget-exhausted cards with NO posted handoff: the worker
    #     ran out of LLM turns before reaching a stop point. There's no
    #     candidate to review and no useful output. Abandon (complete with
    #     note) so the worker frees up; the autoforeman re-picks the next
    #     backlog target which then runs under the bumped 150-turn budget.
    if is_iteration_budget_exhausted(block_reason) and handoff.parse_error:
        return Verdict(
            cid, assignee, title, block_reason, handoff,
            "abandon-iteration-budget",
            f"iteration budget exhausted before worker posted any handoff; abandoning so worker frees up",
        )

    # 0b. Dispatcher-give-up cards: no LLM session was bound to the worker
    #     profile, so the spawned worker process crashed twice and Hermes gave
    #     up. There's no candidate to review or bank — the card never ran.
    if card_show_text and is_dispatcher_give_up(card_show_text):
        # Piece 1: session-broken detection. When the give-up looks
        # definitively stale (>1hr or all recent spawns crashed in <30s),
        # default to unassigning instead of leaving — lets dep-serializer
        # route to whichever worker has a functional gateway. Less
        # destructive than --gave-up-action=abandon.
        #
        # Profile-aware exemption (2026-05-19): if the card is assigned to
        # a Pool B "none-claude-code-manual" profile (workers 5/7/8 today),
        # do NOT stale-unassign. Those profiles HAVE NO HERMES GATEWAY by
        # design — every dispatcher spawn there crashes immediately, which
        # would make the session-broken heuristic ALWAYS fire after 1hr.
        # The playbook's documented manual pickup path (step 2: find the
        # blocked card, unblock+claim yourself) expects the assignee to
        # remain intact. Unassigning would orphan the manual /loop lane.
        if is_session_broken(card_show_text) and not _is_pool_b_profile(assignee):
            return Verdict(
                cid, assignee, title, block_reason, handoff,
                "stale-unassign",
                f"dispatcher gave up AND session looks broken (no recent activity; spawn-crash pattern); unassigning so dep-serializer can re-route",
                gave_up=True,
            )
        action = {
            "leave": "awaiting-session",
            "unblock": "retry-give-up",
            "abandon": "abandon-give-up",
        }.get(gave_up_action, "awaiting-session")
        verdict = Verdict(
            cid, assignee, title, block_reason, handoff,
            action,
            f"dispatcher gave up (no LLM session bound to {assignee or 'profile'})",
            gave_up=True,
        )
        return verdict

    # 1. Cards already in the reviewer's lane stay there; nothing to do.
    if REVIEWER_PROFILE_PATTERN.match(assignee):
        return Verdict(cid, assignee, title, block_reason, handoff,
                       "manual", "reviewer-owned card; needs human/agent review")

    # 2. Worker cards: apply the deterministic decision tree.
    # NOTE: the card title can carry the *previous* reviewer's framing for
    # opt-return cards (e.g. "still below standalone submit floor"). The fresh
    # worker submission lives in the latest block reason + handoff. Match only
    # against those — never against the title.
    handoff_risk_text = " ".join(handoff.risk_notes)

    # 2a. Explicit no-fix signals in the latest block reason or the worker's
    #     own risk_notes acknowledging the previous concern still applies.
    no_fix_text = block_reason + " " + handoff_risk_text
    if has_marker(no_fix_text, NO_FIX_MARKERS):
        return Verdict(cid, assignee, title, block_reason, handoff,
                       "no-fix", f"no-fix marker in latest block-reason/risk-notes: {block_reason[:60]}")

    if handoff.parse_error:
        return Verdict(cid, assignee, title, block_reason, handoff,
                       "manual", f"could not parse handoff: {handoff.parse_error}")

    # 2b. Determine the candidate's expected LB gain.
    gain = handoff.expected_point_gain
    if gain is None:
        gain = handoff.point_delta

    # 2c. Sub-floor candidates -> bank for integration. No standalone submit.
    if gain is not None and gain < STANDALONE_SUBMIT_FLOOR:
        registry_entry = build_banked_registry_entry(card, handoff)
        return Verdict(cid, assignee, title, block_reason, handoff,
                       "bank", f"expected gain {gain:+.4f} < floor {STANDALONE_SUBMIT_FLOOR}; bank-for-integration",
                       new_registry_entry=registry_entry)

    # 2d. Above floor: route through review unless risk markers flag manual.
    risk_text = " ".join(handoff.risk_notes)
    if has_marker(risk_text, HIGH_RISK_MARKERS):
        return Verdict(cid, assignee, title, block_reason, handoff,
                       "manual", f"high-risk markers in risk_notes: gain {gain:+.4f}; leave for reviewer agent")

    # 2e. Above floor, clean: create review card and complete worker card.
    if gain is not None and gain >= STANDALONE_SUBMIT_FLOOR:
        review_card = build_review_card_spec(card, handoff)
        return Verdict(cid, assignee, title, block_reason, handoff,
                       "escalate-to-review", f"expected gain {gain:+.4f} >= floor; create review card",
                       new_review_card=review_card)

    # 2f. No clear delta but we DO have a parsed handoff (worker produced a
    #     review-required candidate; the reviewer can decide). This used to
    #     fall through to "manual" and pile up. Default-escalate when we have
    #     a candidate_id or task_id from the handoff — the reviewer figures it
    #     out. Only "manual" if we truly couldn't parse anything.
    if handoff.candidate_id or handoff.task_id:
        review_card = build_review_card_spec(card, handoff)
        return Verdict(cid, assignee, title, block_reason, handoff,
                       "escalate-to-review",
                       f"no clear point-delta but handoff has candidate_id={handoff.candidate_id or '(missing)'}; let reviewer decide",
                       new_review_card=review_card)

    # 2g. No candidate id, no delta, no clear no-fix signal — truly manual.
    return Verdict(cid, assignee, title, block_reason, handoff,
                   "manual", "no candidate_id in handoff and no decisive signal; needs human review")


def build_banked_registry_entry(card: dict[str, Any], handoff: Handoff) -> dict[str, Any]:
    return {
        "schema": "neurogolf.candidate.v2-draft",
        "candidate_id": handoff.candidate_id or f"{handoff.task_id}_{card.get('id')}",
        "task_id": handoff.task_id,
        "worker_owner": card.get("assignee") or "",
        "expected_point_gain": handoff.expected_point_gain,
        "submit_queue_status": "banked-for-integration",
        "status": "banked-by-auto-triage",
        "approval_mode": "auto-rate-limited",
        "chain_key": f"review:{handoff.candidate_id}" if handoff.candidate_id else "",
        "source_kanban_task_id": card.get("id"),
        "package_sha256": handoff.package_sha,
        "changed_task_ids": handoff.changed_entries or ([handoff.task_id] if handoff.task_id else []),
        "lineage_base_candidate_id": handoff.lineage_base,
        "banked_at": utc_now(),
        "banked_by": "tools/auto_triage_blocked.py",
        "notes": "Auto-banked: expected gain below standalone submit floor. Available for additive stacking.",
    }


def build_review_card_spec(card: dict[str, Any], handoff: Handoff) -> dict[str, Any]:
    candidate_id = handoff.candidate_id
    task_id = handoff.task_id
    chain_key = f"review:{candidate_id}" if candidate_id else f"review:{card.get('id')}"
    reconstructed = isinstance(handoff.parsed, dict) and handoff.parsed.get("_reconstructed_from") == "block_reason_summary"
    title_prefix = "[review-reconstructed" if reconstructed else "[review"
    title = f"{title_prefix} {task_id} {candidate_id}] auto-triaged from {card.get('id')}"[:200]
    family_line = (
        f"algorithm_family: {handoff.algorithm_family}"
        if handoff.algorithm_family
        else "algorithm_family: (worker did not report — handoff predates submit-gate playbook update)"
    )
    attempt_line = (
        f"retry_attempt_index: {handoff.retry_attempt_index}"
        if handoff.retry_attempt_index is not None
        else "retry_attempt_index: (worker did not report)"
    )
    body_lines = [
        f"chain_key: {chain_key}",
        f"source_candidate_id: {candidate_id}",
        f"source_worker_task_id: {card.get('id')}",
        f"source_changed_task_ids: {handoff.changed_entries or [task_id]}",
        family_line,
        attempt_line,
        "duplicate_policy: do-not-create-if-live-card-with-same-chain-key-exists",
        "",
    ]
    if reconstructed:
        body_lines.extend([
            "RECONSTRUCTED HANDOFF — original worker posted a summary-only",
            "block reason without the structured JSON envelope. The fields",
            "below were recovered from the block reason text plus the staged",
            "ONNX file's on-disk SHA. Reviewer must independently verify the",
            "candidate before promoting.",
            "",
        ])
    body_lines.extend([
        "Created by tools/auto_triage_blocked.py — handoff JSON copied verbatim below.",
        "",
        "```json",
        handoff.raw_text,
        "```",
    ])
    return {
        "title": title,
        "body": "\n".join(body_lines),
        "assignee": "neurogolf-reviewer",
        "priority": 75,
        "parent_id": card.get("id") or "",
        "chain_key": chain_key,
    }


def apply_bank(verdict: Verdict, registry_path: Path) -> dict[str, Any]:
    doc = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = doc.setdefault("entries", [])
    existing_ids = {e.get("candidate_id") for e in entries if isinstance(e, dict)}
    new_entry = verdict.new_registry_entry or {}
    if new_entry.get("candidate_id") and new_entry["candidate_id"] not in existing_ids:
        entries.append(new_entry)
    doc["updated_at"] = utc_now()
    # Atomic write: write to tmp first, fsync, then rename. Prevents corruption
    # if the process is killed mid-write (cron jobs that run as no-agent may be
    # SIGKILLed by the dispatcher when the launcher lock is contested).
    tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    tmp.replace(registry_path)
    result_text = (
        f"auto-triage: banked candidate {new_entry.get('candidate_id')} "
        f"(expected gain {new_entry.get('expected_point_gain')}); below standalone floor."
    )
    proc = run_hermes(["complete", verdict.card_id, "--result", result_text])
    return {"complete_rc": proc.returncode, "complete_stderr": _stderr_utf8(proc).strip()[:200]}


def apply_no_fix(verdict: Verdict) -> dict[str, Any]:
    result_text = (
        "auto-triage: closed as no-fix-this-pass; block reason or returned-from-review body "
        "indicates the candidate is not a hidden-safety upgrade. Worker may take a new target."
    )
    proc = run_hermes(["complete", verdict.card_id, "--result", result_text])
    return {"complete_rc": proc.returncode, "complete_stderr": _stderr_utf8(proc).strip()[:200]}


def apply_escalate(verdict: Verdict) -> dict[str, Any]:
    review = verdict.new_review_card or {}
    # Check for an existing review card with the same chain_key before creating.
    list_proc = run_hermes(["list", "--json"])
    if list_proc.returncode == 0:
        try:
            cards = json.loads(_stdout_utf8(list_proc))
        except json.JSONDecodeError:
            cards = []
        for c in cards:
            if not isinstance(c, dict):
                continue
            if (c.get("status") or "").lower() not in ("ready", "running", "todo", "blocked"):
                continue
            if review["chain_key"] in (c.get("body") or "") + (c.get("title") or ""):
                return {"create_rc": 0, "skipped": "duplicate chain_key already exists", "existing_id": c.get("id")}
    create_args = [
        "create",
        review["title"],
        "--body", review["body"],
        "--assignee", review["assignee"],
        "--priority", str(review["priority"]),
        "--created-by", "tools/auto_triage_blocked.py",
        "--idempotency-key", review["chain_key"],
        "--json",
    ]
    proc = run_hermes(create_args)
    new_id = ""
    if proc.returncode == 0:
        try:
            data = json.loads(_stdout_utf8(proc))
            if isinstance(data, dict):
                new_id = data.get("id") or ""
        except json.JSONDecodeError:
            pass
    # Link parent->child if create succeeded and parent_id is present.
    if new_id and review.get("parent_id"):
        run_hermes(["link", review["parent_id"], new_id])
    # Complete the worker card now that work is handed off.
    result_text = (
        f"auto-triage: handed off to review card {new_id or '(create-failed)'}; "
        f"candidate {verdict.handoff.candidate_id} expected gain "
        f"{verdict.handoff.expected_point_gain or verdict.handoff.point_delta}."
    )
    complete_proc = run_hermes(["complete", verdict.card_id, "--result", result_text])
    return {
        "create_rc": proc.returncode,
        "create_stderr": _stderr_utf8(proc).strip()[:200],
        "new_review_card_id": new_id,
        "complete_rc": complete_proc.returncode,
    }


def apply_abandon_iteration_budget(verdict: Verdict) -> dict[str, Any]:
    """Complete an iteration-budget-no-handoff card so the worker frees up."""
    result_text = (
        f"auto-triage: abandoned — worker ran out of LLM iteration budget before "
        f"posting a review-required handoff. Worker is freed; the autoforeman "
        f"will pick the next backlog target under the bumped 150-turn budget. "
        f"Block reason: {verdict.block_reason[:120]}"
    )
    proc = run_hermes(["complete", verdict.card_id, "--result", result_text])
    return {"complete_rc": proc.returncode, "complete_stderr": _stderr_utf8(proc).strip()[:200]}


def apply_retry_give_up(verdict: Verdict) -> dict[str, Any]:
    """Unblock a dispatcher-give-up card so Hermes re-dispatches.

    Only useful after the user binds an LLM session to the worker profile.
    Without a session, the card will crash again and be re-blocked within
    a few minutes.
    """
    proc = run_hermes(["unblock", verdict.card_id])
    return {"unblock_rc": proc.returncode, "unblock_stderr": _stderr_utf8(proc).strip()[:200]}


def apply_abandon_give_up(verdict: Verdict) -> dict[str, Any]:
    """Complete a dispatcher-give-up card as abandoned, freeing the worker.

    The card is closed with status `abandoned-no-llm-session`. The autoforeman
    will pick the next backlog target for the now-idle worker. If sessions
    still aren't bound, the next card will crash too — caller should normally
    prefer `--gave-up-action=leave` instead and bind sessions before retrying.
    """
    result_text = (
        f"auto-triage: abandoned — Hermes dispatcher gave up after 2 spawn crashes "
        f"for {verdict.assignee or 'unbound profile'}. No LLM session was attached. "
        f"Bind a Claude session to this profile and re-route via the backlog."
    )
    proc = run_hermes(["complete", verdict.card_id, "--result", result_text])
    return {"complete_rc": proc.returncode, "complete_stderr": _stderr_utf8(proc).strip()[:200]}


_RETURN_REPAIR_TITLE = re.compile(
    r"^\s*\[(?:repair|optimization[- ]return|task\d+\s+return)\b|^\s*repair\s+task\d+",
    re.IGNORECASE,
)


def is_return_or_repair_card(card_or_title: dict[str, Any] | str) -> bool:
    """True if the card's title looks like a repair / opt-return / task-NNN-return.

    Used by Piece 2 to gate broaden-scope rewriting only on cards where the
    "narrow-scope exhausted" semantics apply (existing return paths). Worker
    cards and bank cards don't qualify — they get scope-broadening through
    submit_gate's retry directive instead.
    """
    title = card_or_title.get("title", "") if isinstance(card_or_title, dict) else card_or_title
    if not isinstance(title, str):
        return False
    return bool(_RETURN_REPAIR_TITLE.match(title))


def count_attempts_this_pass(
    task_outcome_doc: dict[str, Any] | None,
    task_id: str,
    current_pass: int,
) -> int:
    """Return attempts_this_pass for the task — or 0 if stale / missing.

    Stale = the registry's ``last_attempt_pass`` doesn't match ``current_pass``
    (counters from a prior pass don't carry forward). Mirrors the
    submit_gate.py / autoforeman semantics exactly.
    """
    if not isinstance(task_outcome_doc, dict) or not task_id:
        return 0
    entry = (task_outcome_doc.get("tasks") or {}).get(task_id)
    if not isinstance(entry, dict):
        return 0
    last_pass = entry.get("last_attempt_pass")
    try:
        last_pass_int = int(last_pass) if last_pass is not None else None
    except (TypeError, ValueError):
        return 0
    if last_pass_int != current_pass:
        return 0
    attempts = entry.get("attempts_this_pass") or 0
    try:
        return int(attempts)
    except (TypeError, ValueError):
        return 0


def get_attempted_families_this_pass(
    task_outcome_doc: dict[str, Any] | None,
    task_id: str,
    current_pass: int,
) -> list[str]:
    """Return attempted_families for the task in CURRENT pass — or [] if stale."""
    if not isinstance(task_outcome_doc, dict) or not task_id:
        return []
    entry = (task_outcome_doc.get("tasks") or {}).get(task_id)
    if not isinstance(entry, dict):
        return []
    last_pass = entry.get("last_attempt_pass")
    try:
        last_pass_int = int(last_pass) if last_pass is not None else None
    except (TypeError, ValueError):
        return []
    if last_pass_int != current_pass:
        return []
    fams = entry.get("attempted_families")
    return list(fams) if isinstance(fams, list) else []


def should_broaden_scope(
    card: dict[str, Any],
    task_outcome_doc: dict[str, Any] | None,
    current_pass: int,
    *,
    max_attempts_before_broaden: int = 3,
) -> bool:
    """Piece 2 trigger: stale-unassign fires AND card is a return/repair AND
    has accumulated 3+ attempts in the current pass."""
    if not is_return_or_repair_card(card):
        return False
    task_id = _extract_task_id_from_card(card)
    if not task_id:
        return False
    return count_attempts_this_pass(task_outcome_doc, task_id, current_pass) >= max_attempts_before_broaden


def _extract_task_id_from_card(card: dict[str, Any]) -> str:
    """Pull a `taskNNN` id out of the card title or body — best effort."""
    haystack = " ".join(
        str(card.get(k) or "")
        for k in ("title", "body", "description", "latest_summary")
    )
    match = re.search(r"\btask(\d{3})\b", haystack, re.IGNORECASE)
    return f"task{match.group(1)}" if match else ""


def apply_stale_unassign(
    verdict: Verdict,
    *,
    task_outcome_doc: dict[str, Any] | None = None,
    current_pass: int = 0,
) -> dict[str, Any]:
    """Piece 1: unassign + unblock a session-broken give-up card.

    Sequence:
        1. ``hermes kanban assign <id> none`` — clear the (broken) assignee.
        2. (Piece 2) if this is a return/repair card with 3+ attempts this
           pass, also rewrite the card body with a broaden-scope directive
           ("abandon current ONNX, try a different family") BEFORE unblocking.
        3. ``hermes kanban unblock <id>`` — return to ready status.

    After this runs, the dep-serializer picks the card up on its next tick
    and routes it to whichever worker has a functional gateway. Less
    destructive than ``apply_abandon_give_up`` because the work survives.
    """
    audit_note = (
        f"auto-triage stale-unassign: dispatcher gave up AND session looks broken "
        f"(>1hr stale or all recent spawns crashed in <30s). Original assignee "
        f"{verdict.assignee or 'unbound'} cleared; dep-serializer will re-route."
    )
    comment_proc = run_hermes(["comment", verdict.card_id, "--body", audit_note])
    assign_proc = run_hermes(["assign", verdict.card_id, "none"])

    # Piece 2: maybe rewrite body before unblock
    broaden_applied = False
    broaden_proc_rc: int | None = None
    fake_card_for_check = {
        "id": verdict.card_id,
        "title": verdict.title,
    }
    if should_broaden_scope(fake_card_for_check, task_outcome_doc, current_pass):
        task_id = _extract_task_id_from_card(fake_card_for_check)
        families = get_attempted_families_this_pass(task_outcome_doc, task_id, current_pass)
        broaden_body = _build_broaden_scope_append(task_id, families, current_pass)
        broaden_proc = run_hermes(["comment", verdict.card_id, "--body", broaden_body])
        broaden_proc_rc = broaden_proc.returncode
        broaden_applied = True

    unblock_proc = run_hermes(["unblock", verdict.card_id])
    return {
        "comment_rc": comment_proc.returncode,
        "assign_rc": assign_proc.returncode,
        "assign_stderr": _stderr_utf8(assign_proc).strip()[:200],
        "unblock_rc": unblock_proc.returncode,
        "unblock_stderr": _stderr_utf8(unblock_proc).strip()[:200],
        "broaden_scope_applied": broaden_applied,
        "broaden_scope_rc": broaden_proc_rc,
    }


def _build_broaden_scope_append(task_id: str, families_tried: list[str], current_pass: int) -> str:
    """Build the Piece 2 broaden-scope body addendum, reusing the same
    family-directive infrastructure submit_gate uses for retry cards.
    """
    try:
        from tools.algorithm_family_directive import build_directive
    except ImportError:
        # Defensive: if the helper module isn't available for some reason,
        # fall back to a static directive so we still trigger.
        return (
            "## auto_triage broaden-scope directive\n\n"
            f"This task has accumulated 3+ failed narrow-scope attempts in pass "
            f"{current_pass}. Abandon the current ONNX entirely and rebuild "
            f"{task_id} from scratch with a fundamentally different algorithm "
            f"family. Families already tried: {families_tried or '(unknown)'}."
        )
    directive = build_directive(
        families_tried,
        retry_attempt_index=len(families_tried) + 1,
        reason=(
            f"3+ narrow-scope attempts exhausted this pass on {task_id}. "
            f"Broaden scope: abandon current ONNX and rebuild from scratch."
        ),
    )
    return (
        "## auto_triage broaden-scope directive\n\n"
        f"{directive}\n\n"
        "**Specifically**: if the current approach cannot yield a cost reduction, "
        f"abandon the current `{task_id}.onnx` entirely and rebuild it from "
        "scratch with a fundamentally different algorithm family. The narrow "
        "scope of this card's original directive has been exhausted."
    )


def apply(
    verdict: Verdict,
    registry_path: Path,
    *,
    task_outcome_doc: dict[str, Any] | None = None,
    current_pass: int = 0,
) -> dict[str, Any]:
    if verdict.action == "bank":
        return apply_bank(verdict, registry_path)
    if verdict.action == "no-fix":
        return apply_no_fix(verdict)
    if verdict.action == "escalate-to-review":
        return apply_escalate(verdict)
    if verdict.action == "retry-give-up":
        return apply_retry_give_up(verdict)
    if verdict.action == "abandon-give-up":
        return apply_abandon_give_up(verdict)
    if verdict.action == "abandon-iteration-budget":
        return apply_abandon_iteration_budget(verdict)
    if verdict.action == "stale-unassign":
        return apply_stale_unassign(
            verdict, task_outcome_doc=task_outcome_doc, current_pass=current_pass,
        )
    return {"skipped": "manual or awaiting-session — no action taken"}


def render_table(verdicts: list[Verdict]) -> str:
    lines = []
    lines.append(f"  {'card':12}  {'assignee':24}  {'task':8}  {'action':18}  reason")
    lines.append("  " + "-" * 100)
    for v in verdicts:
        task = v.handoff.task_id or "task???"
        lines.append(f"  {v.card_id:12}  {v.assignee:24}  {task:8}  {v.action:18}  {v.reason[:60]}")
    return "\n".join(lines)


def write_log(verdicts: list[Verdict], applied: list[dict[str, Any]], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# auto_triage_blocked  {utc_now()}",
        f"",
        f"Total blocked cards processed: {len(verdicts)}",
        f"",
        "## Verdicts",
        "",
        render_table(verdicts),
        "",
        "## Apply results",
        "",
    ]
    for r in applied:
        lines.append(f"- {r}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    global STANDALONE_SUBMIT_FLOOR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Mutate Hermes + registry. Default is dry-run.")
    parser.add_argument("--json", action="store_true",
                        help="Emit a compact JSON report on stdout (for Hermes cron --no-agent mode).")
    parser.add_argument("--floor", type=float, default=None,
                        help=f"LB floor for bank-vs-escalate (default {STANDALONE_SUBMIT_FLOOR})")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N cards (0 = no limit)")
    parser.add_argument(
        "--gave-up-action",
        choices=("leave", "unblock", "abandon"),
        default="leave",
        help=(
            "How to handle cards Hermes' dispatcher gave up on (no LLM session bound). "
            "leave (default): keep blocked, mark verdict awaiting-session. "
            "unblock: send back to ready so dispatch retries (only useful after binding sessions). "
            "abandon: complete with abandoned note so worker frees up (causes churn until sessions are bound)."
        ),
    )
    parser.add_argument(
        "--skip-review-required",
        action="store_true",
        help=(
            "Phase 3 deployment flag: skip blocked cards with 'review-required' "
            "handoffs (tools/submit_gate.py owns those now). Keeps handling for "
            "dispatcher-give-up, iteration-budget-exhausted, and no-fix cards. "
            "Flip this on once submit_gate is live so the two tools don't race."
        ),
    )
    parser.add_argument("--registry", type=Path, default=CANDIDATE_REGISTRY)
    parser.add_argument("--log", type=Path,
                        default=REPO_ROOT / "Logs" / f"auto_triage_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md")
    args = parser.parse_args(argv)
    if args.floor is not None:
        STANDALONE_SUBMIT_FLOOR = args.floor

    cards = hermes_list_blocked()
    if args.skip_review_required:
        cards = [c for c in cards if not _card_has_review_required(c)]
    if args.limit > 0:
        cards = cards[: args.limit]
    verdicts: list[Verdict] = []
    for card in cards:
        cid = card.get("id") or ""
        if not cid:
            continue
        show = hermes_show(cid)
        block_reason = extract_block_reason(show)
        handoff = parse_handoff(cid, card=card, block_reason=block_reason, card_show=show)
        verdicts.append(
            classify(card, handoff, block_reason, card_show_text=show, gave_up_action=args.gave_up_action)
        )

    if not args.json:
        print(render_table(verdicts))
        print()
    by_action: dict[str, int] = {}
    for v in verdicts:
        by_action[v.action] = by_action.get(v.action, 0) + 1
    if not args.json:
        print("summary:", by_action)
        print(f"mode:    {'APPLY (mutating)' if args.apply else 'DRY-RUN'}")

    applied: list[dict[str, Any]] = []
    no_action_verdicts = {"manual", "awaiting-session"}
    # Piece 2 needs the task_outcome_registry + current pass to decide whether
    # to broaden scope on stale-unassign. Load both once per run; pass_count
    # comes from the same cursor file the autoforeman reads.
    task_outcome_doc = _load_task_outcome_doc()
    current_pass = _load_current_pass_count()
    if args.apply:
        for v in verdicts:
            if v.action in no_action_verdicts:
                applied.append({"card": v.card_id, "action": v.action, "skipped": v.action})
                continue
            result = apply(
                v, args.registry,
                task_outcome_doc=task_outcome_doc,
                current_pass=current_pass,
            )
            result["card"] = v.card_id
            result["action"] = v.action
            applied.append(result)
            if not args.json:
                print(f"  applied {v.action:18} {v.card_id}: {result}")

    write_log(verdicts, applied, args.log)

    if args.json:
        # Cron / no-agent JSON contract used by the Hermes launcher.
        mutating_count = sum(1 for r in applied if "skipped" not in r)
        payload = {
            "schema": "neurogolf.auto_triage.v1",
            "action_count": mutating_count,
            "manual_count": by_action.get("manual", 0),
            "skipped_count": sum(1 for r in applied if "skipped" in r),
            "applied_actions": [
                {
                    "op": r.get("action"),
                    "task_id": r.get("card"),
                    "result": _apply_result_status(r),
                }
                for r in applied if "skipped" not in r
            ],
        }
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(f"log: {args.log.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
