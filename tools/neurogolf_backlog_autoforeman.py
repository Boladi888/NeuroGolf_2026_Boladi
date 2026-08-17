#!/usr/bin/env python3
"""Cheap backlog foreman for NeuroGolf worker-card creation.

This keeps the numeric task cursor moving, fills idle workers with one concrete
card each, and tops up an unassigned bank for future workers. It does not ask an
LLM to pick favorites.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_backlog_candidates import advance_cursor
from tools.neurogolf_board_policy import (
    ACTIVE_CARD_STATUSES,
    BACKLOG_BODY_MARKER,
    POOL_A_SHRINK_WORKERS,
    audit_worker_queue,
)
from tools.neurogolf_cron_snapshot import summarize_task_progress
from tools.neurogolf_dependency_serializer import intended_return_worker, is_unassigned_return_card
from tools.neurogolf_kanban_python import (
    KanbanClient,
    as_text,
    card_assignee,
    card_id,
    card_priority,
    card_status,
    card_text,
    card_title,
    normalize_task_id,
    read_json,
)
from tools.neurogolf_preflight import build_worker_preflight

BOARD_DIR = REPO_ROOT / ".hermes" / "board"
REGISTRY_JSON = BOARD_DIR / "candidate_registry.json"
SUBMIT_QUEUE_JSON = BOARD_DIR / "submit_queue.json"
STATE_JSON = BOARD_DIR / "state.json"
TASK_OUTCOME_JSON = BOARD_DIR / "task_outcome_registry.json"
TASK_RE = re.compile(r"task\d{3}")
BACKLOG_PREVIEW_LIMIT = 192
BACKLOG_CONSIDER_LIMIT = 401
RECENT_DONE_SUPPRESSION_SECONDS = 6 * 60 * 60
DEFER_STATUS_MARKERS = (
    "queued_candidate_member",
    "review-ext",
    "review_ext",
    "in_canonical_best",
    "public_positive_current_best",
    "tested_public_positive_current_best_stack_member",
    "tested_public_negative",
    "tested_non_improving",
    "hidden_positive_organic_replacement",
    "hidden_bad",
    "approved_pending_submit",
    "landed_current_best",
    "current_best",
    "canonical",
    "no_fix",
    "validated_pending_integration",
    "public_positive_pending_integration",
    "pending_integration",
    "avoid_or_retest",
)
HARD_DEFER_NEXT_ACTION_MARKERS = (
    "do not re-mine",
    "do not promote",
    "do not regress",
    "do not stack",
    "no beta action",
    "no action",
    "closed unless",
    "reject cheap",
    "exclude this",
    "exclude from",
    "revisit only",
    "do not work later unless",
    "submit exact",
    "pending submit",
    "integrate ",
    "retest only",
    "if a new source appears",
)
SOFT_DEFER_NEXT_ACTION_MARKERS = (
    "review candidate queue before changing",
    "do not rerun",
    "current canonical",
    "keep the fig",
    "keep current",
    "keep in ",
    "keep organic",
    "unless a new source",
    "new source or organic proof",
)
WORKER_NEXT_ACTION_MARKERS = (
    "extract",
    "shrink",
    "optimize",
    "optimise",
    "rewrite",
    "reduce",
    "current live",
    "current-live",
)

# Pool A cost-floor policy. Per NEUROGOLF_KANBAN_WORKFLOW.md "Bovath high-EV":
# "do not assign fresh worker targets with current cost below 75,000 unless a
# scout/reviewer gives an explicit high-upside structural rationale".
#
# The autoforeman previously had a 30,000 soft-defer threshold gated on a
# specific next_action substring (`leave unchanged unless new alpha/beta...`),
# which most tasks don't have — so sub-floor assignments slipped through and
# workers burned LLM iterations on tiny-cost targets (e.g. task007 at 9k,
# task150 at 5k) where the ceiling gain is ~+0.05 LB at best.
#
# This hard floor fires regardless of next_action. Sub-floor candidates are
# deferred unless the candidate carries an explicit reviewer-exception marker
# (structured field or free-form text token).
POOL_A_COST_FLOOR = 75_000
COST_FLOOR_EXCEPTION_TEXT_MARKERS = (
    "high-upside structural rationale",
    "structural rationale",
    "reviewer-exception",
    "reviewer exception",
    "scout-exception",
    "scout exception",
    "approved-by-bovath",
    "bovath-approved",
    "cost-floor-exception",
    "cost floor exception",
)
COST_FLOOR_EXCEPTION_STRUCTURED_FIELDS = (
    "cost_floor_exception",
    "reviewer_exception",
    "below_floor_rationale",
    "high_upside_rationale",
)


def candidate_has_cost_floor_exception(candidate: dict[str, Any]) -> str:
    """Return a non-empty rationale string when the candidate is exempt from
    the Pool A cost floor, else empty string.

    Two accept paths:
      1. Structured: any of COST_FLOOR_EXCEPTION_STRUCTURED_FIELDS is truthy.
         The value (if string) is returned as the rationale.
      2. Free-form: any of COST_FLOOR_EXCEPTION_TEXT_MARKERS appears in the
         candidate's status / verdict / next_action / notes / body text.
    """
    for field in COST_FLOOR_EXCEPTION_STRUCTURED_FIELDS:
        v = candidate.get(field)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v is True:
            return f"{field}=true"
    text = " ".join(
        as_text(candidate.get(key)).lower()
        for key in (
            "status", "best_known_status", "submit_queue_status", "verdict",
            "next_action", "notes", "body", "rationale", "decision_notes",
        )
    )
    for marker in COST_FLOOR_EXCEPTION_TEXT_MARKERS:
        if marker in text:
            return f"text:{marker}"
    return ""


def candidate_cost_floor_defer_reason(
    candidate: dict[str, Any],
    *,
    floor: int = POOL_A_COST_FLOOR,
) -> str:
    """Hard-defer reason if cost < floor AND no reviewer-exception is present.

    Returns empty string if the candidate has no cost data (don't block on
    missing data — let other gates handle that). Returns empty if cost >= floor.
    Returns empty if a reviewer-exception is present (the candidate is
    explicitly approved as sub-floor work).
    """
    cost_raw = candidate.get("current_cost")
    if cost_raw in (None, ""):
        return ""
    try:
        cost = float(cost_raw)
    except (TypeError, ValueError):
        return ""
    if cost <= 0 or cost >= floor:
        return ""
    if candidate_has_cost_floor_exception(candidate):
        return ""
    return f"defer sub-floor target cost={int(cost)} < {floor} (no reviewer-exception marker)"


def extract_task_id(*values: Any) -> str:
    for value in values:
        match = TASK_RE.search(as_text(value))
        if match:
            return normalize_task_id(match.group(0))
    return ""


def recent_done_task_ids(cards: list[dict[str, Any]], *, now: float | None = None) -> set[str]:
    now = time.time() if now is None else now
    out: set[str] = set()
    for card in cards:
        if as_text(card.get("status")).lower() != "done":
            continue
        try:
            completed_at = float(card.get("completed_at") or 0)
        except (TypeError, ValueError):
            completed_at = 0
        if completed_at <= 0 or now - completed_at > RECENT_DONE_SUPPRESSION_SECONDS:
            continue
        task_id = extract_task_id(card_title(card), card_text(card))
        if task_id:
            out.add(task_id)
    return out


def _as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _body_value(body: str, field: str) -> str:
    prefix = f"{field}:"
    for line in body.splitlines():
        if line.lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def candidate_from_bank_card(card: dict[str, Any]) -> dict[str, Any]:
    body = card_text(card)
    return {
        "task": extract_task_id(card_title(card), body),
        "current_cost": _body_value(body, "current_cost"),
        "current_points": _body_value(body, "current_points"),
        "status": _body_value(body, "status"),
        "next_action": _body_value(body, "next_action"),
    }


def candidate_hard_defer_reason(candidate: dict[str, Any]) -> str:
    status_text = " ".join(
        as_text(candidate.get(key)).lower()
        for key in ("status", "best_known_status", "submit_queue_status", "verdict")
    )
    next_action = as_text(candidate.get("next_action")).lower()
    if any(marker in status_text for marker in DEFER_STATUS_MARKERS):
        if "queued_candidate_member" in status_text:
            return "defer existing candidate queue member to reviewer/submit pipeline"
        if "canonical_best" in status_text or "current_best" in status_text:
            return "defer current-best member; only revisit with explicit replacement evidence"
        if "negative" in status_text or "non_improving" in status_text:
            return "defer previously non-improving public result"
        if "pending_integration" in status_text:
            return "defer pending integration/rebase work to reviewer/submit pipeline"
        if "avoid_or_retest" in status_text:
            return "defer retest-only row until new source evidence exists"
        return "defer protected/organic/review status from worker backlog"
    if any(marker in next_action for marker in HARD_DEFER_NEXT_ACTION_MARKERS):
        return "defer task-state next_action says not to mine this now"
    floor_reason = candidate_cost_floor_defer_reason(candidate)
    if floor_reason:
        return floor_reason
    return ""


def candidate_soft_defer_reason(candidate: dict[str, Any]) -> str:
    next_action = as_text(candidate.get("next_action")).lower()
    cost = _as_float(candidate.get("current_cost"))
    if "leave unchanged unless new alpha/beta work targets this task" in next_action and cost < 30000:
        return "defer low-cost generic no-current target; keep bank for harder tasks"
    if any(marker in next_action for marker in SOFT_DEFER_NEXT_ACTION_MARKERS):
        return "defer task-state next_action says not to mine this now"
    return ""


def candidate_defer_reason(candidate: dict[str, Any], *, allow_soft: bool = False) -> str:
    hard_reason = candidate_hard_defer_reason(candidate)
    if hard_reason:
        return hard_reason
    if allow_soft:
        return ""
    soft_reason = candidate_soft_defer_reason(candidate)
    if soft_reason:
        return soft_reason
    return ""


def task_attempts_exhausted_this_pass(
    task_id: str,
    task_outcome_doc: dict[str, Any] | None,
    current_pass: int,
    *,
    max_attempts: int = 3,
) -> str:
    """Return a defer reason when submit_gate has already burned the per-pass
    attempt budget on this task, else "".

    Reads ``attempts_this_pass`` and ``last_attempt_pass`` from the per-task
    entry written by submit_gate (see tools/task_outcome_registry_io.py).
    The values are stale (carry-over from a prior pass) when
    ``last_attempt_pass != current_pass``; in that case attempts effectively
    reset to 0 and this helper returns "" (the autoforeman re-tries the
    task on the new pass).

    Defensive: missing doc / missing entry / missing fields all return ""
    so the autoforeman behaves identically when submit_gate hasn't started
    populating these fields yet (e.g. during Phase 1 / 2 rollout).
    """
    if not task_id or not isinstance(task_outcome_doc, dict):
        return ""
    entry = (task_outcome_doc.get("tasks") or {}).get(task_id)
    if not isinstance(entry, dict):
        return ""
    last_pass = entry.get("last_attempt_pass")
    try:
        last_pass_int = int(last_pass) if last_pass is not None else None
    except (TypeError, ValueError):
        last_pass_int = None
    if last_pass_int is None or last_pass_int != current_pass:
        return ""
    attempts = entry.get("attempts_this_pass") or 0
    try:
        attempts_int = int(attempts)
    except (TypeError, ValueError):
        return ""
    if attempts_int >= max_attempts:
        summary = entry.get("last_attempt_summary") or "submit_gate gave up for this pass"
        return (
            f"defer attempts_this_pass={attempts_int} >= {max_attempts} "
            f"on pass {current_pass}: {summary}"
        )
    return ""


def _attempted_families_for(
    task_id: str,
    task_outcome_doc: dict[str, Any] | None,
    current_pass: int,
) -> list[str]:
    """Return the attempted-families list for the CURRENT pass (empty when
    the recorded last_attempt_pass differs from current_pass, since attempts
    effectively reset on pass advance)."""
    if not task_id or not isinstance(task_outcome_doc, dict):
        return []
    entry = (task_outcome_doc.get("tasks") or {}).get(task_id)
    if not isinstance(entry, dict):
        return []
    last_pass = entry.get("last_attempt_pass")
    try:
        last_pass_int = int(last_pass) if last_pass is not None else None
    except (TypeError, ValueError):
        last_pass_int = None
    if last_pass_int is None or last_pass_int != current_pass:
        return []
    families = entry.get("attempted_families")
    return list(families) if isinstance(families, list) else []


def next_retry_attempt_index(
    task_id: str,
    task_outcome_doc: dict[str, Any] | None,
    current_pass: int,
) -> int:
    """Return the 1-based attempt index for the next worker card on this task.

    1 when the task has never been tried this pass (or registry is missing).
    N+1 when N attempts have already been recorded for this pass.
    Mirrors the values submit_gate.py writes when it bumps the counter.
    """
    if not task_id or not isinstance(task_outcome_doc, dict):
        return 1
    entry = (task_outcome_doc.get("tasks") or {}).get(task_id)
    if not isinstance(entry, dict):
        return 1
    last_pass = entry.get("last_attempt_pass")
    try:
        last_pass_int = int(last_pass) if last_pass is not None else None
    except (TypeError, ValueError):
        last_pass_int = None
    if last_pass_int is None or last_pass_int != current_pass:
        return 1
    attempts = entry.get("attempts_this_pass") or 0
    try:
        attempts_int = int(attempts)
    except (TypeError, ValueError):
        return 1
    return attempts_int + 1


def priority_for_candidate(candidate: dict[str, Any]) -> int:
    cost = _as_float(candidate.get("current_cost"))
    points = _as_float(candidate.get("current_points"))
    status = as_text(candidate.get("status")).lower()
    next_action = as_text(candidate.get("next_action")).lower()

    priority = 30
    if cost >= 200000:
        priority += 35
    elif cost >= 100000:
        priority += 28
    elif cost >= 75000:
        priority += 23
    elif cost >= 30000:
        priority += 14
    elif cost >= 10000:
        priority += 6

    if any(marker in status for marker in ("wip", "big-delta", "assigned_second_stage_shrink")):
        priority += 8
    if any(marker in next_action for marker in WORKER_NEXT_ACTION_MARKERS):
        priority += 5
    if points >= 18:
        priority += 3
    elif points >= 15:
        priority += 2
    if cost < 1000 and points >= 20:
        priority -= 4
    return max(10, min(priority, 95))


def load_preflight_docs() -> dict[str, Any]:
    return {
        "registry_doc": read_json(REGISTRY_JSON, {}),
        "queue_doc": read_json(SUBMIT_QUEUE_JSON, {}),
        "state_doc": read_json(STATE_JSON, {}),
        "task_outcome_doc": read_json(TASK_OUTCOME_JSON, {}),
    }


def preflight_for(
    *,
    task_id: str,
    assignee: str,
    cards: list[dict[str, Any]],
    docs: dict[str, Any],
    self_kanban_task_id: str = "",
) -> dict[str, Any]:
    return build_worker_preflight(
        task_id=task_id,
        assignee=assignee,
        method_family="current-live structural optimization backlog sweep",
        lineage_base_candidate_id="current-best",
        new_evidence="numeric cursor sweep; worker must document novel hypothesis before submit",
        cards=cards,
        registry_doc=docs.get("registry_doc") if isinstance(docs.get("registry_doc"), dict) else {},
        queue_doc=docs.get("queue_doc") if isinstance(docs.get("queue_doc"), dict) else {},
        state_doc=docs.get("state_doc") if isinstance(docs.get("state_doc"), dict) else {},
        task_outcome_doc=docs.get("task_outcome_doc") if isinstance(docs.get("task_outcome_doc"), dict) else {},
        repo_root=REPO_ROOT,
        self_kanban_task_id=self_kanban_task_id,
    )


def body_for_task(
    *,
    task_id: str,
    assignee: str = "",
    candidate: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
    source_backlog_id: str = "",
    bank: bool = False,
    retry_attempt_index: int | None = None,
    attempted_families: list[str] | None = None,
) -> str:
    candidate = candidate or {}
    preflight = preflight or {}
    marker = f"{BACKLOG_BODY_MARKER} - do not dispatch directly; Python will convert this into one worker card." if bank else ""
    lines = [
        marker,
        f"task_id: {task_id}",
        f"assignee_target: {assignee or '(unassigned-bank)'}",
        f"source_backlog_card: {source_backlog_id or '(none)'}",
        f"current_cost: {candidate.get('current_cost', '')}",
        f"current_points: {candidate.get('current_points', '')}",
        f"status: {candidate.get('status', '')}",
        f"next_action: {candidate.get('next_action', '')}",
    ]
    if retry_attempt_index is not None:
        lines.append(f"retry_attempt_index: {retry_attempt_index}")
    if attempted_families:
        lines.append(f"do_not_use_families: {attempted_families}")
    lines.extend([
        "",
        "Worker instructions:",
        "- Work only this one task unless the card explicitly says Worker1 stack/multitask.",
        "- Preserve context and document every attempted method in The_400_Tasks/Task_by_Task/taskNNN.md.",
        "- Produce local ONNX/submission evidence and a delta packet; do not submit to Kaggle.",
        "- Minimal rebase: verify against the original valid base; submit-captain handles final stacking/rebase only for real submit assembly.",
        "- Set `algorithm_family` in your handoff JSON to a short kebab-case identifier (e.g. `where-select`, `reduce-broadcast`). submit_gate uses this to track diverse exploration across attempts.",
        f"- Copy `retry_attempt_index` ({retry_attempt_index or 1}) from this card body into your handoff JSON so the audit trail can correlate.",
    ])
    soft_defer = candidate_soft_defer_reason(candidate)
    if soft_defer:
        lines.extend(
            [
                "",
                "Loosened backlog note:",
                f"- This task normally soft-defers because: {soft_defer}.",
                "- Proceed only with a materially new, cheaper, strict-clean hypothesis; do not revive old external/current-best payloads unchanged.",
            ]
        )
    footer = preflight.get("required_card_footer")
    if isinstance(footer, list):
        lines.append("")
        lines.append("Preflight:")
        lines.extend(str(item) for item in footer)
    return "\n".join(line for line in lines if line != "")


def plan_backlog(
    cards: list[dict[str, Any]],
    *,
    docs: dict[str, Any] | None = None,
    target_bank: int = 24,
    create_limit: int = 12,
    assign_limit: int = 8,
) -> dict[str, Any]:
    docs = docs or {}
    try:
        task_progress = summarize_task_progress(
            cards,
            preview_limit=BACKLOG_PREVIEW_LIMIT,
            consider_limit=BACKLOG_CONSIDER_LIMIT,
        )
    except TypeError:
        task_progress = summarize_task_progress(cards)
    audit = audit_worker_queue(cards)
    actions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    card_by_id = {card_id(card): card for card in cards}
    cursor_start = as_text(task_progress.get("next_task_id") or "task000")
    cursor_pass = int(task_progress.get("pass_count") or 0)
    cursor_scope = f"pass{cursor_pass}-from{cursor_start}"
    # Pool B (workers 7-8) is reserved for unsolved-task solver hunt; the backlog
    # autoforeman only schedules current-live shrink work, so filter to Pool A.
    pool_a = set(POOL_A_SHRINK_WORKERS)
    idle_workers = [w for w in (audit.get("idle_queue_workers") or []) if w in pool_a]
    return_reserved_workers = sorted(
        {
            intended_return_worker(card)
            for card in cards
            if is_unassigned_return_card(card) and intended_return_worker(card) in idle_workers
        }
    )
    if return_reserved_workers:
        idle_workers = [worker for worker in idle_workers if worker not in return_reserved_workers]
    existing_bank = list(audit.get("unassigned_backlog_bank") or [])
    existing_bank.sort(key=lambda item: (-int(card_priority(card_by_id.get(item["task_id"], {}))), as_text(item.get("task_id"))))
    recently_done = recent_done_task_ids(cards)
    remaining_create_budget = max(0, create_limit)
    remaining_assign_budget = max(0, assign_limit)
    consumed_bank = 0
    deferred_bank_ids: set[str] = set()
    cleanup_budget = max(0, create_limit)

    for bank_item in existing_bank:
        if cleanup_budget <= 0:
            break
        bank_id = as_text(bank_item.get("task_id"))
        bank_card = card_by_id.get(bank_id, {})
        candidate = candidate_from_bank_card(bank_card)
        task_id = normalize_task_id(candidate.get("task"))
        if not task_id:
            continue
        defer_reason = candidate_hard_defer_reason(candidate)
        if not defer_reason:
            # Also check whether submit_gate has already burned the per-pass
            # attempt budget on this task. When yes, defer this pass; the
            # task becomes eligible again on the next pass increment.
            defer_reason = task_attempts_exhausted_this_pass(
                task_id,
                docs.get("task_outcome_doc"),
                cursor_pass,
            )
        if not defer_reason:
            continue
        actions.append(
            {
                "op": "complete_deferred_bank_card",
                "source_backlog_id": bank_id,
                "task_id": task_id,
                "title": card_title(bank_card),
                "reason": defer_reason,
            }
        )
        deferred_bank_ids.add(bank_id)
        cleanup_budget -= 1

    if audit.get("ok"):
        for bank_item in existing_bank:
            if as_text(bank_item.get("task_id")) in deferred_bank_ids:
                continue
            if not idle_workers or remaining_assign_budget <= 0 or remaining_create_budget <= 0:
                break
            bank_card = card_by_id.get(bank_item["task_id"], {})
            task_id = extract_task_id(card_title(bank_card), card_text(bank_card))
            if not task_id:
                skipped.append({"task_id": bank_item["task_id"], "reason": "could not extract task id from backlog card"})
                continue
            worker = idle_workers[0]
            preflight = preflight_for(
                task_id=task_id,
                assignee=worker,
                cards=cards,
                docs=docs,
                self_kanban_task_id=bank_item["task_id"],
            )
            if preflight.get("verdict") == "DENY":
                skipped.append({"task_id": task_id, "assignee": worker, "reason": "preflight DENY", "deny_reasons": preflight.get("deny_reasons")})
                continue
            idle_workers.pop(0)
            priority = max(card_priority(bank_card), 60)
            actions.append(
                {
                    "op": "create_worker_from_bank",
                    "source_backlog_id": bank_item["task_id"],
                    "task_id": task_id,
                    "assignee": worker,
                    "title": f"[worker {task_id} {worker} current-live] cursor backlog task",
                    "body": body_for_task(
                        task_id=task_id,
                        assignee=worker,
                        preflight=preflight,
                        source_backlog_id=bank_item["task_id"],
                        retry_attempt_index=next_retry_attempt_index(
                            task_id, docs.get("task_outcome_doc"), cursor_pass,
                        ),
                        attempted_families=_attempted_families_for(
                            task_id, docs.get("task_outcome_doc"), cursor_pass,
                        ),
                    ),
                    "priority": priority,
                    "idempotency_key": f"ng-worker-{worker}-{task_id}-{bank_item['task_id']}",
                }
            )
            consumed_bank += 1
            remaining_create_budget -= 1
            remaining_assign_budget -= 1

    direct_workers = idle_workers if audit.get("ok") else []
    bank_count = int(audit.get("unassigned_backlog_bank_count") or 0) - consumed_bank - len(deferred_bank_ids)
    cursor_advance_from = ""
    planned_task_ids = {action["task_id"] for action in actions if action.get("task_id")}

    for candidate in task_progress.get("numeric_preview") or []:
        task_id = normalize_task_id(candidate.get("task"))
        if not task_id or task_id in planned_task_ids:
            continue
        if task_id in recently_done:
            skipped.append({"task_id": task_id, "reason": "defer recently completed worker sweep until task_state/docs catch up"})
            cursor_advance_from = task_id
            continue
        if remaining_create_budget <= 0:
            break
        if not direct_workers and bank_count >= target_bank:
            break
        allow_soft_defer = bool(direct_workers and remaining_assign_budget > 0) or bank_count < target_bank
        defer_reason = candidate_defer_reason(candidate, allow_soft=allow_soft_defer)
        if defer_reason:
            skipped.append({"task_id": task_id, "reason": defer_reason})
            cursor_advance_from = task_id
            continue
        if direct_workers and remaining_assign_budget > 0:
            worker = direct_workers[0]
            preflight = preflight_for(task_id=task_id, assignee=worker, cards=cards, docs=docs)
            if preflight.get("verdict") == "DENY":
                skipped.append({"task_id": task_id, "assignee": worker, "reason": "preflight DENY", "deny_reasons": preflight.get("deny_reasons")})
                cursor_advance_from = task_id
                continue
            direct_workers.pop(0)
            actions.append(
                {
                    "op": "create_worker_card",
                    "task_id": task_id,
                    "assignee": worker,
                    "title": f"[worker {task_id} {worker} current-live] cursor backlog task",
                    "body": body_for_task(
                        task_id=task_id,
                        assignee=worker,
                        candidate=candidate,
                        preflight=preflight,
                        retry_attempt_index=next_retry_attempt_index(
                            task_id, docs.get("task_outcome_doc"), cursor_pass,
                        ),
                        attempted_families=_attempted_families_for(
                            task_id, docs.get("task_outcome_doc"), cursor_pass,
                        ),
                    ),
                    "priority": priority_for_candidate(candidate),
                    "idempotency_key": f"ng-worker-{worker}-{task_id}-{cursor_scope}",
                }
            )
            planned_task_ids.add(task_id)
            cursor_advance_from = task_id
            remaining_create_budget -= 1
            remaining_assign_budget -= 1
            continue
        if bank_count >= target_bank:
            break
        preflight = preflight_for(task_id=task_id, assignee="", cards=cards, docs=docs)
        if preflight.get("verdict") == "DENY":
            skipped.append({"task_id": task_id, "reason": "preflight DENY", "deny_reasons": preflight.get("deny_reasons")})
            cursor_advance_from = task_id
            continue
        actions.append(
            {
                "op": "create_bank_card",
                "task_id": task_id,
                "title": f"[BACKLOG single task] {task_id} cursor backlog current-live",
                "body": body_for_task(task_id=task_id, candidate=candidate, preflight=preflight, bank=True),
                "priority": priority_for_candidate(candidate),
                "idempotency_key": f"ng-backlog-{task_id}-{cursor_scope}",
            }
        )
        planned_task_ids.add(task_id)
        cursor_advance_from = task_id
        bank_count += 1
        remaining_create_budget -= 1

    if (
        not cursor_advance_from
        and not actions
        and not (task_progress.get("numeric_preview") or [])
        and (task_progress.get("window_end_task_id"))
    ):
        cursor_advance_from = as_text(task_progress.get("window_end_task_id"))

    return {
        "schema": "neurogolf.backlog_autoforeman.v1",
        "actions": actions,
        "skipped": skipped,
        "queue_ok": audit.get("ok"),
        "idle_workers_start": audit.get("idle_queue_workers") or [],
        "return_reserved_workers": return_reserved_workers,
        "unassigned_bank_start": audit.get("unassigned_backlog_bank_count"),
        "unassigned_bank_planned": bank_count,
        "cursor_next": task_progress.get("next_task_id"),
        "cursor_pass": task_progress.get("pass_count"),
        "cursor_advance_from": cursor_advance_from,
        "action_count": len(actions),
        "skipped_count": len(skipped),
    }


def apply_backlog_plan(client: KanbanClient, plan: dict[str, Any], *, advance: bool = True) -> None:
    created_or_consumed: list[str] = []

    def created_card_is_active(task_id: str, action: dict[str, Any]) -> bool:
        if getattr(client, "dry_run", False):
            return True
        detail = client.show(task_id)
        card = detail.get("task") if isinstance(detail, dict) and isinstance(detail.get("task"), dict) else detail
        if not isinstance(card, dict):
            client.actions.append(
                {
                    "op": "stale_create_result",
                    "task_id": task_id,
                    "idempotency_key": action.get("idempotency_key"),
                    "reason": "create returned a card that could not be read back",
                }
            )
            return False
        status = card_status(card)
        expected_assignee = as_text(action.get("assignee"))
        actual_assignee = card_assignee(card)
        if status not in ACTIVE_CARD_STATUSES:
            client.actions.append(
                {
                    "op": "stale_create_result",
                    "task_id": task_id,
                    "status": status,
                    "assignee": actual_assignee,
                    "idempotency_key": action.get("idempotency_key"),
                    "reason": "create returned an inactive card from an old idempotency key",
                }
            )
            return False
        if expected_assignee and actual_assignee != expected_assignee:
            client.actions.append(
                {
                    "op": "stale_create_result",
                    "task_id": task_id,
                    "status": status,
                    "assignee": actual_assignee,
                    "expected_assignee": expected_assignee,
                    "idempotency_key": action.get("idempotency_key"),
                    "reason": "create returned an active card on the wrong lane",
                }
            )
            return False
        return True

    for action in plan.get("actions", []):
        op = action.get("op")
        if op in {"create_worker_card", "create_worker_from_bank"}:
            worker_id = client.create(
                str(action["title"]),
                body=str(action["body"]),
                assignee=str(action["assignee"]),
                workspace="worktree",
                priority=int(action["priority"]),
                created_by="neurogolf-backlog-autoforeman",
                idempotency_key=str(action["idempotency_key"]),
                max_runtime="6h",
            )
            if not created_card_is_active(worker_id, action):
                continue
            if op == "create_worker_from_bank":
                source_id = str(action["source_backlog_id"])
                client.comment(source_id, f"Python backlog foreman converted this bank item into worker card {worker_id}.")
                client.complete(source_id, f"converted to worker card {worker_id}")
            if action.get("task_id"):
                created_or_consumed.append(str(action["task_id"]))
        elif op == "create_bank_card":
            bank_id = client.create(
                str(action["title"]),
                body=str(action["body"]),
                assignee="",
                workspace="worktree",
                priority=int(action["priority"]),
                created_by="neurogolf-backlog-autoforeman",
                idempotency_key=str(action["idempotency_key"]),
                max_runtime="6h",
            )
            if created_card_is_active(bank_id, action) and action.get("task_id"):
                created_or_consumed.append(str(action["task_id"]))
        elif op == "complete_deferred_bank_card":
            source_id = str(action["source_backlog_id"])
            client.comment(source_id, f"Python backlog foreman closing deferred bank item: {action['reason']}.")
            client.complete(source_id, str(action["reason"]))

    cursor_from = str(plan.get("cursor_advance_from") or "")
    if advance and cursor_from and (created_or_consumed or plan.get("skipped")):
        advance_cursor(
            cursor_from,
            (
                f"python autoforeman created/converted {len(created_or_consumed)} cards, "
                f"skipped {plan.get('skipped_count', 0)} by preflight"
            ),
        )


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"actions={report['action_count']} skipped={report['skipped_count']} queue_ok={report['queue_ok']} cursor={report['cursor_next']} advance_from={report['cursor_advance_from']}",
    ]
    for action in report["actions"]:
        lines.append(f"- {action['op']} {action['task_id']} -> {action.get('assignee') or 'bank'}")
    for item in report["skipped"][:12]:
        lines.append(f"- skipped {item.get('task_id')}: {item.get('reason')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--kanban-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--target-bank", type=int, default=24)
    parser.add_argument("--create-limit", type=int, default=12)
    parser.add_argument("--assign-limit", type=int, default=8)
    parser.add_argument("--no-advance-cursor", action="store_true")
    args = parser.parse_args()

    client = KanbanClient(args.repo_root, dry_run=not args.apply)
    if args.kanban_json:
        cards = json.loads(args.kanban_json.read_text(encoding="utf-8"))
    else:
        cards = client.load_cards()
    if not isinstance(cards, list):
        raise SystemExit("expected Kanban JSON list")

    report = plan_backlog(
        [card for card in cards if isinstance(card, dict)],
        docs=load_preflight_docs(),
        target_bank=args.target_bank,
        create_limit=args.create_limit,
        assign_limit=args.assign_limit,
    )
    if args.apply:
        apply_backlog_plan(client, report, advance=not args.no_advance_cursor)
        report["applied_actions"] = client.actions
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
