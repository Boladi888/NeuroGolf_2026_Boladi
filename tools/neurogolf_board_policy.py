#!/usr/bin/env python3
"""Board/control-plane helpers for NeuroGolf live-card duplicate prevention.

This module keeps the target-claim registry and deterministic chain-key
conventions in one place so orchestrator / watchdog tooling can share the same
precreation guard logic without inventing another registry.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.neurogolf_paths import board_dir, resolve_canonical_root
except ModuleNotFoundError:  # pragma: no cover
    from neurogolf_paths import board_dir, resolve_canonical_root  # type: ignore

REPO_ROOT = resolve_canonical_root(Path(__file__).resolve())
BOARD_DIR = board_dir(REPO_ROOT)
STATE_JSON = BOARD_DIR / "state.json"

LIVE_TERMINAL_STATUSES = {
    "done",
    "closed",
    "archived",
    "submitted-complete",
    "submitted_complete",
    "complete",
    "completed",
    "superseded",
}
ACTIVE_CARD_STATUSES = {"running", "ready", "todo", "blocked"}
RUNNABLE_CARD_STATUSES = {"running", "ready"}
LANE_CONTEXT_STATUSES = ACTIVE_CARD_STATUSES
# Hermes `todo` cards are not inherently parked. They are promoted when their
# dependencies clear, so a shared parent can fan out into many simultaneous
# worker runs. The board list does not expose unsatisfied dependency state, so
# the safe audit default is to count assigned todo as an active worker slot.
PARKED_QUEUE_STATUSES: set[str] = set()
DEFAULT_QUEUE_WORKERS = (
    "neurogolf-worker-1",
    "neurogolf-worker-2",
    "neurogolf-worker-3",
    "neurogolf-worker-4",
    "neurogolf-worker-5",
    "neurogolf-worker-6",
    "neurogolf-worker-7",
    "neurogolf-worker-8",
)
# Pool A: current-live shrink on already-solved tasks (cost floor 75,000).
#   - EMPTY as of 2026-05-18: user shut off the Codex/Hermes-dispatched
#     worker role entirely. The Codex backend still runs the kanban
#     orchestration (reviewers, submit-captain, sync-manager, autoforemen,
#     route-repair, auto-triage, dispatcher daemon) — only the *worker*
#     LLM seat is off. Candidate production now flows exclusively through
#     Claude Code /loop sessions on workers 5/7/8 (Pool B). Pool A
#     autoforeman still runs but produces only bank cards (no idle Pool A
#     worker to assign them to).
#   - workers 1, 2, 3, 4, 6: all draining. They keep any in-flight cards
#     and continue to receive repair / optimization-return cards (those
#     route via DEFAULT_QUEUE_WORKERS, not POOL_A_SHRINK_WORKERS), but
#     the autoforeman will not hand them new backlog targets.
#     BACKLOG bank cards mid-flight on a draining worker are still
#     considered legitimate (the misclaim guard allows them).
# Pool B: candidate work driven by manual Claude Code /loop sessions.
#   - workers 5, 7, 8 as of 2026-05-18 (worker-7 = Sonnet, worker-8 = Opus,
#     worker-5 = additional Sonnet capacity). Profiles point at a closed
#     port / nonexistent provider so the Hermes dispatcher's spawn fails
#     fast; Claude Code claims cards via /loop. The "solver-hunt" naming
#     is historical — workers actually do cost-shrink work on tasks not in
#     the curated SOLVED_TASKS list (every task already has a working
#     baseline solver in submission.zip).
POOL_A_SHRINK_WORKERS: tuple[str, ...] = ()
POOL_B_SOLVER_HUNT_WORKERS: tuple[str, ...] = (
    "neurogolf-worker-5",
    "neurogolf-worker-7",
    "neurogolf-worker-8",
)
# Workers temporarily off the autoforeman list. They keep current cards and
# any repair-card routing, but no new backlog assignments.
POOL_A_DRAINING_WORKERS: tuple[str, ...] = (
    "neurogolf-worker-1",
    "neurogolf-worker-2",
    "neurogolf-worker-3",
    "neurogolf-worker-4",
    "neurogolf-worker-6",
)
NEW_SOLVER_METHOD_FAMILY_MARKERS: tuple[str, ...] = (
    "new-solver",
    "new solver",
    "unsolved",
    "solver-hunt",
    "solver hunt",
)
LOW_CONTEXT_QUEUE_WORKERS: set[str] = set()
BACKLOG_TITLE_PREFIX = "[BACKLOG"
BACKLOG_BODY_MARKER = "UNASSIGNED BACKLOG CARD"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_task_id(task_id: Any) -> str:
    if task_id is None:
        return ""
    text = str(task_id).strip()
    if not text:
        return ""
    if text.startswith("task"):
        suffix = text[4:]
        if suffix.isdigit():
            return f"task{int(suffix):03d}"
        return text
    if text.isdigit():
        return f"task{int(text):03d}"
    return text


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_str_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for value in values:
        text = _as_text(value)
        if text:
            out.append(text)
    return sorted(dict.fromkeys(out))


def _stack_key(stack_candidate_ids: Any) -> str:
    ids = _to_str_list(stack_candidate_ids)
    return "+".join(ids)


def build_worker_chain_key(
    task_id: Any,
    owner_profile: Any,
    lineage_base_candidate_id: Any = "",
    *,
    current_best_candidate_id: Any = "",
) -> str:
    lineage = _as_text(lineage_base_candidate_id) or _as_text(current_best_candidate_id) or "current-best"
    return f"target:{normalize_task_id(task_id)}:{_as_text(owner_profile)}:{lineage}"


def build_review_chain_key(candidate_id: Any) -> str:
    return f"review:{_as_text(candidate_id)}"


def build_worthiness_chain_key(candidate_id: Any) -> str:
    return f"worthiness:{_as_text(candidate_id)}"


def build_submit_chain_key(
    candidate_id: Any = "",
    *,
    stack_candidate_ids: Any = None,
) -> str:
    stack = _stack_key(stack_candidate_ids)
    if stack:
        return f"submit-stack:{stack}"
    return f"submit:{_as_text(candidate_id)}"


def build_chain_keys(candidate: dict[str, Any]) -> dict[str, str]:
    candidate_id = _as_text(candidate.get("candidate_id"))
    task_id = normalize_task_id(candidate.get("task_id"))
    owner_profile = _as_text(
        candidate.get("worker_owner")
        or candidate.get("owner_profile")
        or candidate.get("followup_worker")
        or candidate.get("return_worker")
    )
    lineage = _as_text(candidate.get("lineage_base_candidate_id"))
    if not lineage:
        lineage_base = candidate.get("lineage_base")
        if isinstance(lineage_base, dict):
            lineage = _as_text(lineage_base.get("candidate_id") or lineage_base.get("id"))
    current_best = _as_text(candidate.get("current_best_candidate_id") or candidate.get("current_best") or candidate.get("source_candidate_id"))
    stack_ids = candidate.get("stack_candidate_ids")
    if not stack_ids:
        stack_plan = candidate.get("stack_plan")
        if isinstance(stack_plan, dict):
            stack_ids = stack_plan.get("stack_candidate_ids")
    if not stack_ids:
        submit_execution = candidate.get("submit_execution")
        if isinstance(submit_execution, dict):
            stack_ids = submit_execution.get("stack_candidate_ids")
    keys = {
        "worker": build_worker_chain_key(task_id, owner_profile, lineage, current_best_candidate_id=current_best),
        "review": build_review_chain_key(candidate_id),
        "worthiness": build_worthiness_chain_key(candidate_id),
        "submit": build_submit_chain_key(candidate_id, stack_candidate_ids=stack_ids),
    }
    return keys


def build_target_claim_record(
    task_id: Any,
    owner_profile: Any,
    kanban_task_id: Any,
    *,
    claim_mode: str = "exclusive",
    reason: str = "",
    candidate_scope: str = "single-task candidate",
    lineage_base_candidate_id: Any = "",
    current_best_candidate_id: Any = "",
    duplicate_guard_key: Any = "",
    claimed_at: str | None = None,
) -> dict[str, Any]:
    task_key = normalize_task_id(task_id)
    claim = {
        "task_id": task_key,
        "owner_profile": _as_text(owner_profile),
        "kanban_task_id": _as_text(kanban_task_id),
        "claim_mode": _as_text(claim_mode) or "exclusive",
        "claimed_at": claimed_at or utc_now(),
        "reason": _as_text(reason),
        "candidate_scope": _as_text(candidate_scope) or "single-task candidate",
        "duplicate_guard_key": _as_text(duplicate_guard_key),
        "chain_key": build_worker_chain_key(task_key, owner_profile, lineage_base_candidate_id, current_best_candidate_id=current_best_candidate_id),
    }
    return claim


def read_board_state_doc(path: Path | None = None) -> dict[str, Any]:
    doc = read_json(path or STATE_JSON)
    return doc if isinstance(doc, dict) else {}


def card_status(card: dict[str, Any]) -> str:
    return _as_text(card.get("status")).lower()


def card_assignee(card: dict[str, Any]) -> str:
    return _as_text(card.get("assignee"))


def is_active_card(card: dict[str, Any]) -> bool:
    return card_status(card) in ACTIVE_CARD_STATUSES


def is_backlog_card(card: dict[str, Any]) -> bool:
    title = _as_text(card.get("title"))
    body = _as_text(card.get("body"))
    return title.startswith(BACKLOG_TITLE_PREFIX) or BACKLOG_BODY_MARKER in body


def backlog_card_needs_assignment(card: dict[str, Any]) -> bool:
    status = card_status(card)
    return (
        status == "ready"
        and is_backlog_card(card)
        and bool(card_assignee(card))
    )


def unassigned_backlog_bank_card(card: dict[str, Any]) -> bool:
    return (
        is_active_card(card)
        and is_backlog_card(card)
        and not card_assignee(card)
    )


def backlog_card_sort_key(card: dict[str, Any]) -> tuple[int, int, str]:
    return (
        -_as_int(card.get("priority"), 0),
        _as_int(card.get("created_at"), 0),
        _as_text(card.get("id")),
    )


def active_cards_by_assignee(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        assignee = card_assignee(card)
        if not assignee or not is_active_card(card):
            continue
        grouped.setdefault(assignee, []).append(card)
    for bucket in grouped.values():
        bucket.sort(key=backlog_card_sort_key)
    return grouped


def cards_by_assignee_with_statuses(
    cards: list[dict[str, Any]],
    statuses: set[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        assignee = card_assignee(card)
        if not assignee or card_status(card) not in statuses:
            continue
        grouped.setdefault(assignee, []).append(card)
    for bucket in grouped.values():
        bucket.sort(key=backlog_card_sort_key)
    return grouped


def audit_worker_queue(
    cards: list[dict[str, Any]],
    *,
    queue_workers: tuple[str, ...] = DEFAULT_QUEUE_WORKERS,
    low_context_workers: set[str] | None = None,
    dispatch_driven_workers: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    low_context_workers = low_context_workers or LOW_CONTEXT_QUEUE_WORKERS
    # Lane-context and dispatch-risk violations only apply to workers the
    # Hermes dispatcher actually spawns (Pool A shrink + draining). Pool B
    # workers (5, 7, 8) are manual Claude Code /loop sessions with closed-
    # port profiles — the dispatcher's spawn attempts fail fast and the
    # /loop session works one card at a time. Letting them have a chain of
    # parked opt-return/solver-hunt follow-ups is the dep serializer's
    # intentional behavior, not a violation.
    if dispatch_driven_workers is None:
        dispatch_driven_workers = tuple(POOL_A_SHRINK_WORKERS) + tuple(POOL_A_DRAINING_WORKERS)
    dispatch_driven_set = set(dispatch_driven_workers)
    active_by_assignee = active_cards_by_assignee(cards)
    lane_context_by_assignee = cards_by_assignee_with_statuses(cards, LANE_CONTEXT_STATUSES)
    runnable_by_assignee = cards_by_assignee_with_statuses(cards, RUNNABLE_CARD_STATUSES)
    parked_by_assignee = cards_by_assignee_with_statuses(cards, PARKED_QUEUE_STATUSES)
    pending_assignment_backlog = [card for card in cards if backlog_card_needs_assignment(card)]
    pending_assignment_backlog.sort(key=backlog_card_sort_key)
    unassigned_backlog_bank = [card for card in cards if unassigned_backlog_bank_card(card)]
    unassigned_backlog_bank.sort(key=backlog_card_sort_key)

    pending_by_assignee: dict[str, list[dict[str, Any]]] = {}
    for card in pending_assignment_backlog:
        assignee = card_assignee(card)
        if assignee:
            pending_by_assignee.setdefault(assignee, []).append(card)
    for bucket in pending_by_assignee.values():
        bucket.sort(key=backlog_card_sort_key)

    lane_context_violations: list[dict[str, Any]] = []
    for assignee, context_cards in sorted(lane_context_by_assignee.items()):
        if assignee not in queue_workers or len(context_cards) <= 1:
            continue
        if assignee not in dispatch_driven_set:
            # Non-dispatch-driven worker (Pool B manual /loop) — fanout
            # risk doesn't apply.
            continue
        lane_context_violations.append(
            {
                "assignee": assignee,
                "lane_context_count": len(context_cards),
                "card_ids": [_as_text(card.get("id")) for card in context_cards],
                "statuses": [card_status(card) for card in context_cards],
                "reason": "more than one active assigned card for one worker; use a dependency chain or unassigned backlog bank",
            }
        )

    dispatch_risk_violations: list[dict[str, Any]] = []
    for assignee, runnable_cards in sorted(runnable_by_assignee.items()):
        if assignee not in queue_workers:
            continue
        if assignee not in dispatch_driven_set:
            # Pool B has no Hermes-dispatcher fanout; skip dispatch-risk
            # check for these workers too.
            continue
        ready_cards = [card for card in runnable_cards if card_status(card) == "ready"]
        context_cards = lane_context_by_assignee.get(assignee, [])
        reasons: list[str] = []
        if len(ready_cards) > 1:
            reasons.append("multiple ready cards could dispatch for one worker")
        if ready_cards and any(card_status(card) == "blocked" for card in context_cards):
            reasons.append("ready card exists while blocked context-holder is preserving lane")
        if ready_cards and any(card_status(card) == "running" for card in context_cards):
            reasons.append("ready card exists while worker is already running")
        if not reasons:
            continue
        involved = {id(card): card for card in ready_cards + context_cards}
        involved_cards = sorted(involved.values(), key=backlog_card_sort_key)
        dispatch_risk_violations.append(
            {
                "assignee": assignee,
                "runnable_count": len(runnable_cards),
                "ready_count": len(ready_cards),
                "card_ids": [_as_text(card.get("id")) for card in involved_cards],
                "statuses": [card_status(card) for card in involved_cards],
                "reasons": reasons,
            }
        )

    active_violations: list[dict[str, Any]] = []
    active_violations.extend(
        {
            "assignee": item["assignee"],
            "active_count": item["lane_context_count"],
            "card_ids": item["card_ids"],
            "statuses": item["statuses"],
            "reason": item["reason"],
        }
        for item in lane_context_violations
    )
    active_violations.extend(
        {
            "assignee": item["assignee"],
            "active_count": item["runnable_count"],
            "card_ids": item["card_ids"],
            "statuses": item["statuses"],
            "reason": "; ".join(item["reasons"]),
        }
        for item in dispatch_risk_violations
    )

    assignment_plan: list[dict[str, Any]] = []
    for worker in queue_workers:
        if lane_context_by_assignee.get(worker) or runnable_by_assignee.get(worker) or parked_by_assignee.get(worker):
            continue
        worker_pending = pending_by_assignee.get(worker) or []
        if not worker_pending:
            continue
        head = worker_pending[0]
        assignment_plan.append(
            {
                "assignee": worker,
                "task_id": _as_text(head.get("id")),
                "title": _as_text(head.get("title")),
                "workspace_path": _as_text(head.get("workspace_path")),
            }
        )

    low_context_pending = [
        {
            "assignee": card_assignee(card),
            "task_id": _as_text(card.get("id")),
            "title": _as_text(card.get("title")),
        }
        for card in pending_assignment_backlog
        if card_assignee(card) in low_context_workers
    ]

    return {
        "queue_workers": list(queue_workers),
        "active_by_assignee": {
            assignee: [_as_text(card.get("id")) for card in active_cards]
            for assignee, active_cards in sorted(active_by_assignee.items())
        },
        "lane_context_by_assignee": {
            assignee: [_as_text(card.get("id")) for card in context_cards]
            for assignee, context_cards in sorted(lane_context_by_assignee.items())
        },
        "runnable_by_assignee": {
            assignee: [_as_text(card.get("id")) for card in runnable_cards]
            for assignee, runnable_cards in sorted(runnable_by_assignee.items())
        },
        "parked_by_assignee": {
            assignee: [_as_text(card.get("id")) for card in parked_cards]
            for assignee, parked_cards in sorted(parked_by_assignee.items())
        },
        "lane_context_violations": lane_context_violations,
        "dispatch_risk_violations": dispatch_risk_violations,
        "active_violations": active_violations,
        "pending_assignment_backlog": [
            {
                "assignee": card_assignee(card),
                "task_id": _as_text(card.get("id")),
                "title": _as_text(card.get("title")),
                "status": card_status(card),
                "workspace_path": _as_text(card.get("workspace_path")),
            }
            for card in pending_assignment_backlog
        ],
        "pending_assignment_backlog_count": len(pending_assignment_backlog),
        "unassigned_backlog_bank": [
            {
                "task_id": _as_text(card.get("id")),
                "title": _as_text(card.get("title")),
                "status": card_status(card),
                "workspace_path": _as_text(card.get("workspace_path")),
            }
            for card in unassigned_backlog_bank
        ],
        "unassigned_backlog_bank_count": len(unassigned_backlog_bank),
        "assignment_plan": assignment_plan,
        "lane_free_workers": [
            worker
            for worker in queue_workers
            if not lane_context_by_assignee.get(worker) and not runnable_by_assignee.get(worker)
        ],
        "idle_queue_workers": [
            worker
            for worker in queue_workers
            if (
                not lane_context_by_assignee.get(worker)
                and not runnable_by_assignee.get(worker)
                and not parked_by_assignee.get(worker)
            )
        ],
        "low_context_pending": low_context_pending,
        "ok": not lane_context_violations and not dispatch_risk_violations and not pending_assignment_backlog,
    }


_TASK_ID_PATTERN = re.compile(r"task\s*0*(\d+)", re.IGNORECASE)


def _extract_task_id_from_card(card: dict[str, Any]) -> str:
    for field in ("title", "body", "result"):
        text = _as_text(card.get(field))
        if not text:
            continue
        match = _TASK_ID_PATTERN.search(text)
        if match:
            return f"task{int(match.group(1)):03d}"
    return ""


def _extract_claim_mode_from_card(card: dict[str, Any]) -> str:
    text = (_as_text(card.get("title")) + " " + _as_text(card.get("body"))).lower()
    if "independent-verification" in text or "independent verification" in text:
        return "independent-verification"
    return "exclusive"


def derive_target_claims_from_cards(cards: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Build the per-task claim map directly from live Hermes worker cards.

    Replaces the legacy ``state.json.active_individual_task_locks`` registry.
    A claim is recognized whenever an active card (running/ready/todo/blocked)
    is assigned to one of the worker profiles and its title/body names a
    ``taskNNN``. The first such card per task wins (sorted by claim time).
    """
    if not cards:
        return {}
    workers = set(DEFAULT_QUEUE_WORKERS)
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        if card_status(card) not in ACTIVE_CARD_STATUSES:
            continue
        owner = card_assignee(card)
        if owner not in workers:
            continue
        task_key = _extract_task_id_from_card(card)
        if not task_key:
            continue
        claimed_at = _as_text(card.get("started_at")) or _as_text(card.get("created_at"))
        candidates.append((task_key, claimed_at, card))
    candidates.sort(key=lambda row: (row[0], row[1] or "9999"))
    claims: dict[str, dict[str, Any]] = {}
    for task_key, claimed_at, card in candidates:
        if task_key in claims:
            continue
        owner = card_assignee(card)
        kanban_task_id = _as_text(card.get("id"))
        claim_mode = _extract_claim_mode_from_card(card)
        claims[task_key] = {
            "task_id": task_key,
            "owner_profile": owner,
            "kanban_task_id": kanban_task_id,
            "claim_mode": claim_mode,
            "claimed_at": claimed_at,
            "reason": _as_text(card.get("title"))[:200],
            "candidate_scope": "single-task candidate",
            "duplicate_guard_key": build_worker_chain_key(task_key, owner, "current-best"),
            "chain_key": build_worker_chain_key(task_key, owner, "current-best"),
        }
    return claims


def canonical_target_claims(
    cards: list[dict[str, Any]] | None = None,
    *,
    state_doc: dict[str, Any] | None = None,  # accepted for backward compat; unused
) -> dict[str, dict[str, Any]]:
    """Live worker target claims, derived from Hermes (single source of truth)."""
    del state_doc  # state.json lock sections were retired 2026-05-17
    return derive_target_claims_from_cards(cards)


def _entry_status_text(entry: dict[str, Any]) -> str:
    for field in ("submit_queue_status", "status", "result_status", "verdict"):
        text = _as_text(entry.get(field)).lower()
        if text:
            return text
    return ""


def _entry_is_live(entry: dict[str, Any]) -> bool:
    status = _entry_status_text(entry)
    if not status:
        return True
    return not any(term in status for term in LIVE_TERMINAL_STATUSES)


def _entry_chain_key_map(entry: dict[str, Any]) -> dict[str, str]:
    keys: dict[str, str] = {}
    stored = entry.get("chain_keys")
    if isinstance(stored, dict):
        for stage, value in stored.items():
            stage_text = _as_text(stage)
            value_text = _as_text(value)
            if stage_text and value_text:
                keys[stage_text] = value_text
    primary = _as_text(entry.get("chain_key"))
    primary_stage = _as_text(entry.get("chain_key_stage") or entry.get("stage"))
    if primary and primary_stage:
        keys.setdefault(primary_stage, primary)
    candidate_id = _as_text(entry.get("candidate_id"))
    if candidate_id:
        keys.setdefault("review", build_review_chain_key(candidate_id))
        keys.setdefault("worthiness", build_worthiness_chain_key(candidate_id))
        stack_ids = entry.get("stack_candidate_ids")
        if not stack_ids:
            stack_plan = entry.get("stack_plan")
            if isinstance(stack_plan, dict):
                stack_ids = stack_plan.get("stack_candidate_ids")
        if not stack_ids:
            submit_execution = entry.get("submit_execution")
            if isinstance(submit_execution, dict):
                stack_ids = submit_execution.get("stack_candidate_ids")
        keys.setdefault("submit", build_submit_chain_key(candidate_id, stack_candidate_ids=stack_ids))
    task_id = normalize_task_id(entry.get("task_id"))
    owner_profile = _as_text(entry.get("worker_owner") or entry.get("owner_profile") or entry.get("followup_worker") or entry.get("return_worker") or entry.get("agent_id"))
    lineage = _as_text(entry.get("lineage_base_candidate_id"))
    if not lineage:
        lineage_base = entry.get("lineage_base")
        if isinstance(lineage_base, dict):
            lineage = _as_text(lineage_base.get("candidate_id") or lineage_base.get("id"))
    current_best = _as_text(entry.get("current_best_candidate_id") or entry.get("current_best") or entry.get("source_candidate_id"))
    if task_id and owner_profile:
        keys.setdefault("worker", build_worker_chain_key(task_id, owner_profile, lineage, current_best_candidate_id=current_best))
    return keys


def live_chain_key_matches(
    stage: str,
    chain_key: str,
    *,
    registry_doc: dict[str, Any] | None = None,
    queue_doc: dict[str, Any] | None = None,
    self_candidate_id: str = "",
    self_queue_id: str = "",
) -> list[dict[str, Any]]:
    stage = _as_text(stage)
    chain_key = _as_text(chain_key)
    registry_doc = registry_doc or {}
    queue_doc = queue_doc or {}
    matches: list[dict[str, Any]] = []
    sources: list[tuple[str, list[dict[str, Any]]]] = []
    registry_entries = registry_doc.get("entries") if isinstance(registry_doc.get("entries"), list) else []
    queue_entries = queue_doc.get("queue") if isinstance(queue_doc.get("queue"), list) else []
    if stage in {"review", "worthiness", "submit", "worker"}:
        sources.append(("candidate_registry", [entry for entry in registry_entries if isinstance(entry, dict)]))
        sources.append(("submit_queue", [entry for entry in queue_entries if isinstance(entry, dict)]))
    else:
        sources.append(("candidate_registry", [entry for entry in registry_entries if isinstance(entry, dict)]))
        sources.append(("submit_queue", [entry for entry in queue_entries if isinstance(entry, dict)]))
    for source, entries in sources:
        for entry in entries:
            if not _entry_is_live(entry):
                continue
            if self_candidate_id and _as_text(entry.get("candidate_id")) == self_candidate_id:
                continue
            if self_queue_id and _as_text(entry.get("queue_id")) == self_queue_id:
                continue
            key_map = _entry_chain_key_map(entry)
            entry_key = key_map.get(stage) or _as_text(entry.get("chain_key"))
            if chain_key and entry_key == chain_key:
                matches.append(
                    {
                        "source": source,
                        "candidate_id": _as_text(entry.get("candidate_id")),
                        "task_id": normalize_task_id(entry.get("task_id")),
                        "queue_id": _as_text(entry.get("queue_id")),
                        "chain_key": entry_key,
                        "status": _entry_status_text(entry),
                    }
                )
    return matches


def live_target_claim_match(
    task_id: Any,
    *,
    cards: list[dict[str, Any]] | None = None,
    state_doc: dict[str, Any] | None = None,  # accepted for backward compat; unused
    self_kanban_task_id: str = "",
) -> dict[str, Any] | None:
    del state_doc  # state.json lock sections were retired 2026-05-17
    task_key = normalize_task_id(task_id)
    if not task_key:
        return None
    claims = canonical_target_claims(cards)
    claim = claims.get(task_key)
    if not claim:
        return None
    self_id = _as_text(self_kanban_task_id)
    if self_id and _as_text(claim.get("kanban_task_id")) == self_id:
        return None
    return {
        "task_id": task_key,
        "owner_profile": _as_text(claim.get("owner_profile")),
        "kanban_task_id": _as_text(claim.get("kanban_task_id")),
        "claim_mode": _as_text(claim.get("claim_mode")),
        "claimed_at": _as_text(claim.get("claimed_at")),
        "reason": _as_text(claim.get("reason")),
        "candidate_scope": _as_text(claim.get("candidate_scope") or "single-task candidate"),
        "duplicate_guard_key": _as_text(claim.get("duplicate_guard_key")),
        "chain_key": _as_text(claim.get("chain_key")),
    }


def build_live_card_guard(
    stage: str,
    candidate: dict[str, Any],
    *,
    registry_doc: dict[str, Any] | None = None,
    queue_doc: dict[str, Any] | None = None,
    cards: list[dict[str, Any]] | None = None,
    state_doc: dict[str, Any] | None = None,  # accepted for backward compat; unused
    self_candidate_id: str = "",
    self_queue_id: str = "",
    self_kanban_task_id: str = "",
) -> dict[str, Any]:
    del state_doc  # state.json lock sections were retired 2026-05-17
    stage = _as_text(stage) or "review"
    chain_keys = build_chain_keys(candidate)
    chain_key = chain_keys.get(stage, "")
    live_matches = live_chain_key_matches(
        stage,
        chain_key,
        registry_doc=registry_doc,
        queue_doc=queue_doc,
        self_candidate_id=self_candidate_id,
        self_queue_id=self_queue_id,
    )
    target_claim = (
        live_target_claim_match(candidate.get("task_id"), cards=cards, self_kanban_task_id=self_kanban_task_id)
        if stage == "worker"
        else None
    )
    warnings: list[str] = []
    status = "ok"
    if live_matches:
        status = "duplicate-live-chain-key"
        warnings.append(f"live {stage} card already exists for chain_key={chain_key}")
    if target_claim:
        status = "duplicate-live-target-claim"
        warnings.append(f"live target claim already exists for task_id={target_claim['task_id']}")
    return {
        "ok": status == "ok",
        "status": status,
        "stage": stage,
        "chain_key": chain_key,
        "chain_keys": chain_keys,
        "live_chain_matches": live_matches,
        "live_target_claim": target_claim,
        "warnings": warnings,
        "result": {
            "status": status,
            "stage": stage,
            "chain_key": chain_key,
            "warning_count": len(warnings),
        },
    }
