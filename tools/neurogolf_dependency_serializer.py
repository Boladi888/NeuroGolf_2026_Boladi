#!/usr/bin/env python3
"""Deterministically serialize NeuroGolf worker lanes.

The script keeps one active assigned lane per Alpha worker. Extra future cards
are linked behind the lane head and unassigned back to the backlog bank so
Hermes cannot fan out a whole dependency row when one parent completes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_board_policy import (
    ACTIVE_CARD_STATUSES,
    BACKLOG_TITLE_PREFIX,
    DEFAULT_QUEUE_WORKERS,
    POOL_A_DRAINING_WORKERS,
    POOL_A_SHRINK_WORKERS,
    POOL_B_SOLVER_HUNT_WORKERS,
)
from tools.neurogolf_control_config import primary_reviewer_profile, reviewer_profiles, specialist_profiles
from tools.neurogolf_kanban_python import (
    KanbanClient,
    as_text,
    card_assignee,
    card_body,
    card_created_at,
    card_text,
    card_id,
    card_priority,
    card_status,
    card_title,
    sort_cards,
)

STATUS_RANK = {
    "running": 0,
    "blocked": 1,
    "ready": 2,
    "todo": 3,
}

FUTURE_STATUSES = {"ready", "todo"}
WORKER_RE = re.compile(r"neurogolf-worker-\d+")
WORKER_HINT_RE = re.compile(r"\b(?:neurogolf-)?worker[- ]?([1-8])\b", re.IGNORECASE)
TASK_RE = re.compile(r"\btask\d{3}\b", re.IGNORECASE)
KANBAN_CARD_RE = re.compile(r"\bt_[0-9a-f]{8}\b", re.IGNORECASE)
SOURCE_WORKER_CARD_RE = re.compile(
    r"\b(?:reviewed\s+)?source(?:\s+worker)?\s+card[:\s]+(t_[0-9a-f]{8})\b",
    re.IGNORECASE,
)
REVIEW_SOURCE_LINE_RE = re.compile(
    r"^\s*-\s+(t_[0-9a-f]{8})\s*\|\s*(neurogolf-worker-[1-8])\s*\|\s*tasks=([^|]+)",
    re.IGNORECASE,
)
RETURN_TITLE_PREFIXES = ("[repair]", "[optimization-return]")
# Some submit-captain repair cards open with `[repair taskNNN] ...` (task in
# brackets) or plain `repair task...` (no brackets). The narrow .startswith
# check missed both, causing the serializer to treat them as generic extras
# and strip the assignee — losing the original worker's context. Use a regex
# that captures the [repair / [optimization-return / repair / optimization-return
# variants regardless of what follows.
RETURN_TITLE_REGEX = re.compile(
    r"^\s*\[?(?:repair|optimization[- ]return)\b",
    re.IGNORECASE,
)
SPECIALIST_PROFILES = set(specialist_profiles())
SPECIALIST_ASSIGNEE_ALIASES = {
    "blocked-55": "neurogolf-blocked-55",
    "board-analyst": "neurogolf-board-analyst",
    "orchestrator": "neurogolf-orchestrator",
    "reviewer": primary_reviewer_profile(),
    "reviewer-1": primary_reviewer_profile(),
    "reviewer-2": "neurogolf-reviewer-2",
    "submit-captain": "neurogolf-submit-captain",
    "sync-manager": "neurogolf-sync-manager",
}


def _lane_sort_key(card: dict[str, Any]) -> tuple[int, int, int, int, str]:
    # Lane head priority order:
    #   1. Status: running > blocked > ready > todo (STATUS_RANK)
    #   2. Within FUTURE statuses only (ready/todo): return cards (repair /
    #      opt-return) ahead of non-return at the same status. This makes a
    #      worker's NEXT pick the repair-context card, not a fresh backlog
    #      target. For running/blocked we keep the legacy ordering so two
    #      conflicting in-flight cards still resolve in priority/age order
    #      (covered by the existing cleanup test).
    #   3. Explicit priority (higher wins)
    #   4. Created earlier wins as a final stable tiebreaker.
    status = card_status(card)
    is_future = status in FUTURE_STATUSES
    return_boost = (0 if is_future and is_return_card(card) else 1)
    return (
        STATUS_RANK.get(status, 99),
        return_boost,
        -card_priority(card),
        card_created_at(card),
        card_id(card),
    )


def _active_worker_cards(cards: list[dict[str, Any]], workers: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {worker: [] for worker in workers}
    for card in cards:
        assignee = card_assignee(card)
        if assignee not in grouped:
            continue
        if card_status(card) in ACTIVE_CARD_STATUSES:
            grouped[assignee].append(card)
    for worker in grouped:
        grouped[worker].sort(key=_lane_sort_key)
    return grouped


def canonical_specialist_assignee(assignee: str) -> str:
    return SPECIALIST_ASSIGNEE_ALIASES.get(assignee, assignee if assignee in SPECIALIST_PROFILES else "")


def _active_specialist_cards(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {profile: [] for profile in SPECIALIST_PROFILES}
    for card in cards:
        if card_status(card) not in ACTIVE_CARD_STATUSES:
            continue
        canonical = canonical_specialist_assignee(card_assignee(card))
        if canonical:
            grouped[canonical].append(card)
    for profile in grouped:
        grouped[profile].sort(key=_lane_sort_key)
    return grouped


def is_unassigned_return_card(card: dict[str, Any]) -> bool:
    if card_status(card) != "ready" or card_assignee(card):
        return False
    title = card_title(card) or ""
    if title.lower().startswith(RETURN_TITLE_PREFIXES):
        return True
    return bool(RETURN_TITLE_REGEX.match(title))


def is_return_card(card: dict[str, Any]) -> bool:
    title = card_title(card) or ""
    if title.lower().startswith(RETURN_TITLE_PREFIXES):
        return True
    return bool(RETURN_TITLE_REGEX.match(title))


_SOLVER_HUNT_TITLE_RE = re.compile(r"solver[- ]hunt", re.IGNORECASE)


def is_solver_hunt_card(card: dict[str, Any]) -> bool:
    """Pool B `[worker taskNNN <worker> solver-hunt] …` card.

    The Pool B autoforeman creates these for the unsolved-task hunt
    (workers 5, 7, 8). They carry the same kind of accumulated context as
    repair / optimization-return cards: the worker may have already posted
    a JSON handoff comment with their candidate. The dependency serializer
    must NOT unassign or demote them when a transient cold-start onto a
    BACKLOG bank card makes them an `extra` — the candidate work would be
    lost. Treat them as protected, same as return cards.
    """
    title = card_title(card) or ""
    return bool(_SOLVER_HUNT_TITLE_RE.search(title))


def is_protected_worker_card(card: dict[str, Any]) -> bool:
    """Worker cards whose assignee context must NOT be stripped by the
    dependency serializer's `serialize_future_card` op.

    Currently: return cards (repair, optimization-return) and Pool B
    solver-hunt cards. Both carry per-worker context (candidate ID,
    package SHA, base lineage, partial scratch in the worktree) that
    can't be reconstructed from the kanban after an unassign.
    """
    return is_return_card(card) or is_solver_hunt_card(card)


# BACKLOG bank cards can be cold-started by Pool A workers (the foreman's
# normal `create_worker_from_bank` conversion path) AND by Pool B manual
# /loop workers (5, 7, 8). Pool B inclusion is the operational reality as
# of 2026-05-18: with the Hermes-paid LLM seats off, the only active
# candidate producers are the manual /loop sessions, and they need to be
# able to claim BACKLOG bank cards without the misclaim guard releasing
# them mid-flight. To revive the strict Pool-A-only policy, drop the
# `| set(POOL_B_SOLVER_HUNT_WORKERS)` term.
_BACKLOG_BANK_ALLOWED_WORKERS = (
    set(POOL_A_SHRINK_WORKERS)
    | set(POOL_A_DRAINING_WORKERS)
    | set(POOL_B_SOLVER_HUNT_WORKERS)
)


_WORKER_OWNER_HANDOFF_RE = re.compile(
    r'"worker_owner"\s*:\s*"neurogolf-worker-\d+"',
    re.IGNORECASE,
)
# Workers post their `BLOCKED: review-required: ...` summary into the card's
# `latest_summary` field when they hand off. The JSON handoff itself lives
# in comments (which `hermes kanban list --json` doesn't include), but the
# review-required marker reliably bubbles up to latest_summary AND becomes
# the card's block reason. Treat its presence as "work was done; don't
# strip the assignee" — same intent as the worker_owner JSON check.
_REVIEW_REQUIRED_HANDOFF_RE = re.compile(
    r"review[-\s]required\s*:",
    re.IGNORECASE,
)


def _card_has_worker_handoff(card: dict[str, Any]) -> bool:
    """True iff the card carries evidence of a real worker handoff.

    The reliable signal is **status == "blocked"**. When a Pool B worker
    cold-starts onto a BACKLOG card, the misclaim race goes:

      1. Worker assigns + claims card (status: ready → running)
      2. Worker builds candidate, posts JSON handoff as a comment
      3. Worker runs `hermes kanban block --reason "review-required: ..."`
         (status: running → blocked)

    The misclaim guard's purpose is to catch step 1 before any work
    happens. By the time the status is `blocked`, the worker has
    completed steps 2+3 — the candidate is real and any release would
    orphan it.

    Why not the body-text check? `hermes kanban list --json` doesn't
    include comments, and the JSON handoff is posted as a comment. The
    BACKLOG card's body is the stock "do not dispatch directly" template
    — there's no worker_owner pattern in it. Status is the only signal
    visible in the list snapshot that reliably distinguishes pre-work
    (ready/running) from post-work (blocked).

    The body/text regex check is kept as a belt-and-suspenders signal
    for cards whose body DOES happen to include the handoff JSON (e.g.,
    cards whose body was reconstructed by python-control during a
    misclaim cleanup).
    """
    if card_status(card) == "blocked":
        return True
    body = card_body(card) or ""
    text = card_text(card) or ""
    if _WORKER_OWNER_HANDOFF_RE.search(body) or _WORKER_OWNER_HANDOFF_RE.search(text):
        return True
    if _REVIEW_REQUIRED_HANDOFF_RE.search(body) or _REVIEW_REQUIRED_HANDOFF_RE.search(text):
        return True
    return False


def is_misclaimed_backlog_card(card: dict[str, Any]) -> bool:
    """A `[BACKLOG single task] …` bank card that ended up assigned to a
    worker outside the Pool A shrink + drain set AND has not yet been
    handed off with a real worker JSON.

    The most common cause: a Claude Code Pool B worker's ``/loop`` session
    runs the playbook's cold-start fallback ("if no ready cards for my
    assignee, pull from the unassigned BACKLOG bank"). The /loop wins the
    race against the auto-unblock cron's recovery of its own Pool B card
    and ends up holding both — which the dep serializer then resolves by
    demoting the Pool B card.

    The right fix is to refuse the misclaim at the source: BACKLOG bank
    cards exist for ``create_worker_from_bank`` (Pool A foreman conversion)
    only. A Pool B worker grabbing one is always wrong UNTIL the worker has
    already posted a real review-required JSON handoff — at that point the
    candidate is real work and we should NOT unassign mid-flight, even
    though the card title still says ``[BACKLOG …]``. The handoff signals
    that the worker has committed to this card and the auto-triage cron
    will route it to a reviewer next.

    Draining Pool A workers (4, 6) are allowed to keep BACKLOG cards mid-
    flight because they may already be working on them; the foreman just
    won't hand them new ones.
    """
    title = card_title(card) or ""
    if not title.startswith(BACKLOG_TITLE_PREFIX):
        return False
    assignee = card_assignee(card)
    if not assignee:
        return False
    if card_status(card) not in ACTIVE_CARD_STATUSES:
        return False
    if assignee in _BACKLOG_BANK_ALLOWED_WORKERS:
        return False
    # BUG FIX 2026-05-18: if the worker has already posted a JSON handoff
    # (review-required candidate), do NOT reclaim — that would orphan
    # legitimate work. Observed: worker-7's /loop produced task264 v2
    # candidate (cost -84, 265/265 verify) but the misclaim guard wiped
    # the assignee mid-block, leaving t_dc8111c3 stuck blocked with no
    # owner. The handoff signature is durable enough to detect this.
    if _card_has_worker_handoff(card):
        return False
    return True


def intended_unassigned_specialist(card: dict[str, Any]) -> str:
    if card_assignee(card) or card_status(card) not in FUTURE_STATUSES:
        return ""
    if is_return_card(card):
        return ""
    title = card_title(card).lower()
    # Worker/backlog cards never route to specialists. The body-text fallback
    # below is a safety net for legacy review cards that lack a `[review` title
    # prefix, but it false-positives on backlog cards whose stock body line
    # `next_action: Review candidate queue before changing this task.` contains
    # the literal phrase "Review candidate". When that fires, the lane below
    # rotates through reviewer profiles and dumps a worker build card onto
    # whichever reviewer lane is free (the bug that misrouted t_b8abe4d8 to
    # neurogolf-reviewer-2). Short-circuit on title prefix before consulting
    # the body.
    if title.startswith("[backlog") or title.startswith("[worker"):
        return ""
    text = card_text(card).lower()
    if title.startswith("[review") or "review candidate" in text or "review task" in text:
        return primary_reviewer_profile()
    if title.startswith("[submit") or title.startswith("submit-captain") or title.startswith("[submit-captain"):
        return "neurogolf-submit-captain"
    return ""


def intended_return_worker(card: dict[str, Any], workers: tuple[str, ...] = DEFAULT_QUEUE_WORKERS) -> str:
    haystack = "\n".join(
        str(card.get(key) or "")
        for key in ("title", "body", "description", "latest_summary", "workspace_path", "created_by")
    )
    for match in WORKER_RE.findall(haystack):
        if match in workers:
            return match
    for match in WORKER_HINT_RE.findall(haystack):
        worker = f"neurogolf-worker-{int(match)}"
        if worker in workers:
            return worker
    return ""


def task_ids_from_card(card: dict[str, Any]) -> set[str]:
    haystack = "\n".join(
        str(card.get(key) or "")
        for key in ("title", "body", "description", "latest_summary", "workspace_path")
    )
    return {match.group(0).lower() for match in TASK_RE.finditer(haystack)}


def referenced_card_ids(card: dict[str, Any]) -> list[str]:
    haystack = "\n".join(
        str(card.get(key) or "")
        for key in ("title", "body", "description", "latest_summary", "workspace_path")
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for match in SOURCE_WORKER_CARD_RE.findall(haystack):
        value = match.lower()
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    for match in KANBAN_CARD_RE.findall(haystack):
        value = match.lower()
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def worker_from_source_card(
    source: dict[str, Any],
    target_task_ids: set[str],
    workers: tuple[str, ...],
) -> str:
    assignee = card_assignee(source)
    source_task_ids = task_ids_from_card(source)
    overlaps_target = not target_task_ids or not source_task_ids or bool(target_task_ids & source_task_ids)
    if overlaps_target and assignee in workers:
        return assignee
    # If the source is a review packet, do NOT body-scan it for worker
    # references — a single review packet typically bundles multiple
    # candidates from different workers, and a naive body scan picks the
    # FIRST `neurogolf-worker-N` mentioned regardless of which task it's
    # for. Use worker_from_review_source_lines() instead, which parses the
    # structured `- t_XXX | worker | tasks=...` lines and filters by task
    # overlap. Caller falls through to that path automatically.
    if assignee in reviewer_profiles():
        return ""
    if overlaps_target:
        worker = intended_return_worker(source, workers)
        if worker:
            return worker
    return ""


def worker_from_review_source_lines(
    review_card: dict[str, Any],
    target_task_ids: set[str],
    workers: tuple[str, ...],
) -> str:
    if card_assignee(review_card) not in reviewer_profiles():
        return ""
    for line in card_text(review_card).splitlines():
        match = REVIEW_SOURCE_LINE_RE.match(line)
        if not match:
            continue
        worker = match.group(2)
        if worker not in workers:
            continue
        source_tasks = {found.group(0).lower() for found in TASK_RE.finditer(match.group(3))}
        if target_task_ids and not (target_task_ids & source_tasks):
            continue
        return worker
    return ""


def intended_return_worker_from_context(
    card: dict[str, Any],
    cards_by_id: dict[str, dict[str, Any]],
    workers: tuple[str, ...] = DEFAULT_QUEUE_WORKERS,
) -> str:
    worker = intended_return_worker(card, workers)
    if worker:
        return worker
    target_task_ids = task_ids_from_card(card)
    for source_id in referenced_card_ids(card):
        source = cards_by_id.get(source_id)
        if not source:
            continue
        worker = worker_from_source_card(source, target_task_ids, workers)
        if worker:
            return worker
        worker = worker_from_review_source_lines(source, target_task_ids, workers)
        if worker:
            return worker
    return ""


def source_worker_card_from_context(
    card: dict[str, Any],
    cards_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_task_ids = task_ids_from_card(card)
    for source_id in referenced_card_ids(card):
        source = cards_by_id.get(source_id)
        if source and worker_from_source_card(source, target_task_ids, DEFAULT_QUEUE_WORKERS):
            return source
    return {}


def replacement_workspace_for_return(
    card: dict[str, Any],
    cards_by_id: dict[str, dict[str, Any]],
) -> str:
    if as_text(card.get("workspace_kind")).lower() != "scratch":
        return ""
    source = source_worker_card_from_context(card, cards_by_id)
    source_path = as_text(source.get("workspace_path"))
    if source_path:
        return f"dir:{source_path}"
    return "worktree"


def replacement_body_for_return(card: dict[str, Any]) -> str:
    return (
        f"Python dependency serializer recreated this return card from scratch workspace card {card_id(card)} "
        "so the worker runs inside a project workspace and keeps useful context.\n\n"
        f"{card_body(card)}"
    ).strip()


def plan_serialization(
    cards: list[dict[str, Any]],
    *,
    workers: tuple[str, ...] = DEFAULT_QUEUE_WORKERS,
    max_actions: int = 24,
    running_gateways: list[str] | None = None,
) -> dict[str, Any]:
    grouped = _active_worker_cards(cards, workers)
    specialist_grouped = _active_specialist_cards(cards)
    actions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    return_waiting: list[dict[str, Any]] = []
    cards_by_id = {card_id(card): card for card in cards}
    lanes: dict[str, dict[str, Any]] = {}
    reserved_specialists = {profile for profile, specialist_cards in specialist_grouped.items() if specialist_cards}
    reviewer_profile_set = set(reviewer_profiles())

    # BACKLOG bank cards are for create_worker_from_bank only (Pool A
    # foreman conversion). When a Claude Code Pool B worker's /loop session
    # cold-starts onto a BACKLOG card while waiting for its own Pool B
    # card to be auto-unblocked, that's a race condition we keep observing
    # (worker-7 t_545486fe twice, worker-8 t_30560b2a). Detect + release.
    for card in sort_cards([c for c in cards if is_misclaimed_backlog_card(c)]):
        if len(actions) >= max_actions:
            unresolved.append(
                {
                    "worker": card_assignee(card),
                    "task_id": card_id(card),
                    "status": card_status(card),
                    "title": card_title(card),
                    "reason": "max_actions reached before BACKLOG misclaim could be released",
                }
            )
            continue
        actions.append(
            {
                "op": "unassign_backlog_misclaim",
                "worker": card_assignee(card),
                "task_id": card_id(card),
                "status": card_status(card),
                "title": card_title(card),
            }
        )

    for profile, specialist_cards in sorted(specialist_grouped.items()):
        if not specialist_cards:
            continue
        head = specialist_cards[0]
        head_id = card_id(head)
        lanes[profile] = {
            "head": head_id,
            "head_status": card_status(head),
            "active_ids": [card_id(card) for card in specialist_cards],
        }
        if is_return_card(head) and card_status(head) in ACTIVE_CARD_STATUSES:
            if len(actions) >= max_actions:
                unresolved.append(
                    {
                        "worker": profile,
                        "task_id": head_id,
                        "status": card_status(head),
                        "title": card_title(head),
                        "reason": "max_actions reached before misassigned return card could be removed from specialist lane",
                    }
                )
            else:
                actions.append(
                    {
                        "op": "unassign_misassigned_specialist_return",
                        "worker": profile,
                        "target_worker": intended_return_worker_from_context(head, cards_by_id, workers),
                        "task_id": head_id,
                        "status": card_status(head),
                        "title": card_title(head),
                    }
                )
                reserved_specialists.discard(profile)
            continue
        if card_assignee(head) != profile:
            if len(actions) >= max_actions:
                unresolved.append(
                    {
                        "worker": profile,
                        "task_id": head_id,
                        "status": card_status(head),
                        "title": card_title(head),
                        "reason": "max_actions reached before noncanonical specialist assignee could be repaired",
                    }
                )
            else:
                actions.append(
                    {
                        "op": "normalize_specialist_assignee",
                        "worker": profile,
                        "old_assignee": card_assignee(head),
                        "task_id": head_id,
                        "status": card_status(head),
                        "title": card_title(head),
                    }
                )
        for extra in specialist_cards[1:]:
            if len(actions) >= max_actions:
                unresolved.append(
                    {
                        "worker": profile,
                        "task_id": card_id(extra),
                        "status": card_status(extra),
                        "title": card_title(extra),
                        "reason": "max_actions reached before extra specialist card could be serialized",
                    }
                )
                continue
            status = card_status(extra)
            if is_return_card(extra) and status in ACTIVE_CARD_STATUSES:
                actions.append(
                    {
                        "op": "unassign_misassigned_specialist_return",
                        "worker": profile,
                        "target_worker": intended_return_worker_from_context(extra, cards_by_id, workers),
                        "task_id": card_id(extra),
                        "status": status,
                        "title": card_title(extra),
                    }
                )
                continue
            if status in FUTURE_STATUSES:
                actions.append(
                    {
                        "op": "serialize_specialist_future_card",
                        "worker": profile,
                        "old_assignee": card_assignee(extra),
                        "head_id": head_id,
                        "task_id": card_id(extra),
                        "status": status,
                        "title": card_title(extra),
                    }
                )
            else:
                unresolved.append(
                    {
                        "worker": profile,
                        "task_id": card_id(extra),
                        "status": status,
                        "title": card_title(extra),
                        "reason": "extra running/blocked specialist card needs manual or role-specific cleanup",
                    }
                )

    for worker, worker_cards in grouped.items():
        if not worker_cards:
            continue
        head = worker_cards[0]
        extras = worker_cards[1:]
        lanes[worker] = {
            "head": card_id(head),
            "head_status": card_status(head),
            "active_ids": [card_id(card) for card in worker_cards],
        }
        if is_return_card(head):
            target_worker = intended_return_worker_from_context(head, cards_by_id, workers)
            if target_worker and target_worker != worker:
                if len(actions) >= max_actions:
                    unresolved.append(
                        {
                            "worker": worker,
                            "target_worker": target_worker,
                            "task_id": card_id(head),
                            "status": card_status(head),
                            "title": card_title(head),
                            "reason": "max_actions reached before misassigned worker return could be removed from the wrong lane",
                        }
                    )
                else:
                    actions.append(
                        {
                            "op": "unassign_misassigned_worker_return",
                            "worker": worker,
                            "target_worker": target_worker,
                            "task_id": card_id(head),
                            "status": card_status(head),
                            "title": card_title(head),
                        }
                    )
                continue
        replacement_workspace = replacement_workspace_for_return(head, cards_by_id) if is_return_card(head) else ""
        if replacement_workspace:
            if len(actions) >= max_actions:
                unresolved.append(
                    {
                        "worker": worker,
                        "task_id": card_id(head),
                        "status": card_status(head),
                        "title": card_title(head),
                        "reason": "max_actions reached before scratch return card could be recreated in a project workspace",
                    }
                )
            else:
                actions.append(
                    {
                        "op": "replace_scratch_return_card",
                        "worker": worker,
                        "task_id": card_id(head),
                        "status": card_status(head),
                        "title": card_title(head),
                        "body": replacement_body_for_return(head),
                        "priority": card_priority(head),
                        "workspace": replacement_workspace,
                        "idempotency_key": f"ng-return-workspace-{card_id(head)}",
                    }
                )
            continue
        for extra in extras:
            if len(actions) >= max_actions:
                unresolved.append(
                    {
                        "worker": worker,
                        "task_id": card_id(extra),
                        "status": card_status(extra),
                        "reason": "max_actions reached before this extra card could be serialized",
                    }
                )
                continue
            status = card_status(extra)
            replacement_workspace = replacement_workspace_for_return(extra, cards_by_id) if is_return_card(extra) else ""
            if replacement_workspace:
                actions.append(
                    {
                        "op": "replace_scratch_return_card",
                        "worker": worker,
                        "task_id": card_id(extra),
                        "status": status,
                        "title": card_title(extra),
                        "body": replacement_body_for_return(extra),
                        "priority": card_priority(extra),
                        "workspace": replacement_workspace,
                        "idempotency_key": f"ng-return-workspace-{card_id(extra)}",
                    }
                )
                continue
            if status in FUTURE_STATUSES:
                # Repair / optimization-return / Pool B solver-hunt cards
                # carry per-worker context (failed candidate, package SHA,
                # base lineage, partial scratch in the worktree, posted
                # JSON handoff). Park them behind the lane head as a CHILD
                # but KEEP the assignee — when head completes, the worker
                # picks them up next with their per-profile memory and
                # worktree state intact. Generic backlog extras still get
                # unassigned via serialize_future_card.
                if is_protected_worker_card(extra):
                    actions.append(
                        {
                            "op": "park_future_return_card_assigned",
                            "worker": worker,
                            "head_id": card_id(head),
                            "task_id": card_id(extra),
                            "status": status,
                            "title": card_title(extra),
                        }
                    )
                else:
                    actions.append(
                        {
                            "op": "serialize_future_card",
                            "worker": worker,
                            "head_id": card_id(head),
                            "task_id": card_id(extra),
                            "status": status,
                            "title": card_title(extra),
                        }
                    )
            elif status == "running" and is_return_card(extra):
                actions.append(
                    {
                        "op": "serialize_running_return_card",
                        "worker": worker,
                        "head_id": card_id(head),
                        "task_id": card_id(extra),
                        "status": status,
                        "title": card_title(extra),
                    }
                )
            else:
                unresolved.append(
                    {
                        "worker": worker,
                        "task_id": card_id(extra),
                        "status": status,
                        "title": card_title(extra),
                        "reason": "extra running/blocked cards preserve context and need reviewer/blocked cleanup, not automatic unassignment",
                    }
                )

    reserved_workers = {worker for worker, worker_cards in grouped.items() if worker_cards}
    for card in sort_cards([card for card in cards if intended_unassigned_specialist(card)]):
        profile = intended_unassigned_specialist(card)
        if profile in reviewer_profile_set:
            for reviewer in reviewer_profiles():
                if reviewer not in reserved_specialists:
                    profile = reviewer
                    break
        item = {
            "worker": profile,
            "task_id": card_id(card),
            "status": card_status(card),
            "title": card_title(card),
        }
        if profile in reserved_specialists:
            continue
        if len(actions) >= max_actions:
            unresolved.append({**item, "reason": "max_actions reached before specialist continuation could be assigned"})
            continue
        actions.append(
            {
                "op": "assign_specialist_continuation",
                **item,
            }
        )
        reserved_specialists.add(profile)

    for card in sort_cards([card for card in cards if is_unassigned_return_card(card)]):
        worker = intended_return_worker_from_context(card, cards_by_id, workers)
        item = {
            "worker": worker,
            "task_id": card_id(card),
            "status": card_status(card),
            "title": card_title(card),
        }
        if not worker:
            unresolved.append({**item, "reason": "unassigned return card has no deterministic target worker"})
            continue
        if worker in reserved_workers:
            return_waiting.append({**item, "reason": "target worker already has an active context lane"})
            continue
        if len(actions) >= max_actions:
            unresolved.append({**item, "reason": "max_actions reached before return card could be assigned"})
            continue
        replacement_workspace = replacement_workspace_for_return(card, cards_by_id)
        if replacement_workspace:
            actions.append(
                {
                    "op": "replace_scratch_return_card",
                    "worker": worker,
                    "task_id": card_id(card),
                    "status": card_status(card),
                    "title": card_title(card),
                    "body": replacement_body_for_return(card),
                    "priority": card_priority(card),
                    "workspace": replacement_workspace,
                    "idempotency_key": f"ng-return-workspace-{card_id(card)}",
                }
            )
            reserved_workers.add(worker)
            continue
        actions.append(
            {
                "op": "assign_return_card",
                "worker": worker,
                "task_id": card_id(card),
                "status": card_status(card),
                "title": card_title(card),
            }
        )
        reserved_workers.add(worker)

    # Final fallback (2026-05-19): any ready+unassigned card that didn't get
    # routed by the above passes (no clear lineage, not a known specialist
    # type, not a return card) gets assigned to a running gateway profile.
    # This is what makes "auto-assign to available workers" actually work —
    # without it, an unassigned ready card just sits there until a human
    # picks it. The set of "available" profiles comes from
    # ``hermes gateway list`` parsed by ``running_gateway_profiles()``;
    # caller can override via ``running_gateways`` arg (used by tests).
    available = running_gateways if running_gateways is not None else running_gateway_profiles()
    if available:
        already_targeted = {a.get("task_id") for a in actions if a.get("task_id")}
        for card in sort_cards([c for c in cards if _is_orphan_ready_card(c)]):
            tid = card_id(card)
            if tid in already_targeted:
                continue
            if len(actions) >= max_actions:
                unresolved.append({
                    "worker": "",
                    "task_id": tid,
                    "status": card_status(card),
                    "title": card_title(card),
                    "reason": "max_actions reached before orphan ready card could be assigned to a running gateway",
                })
                continue
            target = _pick_gateway_for_card(card, available, reserved_workers, reserved_specialists)
            if not target:
                unresolved.append({
                    "worker": "",
                    "task_id": tid,
                    "status": card_status(card),
                    "title": card_title(card),
                    "reason": "no running gateway available to take this orphan ready card",
                })
                continue
            actions.append({
                "op": "assign_orphan_to_running_gateway",
                "worker": target,
                "task_id": tid,
                "status": card_status(card),
                "title": card_title(card),
            })
            if target in reviewer_profile_set or target in SPECIALIST_PROFILES:
                reserved_specialists.add(target)
            else:
                reserved_workers.add(target)

    return {
        "schema": "neurogolf.dependency_serializer.v1",
        "ok": not unresolved,
        "lanes": lanes,
        "actions": actions,
        "unresolved": unresolved,
        "return_waiting": return_waiting,
        "action_count": len(actions),
        "unresolved_count": len(unresolved),
        "return_waiting_count": len(return_waiting),
    }


def _is_orphan_ready_card(card: dict[str, Any]) -> bool:
    """A ready+unassigned card that nothing else has claimed.

    Excludes:
        - return / repair / opt-return cards (handled by parent-lineage routing)
        - cards with a specialist-typed title (review / submit / backlog / worker)
          — those flow through other branches above
    """
    if card_assignee(card) or card_status(card) != "ready":
        return False
    if is_return_card(card):
        return False
    title = (card_title(card) or "").lower()
    # Skip cards that have a structured title; those have dedicated routing
    # in earlier branches. Only catch the truly orphan ones (titles that
    # don't match any of the known patterns).
    skip_prefixes = ("[review", "[submit", "[backlog", "[worker", "[repair")
    if any(title.startswith(p) for p in skip_prefixes):
        return False
    return True


def _pick_gateway_for_card(
    card: dict[str, Any],
    available_profiles: list[str],
    reserved_workers: set[str],
    reserved_specialists: set[str],
) -> str:
    """Pick a running gateway profile that doesn't already have an active lane.

    Preference order:
        1. A worker profile (workers 1-6) — they're built for ONNX-building.
        2. Default profile — generic fallback ("right now that's only
           default which is fine" — 2026-05-19 user policy).
    Specialist profiles (reviewer / submit-captain / etc.) are NOT picked
    here because they have specialized roles and dedicated routing above.
    """
    workers_in_pool = [
        p for p in available_profiles
        if p.startswith("neurogolf-worker-") and p not in reserved_workers
    ]
    if workers_in_pool:
        return workers_in_pool[0]
    if "default" in available_profiles:
        return "default"
    return ""


def running_gateway_profiles() -> list[str]:
    """Query ``hermes gateway list`` for currently-running gateway profiles.

    Returns the profile names that show as running (✓). On any error
    (subprocess failure, parse failure), returns an empty list — the
    fallback assignment path silently skips when no profiles are available,
    which matches "leave orphan cards alone if we can't confirm a target".
    """
    try:
        proc = subprocess.run(
            ["hermes", "gateway", "list"],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, FileNotFoundError):
        return []
    if proc.returncode != 0:
        return []
    out: list[str] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        # Format from `hermes gateway list`:
        #   ✓ default (current)        — PID 24572
        #   ✗ neurogolf-worker-1       — not running
        if not line or not line.startswith("✓"):
            continue
        # Strip the leading ✓ and split on whitespace; first token is the profile name.
        rest = line.lstrip("✓").strip()
        if not rest:
            continue
        # Profile name is the first whitespace-separated token; strip trailing
        # decorations like "(current)".
        name = rest.split()[0]
        if name:
            out.append(name)
    return out


def manual_claude_code_profiles() -> list[str]:
    """Profiles configured as ``none-claude-code-manual``.

    These are Pool B workers driven manually via a Claude Code ``/loop``
    window — they have no Hermes gateway daemon (by design), so
    ``hermes gateway list`` shows them as ✗. But the profile model being
    ``none-claude-code-manual`` is the strongest available signal that
    the operator INTENDS to drive them, so we include them in the
    available-profile pool for orphan-fallback routing. Picked AFTER
    running gateways (which are more immediate-reactive), but BEFORE
    ``default`` (so default genuinely stays last-resort).

    Returns [] on subprocess / parse failure — silently degrades to
    gateway-only pool.
    """
    try:
        proc = subprocess.run(
            ["hermes", "profile", "list"],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, FileNotFoundError):
        return []
    if proc.returncode != 0:
        return []
    out: list[str] = []
    for line in (proc.stdout or "").splitlines():
        if "none-claude-code-manual" not in line:
            continue
        # Line format: "  neurogolf-worker-7 none-claude-code-manual      stopped ..."
        # Take the first non-decoration token as the profile name.
        for token in line.split():
            cleaned = token.lstrip("◆ ").strip()
            if cleaned.startswith("neurogolf-") or cleaned == "default":
                out.append(cleaned)
                break
    return out


def available_profile_pool() -> list[str]:
    """Combine running gateways + manual Claude Code profiles into the
    pool that the orphan-fallback picker sees.

    Order matters — earlier entries win when ``_pick_gateway_for_card``
    iterates. Policy (2026-05-19):
        1. Running gateway profiles (probe ✓) — they're immediately
           dispatchable by the default gateway. Worker-N profiles among
           these are preferred over default in the picker.
        2. Manually-driven Pool B profiles (workers 7/8) — picked next
           when no running gateway worker is available. They DON'T have
           a daemon but their Claude Code ``/loop`` window picks them up.
        3. ``default`` — true last resort. Only fires if neither of
           the above pools has a worker-class profile.

    The picker function further preferences ``neurogolf-worker-*`` over
    ``default`` within whatever set it receives. So the net order is:

        running worker-N  >  manual worker-7/8  >  default
    """
    running = running_gateway_profiles()
    seen = set(running)
    manual = [p for p in manual_claude_code_profiles() if p not in seen]
    return running + manual


def _running_worker_pids(detail: dict[str, Any]) -> list[int]:
    pids: list[int] = []
    for run in detail.get("runs") or []:
        if not isinstance(run, dict):
            continue
        pid = run.get("worker_pid")
        if isinstance(pid, int) and pid > 0:
            pids.append(pid)
    for event in detail.get("events") or []:
        if not isinstance(event, dict) or event.get("kind") != "spawned":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        pid = payload.get("pid")
        if isinstance(pid, int) and pid > 0:
            pids.append(pid)
    return sorted(set(pids))


def _terminate_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        except OSError:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    text=True,
                    capture_output=True,
                    check=False,
                )


def apply_serialization_plan(client: KanbanClient, plan: dict[str, Any]) -> None:
    for action in plan.get("actions", []):
        task_id = str(action["task_id"])
        worker = str(action["worker"])
        if action.get("op") == "normalize_specialist_assignee":
            old_assignee = str(action.get("old_assignee") or "")
            client.comment(
                task_id,
                (
                    "Python dependency serializer repaired a noncanonical specialist assignee "
                    f"from {old_assignee or '(blank)'} to {worker}; Hermes profiles require the full machine profile name."
                ),
            )
            client.assign(task_id, worker)
        elif action.get("op") == "serialize_specialist_future_card":
            head_id = str(action["head_id"])
            old_assignee = str(action.get("old_assignee") or "")
            client.link(head_id, task_id)
            client.comment(
                task_id,
                (
                    "Python dependency serializer parked this extra specialist card behind "
                    f"{head_id} and unassigned it so {worker} has exactly one active lane. "
                    f"Previous assignee was {old_assignee or '(blank)'}."
                ),
            )
            client.assign(task_id, "none")
        elif action.get("op") in {"unassign_misassigned_specialist_return", "unassign_misassigned_worker_return"}:
            target_worker = str(action.get("target_worker") or "")
            if str(action.get("status") or "") == "running":
                detail = client.show(task_id)
                pids = _running_worker_pids(detail)
                _terminate_pids(pids)
                client.reclaim(
                    task_id,
                    (
                        "Python dependency serializer stopped this repair/optimization return "
                        f"because it was running under {worker}, not its deterministic origin lane."
                    ),
                )
            tail = (
                f" It will return to {target_worker} when that worker lane is free."
                if target_worker
                else " It stays unassigned until a deterministic origin worker is known."
            )
            client.comment(
                task_id,
                (
                    "Python dependency serializer removed this repair/optimization return from "
                    f"{worker}; return cards must stay on their deterministic origin worker lane."
                    f"{tail}"
                ),
            )
            client.assign(task_id, "none")
        elif action.get("op") == "serialize_future_card":
            head_id = str(action["head_id"])
            client.link(head_id, task_id)
            client.comment(
                task_id,
                (
                    "Python dependency serializer moved this future worker card behind "
                    f"{head_id} and unassigned it so {worker} keeps one active context lane."
                ),
            )
            client.assign(task_id, "none")
        elif action.get("op") == "park_future_return_card_assigned":
            head_id = str(action["head_id"])
            # Idempotency at the application layer: hermes emits a
            # `linked` event AND records a `commented` event every time
            # we call link()+comment(), even when the parent-child edge
            # already exists. Without this guard, the serializer floods
            # the card with hundreds of duplicate park comments every tick
            # (462 on a single card before the fix). Check the card's
            # existing parents and skip the link+comment if head_id is
            # already there.
            try:
                detail = client.show(task_id)
                existing_parents = detail.get("parents") or []
                if not isinstance(existing_parents, list):
                    existing_parents = []
            except Exception:
                # If `hermes kanban show` fails for any reason, fall
                # through to the original behavior (link + comment).
                # Worst case: we add one redundant comment.
                existing_parents = []
            if head_id in existing_parents:
                # Already parked; no-op. Don't emit a new comment.
                pass
            else:
                client.link(head_id, task_id)
                client.comment(
                    task_id,
                    (
                        "Python dependency serializer parked this repair / optimization-return "
                        f"card behind active lane {head_id} but KEPT {worker} as assignee — "
                        "repair-card context (failed candidate, base lineage, worker memory) "
                        "must stay with the original worker. Card becomes runnable when the "
                        "parent lane completes."
                    ),
                )
            # Intentionally NOT calling client.assign(task_id, "none").
        elif action.get("op") == "unassign_backlog_misclaim":
            status = str(action.get("status") or "")
            # If running, release the lock first so the BACKLOG bank card
            # is unclaimable until the Pool A foreman next runs `create_
            # worker_from_bank` against it (which goes through the normal
            # idempotency-key flow with a Pool A assignee).
            if status == "running":
                client.reclaim(
                    task_id,
                    (
                        "Python dependency serializer released BACKLOG misclaim — "
                        f"bank cards must not be held by {worker} (Pool B worker). "
                        "Pool A foreman owns conversion of bank cards into worker cards."
                    ),
                )
            client.comment(
                task_id,
                (
                    "Python dependency serializer released BACKLOG misclaim: this is a "
                    f"`[BACKLOG single task]` bank card, but it was assigned to {worker} "
                    "who is not in POOL_A_SHRINK_WORKERS. BACKLOG bank cards exist for "
                    "the Pool A foreman's `create_worker_from_bank` conversion only; "
                    "Pool B workers should not cold-start onto them. Returning the card "
                    "to the unassigned bank."
                ),
            )
            client.assign(task_id, "none")
        elif action.get("op") == "serialize_running_return_card":
            head_id = str(action["head_id"])
            detail = client.show(task_id)
            pids = _running_worker_pids(detail)
            _terminate_pids(pids)
            reason = (
                "Python dependency serializer stopped duplicate running return card "
                f"behind active lane {head_id}."
            )
            client.reclaim(task_id, reason)
            client.link(head_id, task_id)
            client.comment(
                task_id,
                (
                    f"{reason} It is unassigned so {worker} keeps one active context lane; "
                    "assign it back only after the parent lane completes."
                ),
            )
            client.assign(task_id, "none")
        elif action.get("op") == "replace_scratch_return_card":
            if str(action.get("status") or "") == "running":
                detail = client.show(task_id)
                pids = _running_worker_pids(detail)
                _terminate_pids(pids)
                client.reclaim(
                    task_id,
                    "Python dependency serializer is recreating this return card inside a project workspace.",
                )
            new_id = client.create(
                str(action["title"]),
                body=str(action["body"]),
                assignee=worker,
                workspace=str(action["workspace"]),
                priority=int(action.get("priority") or 0),
                created_by="neurogolf-python-control",
                idempotency_key=str(action["idempotency_key"]),
                max_runtime="6h",
            )
            client.comment(
                new_id,
                (
                    "Python dependency serializer recreated this worker return from "
                    f"{task_id}; original scratch workspace was not a valid NeuroGolf project lane."
                ),
            )
            client.complete(
                task_id,
                f"recreated as project-workspace return card {new_id}",
            )
        elif action.get("op") == "assign_return_card":
            client.comment(
                task_id,
                f"Python dependency serializer assigned this repair/optimization return to {worker}; one active lane per worker still enforced.",
            )
            client.assign(task_id, worker)
        elif action.get("op") == "assign_specialist_continuation":
            client.comment(
                task_id,
                (
                    "Python dependency serializer assigned this parked specialist continuation to "
                    f"{worker}; one active specialist lane is free now."
                ),
            )
            client.assign(task_id, worker)
        elif action.get("op") == "assign_orphan_to_running_gateway":
            client.comment(
                task_id,
                (
                    "Python dependency serializer assigned this orphan ready card to "
                    f"{worker} — the only matching running gateway. No parent-lineage "
                    "worker hint was found, so this is the fallback routing."
                ),
            )
            client.assign(task_id, worker)


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"actions={report['action_count']} unresolved={report['unresolved_count']} ok={report['ok']}",
    ]
    for action in report["actions"]:
        if action["op"] == "normalize_specialist_assignee":
            lines.append(
                f"- normalize {action['task_id']} {action.get('old_assignee') or '(blank)'} -> {action['worker']} ({action['status']})"
            )
        elif action["op"] == "serialize_specialist_future_card":
            lines.append(
                f"- serialize specialist {action['task_id']} behind {action['head_id']} for {action['worker']} ({action['status']})"
            )
        elif action["op"] == "unassign_misassigned_specialist_return":
            target = action.get("target_worker") or "(unknown origin)"
            lines.append(
                f"- unassign return {action['task_id']} from {action['worker']} toward {target} ({action['status']})"
            )
        elif action["op"] == "serialize_future_card":
            lines.append(
                f"- serialize {action['task_id']} behind {action['head_id']} for {action['worker']} ({action['status']})"
            )
        elif action["op"] == "park_future_return_card_assigned":
            lines.append(
                f"- park return {action['task_id']} behind {action['head_id']} for {action['worker']} (kept assigned, {action['status']})"
            )
        elif action["op"] == "serialize_running_return_card":
            lines.append(
                f"- reclaim+serialize running return {action['task_id']} behind {action['head_id']} for {action['worker']}"
            )
        elif action["op"] == "replace_scratch_return_card":
            lines.append(
                f"- recreate scratch return {action['task_id']} for {action['worker']} in {action.get('workspace')} ({action['status']})"
            )
        elif action["op"] == "assign_return_card":
            lines.append(f"- assign return {action['task_id']} to {action['worker']} ({action['status']})")
        elif action["op"] == "assign_specialist_continuation":
            lines.append(f"- assign specialist continuation {action['task_id']} to {action['worker']} ({action['status']})")
    for item in report["unresolved"]:
        lines.append(f"- unresolved {item['worker']} {item['task_id']} ({item['status']}): {item['reason']}")
    for item in report.get("return_waiting", []):
        lines.append(f"- waiting {item['worker']} {item['task_id']} ({item['status']}): {item['reason']}")
    if len(lines) == 1:
        lines.append("no serialization needed")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--kanban-json", type=Path, help="Saved hermes kanban list --json output")
    parser.add_argument("--apply", action="store_true", help="Mutate the Hermes board")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run output")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-actions", type=int, default=24)
    args = parser.parse_args()

    client = KanbanClient(args.repo_root, dry_run=not args.apply)
    if args.kanban_json:
        cards = json.loads(args.kanban_json.read_text(encoding="utf-8"))
    else:
        cards = client.load_cards()
    if not isinstance(cards, list):
        raise SystemExit("expected Kanban JSON list")

    # Probe live gateway state PLUS manual Claude Code profiles. The
    # combined pool (running gateways first, then manual Pool B workers)
    # is what the orphan-fallback picker consults. See
    # ``available_profile_pool`` docstring for the precedence policy.
    # On subprocess failure both probes return [] and the fallback
    # silently disables (safe).
    report = plan_serialization(
        [card for card in cards if isinstance(card, dict)],
        max_actions=args.max_actions,
        running_gateways=available_profile_pool(),
    )
    if args.apply:
        apply_serialization_plan(client, report)
        report["applied_actions"] = client.actions
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
