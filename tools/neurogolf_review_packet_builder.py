#!/usr/bin/env python3
"""Build one deterministic reviewer handoff card from blocked worker cards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_board_policy import ACTIVE_CARD_STATUSES
from tools.neurogolf_control_config import (
    primary_reviewer_profile,
    review_packet_limit,
    reviewer_max_active_per_profile,
    reviewer_profiles,
)
from tools.neurogolf_kanban_python import (
    KanbanClient,
    as_text,
    card_assignee,
    card_id,
    card_status,
    card_text,
    card_title,
    sort_cards,
)

REVIEWER_PROFILE = primary_reviewer_profile()
WORKER_RE = re.compile(r"^neurogolf-worker-\d+$")
TASK_RE = re.compile(r"task\d{3}")
REVIEW_MARKERS = (
    "review-required",
    "review required",
    "needs reviewer",
    "needs-review",
    "send to reviewer",
    "reviewer requested",
)
HANDOFF_MARKERS = (
    "review handoff moved to",
    "python review packet",
)


def task_ids_from_text(value: str) -> list[str]:
    return sorted(dict.fromkeys(TASK_RE.findall(value)))


def card_needs_review(card: dict[str, Any]) -> bool:
    if card_status(card) != "blocked":
        return False
    if not WORKER_RE.match(card_assignee(card)):
        return False
    text = card_text(card).lower()
    if any(marker in text for marker in HANDOFF_MARKERS):
        return False
    return any(marker in text for marker in REVIEW_MARKERS)


def active_reviewer_cards(
    cards: list[dict[str, Any]],
    *,
    profiles: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    profiles = profiles or reviewer_profiles()
    return [
        card
        for card in cards
        if card_assignee(card) in profiles and card_status(card) in ACTIVE_CARD_STATUSES
    ]


def reviewer_loads(
    cards: list[dict[str, Any]],
    *,
    profiles: tuple[str, ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    profiles = profiles or reviewer_profiles()
    loads: dict[str, list[dict[str, Any]]] = {profile: [] for profile in profiles}
    for card in active_reviewer_cards(cards, profiles=profiles):
        loads.setdefault(card_assignee(card), []).append(card)
    for profile in loads:
        loads[profile] = sort_cards(loads[profile])
    return loads


def choose_reviewer_profile(
    cards: list[dict[str, Any]],
    *,
    profiles: tuple[str, ...] | None = None,
    max_active_per_profile: int | None = None,
) -> str:
    profiles = profiles or reviewer_profiles()
    max_active = max_active_per_profile or reviewer_max_active_per_profile()
    loads = reviewer_loads(cards, profiles=profiles)
    candidates = [profile for profile in profiles if len(loads.get(profile, [])) < max_active]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda profile: (len(loads.get(profile, [])), profiles.index(profile)))[0]


def _source_digest(cards: list[dict[str, Any]]) -> str:
    source_ids = sorted(card_id(card) for card in cards)
    return hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()[:12]


def build_reviewer_body(source_cards: list[dict[str, Any]]) -> str:
    lines = [
        "PYTHON REVIEW PACKET",
        "",
        "Purpose: verify local delta evidence from blocked workers and decide whether each candidate should enter submit_queue.json.",
        "",
        "Rules:",
        "- Do not edit ONNX files.",
        "- Do not submit to Kaggle.",
        "- Verify against the worker's original valid base; keep minimal-rebase policy intact.",
        "- If approved, write/update the candidate registry and submit queue with package path, sha, expected gain, changed_task_ids, lineage, and origin worker.",
        "- Do not create loose Kanban submit cards. If an exceptional card is unavoidable, the only spawnable assignee is `neurogolf-submit-captain`, never `submit-captain`.",
        "- If weak/missing gain, return the candidate to the same origin worker with concrete next optimization notes.",
        "",
        "Source cards:",
    ]
    for card in source_cards:
        text = card_text(card)
        task_ids = ",".join(task_ids_from_text(text)) or "(unknown)"
        snippet = " ".join(text.split())[:420]
        lines.append(
            f"- {card_id(card)} | {card_assignee(card)} | tasks={task_ids} | {card_title(card)}"
        )
        if snippet:
            lines.append(f"  evidence: {snippet}")
    return "\n".join(lines)


def plan_review_packet(
    cards: list[dict[str, Any]],
    *,
    limit: int | None = None,
    allow_when_reviewer_active: bool = False,
    profiles: tuple[str, ...] | None = None,
    max_active_per_profile: int | None = None,
) -> dict[str, Any]:
    profiles = profiles or reviewer_profiles()
    limit = limit or review_packet_limit()
    max_active = max_active_per_profile or reviewer_max_active_per_profile()
    reviewer_active = sort_cards(active_reviewer_cards(cards, profiles=profiles))
    reviewer_load_report = {
        profile: [card_id(card) for card in reviewer_loads(cards, profiles=profiles).get(profile, [])]
        for profile in profiles
    }
    assignee = choose_reviewer_profile(cards, profiles=profiles, max_active_per_profile=max_active)
    if reviewer_active and not assignee and not allow_when_reviewer_active:
        return {
            "schema": "neurogolf.review_packet_builder.v1",
            "ok": True,
            "actions": [],
            "source_cards": [],
            "reviewer_active": [card_id(card) for card in reviewer_active],
            "reviewer_load": reviewer_load_report,
            "skip_reason": "all reviewer lanes busy",
        }
    if not assignee:
        assignee = profiles[0] if profiles else REVIEWER_PROFILE

    sources = sort_cards([card for card in cards if card_needs_review(card)])[:limit]
    if not sources:
        return {
            "schema": "neurogolf.review_packet_builder.v1",
            "ok": True,
            "actions": [],
            "source_cards": [],
            "reviewer_active": [card_id(card) for card in reviewer_active],
            "reviewer_load": reviewer_load_report,
            "skip_reason": "no blocked worker review-required cards",
        }

    digest = _source_digest(sources)
    source_ids = [card_id(card) for card in sources]
    return {
        "schema": "neurogolf.review_packet_builder.v1",
        "ok": True,
        "reviewer_active": [card_id(card) for card in reviewer_active],
        "source_cards": source_ids,
        "actions": [
            {
                "op": "create_reviewer_packet",
                "title": f"[review packet] blocked local-delta candidates {digest}",
                "assignee": assignee,
                "priority": 85,
                "idempotency_key": f"ng-review-packet-{digest}",
                "source_cards": source_ids,
                "body": build_reviewer_body(sources),
            }
        ],
        "reviewer_load": reviewer_load_report,
    }


def apply_review_packet_plan(client: KanbanClient, plan: dict[str, Any]) -> None:
    for action in plan.get("actions", []):
        if action.get("op") != "create_reviewer_packet":
            continue
        reviewer_id = client.create(
            str(action["title"]),
            body=str(action["body"]),
            assignee=str(action["assignee"]),
            workspace="worktree",
            priority=int(action["priority"]),
            created_by="neurogolf-review-packet-builder",
            idempotency_key=str(action["idempotency_key"]),
            max_runtime="2h",
        )
        for source_id in action.get("source_cards", []):
            client.comment(
                str(source_id),
                f"Python review packet builder handed this blocked candidate to {reviewer_id}.",
            )
            client.complete(
                str(source_id),
                f"review handoff moved to {reviewer_id}; source card closed to free worker lane",
            )


def enrich_blocked_cards(client: KanbanClient, cards: list[dict[str, Any]], *, limit: int = 32) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    count = 0
    for card in cards:
        if card_status(card) != "blocked":
            enriched.append(card)
            continue
        if count >= limit:
            enriched.append(card)
            continue
        count += 1
        try:
            shown = client.show(card_id(card))
        except SystemExit:
            enriched.append(card)
            continue
        merged = dict(card)
        merged.update({key: value for key, value in shown.items() if value not in (None, "")})
        # The detailed show payload may keep comments/events under nested lists.
        extra = []
        for key in ("comments", "events", "runs"):
            value = shown.get(key)
            if isinstance(value, list):
                extra.extend(as_text(item) for item in value[:6])
        try:
            log_text = client.log(card_id(card))
        except (AttributeError, SystemExit):
            log_text = ""
        if log_text.strip():
            extra.append(log_text)
        if extra:
            merged["latest_summary"] = "\n".join(extra)
        enriched.append(merged)
    return enriched


def render_text(report: dict[str, Any]) -> str:
    if not report.get("actions"):
        return f"no reviewer packet created: {report.get('skip_reason') or 'no action'}"
    action = report["actions"][0]
    return (
        f"create reviewer packet {action['idempotency_key']} for "
        f"{len(action['source_cards'])} blocked source card(s): {','.join(action['source_cards'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--kanban-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-enrich", action="store_true")
    parser.add_argument("--allow-when-reviewer-active", action="store_true")
    args = parser.parse_args()

    client = KanbanClient(args.repo_root, dry_run=not args.apply)
    if args.kanban_json:
        cards = json.loads(args.kanban_json.read_text(encoding="utf-8"))
    else:
        cards = client.load_cards()
        if not args.no_enrich:
            cards = enrich_blocked_cards(client, cards)
    if not isinstance(cards, list):
        raise SystemExit("expected Kanban JSON list")

    report = plan_review_packet(
        [card for card in cards if isinstance(card, dict)],
        limit=args.limit,
        allow_when_reviewer_active=args.allow_when_reviewer_active,
    )
    if args.apply:
        apply_review_packet_plan(client, report)
        report["applied_actions"] = client.actions
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
