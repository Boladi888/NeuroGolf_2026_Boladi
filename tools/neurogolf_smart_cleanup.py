#!/usr/bin/env python3
"""Shared cheap cleanup pass for NeuroGolf cron control jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_blocked_mechanical_cleanup import apply_blocked_cleanup_plan, current_best_is_synced, plan_blocked_cleanup
from tools.neurogolf_cron_snapshot import build_snapshot
from tools.neurogolf_dependency_serializer import apply_serialization_plan, plan_serialization
from tools.neurogolf_kanban_python import KanbanClient
from tools.neurogolf_review_packet_builder import enrich_blocked_cards
from tools.neurogolf_submit_trigger import _active_submit_card_ids, apply_submit_trigger_plan, plan_waiting_submit_cleanup


def _compact_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        compact.append(
            {
                key: action.get(key)
                for key in (
                    "op",
                    "task_id",
                    "title",
                    "assignee",
                    "worker",
                    "target_worker",
                    "reviewer_task_id",
                    "idempotency_key",
                    "reason",
                    "result",
                )
                if action.get(key) not in (None, "")
            }
        )
    return compact


def run_smart_cleanup(
    *,
    repo_root: Path = REPO_ROOT,
    apply: bool = False,
    include_serialization: bool = True,
    include_blocked: bool = True,
    include_submit_lane: bool = True,
) -> dict[str, Any]:
    client = KanbanClient(repo_root, dry_run=not apply)
    sections: list[dict[str, Any]] = []

    if include_serialization:
        cards = client.load_cards()
        serialization = plan_serialization(cards)
        if apply and serialization.get("actions"):
            apply_serialization_plan(client, serialization)
        sections.append(
            {
                "name": "serialization",
                "action_count": len(serialization.get("actions") or []),
                "unresolved_count": serialization.get("unresolved_count", 0),
                "actions": serialization.get("actions") or [],
            }
        )

    if include_blocked:
        cards = client.load_cards()
        cards = enrich_blocked_cards(client, cards)
        blocked = plan_blocked_cleanup(cards, best_synced=current_best_is_synced())
        if apply and blocked.get("actions"):
            apply_blocked_cleanup_plan(client, blocked)
        sections.append(
            {
                "name": "blocked",
                "action_count": len(blocked.get("actions") or []),
                "manual_count": blocked.get("manual_count", 0),
                "reviewer_active": blocked.get("reviewer_active") or [],
                "waiting_review_sources": blocked.get("waiting_review_sources") or [],
                "actions": blocked.get("actions") or [],
            }
        )

    if include_submit_lane:
        submit_cleanup_actions: list[dict[str, Any]] = []
        snapshot = build_snapshot("submit")
        for task_id in _active_submit_card_ids(snapshot):
            detail = client.show(task_id)
            action = plan_waiting_submit_cleanup(detail, snapshot=snapshot)
            if action:
                submit_cleanup_actions.append(action)
        if apply and submit_cleanup_actions:
            apply_submit_trigger_plan(client, {"actions": submit_cleanup_actions})
        sections.append(
            {
                "name": "submit_lane",
                "action_count": len(submit_cleanup_actions),
                "actions": submit_cleanup_actions,
            }
        )

    actions = _compact_actions(client.actions)
    return {
        "schema": "neurogolf.smart_cleanup.v1",
        "applied": apply,
        "sections": sections,
        "action_count": len(actions),
        "actions": actions,
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [f"smart_cleanup actions={report.get('action_count', 0)} applied={report.get('applied')}"]
    for section in report.get("sections") or []:
        lines.append(
            f"- {section.get('name')}: actions={section.get('action_count', 0)}"
            + (f" unresolved={section.get('unresolved_count')}" if "unresolved_count" in section else "")
            + (f" manual={section.get('manual_count')}" if "manual_count" in section else "")
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-serialization", action="store_true")
    parser.add_argument("--no-blocked", action="store_true")
    parser.add_argument("--no-submit-lane", action="store_true")
    args = parser.parse_args()

    report = run_smart_cleanup(
        repo_root=args.repo_root,
        apply=args.apply,
        include_serialization=not args.no_serialization,
        include_blocked=not args.no_blocked,
        include_submit_lane=not args.no_submit_lane,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
