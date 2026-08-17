#!/usr/bin/env python3
"""Audit NeuroGolf Kanban queue assignment/serialization safety.

This helper is read-only by default. It inspects `hermes kanban list --json`
or a saved JSON export and reports the exact failure modes Bovath flagged:
- runnable backlog cards that still say "UNASSIGNED BACKLOG CARD"
- more than one running/blocked context-holder per worker
- ready cards that could dispatch while a worker is already running/blocked
- dependency-gated or unassigned queues, because assigned todo cards can fan out
  into runnable work as soon as their parents clear
- idle Alpha queue workers that could receive exactly one concrete backlog head

`--strict` is intentionally narrow: it exits non-zero only when the live queue
needs reconciliation/serialization before more queue-worker dispatch should
happen. It is meant to stop assign-queue / dispatch bursts until conflicts are
resolved; it is not a blanket ban on every unrelated singleton specialist
routing action unless the caller decides the queue failure creates real dispatch
risk for that action.

Examples:
  .venv/Scripts/python tools/neurogolf_queue_audit.py
  .venv/Scripts/python tools/neurogolf_queue_audit.py --json
  hermes kanban list --json > /tmp/board.json
  .venv/Scripts/python tools/neurogolf_queue_audit.py --kanban-json /tmp/board.json --strict
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tools.neurogolf_board_policy import audit_worker_queue
    from tools.neurogolf_paths import resolve_canonical_root
except ModuleNotFoundError:  # pragma: no cover
    from neurogolf_board_policy import audit_worker_queue
    from neurogolf_paths import resolve_canonical_root

DEFAULT_REPO_ROOT = resolve_canonical_root(Path(__file__).resolve())


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def load_cards(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.kanban_json:
        return json.loads(args.kanban_json.read_text(encoding="utf-8"))
    proc = run(["hermes", "kanban", "list", "--json"], args.repo_root.resolve())
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)
    data = json.loads(proc.stdout)
    if not isinstance(data, list):
        raise SystemExit("expected list from hermes kanban list --json")
    return data


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"ok: {report['ok']}")
    lines.append(f"assigned_ready_backlog_count: {report['pending_assignment_backlog_count']}")
    lines.append(f"unassigned_backlog_bank_count: {report['unassigned_backlog_bank_count']}")
    lines.append(f"idle_queue_workers: {', '.join(report['idle_queue_workers']) or '(none)'}")
    lines.append(f"lane_free_workers: {', '.join(report['lane_free_workers']) or '(none)'}")
    lines.append("")

    lines.append("active_by_assignee:")
    for assignee, card_ids in report["active_by_assignee"].items():
        lines.append(f"  - {assignee}: {', '.join(card_ids)}")
    if not report["active_by_assignee"]:
        lines.append("  (none)")
    lines.append("")

    lines.append("lane_context_violations:")
    if report["lane_context_violations"]:
        for item in report["lane_context_violations"]:
            lines.append(
                f"  - {item['assignee']}: lane_context_count={item['lane_context_count']}"
                f" statuses={','.join(item['statuses'])} ids={','.join(item['card_ids'])}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("dispatch_risk_violations:")
    if report["dispatch_risk_violations"]:
        for item in report["dispatch_risk_violations"]:
            lines.append(
                f"  - {item['assignee']}: ready_count={item['ready_count']}"
                f" statuses={','.join(item['statuses'])} ids={','.join(item['card_ids'])}"
                f" reasons={'; '.join(item['reasons'])}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("legacy_active_violations:")
    if report["active_violations"]:
        for item in report["active_violations"]:
            lines.append(
                f"  - {item['assignee']}: active_count={item['active_count']}"
                f" statuses={','.join(item['statuses'])} ids={','.join(item['card_ids'])}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("assigned_ready_backlog:")
    if report["pending_assignment_backlog"]:
        for item in report["pending_assignment_backlog"]:
            lines.append(
                f"  - {item['task_id']} | {item['assignee']} | {item['status']} | {item['title']}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("parked_by_assignee:")
    for assignee, card_ids in report["parked_by_assignee"].items():
        lines.append(f"  - {assignee}: {', '.join(card_ids)}")
    if not report["parked_by_assignee"]:
        lines.append("  (none)")
    lines.append("")

    lines.append("assignment_plan:")
    if report["assignment_plan"]:
        for item in report["assignment_plan"]:
            lines.append(
                f"  - assign {item['task_id']} to {item['assignee']}"
                f" using workspace {item['workspace_path']}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("low_context_pending:")
    if report["low_context_pending"]:
        for item in report["low_context_pending"]:
            lines.append(f"  - {item['task_id']} | {item['assignee']} | {item['title']}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--kanban-json", type=Path, default=None, help="Optional saved output from hermes kanban list --json")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when queue-worker assignment/serialization violations require reconciliation before more queue dispatch")
    args = parser.parse_args()

    cards = load_cards(args)
    report = audit_worker_queue(cards)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))

    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
