#!/usr/bin/env python3
"""Suggest NeuroGolf backlog targets from the durable numeric task cursor.

This is intentionally conservative: it does not create Kanban cards. The
backlog foreman can run it before card creation, then advance the cursor after
creating concrete cards so the next run does not keep rereading the same patch
of the spreadsheet.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_cron_snapshot import (
    TASK_BACKLOG_CURSOR,
    load_cards,
    read_json,
    read_task_rows,
    summarize_task_progress,
    task_sort_key,
)


def next_task_after(task_id: str) -> tuple[str, int]:
    rows = read_task_rows()
    task_ids = [row.get("task") or "" for row in rows if row.get("task")]
    if not task_ids:
        return "", 0
    task_ids = sorted(task_ids, key=task_sort_key)
    try:
        index = task_ids.index(task_id)
    except ValueError:
        return task_ids[0], 0
    next_index = (index + 1) % len(task_ids)
    wrapped = 1 if next_index == 0 else 0
    return task_ids[next_index], wrapped


def advance_cursor(task_id: str, reason: str) -> dict[str, object]:
    current = read_json(TASK_BACKLOG_CURSOR, {})
    next_task, wrapped = next_task_after(task_id)
    pass_count = int(current.get("pass_count", 0) or 0) + wrapped
    payload = {
        "schema": "neurogolf.task_backlog_cursor.v1",
        "next_task_id": next_task,
        "pass_count": pass_count,
        "last_advanced_from": task_id,
        "last_advanced_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M EST"),
        "last_reason": reason,
        "policy": "Scan task000..task400 in numeric order from next_task_id; high-EV priority can reorder only within the visible candidate window.",
    }
    TASK_BACKLOG_CURSOR.parent.mkdir(parents=True, exist_ok=True)
    TASK_BACKLOG_CURSOR.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=16, help="number of preview candidates to print")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--advance-to", metavar="taskNNN", help="advance cursor past a created/considered task")
    parser.add_argument("--advance-window", action="store_true", help="advance cursor past the current numeric scan window")
    parser.add_argument("--reason", default="backlog foreman considered/created cards")
    args = parser.parse_args()

    if args.advance_to:
        payload = advance_cursor(args.advance_to, args.reason)
        print(json.dumps(payload, indent=2) if args.json else f"advanced next_task_id={payload['next_task_id']} pass={payload['pass_count']}")
        return 0

    if args.advance_window:
        snapshot = summarize_task_progress(load_cards())
        window_end = str(snapshot.get("window_end_task_id") or snapshot.get("next_task_id") or "")
        if not window_end:
            raise SystemExit("No task window available to advance.")
        payload = advance_cursor(window_end, args.reason)
        output = {
            "advanced_from_window_start": snapshot.get("next_task_id"),
            "advanced_past": window_end,
            "next_task_id": payload["next_task_id"],
            "pass_count": payload["pass_count"],
            "considered": snapshot.get("considered_until_preview"),
            "skipped_counts": snapshot.get("skipped_counts", {}),
        }
        print(json.dumps(output, indent=2) if args.json else f"advanced past {window_end}; next_task_id={payload['next_task_id']} pass={payload['pass_count']}")
        return 0

    snapshot = summarize_task_progress(load_cards())
    snapshot["numeric_preview"] = snapshot["numeric_preview"][: args.limit]
    if args.json:
        print(json.dumps(snapshot, indent=2))
    else:
        print(f"Task cursor: next={snapshot['next_task_id']} pass={snapshot['pass_count']} rows={snapshot['rows']} window_end={snapshot['window_end_task_id']}")
        print(f"Sources: {snapshot['source_csv']} ; xlsx_exists={snapshot['source_xlsx_exists']}")
        for item in snapshot["numeric_preview"]:
            print(
                f"- {item['task']}: cost={item['current_cost']} points={item['current_points']} "
                f"status={item['status'] or '(blank)'} next={item['next_action'] or '(none)'}"
            )
        if snapshot.get("skipped_counts"):
            print(f"Skipped in window: {snapshot['skipped_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
