#!/usr/bin/env python3
"""Detect orphan candidates: approved in candidate_registry.json with
``submit_queue_status: ready`` but missing from ``submit_queue.json::queue``.

Discovered 2026-05-18: 6 candidates (+4.42 expected pts total, including
task388 at +3.27 pts) were sitting in the registry as reviewer-approved
but never made it into the submit_queue. The submit-captain only reads
the queue, so these candidates can't be submitted to Kaggle — they're
silently lost.

Root cause is somewhere in the review-packet-builder / approval flow
(transactional gap between registry write and queue write). This cron
catches the gap by comparing the two files every tick.

Output: ``.hermes/board/orphan_candidates.json`` containing the orphan
list with EV, package info, and risk flags. ``--apply`` writes the file;
default is dry-run.

When emitted in JSON cron mode, the cron output also surfaces an
``orphan_count`` and ``total_orphan_ev`` so the daily digest can include
a one-line summary.

Usage:
    python tools/neurogolf_orphan_candidates.py            # dry-run, text
    python tools/neurogolf_orphan_candidates.py --apply
    python tools/neurogolf_orphan_candidates.py --apply --json   # cron
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = REPO_ROOT / ".hermes" / "board"
REGISTRY_PATH = BOARD_DIR / "candidate_registry.json"
SUBMIT_QUEUE_PATH = BOARD_DIR / "submit_queue.json"
OUTPUT_PATH = BOARD_DIR / "orphan_candidates.json"

SCHEMA_VERSION = "neurogolf.orphan_candidates.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _ev_for_entry(entry: dict[str, Any]) -> float:
    """Same priority chain the submit-captain uses."""
    for key in ("priority_score", "expected_point_gain", "expected_gain", "point_delta"):
        value = entry.get(key)
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f != 0:
            return f
    return 0.0


def find_orphans(
    registry_doc: dict[str, Any] | None,
    queue_doc: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return orphan candidate summaries — registry entries with status
    ``ready`` that have no matching ``candidate_id`` in the submit_queue.
    """
    registry = registry_doc if isinstance(registry_doc, dict) else {}
    queue = queue_doc if isinstance(queue_doc, dict) else {}
    reg_entries = registry.get("entries") or []
    queue_entries = queue.get("queue") or []
    queue_cids = {
        e.get("candidate_id")
        for e in queue_entries
        if isinstance(e, dict) and e.get("candidate_id")
    }
    orphans: list[dict[str, Any]] = []
    for entry in reg_entries:
        if not isinstance(entry, dict):
            continue
        status = entry.get("submit_queue_status") or entry.get("status")
        if status != "ready":
            continue
        cid = entry.get("candidate_id")
        if not cid or cid in queue_cids:
            continue
        ap = entry.get("approval_packet") or {}
        # Package path can live at top level or nested in approval_packet.
        pkg_path = entry.get("package_path") or ap.get("package_path") or ""
        pkg_sha = entry.get("package_sha256") or ap.get("package_sha256") or ""
        orphans.append({
            "candidate_id": cid,
            "task_id": entry.get("task_id"),
            "worker_owner": entry.get("worker_owner"),
            "expected_point_gain": _ev_for_entry(entry),
            "package_path": pkg_path,
            "package_sha256": pkg_sha,
            "package_in_staged_candidates": "/staged_candidates/" in (pkg_path or "").replace("\\", "/") or "/staged_submits/" in (pkg_path or "").replace("\\", "/"),
            "package_in_worktree": "/.worktrees/" in (pkg_path or "").replace("\\", "/"),
            "lineage_base_candidate_id": entry.get("lineage_base_candidate_id"),
            "risk_flags": list(entry.get("risk_flags") or []),
            "reviewer_disposition": entry.get("reviewer_disposition"),
            "approval_mode": entry.get("approval_mode"),
            "reviewer_decided_at": entry.get("reviewer_decided_at"),
        })
    # Sort highest EV first so the daily digest leads with the big ones.
    orphans.sort(key=lambda o: -o["expected_point_gain"])
    return orphans


def build_report(
    registry_doc: dict[str, Any] | None,
    queue_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    orphans = find_orphans(registry_doc, queue_doc)
    total_ev = round(sum(o["expected_point_gain"] for o in orphans), 6)
    top_ev = max((o["expected_point_gain"] for o in orphans), default=0.0)
    return {
        "schema": SCHEMA_VERSION,
        "computed_at": _utc_now(),
        "orphan_count": len(orphans),
        "total_orphan_ev": total_ev,
        "top_orphan_ev": top_ev,
        "orphans": orphans,
    }


def write_report(report: dict[str, Any], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the orphan report file.")
    parser.add_argument("--json", action="store_true", help="One-line JSON summary for cron.")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--queue", type=Path, default=SUBMIT_QUEUE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build_report(
        registry_doc=_read_json(args.registry),
        queue_doc=_read_json(args.queue),
    )

    if args.apply:
        write_report(report, args.output)

    if args.json:
        # Compact summary so cron output stays small (full list on disk).
        summary = {
            "schema": SCHEMA_VERSION,
            "applied": args.apply,
            "computed_at": report["computed_at"],
            "orphan_count": report["orphan_count"],
            "total_orphan_ev": report["total_orphan_ev"],
            "top_orphan_ev": report["top_orphan_ev"],
            "top_3_task_ids": [o["task_id"] for o in report["orphans"][:3]],
        }
        json.dump(summary, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(f"orphan_count={report['orphan_count']}  total_ev=+{report['total_orphan_ev']}  top_ev=+{report['top_orphan_ev']}")
        for o in report["orphans"]:
            risk_str = ",".join(o["risk_flags"][:3])
            loc = "staged" if o["package_in_staged_candidates"] else ("worktree" if o["package_in_worktree"] else "other")
            print(f"  +{o['expected_point_gain']:6.3f}  {o['task_id']:10}  pkg={loc}  {(o['candidate_id'] or '')[:55]}  flags=[{risk_str}]")
        print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
