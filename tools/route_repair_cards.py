#!/usr/bin/env python3
"""Route post-submit repair cards back to their original worker.

The submit-captain creates `[repair taskNNN] ...` cards after a public-negative
Kaggle result. The card body includes the full context (failed candidate ID,
package SHA, base lineage, original worker profile) — but the `assignee`
field is left empty. The autoforeman then picks the card up as backlog and
routes it to whichever worker is idle, **losing the original worker's
context** (their per-profile memory, any partial scratch in their worktree,
their accumulated learning about this task family).

This script scans for such cards and assigns each one back to the original
worker named in the body. Idempotent — if a card is already assigned, it's
skipped.

Matched patterns in the body (regex, case-insensitive):
    Original worker card: t_xxx (neurogolf-worker-N)
    Original worker: neurogolf-worker-N
    route ... back to worker-N
    route ... back to neurogolf-worker-N

Run:
    python tools/route_repair_cards.py           # dry-run
    python tools/route_repair_cards.py --apply   # mutate
    python tools/route_repair_cards.py --apply --json   # cron-style output
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Patterns ordered by specificity. First match wins.
# Per the KanbanWorkerLoop playbook, Claude Code workers MUST include
# `"worker_owner": "neurogolf-worker-N"` in their handoff JSON — that line
# matches the `worker_owner` pattern below. This is the canonical signal
# when the kanban `task.assignee` field is empty (e.g. a card that was
# claimed before being assigned, which the kanban refuses to retroactively
# reassign while the lock is held).
ORIGINAL_WORKER_PATTERNS = (
    re.compile(r"\"worker_owner\"\s*:\s*\"(neurogolf-worker-\d+)\"", re.IGNORECASE),
    re.compile(r"worker_owner\s*[:=]\s*(neurogolf-worker-\d+)", re.IGNORECASE),
    re.compile(r"original\s+worker\s+card:\s*\S+\s*\((neurogolf-worker-\d+)\)", re.IGNORECASE),
    re.compile(r"original\s+worker:\s*(neurogolf-worker-\d+)", re.IGNORECASE),
    re.compile(r"route\s+\S+\s+(?:hidden-safety|investigation)?\s*back to\s+(neurogolf-worker-\d+)", re.IGNORECASE),
    re.compile(r"route\s+\S+\s+(?:hidden-safety|investigation)?\s*back to\s+worker[-\s]?(\d+)", re.IGNORECASE),
    re.compile(r"\bworker[-\s]?(\d+)\b", re.IGNORECASE),  # last-resort
)

REPAIR_TITLE_PATTERNS = (
    re.compile(r"^\[repair\s+task\d+\]", re.IGNORECASE),
    re.compile(r"^repair\s+task\d+", re.IGNORECASE),
    re.compile(r"^\[optimization[- ]return\]", re.IGNORECASE),
)


def run_hermes(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["hermes", "kanban", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )


def stdout_utf8(proc: subprocess.CompletedProcess[bytes]) -> str:
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def hermes_list() -> list[dict[str, Any]]:
    proc = run_hermes(["list", "--json"])
    if proc.returncode != 0:
        return []
    try:
        return [c for c in json.loads(stdout_utf8(proc)) if isinstance(c, dict)]
    except json.JSONDecodeError:
        return []


def extract_target_worker(body: str) -> str:
    """Return profile name or '' if no worker can be extracted."""
    for pat in ORIGINAL_WORKER_PATTERNS:
        m = pat.search(body)
        if not m:
            continue
        cap = m.group(1)
        if cap.isdigit():
            return f"neurogolf-worker-{cap}"
        return cap.lower()
    return ""


def is_repair_card(card: dict[str, Any]) -> bool:
    title = card.get("title") or ""
    for pat in REPAIR_TITLE_PATTERNS:
        if pat.match(title):
            return True
    return False


def parent_ids_of(card_id: str) -> list[str]:
    """Extract parent task IDs from `hermes kanban show <id>` output."""
    proc = run_hermes(["show", card_id])
    if proc.returncode != 0:
        return []
    show = stdout_utf8(proc)
    m = re.search(r"^\s*parents:\s*(.+?)$", show, re.MULTILINE)
    if not m:
        return []
    return [p.strip() for p in m.group(1).split(",") if p.strip().startswith("t_")]


def worker_assignee_from_parents(card_id: str, cards_by_id: dict[str, dict[str, Any]]) -> str:
    """Look at the card's parent cards; return the first worker-profile assignee.

    Two-stage lookup per parent:
      1. The parent's `task.assignee` field (works for normally-assigned cards).
      2. If empty, the parent card's full `hermes kanban show` text — which
         includes comments. Claude Code worker handoffs include a
         `"worker_owner": "neurogolf-worker-N"` line in their JSON handoff
         comment, so the worker identity is recoverable even when the kanban
         row's assignee is empty (e.g. cards that were claimed before being
         assigned, which Hermes refuses to retroactively reassign).
    """
    for pid in parent_ids_of(card_id):
        parent = cards_by_id.get(pid)
        if parent:
            a = (parent.get("assignee") or "").strip().lower()
            if a.startswith("neurogolf-worker-"):
                return a
        # Fall back to the parent's full show text — scan for worker_owner /
        # other patterns. This is one extra `hermes kanban show` per parent;
        # acceptable since route_repair runs every 2 min on a small set.
        proc = run_hermes(["show", pid])
        if proc.returncode != 0:
            continue
        worker = extract_target_worker(stdout_utf8(proc))
        if worker:
            return worker
    return ""


def find_misrouted_repair_cards() -> list[dict[str, Any]]:
    cards = hermes_list()
    cards_by_id = {c.get("id"): c for c in cards if c.get("id")}
    out: list[dict[str, Any]] = []
    for c in cards:
        if not is_repair_card(c):
            continue
        status = (c.get("status") or "").lower()
        # Skip cards that are already done/archived OR already assigned.
        if status in ("done", "archived"):
            continue
        if (c.get("assignee") or "").strip():
            continue
        body = c.get("body") or ""
        title = c.get("title") or ""
        target = extract_target_worker(body) or extract_target_worker(title)
        # Fallback: parent-card chain. submit-captain creates repairs with
        # `--parent t_<submit_id>` and `--parent t_<original_worker_id>`,
        # so the original worker's assignee is reachable.
        if not target:
            target = worker_assignee_from_parents(c.get("id"), cards_by_id)
        if not target:
            continue
        out.append(
            {
                "id": c.get("id"),
                "title": title[:80],
                "status": status,
                "target": target,
            }
        )
    return out


def apply_assignment(card_id: str, assignee: str) -> tuple[bool, str]:
    proc = run_hermes(["assign", card_id, assignee])
    return proc.returncode == 0, (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()[:200]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply assignments. Default is dry-run.")
    parser.add_argument("--json", action="store_true", help="JSON output for cron.")
    args = parser.parse_args(argv)

    matches = find_misrouted_repair_cards()
    applied: list[dict[str, Any]] = []
    for m in matches:
        result: dict[str, Any] = {**m, "applied": False, "msg": ""}
        if args.apply:
            ok, msg = apply_assignment(m["id"], m["target"])
            result["applied"] = ok
            result["msg"] = msg
        applied.append(result)

    if args.json:
        payload = {
            "schema": "neurogolf.route_repair_cards.v1",
            "action_count": sum(1 for r in applied if r["applied"]),
            "candidates": applied,
        }
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
    else:
        if not matches:
            print("No unassigned repair cards found.")
            return 0
        print(f"{'card':12}  {'->worker':24}  {'status':8}  title")
        print("-" * 100)
        for r in applied:
            mark = "OK" if r["applied"] else ("would-assign" if not args.apply else "FAIL")
            print(f"{r['id']:12}  {r['target']:24}  {mark:12}  {r['title']}")
            if r.get("msg"):
                print(f"    {r['msg']}")
        print(f"\nmode: {'APPLY' if args.apply else 'DRY-RUN'}  matched {len(matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
