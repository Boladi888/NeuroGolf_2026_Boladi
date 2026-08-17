#!/usr/bin/env python3
"""Regenerate CURRENT_STATE.md from authoritative sources.

The file is kept to a strict ~60-line schema: current best + canonical SHA +
8 worker assignments (one line each) + active blockers + last 5 submits.
Everything else belongs in Logs/sessions/.

Run with --check to fail when the file would change (useful in pre-commit).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "submissions" / "current_best_manifest.json"
PACKAGE = REPO_ROOT / "submissions" / "submission.zip"
LEDGER = REPO_ROOT / "Logs" / "submissions_ledger.csv"
SUBMIT_CREDIT = REPO_ROOT / ".hermes" / "board" / "submit_credit_state.json"
CURRENT_STATE = REPO_ROOT / "CURRENT_STATE.md"

WORKERS = [f"neurogolf-worker-{i}" for i in range(1, 9)]
ACTIVE_STATUSES = {"running", "ready", "todo", "blocked"}
TASK_ID_PATTERN = re.compile(r"task\s*0*(\d+)", re.IGNORECASE)


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def hermes_cards() -> list[dict[str, Any]]:
    # Force UTF-8 decode — hermes kanban list --json can include emoji or other
    # extended characters in card bodies which crash the Windows cp1252 default.
    proc = subprocess.run(
        ["hermes", "kanban", "list", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return []
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    return [c for c in data if isinstance(c, dict)]


def extract_task_id(card: dict[str, Any]) -> str:
    text = " ".join(str(card.get(k, "")) for k in ("title", "body", "result"))
    match = TASK_ID_PATTERN.search(text)
    return f"task{int(match.group(1)):03d}" if match else ""


def worker_assignment_line(worker: str, cards: list[dict[str, Any]]) -> str:
    owned = [c for c in cards if c.get("assignee") == worker and (c.get("status") or "").lower() in ACTIVE_STATUSES]
    if not owned:
        return f"- {worker:24} (idle)"
    owned.sort(key=lambda c: c.get("started_at") or c.get("created_at") or "")
    card = owned[0]
    task = extract_task_id(card) or "task???"
    status = (card.get("status") or "?").lower()
    cid = card.get("id") or "?"
    title = (card.get("title") or "")[:60]
    return f"- {worker:24} {task:8} [{status:7}] {cid:12} {title}"


def recent_submits(n: int = 5) -> list[tuple[str, str, str]]:
    if not LEDGER.exists():
        return []
    rows: list[dict[str, str]] = []
    with LEDGER.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows = rows[-n:]
    out: list[tuple[str, str, str]] = []
    for row in rows:
        date = row.get("date", "")
        kind = row.get("kind", "")[:50]
        verdict = row.get("verdict", "")[:60]
        out.append((date, kind, verdict))
    return out


def active_blockers(cards: list[dict[str, Any]]) -> list[str]:
    blocked = [c for c in cards if (c.get("status") or "").lower() == "blocked"]
    out: list[str] = []
    for c in blocked:
        title = (c.get("title") or "")[:70]
        assignee = c.get("assignee") or "-"
        cid = c.get("id") or "?"
        out.append(f"- {cid:12} {assignee:24} {title}")
    return out


def render(now: str, cards: list[dict[str, Any]]) -> str:
    manifest = read_json(MANIFEST, {}) or {}
    best = manifest.get("current_best") or {}
    score = best.get("score", "?")
    candidate = best.get("candidate_id", "?")
    sha = (best.get("package_sha256") or "")[:16]
    credit = read_json(SUBMIT_CREDIT, {}) or {}
    credits = credit.get("credits") or {}
    available = credits.get("available", "?")
    daily_used = credits.get("daily_used", "?")

    lines: list[str] = []
    lines.append("# CURRENT_STATE.md — auto-generated snapshot")
    lines.append("")
    lines.append(f"> Generated: {now}. Source: `tools/update_current_state.py`. DO NOT EDIT BY HAND.")
    lines.append("> Narrative history → `Logs/sessions/`. Lessons → `Logs/lessons.md`.")
    lines.append("")
    lines.append("## Best")
    lines.append("")
    lines.append(f"- Kaggle LB: **{score}**  candidate `{candidate}`")
    lines.append(f"- Package: `submissions/submission.zip`  sha256 `{sha}...`")
    lines.append(f"- Submit credits: {available} available, {daily_used} used today")
    lines.append("")
    lines.append("## Workers (one line each)")
    lines.append("")
    for worker in WORKERS:
        lines.append(worker_assignment_line(worker, cards))
    lines.append("")
    blockers = active_blockers(cards)
    lines.append("## Blocked cards")
    lines.append("")
    if not blockers:
        lines.append("- (none)")
    else:
        lines.extend(blockers[:12])
    lines.append("")
    lines.append("## Last 5 submits")
    lines.append("")
    for date, kind, verdict in recent_submits(5):
        lines.append(f"- {date}  {kind}  →  {verdict}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit non-zero if file would change")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = hermes_cards()
    content = render(now, cards)
    if args.stdout:
        sys.stdout.write(content)
        return 0
    existing = CURRENT_STATE.read_text(encoding="utf-8") if CURRENT_STATE.exists() else ""
    if existing == content:
        return 0
    if args.check:
        sys.stderr.write("CURRENT_STATE.md is out of date; run tools/update_current_state.py\n")
        return 1
    CURRENT_STATE.write_text(content, encoding="utf-8")
    line_count = content.count("\n")
    print(f"Wrote CURRENT_STATE.md ({line_count} lines).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
