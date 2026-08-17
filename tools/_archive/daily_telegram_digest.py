#!/usr/bin/env python3
"""Once-a-day NeuroGolf digest sized for Telegram.

Output is intentionally short (under ~500 chars). Just the numbers I'd want
to glance at on a phone, no event log. Run via Hermes cron --no-agent and
deliver to telegram.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "submissions" / "current_best_manifest.json"
CREDIT = REPO_ROOT / ".hermes" / "board" / "submit_credit_state.json"
LEDGER = REPO_ROOT / "Logs" / "submissions_ledger.csv"
CANDIDATES = REPO_ROOT / ".hermes" / "board" / "candidate_registry.json"


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def hermes_cards() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["hermes", "kanban", "list", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return []
    try:
        return [c for c in json.loads(proc.stdout.decode("utf-8", errors="replace")) if isinstance(c, dict)]
    except json.JSONDecodeError:
        return []


def hermes_diag_count() -> int:
    proc = subprocess.run(
        ["hermes", "kanban", "diag", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return 0
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
        return len(data) if isinstance(data, list) else 0
    except json.JSONDecodeError:
        return 0


def submits_today() -> tuple[int, list[tuple[str, str]]]:
    """(count_today, last 3 (date, kind+verdict) entries)."""
    if not LEDGER.exists():
        return 0, []
    today = datetime.now().strftime("%Y-%m-%d")
    count = 0
    recent: list[tuple[str, str]] = []
    with LEDGER.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")
            if d == today:
                count += 1
            kind = row.get("kind", "")[:25]
            verdict = row.get("verdict", "")[:35]
            recent.append((d, f"{kind} ->{verdict}"))
    return count, recent[-3:]


def top_pending_gains(n: int = 3) -> list[tuple[str, float]]:
    doc = read_json(CANDIDATES, {}) or {}
    entries = doc.get("entries") or []
    out: list[tuple[str, float]] = []
    for e in entries:
        st = (e.get("submit_queue_status") or "").lower()
        if st not in ("ready", "approved", "pending"):
            continue
        cid = e.get("candidate_id") or ""
        raw = e.get("expected_point_gain")
        try:
            gain = float(raw) if raw not in (None, "") else 0.0
        except (TypeError, ValueError):
            gain = 0.0
        if cid and gain > 0:
            out.append((cid, gain))
    out.sort(key=lambda x: -x[1])
    return out[:n]


def main() -> int:
    # Force UTF-8 stdout. The launcher / cron pipe goes through Hermes which
    # handles UTF-8 fine; only the local Windows cp1252 console crashes on
    # arrows/emoji. Reconfigure if available (Python 3.7+).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    now = datetime.now()
    cards = hermes_cards()
    by_status = Counter(c.get("status") for c in cards)
    workers_active = sum(
        1
        for c in cards
        if (c.get("assignee") or "").startswith("neurogolf-worker-")
        and (c.get("status") or "") in ("running", "ready", "blocked", "todo")
    )

    manifest = read_json(MANIFEST, {}) or {}
    best = manifest.get("current_best") or {}
    lb = best.get("score", "?")
    cand = (best.get("candidate_id") or "")[:48]

    credit = read_json(CREDIT, {}) or {}
    cr = credit.get("credits") or {}
    available = cr.get("available", "?")
    daily_used = cr.get("daily_used", "?")
    daily_cap = (credit.get("policy") or {}).get("daily_submit_cap", "?")

    diag_n = hermes_diag_count()
    blocked = by_status.get("blocked", 0)
    running = by_status.get("running", 0)
    ready = by_status.get("ready", 0)
    submits_n, recent = submits_today()
    top = top_pending_gains(3)

    lines = [
        f"NeuroGolf daily digest — {now.strftime('%Y-%m-%d %H:%M %Z').strip()}",
        f"",
        f"LB: {lb}  cand: {cand}",
        f"Submits today: {submits_n} (credits {available}, daily {daily_used}/{daily_cap})",
        f"Workers active: {workers_active}/8   running={running} ready={ready} blocked={blocked}   diag warnings: {diag_n}",
    ]
    if top:
        lines.append("Top pending:")
        for cid, g in top:
            lines.append(f"  +{g:.2f}  {cid[:55]}")
    if recent:
        lines.append("Recent submits:")
        for d, summary in recent:
            lines.append(f"  {d}  {summary[:55]}")
    out = "\n".join(lines) + "\n"
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
