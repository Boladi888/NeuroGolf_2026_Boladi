#!/usr/bin/env python3
"""One-page NeuroGolf metrics dashboard.

Sources:
  - Logs/submissions_ledger.csv          : predicted vs actual LB delta history
  - .hermes/board/candidate_registry.json: pending candidates, worker attribution
  - .hermes/board/submit_credit_state.json: credit utilization
  - hermes kanban list --json            : reviewer queue, blocked depth, active

Run:
  .venv/Scripts/python tools/dashboard.py
  .venv/Scripts/python tools/dashboard.py --json  (machine-readable)

Prints to stdout. Intended for quick read by humans + ingest by automation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER = REPO_ROOT / "Logs" / "submissions_ledger.csv"
CANDIDATES = REPO_ROOT / ".hermes" / "board" / "candidate_registry.json"
CREDIT = REPO_ROOT / ".hermes" / "board" / "submit_credit_state.json"
MANIFEST = REPO_ROOT / "submissions" / "current_best_manifest.json"

WORKERS = [f"neurogolf-worker-{i}" for i in range(1, 9)]
ACTIVE_STATUSES = {"running", "ready", "todo", "blocked"}
PREDICTED_PATTERN = re.compile(r"\+?(-?\d+\.\d+)")
ACTUAL_PATTERN = re.compile(r"\+?(-?\d+\.\d+)\s+vs\s+\d+\.\d+", re.IGNORECASE)
EXPECTED_INLINE = re.compile(r"expected\s+\+?(-?\d+\.\d+)", re.IGNORECASE)
PROMOTED_PATTERN = re.compile(r"promoted", re.IGNORECASE)
FLAT_PATTERN = re.compile(r"public[- ]?tie|public[- ]?flat", re.IGNORECASE)
HIDDEN_BAD_PATTERN = re.compile(r"-\d+\.\d{2,}", re.IGNORECASE)


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def _run_hermes_json(args: list[str]) -> Any:
    """Force UTF-8 decoding of hermes stdout; Hermes JSON often contains emoji
    (and other extended characters) in card bodies/comments which the default
    Windows cp1252 decoder can't handle. Returns None on any failure."""
    proc = subprocess.run(
        ["hermes", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def hermes_cards() -> list[dict[str, Any]]:
    data = _run_hermes_json(["kanban", "list", "--json"])
    if not isinstance(data, list):
        return []
    return [c for c in data if isinstance(c, dict)]


def hermes_diagnostics() -> list[dict[str, Any]]:
    """Source-of-truth for board diagnostics — defer to Hermes' own report."""
    data = _run_hermes_json(["kanban", "diag", "--json"])
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def parse_ledger_row(row: dict[str, str], worker_index: dict[str, str]) -> dict[str, Any] | None:
    if not row.get("date") or row.get("date") == "date":
        return None
    pred = row.get("local_prediction", "")
    score = row.get("kaggle_score", "")
    verdict = row.get("verdict", "")
    kind = row.get("kind", "")
    fruit = row.get("fruit", "")  # candidate_id-like
    expected_match = EXPECTED_INLINE.search(verdict) or PREDICTED_PATTERN.search(pred)
    actual_match = ACTUAL_PATTERN.search(verdict)
    try:
        kaggle = float(score)
    except (TypeError, ValueError):
        kaggle = None
    worker = worker_index.get(fruit) or extract_worker(kind)
    return {
        "date": row.get("date"),
        "kind": kind,
        "candidate_id": fruit,
        "expected_delta": float(expected_match.group(1)) if expected_match else None,
        "actual_delta": float(actual_match.group(1)) if actual_match else None,
        "kaggle_score": kaggle,
        "verdict_raw": verdict,
        "is_promoted": bool(PROMOTED_PATTERN.search(verdict)),
        "is_flat": bool(FLAT_PATTERN.search(verdict)),
        "worker": worker,
        "task": extract_task(fruit + " " + kind + " " + verdict),
    }


def extract_worker(text: str) -> str:
    for w in WORKERS:
        if w in text:
            return w
    if "submit-captain" in text:
        return "neurogolf-submit-captain"
    if "Worker" in text:
        m = re.search(r"Worker\s*(\d)", text)
        if m:
            return f"neurogolf-worker-{m.group(1)}"
    if "5424" in text:
        return "neurogolf-worker-4 (5424)"
    if "codex-787f" in text:
        return "neurogolf-worker-2 (codex-787f)"
    if "Sabrina" in text:
        return "Sabrina"
    if "Valerie" in text:
        return "Valerie"
    return "unknown"


def extract_task(text: str) -> str:
    match = re.search(r"task\s*0*(\d+)", text.lower())
    return f"task{int(match.group(1)):03d}" if match else "task???"


def build_worker_index() -> dict[str, str]:
    """Map candidate_id -> worker_owner from the candidate registry."""
    doc = read_json(CANDIDATES, {}) or {}
    out: dict[str, str] = {}
    for entry in doc.get("entries") or []:
        cid = entry.get("candidate_id")
        owner = entry.get("worker_owner") or entry.get("followup_worker") or ""
        if cid and owner:
            out[cid] = owner
    return out


def load_ledger() -> list[dict[str, Any]]:
    if not LEDGER.exists():
        return []
    worker_index = build_worker_index()
    rows: list[dict[str, Any]] = []
    with LEDGER.open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            parsed = parse_ledger_row(raw, worker_index)
            if parsed:
                rows.append(parsed)
    return rows


def points_per_worker(ledger: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: {"promoted": 0.0, "flat": 0, "hidden_bad": 0, "submits": 0})
    for row in ledger:
        worker = row["worker"]
        out[worker]["submits"] += 1
        if row["is_promoted"] and row["actual_delta"] is not None and row["actual_delta"] > 0:
            out[worker]["promoted"] += row["actual_delta"]
        if row["is_flat"]:
            out[worker]["flat"] += 1
        if row["actual_delta"] is not None and row["actual_delta"] < -1.0:
            out[worker]["hidden_bad"] += 1
    return {k: dict(v) for k, v in out.items()}


def prediction_accuracy(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [(r["expected_delta"], r["actual_delta"]) for r in ledger
             if r["expected_delta"] is not None and r["actual_delta"] is not None]
    if not pairs:
        return {"n": 0}
    n = len(pairs)
    errors = [a - e for e, a in pairs]
    abs_errors = [abs(x) for x in errors]
    abs_errors_sorted = sorted(abs_errors)
    median = abs_errors_sorted[n // 2]
    mean = sum(errors) / n
    # Hidden-bad: predicted positive, actual <= -1 (large negative surprise)
    hidden_bad = sum(1 for e, a in pairs if e is not None and e > 0 and a is not None and a < -1.0)
    # Misses: predicted positive >= 0.3 but actual within +/-0.05 (public-flat)
    flat_surprises = sum(1 for e, a in pairs if e is not None and e >= 0.3 and a is not None and -0.05 <= a <= 0.05)
    return {
        "n": n,
        "mean_signed_error": round(mean, 3),
        "median_abs_error": round(median, 3),
        "hidden_bad_count": hidden_bad,
        "flat_surprise_count": flat_surprises,
        "hidden_bad_rate": round(hidden_bad / n, 3),
    }


def credit_status() -> dict[str, Any]:
    doc = read_json(CREDIT, {}) or {}
    credits = doc.get("credits") or {}
    return {
        "available": credits.get("available"),
        "reserved": credits.get("reserved"),
        "daily_used": credits.get("daily_used"),
        "daily_cap": (doc.get("policy") or {}).get("daily_submit_cap"),
        "last_submit_at": credits.get("last_submit_at"),
    }


def reviewer_queue(cards: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    by_assignee_active: Counter[str] = Counter()
    review_pending = 0
    review_queued = 0  # [review ...] cards in todo, dependency-serialized behind a head
    submit_pending = 0
    unassigned_active = 0
    unassigned_backlog = 0
    worker_capacity_violations: list[str] = []
    worker_counts: Counter[str] = Counter()
    for c in cards:
        status = (c.get("status") or "").lower()
        if status not in ACTIVE_STATUSES:
            continue
        assignee = c.get("assignee") or ""
        title_lower = (c.get("title") or "").lower()
        is_review_card = title_lower.startswith("[review")
        is_submit_card = title_lower.startswith("[submit")
        if is_review_card:
            review_pending += 1
            if not assignee or status == "todo":
                review_queued += 1
        if is_submit_card or assignee == "neurogolf-submit-captain":
            submit_pending += 1
        if not assignee:
            unassigned_active += 1
            if not is_review_card and not is_submit_card:
                unassigned_backlog += 1
            continue
        by_assignee_active[assignee] += 1
        if assignee in WORKERS:
            worker_counts[assignee] += 1
    for w, n in worker_counts.items():
        if n > 1:
            worker_capacity_violations.append(f"{w} has {n} active cards (one-task rule)")
    # Defer to Hermes' own diagnostic engine for the "agents crashed" warning.
    # If `hermes kanban diag` reports a repeated_crashes error on a worker
    # card, surface it. Otherwise stay silent.
    awaiting_session_workers: list[str] = []
    for d in diagnostics:
        for sub in d.get("diagnostics") or []:
            if not isinstance(sub, dict):
                continue
            if sub.get("kind") == "repeated_crashes" and (d.get("status") or "").lower() in ACTIVE_STATUSES:
                assignee = d.get("assignee") or "?"
                awaiting_session_workers.append(assignee)
                break
    return {
        "review_pending": review_pending,
        "review_queued": review_queued,
        "submit_pending": submit_pending,
        "unassigned_active": unassigned_active,
        "unassigned_backlog": unassigned_backlog,
        "awaiting_session_workers": awaiting_session_workers,
        "by_assignee": dict(by_assignee_active),
        "worker_capacity_violations": worker_capacity_violations,
        "idle_workers": [w for w in WORKERS if worker_counts.get(w, 0) == 0],
    }


def candidate_pipeline() -> dict[str, Any]:
    doc = read_json(CANDIDATES, {}) or {}
    entries = doc.get("entries") or []
    by_status: Counter[str] = Counter()
    expected_gain = 0.0
    for e in entries:
        st = (e.get("submit_queue_status") or e.get("status") or "unknown")
        by_status[st] += 1
        # Legacy candidate_registry entries sometimes have expected_point_gain
        # stored as a string. Coerce safely; bad values get ignored.
        raw_gain = e.get("expected_point_gain")
        try:
            gain = float(raw_gain) if raw_gain not in (None, "") else 0.0
        except (TypeError, ValueError):
            gain = 0.0
        if st in ("ready", "approved", "pending"):
            expected_gain += gain
    return {
        "total_entries": len(entries),
        "by_status": dict(by_status.most_common()),
        "ready_expected_gain": round(expected_gain, 3),
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"  NEUROGOLF DASHBOARD  --  {report['generated_at']}  (LB: {report['lb']})")
    lines.append("=" * 78)
    lines.append("")
    pa = report["prediction_accuracy"]
    lines.append("PREDICTED vs ACTUAL LB DELTA")
    lines.append(f"  Sample size:           {pa['n']} submits with both fields")
    lines.append(f"  Mean signed error:     {pa.get('mean_signed_error','-')}  (actual - expected)")
    lines.append(f"  Median |error|:        {pa.get('median_abs_error','-')}")
    lines.append(f"  Hidden-bad incidents:  {pa.get('hidden_bad_count','-')}  ({(pa.get('hidden_bad_rate') or 0) * 100:.1f}%)")
    lines.append(f"  Public-flat surprises: {pa.get('flat_surprise_count','-')}  (predicted >= +0.3, actual ~ 0)")
    lines.append("")
    cr = report["credit"]
    lines.append("SUBMIT CREDIT")
    lines.append(f"  Available:    {cr.get('available'):<5}  Reserved: {cr.get('reserved')}")
    lines.append(f"  Daily used:   {cr.get('daily_used'):<5}/ cap {cr.get('daily_cap')}")
    lines.append(f"  Last submit:  {cr.get('last_submit_at')}")
    lines.append("")
    rq = report["reviewer_queue"]
    lines.append("REVIEWER / SUBMIT QUEUE")
    lines.append(f"  Review cards total:      {rq['review_pending']}  ({rq['review_queued']} dep-serialized behind lane head)")
    lines.append(f"  Submit cards pending:    {rq['submit_pending']}")
    lines.append(f"  Unassigned backlog:      {rq['unassigned_backlog']}  (worker targets awaiting orchestrator)")
    lines.append(f"  Idle workers (Pool A+B): {len(rq['idle_workers'])} / 8")
    if rq["idle_workers"]:
        lines.append(f"    {', '.join(rq['idle_workers'])}")
    if rq["awaiting_session_workers"]:
        lines.append(f"  AWAITING LLM SESSION:    {len(rq['awaiting_session_workers'])} profile(s) have dispatcher-give-up blocks")
        for w in rq["awaiting_session_workers"]:
            lines.append(f"    !! {w} -- bind a Claude session OR run tools/auto_triage_blocked.py --apply --gave-up-action=abandon")
    if rq["worker_capacity_violations"]:
        lines.append("  CAPACITY VIOLATIONS:")
        for v in rq["worker_capacity_violations"]:
            lines.append(f"    !! {v}")
    if rq["by_assignee"]:
        lines.append("  Per-assignee active card counts:")
        for assignee, count in sorted(rq["by_assignee"].items(), key=lambda x: -x[1])[:8]:
            lines.append(f"    {assignee:30} {count}")
    lines.append("")
    cp = report["candidate_pipeline"]
    lines.append("CANDIDATE REGISTRY")
    lines.append(f"  Total entries:               {cp['total_entries']}")
    lines.append(f"  Ready-state expected gain:   +{cp['ready_expected_gain']} LB if all submit positive")
    for st, n in list(cp["by_status"].items())[:6]:
        lines.append(f"    {st:30} {n}")
    lines.append("")
    pw = report["points_per_worker"]
    lines.append("POINTS PER WORKER (cumulative, from ledger)")
    sorted_workers = sorted(pw.items(), key=lambda x: -x[1].get("promoted", 0))
    lines.append(f"  {'Worker':30} {'+LB':>8}  {'submits':>8}  {'flat':>5}  {'hbad':>5}")
    for w, m in sorted_workers[:12]:
        lines.append(f"  {w:30} {m.get('promoted', 0):>+8.2f}  {m.get('submits', 0):>8}  {m.get('flat', 0):>5}  {m.get('hidden_bad', 0):>5}")
    lines.append("")
    lines.append("UNSOLVED-TASK OPPORTUNITY")
    u = report["unsolved"]
    lines.append(f"  Tasks at floor (1 pt):     {u['unsolved_count']} / 400")
    lines.append(f"  Avg pts/task if solved at cost 50k: ~14  ->  ceiling ~{u['unsolved_count'] * 14:.0f} LB")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def unsolved_summary() -> dict[str, Any]:
    try:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from inspect_tasks import SOLVED_TASKS  # type: ignore
    except Exception:
        return {"unsolved_count": -1, "note": "inspect_tasks.SOLVED_TASKS unavailable"}
    solved = set(SOLVED_TASKS)
    unsolved = [t for t in range(1, 401) if t not in solved]
    return {"unsolved_count": len(unsolved), "solved_count": len(solved)}


def build_report() -> dict[str, Any]:
    ledger = load_ledger()
    cards = hermes_cards()
    diagnostics = hermes_diagnostics()
    manifest = read_json(MANIFEST, {}) or {}
    best = manifest.get("current_best") or {}
    return {
        "schema": "neurogolf.dashboard.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "lb": best.get("score", "?"),
        "prediction_accuracy": prediction_accuracy(ledger),
        "credit": credit_status(),
        "reviewer_queue": reviewer_queue(cards, diagnostics),
        "candidate_pipeline": candidate_pipeline(),
        "points_per_worker": points_per_worker(ledger),
        "unsolved": unsolved_summary(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)
    report = build_report()
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
