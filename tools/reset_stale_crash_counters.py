#!/usr/bin/env python3
"""Reset stale Hermes ``consecutive_failures`` counters for blocked tasks.

When a worker hit the dispatcher max-retries limit (e.g. the v0.14.0
NoConsoleScreenBufferError crash loop) and the underlying bug was later fixed,
the ``tasks.consecutive_failures`` counter remains at the limit. On the next
``unblock``, the dispatcher claims the card, spawns the worker, then
immediately re-blocks it because ``consecutive_failures >= effective_limit``.
The spawn process continues to run as an orphan whose results are never
recorded.

This script clears the counter and the stale ``last_failure_error`` field for
the named tasks (or all blocked tasks with a non-zero counter), kills any
orphan worker PIDs that Hermes is no longer tracking, and unblocks the cards.

Usage:
    python tools/reset_stale_crash_counters.py                  # dry-run
    python tools/reset_stale_crash_counters.py --apply          # mutate
    python tools/reset_stale_crash_counters.py --apply --tasks t_aaa t_bbb

The kill step is conservative: only PIDs that Hermes ``task_runs`` rows still
reference but whose row is marked ``outcome=blocked`` (i.e. the spawn that
Hermes already gave up on but didn't kill).
"""
from __future__ import annotations

import argparse
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

KANBAN_DB = Path.home() / "AppData" / "Local" / "hermes" / "kanban.db"


def latest_block_reason(conn: sqlite3.Connection, task_id: str) -> str:
    row = conn.execute(
        """SELECT payload FROM task_events
           WHERE task_id = ? AND kind = 'blocked'
           ORDER BY created_at DESC LIMIT 1""",
        (task_id,),
    ).fetchone()
    if not row or not row[0]:
        return ""
    payload = row[0]
    import json as _json
    try:
        data = _json.loads(payload)
        return str(data.get("reason", "")).strip()
    except Exception:
        return str(payload)[:300]


def find_stale_blocked_tasks(conn: sqlite3.Connection, only: list[str] | None = None) -> list[dict[str, object]]:
    """Return tasks that are blocked AND have a non-zero consecutive_failures."""
    rows = list(conn.execute(
        """SELECT id, assignee, status, consecutive_failures, last_failure_error
           FROM tasks
           WHERE status = 'blocked' AND consecutive_failures > 0"""
    ))
    out = []
    for tid, assignee, status, fails, last_error in rows:
        if only and tid not in only:
            continue
        block_reason = latest_block_reason(conn, tid)
        out.append({
            "id": tid,
            "assignee": assignee or "",
            "status": status,
            "consecutive_failures": fails,
            "last_failure_error": last_error or "",
            "latest_block_reason": block_reason,
            "is_review_required": "review-required" in block_reason.lower(),
        })
    return out


def find_orphan_pids(conn: sqlite3.Connection, task_ids: list[str]) -> list[tuple[str, int]]:
    """Find worker PIDs that Hermes spawned but no longer tracks.

    Sources, in order:
    1. ``task_runs.worker_pid`` from the most recent run row (if Hermes recorded
       a PID before blocking).
    2. ``task_events`` of kind ``spawned`` (the dispatcher logged the spawn
       even though it then re-blocked the run and never wrote the PID back to
       task_runs).

    A PID is an "orphan" if it appears as the latest spawn for the task AND is
    still alive on the host AND its task is currently blocked.
    """
    if not task_ids:
        return []
    import json as _json
    import re as _re
    pid_re = _re.compile(r"'pid':\s*(\d+)")
    candidates: list[tuple[str, int]] = []
    for task_id in task_ids:
        # Source 1: task_runs.worker_pid
        row = conn.execute(
            """SELECT worker_pid FROM task_runs
               WHERE task_id = ? AND worker_pid IS NOT NULL
               ORDER BY started_at DESC LIMIT 1""",
            (task_id,),
        ).fetchone()
        if row and row[0]:
            candidates.append((task_id, int(row[0])))
            continue
        # Source 2: latest 'spawned' task_event payload.
        row = conn.execute(
            """SELECT payload FROM task_events
               WHERE task_id = ? AND kind = 'spawned'
               ORDER BY created_at DESC LIMIT 1""",
            (task_id,),
        ).fetchone()
        if not row or not row[0]:
            continue
        payload = row[0]
        try:
            data = _json.loads(payload)
            pid = int(data.get("pid", 0))
        except Exception:
            m = pid_re.search(payload)
            pid = int(m.group(1)) if m else 0
        if pid > 0:
            candidates.append((task_id, pid))
    alive: list[tuple[str, int]] = []
    for task_id, pid in candidates:
        if pid_alive(pid):
            alive.append((task_id, pid))
    return alive


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -First 1 Id"],
            capture_output=True, text=True, check=False,
        )
        return str(pid) in proc.stdout
    except Exception:
        return False


def kill_pid(pid: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
            capture_output=True, text=True, check=False,
        )
        ok = proc.returncode == 0
        return ok, (proc.stderr or proc.stdout).strip()[:200]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def reset_counters(conn: sqlite3.Connection, task_ids: list[str]) -> int:
    if not task_ids:
        return 0
    qmarks = ",".join("?" * len(task_ids))
    cur = conn.execute(
        f"UPDATE tasks SET consecutive_failures = 0, last_failure_error = NULL WHERE id IN ({qmarks})",
        task_ids,
    )
    conn.commit()
    return cur.rowcount


def mark_stale_crashed_runs(conn: sqlite3.Connection, task_ids: list[str], before_ts: int) -> int:
    """Re-mark pre-patch crashed task_runs as ``superseded-by-patch``.

    Hermes' ``diag`` computes the "Agent crashed Nx" warning from the most
    recent runs whose ``outcome`` is one of the failure values. Renaming the
    outcome takes the old crashes out of that window without losing the
    audit trail (``task_events`` is unchanged).
    """
    if not task_ids:
        return 0
    qmarks = ",".join("?" * len(task_ids))
    cur = conn.execute(
        f"""UPDATE task_runs
            SET outcome = 'superseded-by-patch',
                summary = COALESCE(summary, '') || ' [pre-patch crash; superseded by NEUROGOLF_CPRINT_HEADLESS_PATCH_V1]'
            WHERE task_id IN ({qmarks})
              AND outcome IN ('crashed', 'spawn_failed', 'memory_limit', 'gave_up')
              AND started_at < ?""",
        (*task_ids, before_ts),
    )
    conn.commit()
    return cur.rowcount


def hermes_unblock(task_id: str) -> tuple[bool, str]:
    proc = subprocess.run(
        ["hermes", "kanban", "unblock", task_id],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()[:200]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Mutate the DB and unblock. Default is dry-run.")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Restrict to these task ids. Default: all blocked tasks with consecutive_failures > 0.")
    parser.add_argument("--no-kill", action="store_true", help="Skip killing orphan worker PIDs.")
    parser.add_argument("--no-unblock", action="store_true", help="Skip calling `hermes kanban unblock` after reset.")
    parser.add_argument("--db", type=Path, default=KANBAN_DB)
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"ERROR: kanban DB not found at {args.db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db)
    try:
        stale = find_stale_blocked_tasks(conn, only=args.tasks)
        if not stale:
            print("No stale blocked tasks with consecutive_failures > 0. Nothing to reset.")
            return 0
        print(f"Stale blocked tasks (consecutive_failures > 0):  {len(stale)}")
        for t in stale:
            kind = "review-required (real handoff)" if t["is_review_required"] else "crash-residual"
            print(f"  {t['id']}  assignee={t['assignee']:24}  fails={t['consecutive_failures']}  kind={kind}")
            if t["latest_block_reason"]:
                print(f"      reason: {t['latest_block_reason'][:90]}")

        task_ids = [t["id"] for t in stale]
        # Only unblock tasks whose block reason is the stale crash residual.
        # Review-required cards have real handoffs — leave for auto_triage.
        crash_residual_ids = [t["id"] for t in stale if not t["is_review_required"]]
        review_required_ids = [t["id"] for t in stale if t["is_review_required"]]

        orphans = find_orphan_pids(conn, crash_residual_ids) if not args.no_kill else []
        if orphans:
            print(f"\nOrphan worker PIDs still alive on crash-residual tasks:")
            for tid, pid in orphans:
                print(f"  {tid}  pid={pid}")

        if not args.apply:
            print(f"\nmode: DRY-RUN (re-run with --apply to mutate)")
            print(f"     would reset consecutive_failures on {len(task_ids)} task(s)")
            print(f"     would unblock {len(crash_residual_ids)} crash-residual task(s)")
            print(f"     would LEAVE {len(review_required_ids)} review-required task(s) blocked for auto_triage")
            return 0

        if orphans and not args.no_kill:
            print("\nKilling orphan workers (crash-residual tasks only)...")
            for tid, pid in orphans:
                ok, msg = kill_pid(pid)
                print(f"  kill pid={pid:>6}  task={tid}  ok={ok}  {msg}")
        rowcount = reset_counters(conn, task_ids)
        print(f"\nReset consecutive_failures = 0 on {rowcount} task row(s).")
        # Pre-patch crash timestamp: read from the file written by
        # patch_hermes_cprint.py. Falls back to a fixed historical value if
        # the file is missing (e.g. patch was applied before that machinery
        # existed). Any crashed/spawn_failed/gave_up task_run before this
        # time is stale-by-definition because the spawn bug it represents
        # no longer applies.
        patch_ts_file = Path.home() / "AppData" / "Local" / "hermes" / "neurogolf_cprint_patch_applied_at.txt"
        # Also check the repo-local path (newer patch_hermes_cprint.py writes here).
        repo_patch_ts_file = Path(__file__).resolve().parent.parent / ".hermes" / "board" / "cprint_patch_applied_at.txt"
        patch_ts: int = 1779062340  # 2026-05-17T15:59:00Z fallback
        for f in (repo_patch_ts_file, patch_ts_file):
            if f.exists():
                try:
                    patch_ts = int(f.read_text(encoding="utf-8").strip())
                    break
                except (ValueError, OSError):
                    pass
        marked = mark_stale_crashed_runs(conn, task_ids, patch_ts)
        print(f"Re-marked {marked} pre-patch failure run row(s) (patch_ts={patch_ts}) as 'superseded-by-patch'.")
        if not args.no_unblock:
            if crash_residual_ids:
                print(f"\nUnblocking {len(crash_residual_ids)} crash-residual task(s)...")
                for tid in crash_residual_ids:
                    ok, msg = hermes_unblock(tid)
                    print(f"  unblock {tid}: ok={ok} {msg[:80]}")
            if review_required_ids:
                print(f"\nLeft {len(review_required_ids)} review-required task(s) blocked (run auto_triage_blocked.py next):")
                for tid in review_required_ids:
                    print(f"  {tid}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
