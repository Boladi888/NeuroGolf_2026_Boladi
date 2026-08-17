"""Sync the live task_state.csv into NeuroGolf_Task_Status.xlsx — idempotent, fast.

Direction (per the 2026-06-07 flip in merge_task_status.py): the CSV is the live
source of truth for cost/points/src/split columns — the captain reconciles it after
every ship. The xlsx is the owner's spreadsheet VIEW of that truth plus manually
maintained workflow columns (status, priority, family, notes, ...) which this tool
NEVER touches.

What it updates:
  - Tasks sheet: the VOLATILE columns only (current_cost, current_points, memory,
    params, current_src, best_known_status, verdict, next_action, team_split,
    last_attempted_at). Adds the team_split column if missing.
  - Summary sheet: the "Generated:" stamp, Sum of current_cost, Sum of
    current_points (LB) — so the headline matches the leaderboard.

Usage:
    python tools/sync_task_xlsx.py            # sync; prints one summary line
    python tools/sync_task_xlsx.py --check    # exit 1 if out of sync (no writes)
    python tools/sync_task_xlsx.py --lb 6659.34   # optional: stamp the actual LB

Automation: the pre-commit hook (.git/hooks/pre-commit; tracked copy at
tools/hooks/pre-commit) runs this automatically whenever task_state.csv is staged,
then stages the refreshed xlsx into the same commit. Fail-open: a sync error WARNS
loudly but never blocks a commit — run manually if you see the warning.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "The_400_Tasks" / "task_state.csv"
XLSX_PATH = REPO / "The_400_Tasks" / "NeuroGolf_Task_Status.xlsx"

VOLATILE = [
    "current_cost", "current_points", "memory", "params", "current_src",
    "best_known_status", "verdict", "next_action", "team_split", "last_attempted_at",
]
NUMERIC = {"current_cost", "current_points", "memory", "params"}


def load_csv() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["task"]] = row
    return rows


def coerce(col: str, val: str):
    if col in NUMERIC:
        try:
            f = float(val)
            return int(f) if col != "current_points" and f == int(f) else f
        except ValueError:
            return val
    return val


def same(old, new) -> bool:
    if old is None and (new == "" or new is None):
        return True
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return abs(float(old) - float(new)) < 1e-9
    return old == new


def sync(check_only: bool = False, lb: str | None = None) -> tuple[int, int]:
    import openpyxl
    from openpyxl.styles import Font, PatternFill

    csv_rows = load_csv()
    wb = openpyxl.load_workbook(XLSX_PATH)
    ws = wb["Tasks"]
    hdr = {c.value: i + 1 for i, c in enumerate(ws[1])}

    # ensure team_split column exists
    added_col = False
    if "team_split" not in hdr:
        col = ws.max_column + 1
        h = ws.cell(row=1, column=col, value="team_split")
        h.fill = PatternFill("solid", fgColor="1F4E78")
        h.font = Font(color="FFFFFF", bold=True)
        hdr["team_split"] = col
        added_col = True

    changed = 0
    rows_touched = 0
    for r in range(2, ws.max_row + 1):
        task = ws.cell(row=r, column=hdr["task"]).value
        src = csv_rows.get(task)
        if not src:
            continue
        row_changed = False
        for col in VOLATILE:
            if col not in hdr:
                continue
            raw = src.get(col)
            if raw is None or raw == "":
                continue
            new = coerce(col, raw)
            cell = ws.cell(row=r, column=hdr[col])
            if not same(cell.value, new):
                if not check_only:
                    cell.value = new
                changed += 1
                row_changed = True
        rows_touched += row_changed

    # Summary refresh
    total_cost = sum(float(v["current_cost"]) for v in csv_rows.values() if v["task"] != "task000")
    total_pts = sum(float(v["current_points"]) for v in csv_rows.values() if v["task"] != "task000")
    stamp = (f"Generated: {datetime.date.today().isoformat()} (sum-points {total_pts:.2f}"
             + (f"; Kaggle LB {lb}" if lb else "") + "). Tasks: 401. Auto-synced from task_state.csv.")
    ws2 = wb["Summary"]
    have_pts_row = False
    for row in ws2.iter_rows(min_row=1, max_row=ws2.max_row):
        a = row[0].value
        if isinstance(a, str) and a.startswith("Generated:"):
            # The stamp carries today's date + optional --lb suffix, so it differs
            # daily and across --lb/no-lb invocations. Refresh it on apply, but do
            # NOT count it as drift in --check (else --check false-positives every
            # day and whenever --lb is omitted). The data cells below are the real
            # drift signal.
            if not check_only and not same(a, stamp):
                row[0].value = stamp
        elif a == "Sum of current_cost (LB-approx)":
            if not same(row[1].value, int(total_cost)):
                if not check_only:
                    row[1].value = int(total_cost)
                changed += 1
        elif a == "Sum of current_points (LB)":
            have_pts_row = True
            if not same(row[1].value, round(total_pts, 2)):
                if not check_only:
                    row[1].value = round(total_pts, 2)
                changed += 1
    if not have_pts_row and not check_only:
        for idx, row in enumerate(ws2.iter_rows(min_row=1, max_row=ws2.max_row), start=1):
            if row[0].value == "Sum of current_cost (LB-approx)":
                ws2.insert_rows(idx + 1)
                ws2.cell(row=idx + 1, column=1, value="Sum of current_points (LB)")
                ws2.cell(row=idx + 1, column=2, value=round(total_pts, 2))
                changed += 1
                break

    if not check_only and (changed or added_col):
        wb.save(XLSX_PATH)
    return changed, rows_touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if out of sync; write nothing")
    ap.add_argument("--lb", default=None, help="actual Kaggle LB to stamp into the Summary line")
    args = ap.parse_args()
    changed, rows = sync(check_only=args.check, lb=args.lb)
    if args.check:
        print(f"sync_task_xlsx --check: {changed} cell(s) out of sync across {rows} row(s)")
        return 1 if changed else 0
    print(f"sync_task_xlsx: {changed} cell(s) updated across {rows} row(s)"
          + (" (in sync — no write)" if changed == 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
