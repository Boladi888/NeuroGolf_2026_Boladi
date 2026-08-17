#!/usr/bin/env python3
"""Build an orange5-centered ablation queue from tracked scoring ledgers."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATE = "2026-05-08"
CURRENT_ROLE = "current_valid_best"
TESTED_ROLE = "tested_non_improving"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _score_from_text(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value or "")
    if not match:
        return None
    return float(match.group(0))


def _task_row_key(row: dict[str, str]) -> tuple[float, int]:
    delta = _float_or_none(row.get("delta")) or -999.0
    cost = _int_or_none(row.get("cost")) or 999_999_999
    return (delta, -cost)


def _candidate_task_label(row: dict[str, str]) -> str:
    task = row["task"]
    delta = _float_or_none(row.get("delta"))
    cost = row.get("cost", "")
    source = row.get("source", "")
    bits = [f"task{task}"]
    if delta is not None:
        bits.append(f"+{delta:.4f} local")
    if cost:
        bits.append(f"cost {cost}")
    if source:
        bits.append(source)
    return " / ".join(bits)


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_queue(date: str) -> tuple[list[dict[str, str]], str]:
    candidates_path = ROOT / "The_400_Tasks" / "submission_candidates.csv"
    tasks_path = ROOT / "The_400_Tasks" / "submission_candidate_tasks.csv"
    state_path = ROOT / "The_400_Tasks" / "task_state.csv"

    candidates = _read_csv(candidates_path)
    task_rows = _read_csv(tasks_path)
    task_state = _read_csv(state_path)

    tasks_by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in task_rows:
        tasks_by_candidate[row.get("candidate", "")].append(row)

    current = next(row for row in candidates if row.get("role") == CURRENT_ROLE)
    current_name = current["candidate"]
    current_score = _score_from_text(current.get("estimated_result", ""))
    current_tasks = tasks_by_candidate[current_name]
    current_task_ids = {row["task"] for row in current_tasks}

    coconut = next(row for row in candidates if row["candidate"] == "coconut4_known_best_drop383")
    coconut_score = _score_from_text(coconut.get("estimated_result", ""))

    tested = [
        row
        for row in candidates
        if row.get("role") == TESTED_ROLE and (_int_or_none(row.get("priority")) or 999) < 100
    ]
    tested.sort(key=lambda row: _int_or_none(row.get("priority")) or 999)

    avoid_tasks = {
        row["task"]
        for row in task_state
        if row.get("best_known_status") == "avoid_or_retest"
    }

    tested_memberships: dict[str, list[str]] = defaultdict(list)
    best_row_by_task: dict[str, dict[str, str]] = {}
    for candidate in tested:
        for row in tasks_by_candidate[candidate["candidate"]]:
            task = row["task"]
            tested_memberships[task].append(candidate["candidate"])
            if task in current_task_ids or task in avoid_tasks:
                continue
            if task not in best_row_by_task or _task_row_key(row) > _task_row_key(best_row_by_task[task]):
                best_row_by_task[task] = row

    queue: list[dict[str, str]] = []
    priority = 10
    for row in sorted(current_tasks, key=lambda r: _float_or_none(r.get("delta")) or 999.0):
        delta = _float_or_none(row.get("delta"))
        queue.append(
            {
                "priority": str(priority),
                "tier": "A",
                "candidate_stub": f"orange5_minus_task{row['task']}_leaveout",
                "action": "drop_from_orange5",
                "task": row["task"],
                "local_delta_vs_coconut": f"{delta:.4f}" if delta is not None else "",
                "cost": row.get("cost", ""),
                "source": row.get("source", ""),
                "payload_path": row.get("payload_path", ""),
                "evidence": "orange5 is positive as a 5-task chunk; leave-one-out tests reveal hidden contribution of each current base member.",
                "submit_goal": "If score improves over orange5, remove this task from the base; if it drops, keep it.",
                "risk": "Uses an already-positive base; one upload per task.",
            }
        )
        priority += 1

    papaya_extra = [
        row
        for row in tasks_by_candidate["papaya5_coconut_plus_top12_verified"]
        if row["task"] not in current_task_ids and row["task"] not in avoid_tasks
    ]
    priority = 20
    for row in sorted(papaya_extra, key=_task_row_key, reverse=True):
        delta = _float_or_none(row.get("delta"))
        queue.append(
            {
                "priority": str(priority),
                "tier": "B",
                "candidate_stub": f"orange5_plus_task{row['task']}_isolated",
                "action": "add_to_orange5",
                "task": row["task"],
                "local_delta_vs_coconut": f"{delta:.4f}" if delta is not None else "",
                "cost": row.get("cost", ""),
                "source": row.get("source", ""),
                "payload_path": row.get("payload_path", ""),
                "evidence": "papaya5 added this task family on top of orange5 and scored 13.13 below orange5 as a group.",
                "submit_goal": "Find any survivor hidden-positive task inside the failed papaya add-on set.",
                "risk": "The group was hidden-negative; isolated retest required before promotion.",
            }
        )
        priority += 1

    remaining = [
        row
        for task, row in best_row_by_task.items()
        if task not in {r["task"] for r in papaya_extra}
    ]
    priority = 40
    for row in sorted(remaining, key=_task_row_key, reverse=True):
        task = row["task"]
        delta = _float_or_none(row.get("delta"))
        memberships = "; ".join(sorted(set(tested_memberships[task])))
        tier = "C" if (delta or 0.0) >= 0.5 else "D"
        queue.append(
            {
                "priority": str(priority),
                "tier": tier,
                "candidate_stub": f"orange5_plus_task{task}_isolated",
                "action": "add_to_orange5",
                "task": task,
                "local_delta_vs_coconut": f"{delta:.4f}" if delta is not None else "",
                "cost": row.get("cost", ""),
                "source": row.get("source", ""),
                "payload_path": row.get("payload_path", ""),
                "evidence": f"appeared only in tested non-improving chunks: {memberships}",
                "submit_goal": "Retest only after Tier A/B, or if the source gets a new reason to trust it.",
                "risk": "Lower signal; prior chunk evidence is negative.",
            }
        )
        priority += 1

    score_rows: list[str] = []
    if coconut_score is not None and current_score is not None:
        score_rows.append(
            f"- `orange5` scored **{current_score:.2f}**, "
            f"which is **{current_score - coconut_score:+.2f}** vs `coconut4`."
        )
    for candidate in tested:
        score = _score_from_text(candidate.get("estimated_result", ""))
        if score is None:
            continue
        if current_score is not None:
            score_rows.append(
                f"- `{candidate['candidate']}` scored **{score:.2f}**, "
                f"which is **{score - current_score:+.2f}** vs `orange5`."
            )

    md_lines = [
        f"# Orange5 Ablation Queue - {date}",
        "",
        "Generated from tracked CSV metadata only:",
        "",
        "- `The_400_Tasks/submission_candidates.csv`",
        "- `The_400_Tasks/submission_candidate_tasks.csv`",
        "- `The_400_Tasks/task_state.csv`",
        "",
        "This report does not inspect, copy, or create submission payloads.",
        "",
        "## Current Evidence",
        "",
        *score_rows,
        "",
        "Interpretation: `orange5` is the only proven positive base after the reset-time queue. "
        "The broader chunks are diagnostic failures, so every non-orange addition needs isolated proof.",
        "",
        "## Recommended Upload Order",
        "",
        "| Priority | Tier | Candidate stub | Action | Task | Local delta | Cost | Source | Why |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in queue:
        md_lines.append(
            "| {priority} | {tier} | `{candidate_stub}` | {action} | {task} | "
            "{local_delta_vs_coconut} | {cost} | {source} | {evidence} |".format(**row)
        )

    md_lines += [
        "",
        "## Tier Notes",
        "",
        "- **Tier A:** leave-one-out probes for the five current orange5 members. These establish whether any current member is hidden-negative despite the chunk being net-positive.",
        "- **Tier B:** papaya-only extras. They have the strongest local deltas among failed additions, but the group scored below orange5.",
        "- **Tier C:** remaining single-task add-backs with local delta >= 0.5. Treat as quota-fill after Tier A/B.",
        "- **Tier D:** tiny-delta tail. Do not spend submissions here unless a source refresh changes the evidence.",
        "- Keep task089, task241, and task380 out of this queue unless a fresh source plus isolated verification changes their ledger status.",
        "",
        "## Packaging Rule",
        "",
        "Build each candidate as a single-task modification against the exact orange5 base. "
        "After any upload, update `The_400_Tasks/submission_candidates.csv`, "
        "`The_400_Tasks/task_state.csv`, and regenerate the candidate digest.",
        "",
    ]

    return queue, "\n".join(md_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument(
        "--out-prefix",
        default=f"Logs/orange5_ablation_queue_{DEFAULT_DATE}",
        help="Output path without extension, relative to repository root.",
    )
    args = parser.parse_args()

    rows, markdown = build_queue(args.date)
    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "priority",
        "tier",
        "candidate_stub",
        "action",
        "task",
        "local_delta_vs_coconut",
        "cost",
        "source",
        "payload_path",
        "evidence",
        "submit_goal",
        "risk",
    ]
    _write_csv(out_prefix.with_suffix(".csv"), rows, fieldnames)
    out_prefix.with_suffix(".md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out_prefix.with_suffix('.csv').relative_to(ROOT)}")
    print(f"Wrote {out_prefix.with_suffix('.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
