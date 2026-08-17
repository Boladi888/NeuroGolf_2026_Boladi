#!/usr/bin/env python3
"""Summarize the retired 2026-05 submission candidate queue from archived CSV ledgers.

This intentionally reads only archived CSV metadata plus task_state.csv. It does
not inspect local submission zips, ONNX payloads, or submissions/* task metadata
files. Current shipping truth moved to the OnnxlyFans Team_Submission flow.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATE = "2026-05-08"
RETIRED_LEDGER_DIR = ROOT / "The_400_Tasks" / "_archive" / "retired_top_level_ledgers_2026-06-21"


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


def _priority_sort(row: dict[str, str]) -> int:
    priority = _int_or_none(row.get("priority"))
    return priority if priority is not None else 999999


def _candidate_sort(row: dict[str, str]) -> tuple[int, int]:
    role_rank = 0 if row.get("role") == "current_valid_best" else 1
    return role_rank, _priority_sort(row)


def _task_label(row: dict[str, str]) -> str:
    task = row.get("task", "")
    delta = _float_or_none(row.get("delta"))
    source = row.get("source", "")
    cost = row.get("cost", "")
    if delta is None:
        return task
    cost_note = f", cost {cost}" if cost else ""
    source_note = f" {source}" if source else ""
    return f"{task} (+{delta:.4f}{source_note}{cost_note})"


def build_digest(date: str) -> tuple[list[dict[str, str]], str]:
    candidates_path = RETIRED_LEDGER_DIR / "submission_candidates.csv"
    tasks_path = RETIRED_LEDGER_DIR / "submission_candidate_tasks.csv"
    state_path = ROOT / "The_400_Tasks" / "task_state.csv"

    candidates = _read_csv(candidates_path)
    candidate_tasks = _read_csv(tasks_path)
    task_state = _read_csv(state_path)

    tasks_by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_tasks:
        tasks_by_candidate[row.get("candidate", "")].append(row)

    active = [row for row in candidates if _int_or_none(row.get("priority")) is not None]
    active.sort(key=_candidate_sort)

    digest_rows: list[dict[str, str]] = []
    for row in active:
        name = row.get("candidate", "")
        rows = tasks_by_candidate.get(name, [])
        delta_rows = [r for r in rows if _float_or_none(r.get("delta")) is not None]
        deltas = [_float_or_none(r.get("delta")) for r in delta_rows]
        costs = [_int_or_none(r.get("cost")) for r in delta_rows]
        deltas = [d for d in deltas if d is not None]
        costs = [c for c in costs if c is not None]
        top_rows = sorted(
            delta_rows,
            key=lambda r: _float_or_none(r.get("delta")) or -1.0,
            reverse=True,
        )[:8]
        sources = sorted({r.get("source", "") for r in delta_rows if r.get("source", "")})
        digest_rows.append(
            {
                "priority": row.get("priority", ""),
                "candidate": name,
                "role": row.get("role", ""),
                "declared_task_count": row.get("task_count", ""),
                "metadata_rows": str(len(rows)),
                "estimated_result": row.get("estimated_result", ""),
                "numeric_delta_sum": f"{sum(deltas):.4f}" if deltas else "",
                "min_delta": f"{min(deltas):.4f}" if deltas else "",
                "max_delta": f"{max(deltas):.4f}" if deltas else "",
                "min_cost": str(min(costs)) if costs else "",
                "max_cost": str(max(costs)) if costs else "",
                "sources": "; ".join(sources),
                "top_tasks": "; ".join(_task_label(r) for r in top_rows),
                "local_payload_path": row.get("local_payload_path", ""),
                "tasks_metadata_path": row.get("tasks_metadata_path", ""),
                "known_risks": row.get("known_risks", ""),
                "next_action": row.get("next_action", ""),
                "reference": row.get("reference", ""),
            }
        )

    retest_rows = [
        row
        for row in task_state
        if row.get("best_known_status") == "avoid_or_retest"
        or "retest" in row.get("next_action", "").lower()
        or "avoid" in row.get("verdict", "").lower()
    ]
    retest_rows.sort(key=lambda row: row.get("task", ""))

    md_lines = [
        f"# Submission Candidate Digest - {date}",
        "",
        "Generated from retired archived CSV metadata only:",
        "",
        "- `The_400_Tasks/_archive/retired_top_level_ledgers_2026-06-21/submission_candidates.csv`",
        "- `The_400_Tasks/_archive/retired_top_level_ledgers_2026-06-21/submission_candidate_tasks.csv`",
        "- `The_400_Tasks/task_state.csv`",
        "",
        "This digest does not inspect or copy `submissions/`, `.zip`, or `.onnx` payloads. It is historical evidence, not the current upload queue.",
        "",
        "## Candidate State",
        "",
        "| Priority | Candidate | Role | Tasks | Estimate | Next action | Risk | Local payload path |",
        "|---:|---|---|---:|---|---|---|---|",
    ]
    for row in digest_rows:
        md_lines.append(
            "| {priority} | `{candidate}` | {role} | {declared_task_count} "
            "| {estimated_result} | {next_action} | {known_risks} | `{local_payload_path}` |".format(
                **row
            )
        )

    md_lines += [
        "",
        "## Chunk Contents",
        "",
    ]
    for row in digest_rows:
        md_lines += [
            f"### {row['priority']}. `{row['candidate']}`",
            "",
            f"- Metadata rows: {row['metadata_rows']} (declared task count: {row['declared_task_count']})",
            f"- Numeric local delta sum in metadata: {row['numeric_delta_sum'] or 'n/a'}",
            f"- Delta range: {row['min_delta'] or 'n/a'} to {row['max_delta'] or 'n/a'}",
            f"- Cost range: {row['min_cost'] or 'n/a'} to {row['max_cost'] or 'n/a'}",
            f"- Sources: {row['sources'] or 'n/a'}",
            f"- Top tasks by local delta: {row['top_tasks'] or 'n/a'}",
            f"- Reference: `{row['reference']}`",
            "",
        ]

    md_lines += [
        "## Retest Guardrails",
        "",
        "| Task | Status | Verdict | Next action | Reference |",
        "|---|---|---|---|---|",
    ]
    for row in retest_rows:
        md_lines.append(
            "| {task} | {best_known_status} | {verdict} | {next_action} | `{reference}` |".format(
                **row
            )
        )

    md_lines += [
        "",
        "## Historical Checklist",
        "",
        "1. Read the top rows as retired 2026-05 evidence, not an automatic upload command.",
        "2. For current uploads, use the OnnxlyFans `Team_Submission` manifest / current leaderboard state.",
        "3. If replaying the old queue as research, write new notes to `Logs/` and update `The_400_Tasks/task_state.csv` only for per-task outcomes.",
        "4. Treat task089, task241, and task380 as avoid/retest until a fresh isolated verification says otherwise.",
        "5. Stage only CSV/Markdown/script changes; never stage payload zips, ONNX files, `data/`, or `submissions/`.",
        "",
    ]

    return digest_rows, "\n".join(md_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument(
        "--out-prefix",
        default=f"Logs/submission_candidate_digest_{DEFAULT_DATE}",
        help="Output path without extension, relative to repository root.",
    )
    args = parser.parse_args()

    rows, markdown = build_digest(args.date)
    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    fieldnames = [
        "priority",
        "candidate",
        "role",
        "declared_task_count",
        "metadata_rows",
        "estimated_result",
        "numeric_delta_sum",
        "min_delta",
        "max_delta",
        "min_cost",
        "max_cost",
        "sources",
        "top_tasks",
        "local_payload_path",
        "tasks_metadata_path",
        "known_risks",
        "next_action",
        "reference",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
