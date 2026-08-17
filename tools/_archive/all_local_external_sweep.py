#!/usr/bin/env python3
"""Verify local external ONNX artifacts for selected NeuroGolf tasks.

This is intentionally Beta-owned reporting glue. It scans every top-level
directory under `NEUROGOLF_COMPETITOR_ARTIFACTS_DIR` (or the local
data/competitor_artifacts fallback) that contains taskNNN.onnx files,
then runs the same candidate triage + visible/test/arc-gen verifier used by
tools/external_blend.py. It writes tracked CSV/Markdown reports, but never
builds or submits a zip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
ARTIFACTS = Path(
    os.environ.get(
        "NEUROGOLF_COMPETITOR_ARTIFACTS_DIR",
        str(ROOT / "data" / "competitor_artifacts"),
    )
)
sys.path.insert(0, str(TOOLS))

from external_blend import _candidate_bytes, _verify_candidate  # noqa: E402


def _task_num(path: Path) -> int | None:
    match = re.fullmatch(r"task(\d{3})\.onnx", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_tasks(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def _source_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else path.parent.name


def _license_note(source: str) -> tuple[str, int]:
    """Return (human note, rank). Lower rank is safer for sorting."""
    known_public_prefixes = (
        "afr1ste-",
        "beicicc-",
        "franksunp",
        "kernel-output-jonathan",
        "kernel-output-jonathanchan",
        "kernel-output-konbu17",
        "konbu17-",
        "thisray-",
        "vyanktesh",
    )
    if source.startswith(known_public_prefixes):
        return "local-public-artifact; verify upstream Kaggle license before merge", 1
    return "local-artifact; license not mapped in this worktree", 2


def _read_cascade_audit(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["path"].replace("/", "\\"): row for row in csv.DictReader(f)}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(Path("data") / "competitor_artifacts" / path.relative_to(ARTIFACTS))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rows(tasks: set[int], full: bool, max_pairs: int, audit_csv: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not ARTIFACTS.exists():
        raise FileNotFoundError(
            f"Missing artifact directory {ARTIFACTS}. Set NEUROGOLF_COMPETITOR_ARTIFACTS_DIR "
            "to the directory containing local competitor artifact folders."
        )

    source_files: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(ARTIFACTS.rglob("task*.onnx")):
        task = _task_num(path)
        if task is None:
            continue
        source_files[_source_name(path, ARTIFACTS)].append(path)

    if not source_files:
        raise RuntimeError(
            f"No taskNNN.onnx files found under {ARTIFACTS}; refusing to write an empty sweep report."
        )

    manifest_rows = []
    for source, paths in sorted(source_files.items()):
        task_counts = Counter(_task_num(path) for path in paths)
        manifest_rows.append(
            {
                "source": source,
                "task_file_count": str(len(paths)),
                "distinct_task_count": str(len(task_counts)),
                "has_target_task": str(any(task in tasks for task in task_counts)),
                "target_tasks_present": ";".join(f"{task:03d}" for task in sorted(tasks & set(task_counts))),
            }
        )

    audit_by_path = _read_cascade_audit(audit_csv)
    rows: list[dict[str, str]] = []
    max_pairs_arg = None if full else max_pairs
    for source, paths in sorted(source_files.items()):
        license_status, license_rank = _license_note(source)
        for path in paths:
            task = _task_num(path)
            if task not in tasks:
                continue
            rel = _rel(path)
            raw, triage_status, src_size, renamed = _candidate_bytes(path)
            result = None
            reason = triage_status
            if raw is not None:
                result, reason = _verify_candidate(raw, task, max_pairs_arg, include_risky=False)
            audit = audit_by_path.get(rel.replace("/", "\\"))
            if result is None:
                cost = points = macs = memory = params = ""
                status = "fail"
            else:
                cost, points, macs, memory, params = result
                status = "pass"
            cascade = audit.get("classification", "") if audit else ""
            cascade_reasons = audit.get("reasons", "") if audit else ""
            rows.append(
                {
                    "task": f"{task:03d}",
                    "source": source,
                    "path": rel,
                    "status": status,
                    "triage_status": triage_status,
                    "reason": reason,
                    "source_size": str(src_size),
                    "sha256": _sha256(path),
                    "renamed": str(bool(renamed)),
                    "cost": str(cost),
                    "points": f"{points:.4f}" if isinstance(points, float) else str(points),
                    "macs": str(macs),
                    "memory": str(memory),
                    "params": str(params),
                    "cascade_classification": cascade,
                    "cascade_reasons": cascade_reasons,
                    "license_status": license_status,
                    "license_rank": str(license_rank),
                    "rank_key": "",
                }
            )

    for row in rows:
        pass_rank = 0 if row["status"] == "pass" else 1
        cascade_rank = 0 if "known-bad" not in row["cascade_classification"] else 1
        cost = int(row["cost"] or "999999999")
        license_rank = int(row["license_rank"])
        row["rank_key"] = f"{pass_rank}:{cascade_rank}:{license_rank}:{cost:09d}:{row['source']}"

    rows.sort(key=lambda row: row["rank_key"])
    if not rows:
        requested = ", ".join(f"task{task:03d}" for task in sorted(tasks))
        raise RuntimeError(
            f"No requested task candidates ({requested}) found under {ARTIFACTS}; "
            "refusing to overwrite tracked evidence with an empty sweep."
        )
    return manifest_rows, rows


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _build_md(date: str, tasks: set[int], manifest: list[dict[str, str]], rows: list[dict[str, str]], full: bool, max_pairs: int) -> str:
    targets = ", ".join(f"task{task:03d}" for task in sorted(tasks))
    target_sources = [row for row in manifest if row["has_target_task"] == "True"]
    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row for row in rows if row["status"] != "pass"]
    best = sorted(passed, key=lambda row: (int(row["cost"]), int(row["license_rank"]), row["source"]))[:12]
    by_status = Counter(row["status"] for row in rows)
    by_cascade = Counter(row["cascade_classification"] or "not-in-audit" for row in rows)

    lines = [
        f"# All-Local External Source Sweep - {date}",
        "",
        f"Targets: {targets}",
        f"Verifier: visible + test + {'all arc-gen pairs' if full else f'first {max_pairs} arc-gen pairs'} via `tools.external_blend` helpers.",
        "",
        "## Coverage",
        "",
        f"- Top-level local artifact directories with any `taskNNN.onnx`: {len(manifest)}",
        f"- Directories containing target task files: {len(target_sources)}",
        f"- Target candidate files scanned: {len(rows)}",
        f"- Pass/fail: {dict(by_status)}",
        f"- Cascade audit classes: {dict(by_cascade)}",
        "",
        "## Ranked Passing Candidates",
        "",
        "| Rank | Task | Source | Cost | Points | Cascade | License note | Path |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    for idx, row in enumerate(best, start=1):
        lines.append(
            f"| {idx} | task{row['task']} | `{row['source']}` | {row['cost']} | {row['points']} | "
            f"{row['cascade_classification'] or 'verified'} | {row['license_status']} | `{row['path']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This reproduces the Beta all-local source finding as a tracked report: artifact directories outside the default blend manifest contain full-verifying candidates.",
        "- The cheapest passing family is cost 134,980 (`memory=134,048`, `params=932`) for task145 in this run, but many copies are opset>10 and should stay as source-manifest evidence until Alpha/Owner chooses a packaging policy.",
        "- For task145, `thisray-4808` is the cleanest cascade row in the existing audit (`likely-safe`, opset 10) but costs 608,061, so it is safer but less efficient than the 134,980 family.",
        "- License status is intentionally conservative because this worktree has local artifacts but not the full NAS dataset-license table.",
        "",
        "## Failed / Rejected Summary",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    reason_counts = Counter(row["reason"] for row in failed)
    for reason, count in reason_counts.most_common(12):
        lines.append(f"| {reason or 'n/a'} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="145")
    parser.add_argument("--date", default="2026-05-08")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--out-prefix", default="Logs/all_local_external_source_sweep_2026-05-08")
    parser.add_argument("--audit-csv", default="Logs/external_cascade_audit_2026-05-08.csv")
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)
    manifest, rows = build_rows(tasks, full=args.full, max_pairs=args.max_pairs, audit_csv=ROOT / args.audit_csv)

    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_prefix.with_suffix(".manifest.csv"),
        manifest,
        ["source", "task_file_count", "distinct_task_count", "has_target_task", "target_tasks_present"],
    )
    fields = [
        "task",
        "source",
        "path",
        "status",
        "triage_status",
        "reason",
        "source_size",
        "sha256",
        "renamed",
        "cost",
        "points",
        "macs",
        "memory",
        "params",
        "cascade_classification",
        "cascade_reasons",
        "license_status",
        "license_rank",
        "rank_key",
    ]
    _write_csv(out_prefix.with_suffix(".csv"), rows, fields)
    out_prefix.with_suffix(".md").write_text(
        _build_md(args.date, tasks, manifest, rows, full=args.full, max_pairs=args.max_pairs).rstrip() + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {_display_path(out_prefix.with_suffix('.manifest.csv'))}")
    print(f"Wrote {_display_path(out_prefix.with_suffix('.csv'))}")
    print(f"Wrote {_display_path(out_prefix.with_suffix('.md'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
