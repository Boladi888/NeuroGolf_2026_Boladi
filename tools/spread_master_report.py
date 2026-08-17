#!/usr/bin/env python3
"""Generate the master spread-knowledge report from a scan JSON.

The report rolls up:
  * per-pattern totals (how many tasks the pattern applies to, total bytes saved)
  * top-N tasks by total savings
  * top-N advisory candidates (crop-to-size opportunities)
  * a section per pattern listing all task hits with their cost deltas

Output:
  Research_Archive/notes/spread_knowledge_findings_<UTC date>.md
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "Logs" / "spread_scan_results"
NOTES_DIR = REPO_ROOT / "Research_Archive" / "notes"


def _latest_scan() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    cands = sorted(RESULTS_DIR.glob("spread_scan_*.json"))
    return cands[-1] if cands else None


def _latest_polish_sweep() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    cands = sorted(RESULTS_DIR.glob("polish_sweep_*.csv"))
    return cands[-1] if cands else None


def _load_polish_sweep(csv_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("task_id")
            if not tid:
                continue
            try:
                delta = int(row.get("delta") or 0)
            except ValueError:
                delta = 0
            try:
                orig = int(row.get("orig_cost") or 0)
            except ValueError:
                orig = 0
            try:
                polished = int(row.get("polished_cost") or 0)
            except ValueError:
                polished = 0
            out[tid] = {
                "delta": delta,
                "orig_cost": orig,
                "polished_cost": polished,
                "applied_passes": [p for p in (row.get("applied_passes") or "").split(",") if p],
            }
    return out


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_report(scan_json: Path, polish_csv: Path | None = None) -> str:
    data = json.loads(scan_json.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    scan_date = scan_json.stem.replace("spread_scan_", "")
    polish_data = _load_polish_sweep(polish_csv) if polish_csv and polish_csv.exists() else {}

    # Pattern roll-ups.
    per_pattern_savings: dict[str, list[tuple[str, int]]] = defaultdict(list)
    per_pattern_advisory: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    per_pattern_verdicts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Per-task total savings if all applicable patterns shipped (independent / upper bound).
    per_task_savings_upper: dict[str, int] = {}
    per_task_orig_cost: dict[str, int] = {}

    for t in tasks:
        tid = t["task_id"]
        orig = t.get("original") or {}
        per_task_orig_cost[tid] = int(orig.get("cost", 0) or 0)
        total = 0
        for p in t.get("patterns", []):
            verdict = p.get("verdict", "?")
            per_pattern_verdicts[p["name"]][verdict] += 1
            if verdict == "applied" and p.get("cost_delta") is not None and p["cost_delta"] < 0:
                per_pattern_savings[p["name"]].append((tid, int(p["cost_delta"])))
                total += -int(p["cost_delta"])
            elif verdict == "advisory" and p.get("cost_delta") is not None and p["cost_delta"] < 0:
                per_pattern_advisory[p["name"]].append((tid, int(p["cost_delta"]), p.get("notes", "")))
        per_task_savings_upper[tid] = total

    # True sequential-pipeline savings (lower than the per-pattern sum because
    # onnxsim subsumes many of safe_opt's identities; etc.).
    per_task_savings_true: dict[str, int] = {
        tid: max(0, -info["delta"]) for tid, info in polish_data.items()
    }

    # ---- markdown render ----
    lines: list[str] = []
    lines.append(f"# Spread-knowledge findings — {scan_date}")
    lines.append("")
    try:
        scan_rel = str(scan_json.relative_to(REPO_ROOT))
    except ValueError:
        scan_rel = str(scan_json)
    lines.append(
        f"Generated from `{scan_rel}` via "
        "`tools/spread_master_report.py`. The scan walks every task in "
        "`submissions/submission.zip` and tries every "
        "`tools/spread_patterns/pass_*.py` module against it; this report "
        "rolls up the results."
    )
    lines.append("")

    # Headline totals.
    total_tasks = len(tasks)
    upper_with_savings = sum(1 for v in per_task_savings_upper.values() if v > 0)
    upper_grand_total = sum(per_task_savings_upper.values())
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Tasks scanned: **{total_tasks}**")
    if polish_data:
        true_with_savings = sum(1 for v in per_task_savings_true.values() if v > 0)
        true_total = sum(per_task_savings_true.values())
        lines.append(
            f"- Tasks where the **sequential polish pipeline** "
            f"(`tools/polish_candidate.py`) saves bytes: **{true_with_savings}**"
        )
        lines.append(
            f"- **Real recoverable bytes** if every task is polished: **{true_total:,}** "
            f"(this is the truth — patterns subsume each other when applied in order)"
        )
        lines.append(
            f"- Per-pattern upper-bound sum (each pattern measured INDEPENDENTLY vs original): "
            f"{upper_grand_total:,} across {upper_with_savings} tasks. **This number OVERSTATES** "
            "the achievable savings because onnxsim already covers most of safe_opt's "
            "algebraic identities, etc."
        )
    else:
        lines.append(f"- Tasks with at least one math-equivalent shrink available: **{upper_with_savings}**")
        lines.append(
            f"- Per-pattern upper-bound sum: **{upper_grand_total:,}** "
            "(this is an UPPER BOUND — run `tools/spread_polish_sweep.py` "
            "to get the true sequential-pipeline total)"
        )
    lines.append("")

    # Per-pattern verdict table.
    lines.append("## Pattern verdict breakdown")
    lines.append("")
    lines.append(
        "| Pattern | Applied | No-op | Advisory | Regression | Verify-failed | Score-failed | Error | Skipped | Bytes saved (sum) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for pname in sorted(per_pattern_verdicts.keys()):
        v = per_pattern_verdicts[pname]
        bytes_saved = sum(-d for _, d in per_pattern_savings.get(pname, []))
        lines.append(
            f"| `{pname}` | {v.get('applied', 0)} | {v.get('no-op', 0)} | "
            f"{v.get('advisory', 0)} | {v.get('regression', 0)} | "
            f"{v.get('verify-failed', 0)} | {v.get('score-failed', 0)} | "
            f"{v.get('error', 0)} | {v.get('skipped', 0)} | "
            f"{bytes_saved:,} |"
        )
    lines.append("")

    # Top-20 tasks. Prefer the polish-sweep "true" delta when available,
    # otherwise fall back to the per-pattern upper bound.
    source_label = "sequential polish pipeline" if polish_data else "per-pattern upper bound — overstates real savings"
    per_task_for_topn = per_task_savings_true if polish_data else per_task_savings_upper
    top_n = sorted(per_task_for_topn.items(), key=lambda kv: kv[1], reverse=True)[:20]
    lines.append(f"## Top 20 tasks by recoverable bytes ({source_label})")
    lines.append("")
    lines.append("| Task | Original cost | Recoverable bytes | New cost | Δ points | Passes |")
    lines.append("|---|---:|---:|---:|---:|---|")
    import math
    for tid, saved in top_n:
        if saved == 0:
            continue
        orig_cost = per_task_orig_cost.get(tid, 0)
        polish_entry = polish_data.get(tid, {})
        if polish_data and polish_entry.get("polished_cost"):
            new_cost = polish_entry["polished_cost"]
        else:
            new_cost = max(1, orig_cost - saved)
        orig_pts = max(1.0, 25.0 - math.log(max(1.0, orig_cost)))
        new_pts = max(1.0, 25.0 - math.log(max(1.0, new_cost)))
        dpts = new_pts - orig_pts
        passes = "+".join(polish_entry.get("applied_passes", [])) or "?"
        lines.append(
            f"| {tid} | {orig_cost:,} | -{saved:,} | {new_cost:,} | +{dpts:.4f} | {passes} |"
        )
    lines.append("")

    # Per-pattern detail.
    lines.append("## Per-pattern detail")
    lines.append("")
    for pname in sorted(per_pattern_savings.keys()):
        hits = sorted(per_pattern_savings[pname], key=lambda kv: kv[1])
        if not hits:
            continue
        total = sum(-d for _, d in hits)
        lines.append(f"### `{pname}` — {len(hits)} task(s), {total:,} bytes total")
        lines.append("")
        lines.append("| Task | Δ bytes |")
        lines.append("|---|---:|")
        for tid, d in hits:
            lines.append(f"| {tid} | {d} |")
        lines.append("")

    # Advisory section.
    lines.append("## Advisory opportunities (manual work required)")
    lines.append("")
    for pname in sorted(per_pattern_advisory.keys()):
        hits = sorted(per_pattern_advisory[pname], key=lambda kv: kv[1])
        if not hits:
            continue
        est_total = sum(-d for _, d, _ in hits)
        lines.append(f"### `{pname}` — {len(hits)} task(s), ~{est_total:,} bytes estimated")
        lines.append("")
        lines.append("| Task | Est. Δ | Notes |")
        lines.append("|---|---:|---|")
        for tid, d, n in hits[:30]:
            n_clean = n.replace("|", "\\|")
            if len(n_clean) > 100:
                n_clean = n_clean[:97] + "..."
            lines.append(f"| {tid} | {d} | {n_clean} |")
        if len(hits) > 30:
            lines.append(f"| ... | | ({len(hits) - 30} more) |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**Re-run**: `.venv/Scripts/python.exe tools/spread_scan.py && "
                 ".venv/Scripts/python.exe tools/spread_master_report.py`.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=None, help="Scan JSON path (default: latest)")
    parser.add_argument("--polish-csv", type=Path, default=None,
                        help="polish_sweep_*.csv path for sequential-pipeline truth (default: latest)")
    parser.add_argument("--out", type=Path, default=None, help="Output markdown path")
    args = parser.parse_args(argv)
    scan = args.scan or _latest_scan()
    if scan is None or not scan.exists():
        print(f"No scan JSON found in {RESULTS_DIR}; run tools/spread_scan.py first.")
        return 2
    polish_csv = args.polish_csv if args.polish_csv else _latest_polish_sweep()
    md = build_report(scan, polish_csv=polish_csv)
    out = args.out
    if out is None:
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        out = NOTES_DIR / f"spread_knowledge_findings_{_utc_today()}.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
