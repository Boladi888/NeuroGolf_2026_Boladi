#!/usr/bin/env python3
"""Audit external ONNX artifacts for NeuroGolf cascade risk."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import onnx
from google.protobuf.message import DecodeError

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from cost_predictor import predict  # noqa: E402


DEFAULT_EXTERNALS_DIR = ROOT / "data" / "competitor_artifacts"
DEFAULT_OUT = ROOT / "Logs" / "external_cascade_audit.csv"
DEFAULT_MD = ROOT / "Logs" / "external_cascade_audit.md"
DEFAULT_THRESHOLD = 1_000_000_000
DEFAULT_HIGH_MEMORY = 750_000
CASCADE_FLOOR = 1_000
FILE_SIZE_CAP = 1_509_949
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}


def _task_num(path: Path) -> int | None:
    match = re.search(r"task(\d{1,3})\.onnx$", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _source_name(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name
    parts = rel.parts
    if len(parts) <= 1:
        return root.name
    return parts[0]


def _find_onnx_files(roots: list[Path], limit: int | None) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".onnx":
            found.append((root.parent, root))
        elif root.is_dir():
            for path in sorted(root.rglob("task*.onnx")):
                found.append((root, path))
                if limit is not None and len(found) >= limit:
                    return found
        else:
            print(f"warning: missing external path: {root}", file=sys.stderr)
    return found


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _static_scan(path: Path) -> dict[str, str]:
    try:
        model = onnx.load(str(path))
    except (OSError, DecodeError, ValueError) as exc:
        return {
            "static_error": f"load: {exc}",
            "opset": "",
            "domains": "",
            "banned_ops": "",
            "op_count": "",
            "has_functions": "",
            "has_custom_domain": "",
        }

    opsets = []
    domains = []
    for item in model.opset_import:
        domain = item.domain or ""
        domains.append(domain or "ai.onnx")
        if domain in ALLOWED_DOMAINS:
            opsets.append(item.version)

    op_types = [node.op_type.upper() for node in model.graph.node]
    banned = sorted(set(op_types) & BANNED_OPS)
    custom_domains = sorted(
        {
            node.domain
            for node in model.graph.node
            if node.domain and node.domain not in ALLOWED_DOMAINS
        }
    )
    return {
        "static_error": "",
        "opset": str(max(opsets) if opsets else ""),
        "domains": ";".join(sorted(set(domains))),
        "banned_ops": ";".join(banned),
        "op_count": str(len(op_types)),
        "has_functions": str(bool(model.functions)),
        "has_custom_domain": str(bool(custom_domains)),
    }


def _profile_v2(path: Path) -> dict[str, str]:
    try:
        from _score_v2 import score_network_v2
    except ModuleNotFoundError as exc:
        return {
            "profile_status": f"unavailable:{exc.name}",
            "profile_macs": "",
            "profile_memory": "",
            "profile_params": "",
            "profile_cost": "",
        }
    result = score_network_v2(str(path))
    if result is None:
        return {
            "profile_status": "rejected",
            "profile_macs": "",
            "profile_memory": "",
            "profile_params": "",
            "profile_cost": "",
        }
    macs, memory, params = result
    return {
        "profile_status": "ok",
        "profile_macs": str(int(macs)),
        "profile_memory": str(int(memory)),
        "profile_params": str(int(params)),
        "profile_cost": str(int(memory + params)),
    }


def _current_cost(memory: int, params: int) -> int:
    return int(memory + params)


def _current_points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, cost)))


def _classify(row: dict[str, str], threshold: int, high_memory: int) -> tuple[str, str]:
    reasons: list[str] = []
    if row["static_error"]:
        reasons.append("static_load_error")
    if row["banned_ops"]:
        reasons.append(f"banned_ops:{row['banned_ops']}")
    if row["has_functions"] == "True":
        reasons.append("model_functions")
    if row["has_custom_domain"] == "True":
        reasons.append("custom_domain")
    opset = int(row["opset"]) if row["opset"].isdigit() else 0
    if opset > 10:
        reasons.append(f"opset>{10}")
    size = int(row["file_size"])
    if size > FILE_SIZE_CAP:
        reasons.append("file_size_cap")
    if row["predict_error"]:
        reasons.append("predict_error")
    memory = int(row["memory"] or 0)
    params = int(row["params"] or 0)
    cost = _current_cost(memory, params)
    cumulative = int(row["source_cumulative_memory"] or 0)
    if cost < CASCADE_FLOOR and not row["predict_error"]:
        reasons.append("near_zero_cost")
    if memory >= high_memory:
        reasons.append("high_memory")
    if cumulative >= threshold:
        reasons.append("source_cumulative_threshold")

    if any(
        reason in reasons
        for reason in ("static_load_error", "model_functions", "custom_domain", "file_size_cap", "predict_error")
    ) or row["banned_ops"]:
        return "known-bad", ";".join(reasons)
    if reasons:
        return "suspect", ";".join(reasons)
    return "likely-safe", ""


def build_audit(
    roots: list[Path],
    threshold: int,
    high_memory: int,
    profile_limit: int,
    limit: int | None,
) -> list[dict[str, str]]:
    files = _find_onnx_files(roots, limit)
    rows: list[dict[str, str]] = []
    for root, path in files:
        pred = predict(path)
        row = {
            "source": _source_name(root, path),
            "task": f"{_task_num(path):03d}" if _task_num(path) is not None else "",
            "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "file_size": str(path.stat().st_size),
            "sha256": _sha256(path),
        }
        row.update(_static_scan(path))
        if pred.get("error"):
            row.update(
                {
                    "predict_error": str(pred["error"]),
                    "macs": "",
                    "memory": "",
                    "params": "",
                    "cost": "",
                    "points": "",
                    "cascade_risk": "",
                    "ops": "",
                }
            )
        else:
            memory = int(pred["memory"])
            params = int(pred["params"])
            cost = _current_cost(memory, params)
            row.update(
                {
                    "predict_error": "",
                    "macs": str(pred["macs"]),
                    "memory": str(memory),
                    "params": str(params),
                    "cost": str(cost),
                    "points": f"{_current_points(cost):.4f}",
                    "cascade_risk": str(cost < CASCADE_FLOOR),
                    "ops": ";".join(f"{k}:{v}" for k, v in pred["ops"].items()),
                }
            )
        row.update(
            {
                "source_cumulative_memory": "0",
                "source_cumulative_flag": "False",
                "classification": "",
                "reasons": "",
                "profile_status": "not_run",
                "profile_macs": "",
                "profile_memory": "",
                "profile_params": "",
                "profile_cost": "",
                "mini_dry_run": "not_run",
            }
        )
        rows.append(row)

    cumulative_by_source: dict[str, int] = defaultdict(int)
    for row in sorted(rows, key=lambda r: (r["source"], r["task"], r["path"])):
        memory = int(row["memory"] or 0)
        cumulative_by_source[row["source"]] += memory
        row["source_cumulative_memory"] = str(cumulative_by_source[row["source"]])
        row["source_cumulative_flag"] = str(cumulative_by_source[row["source"]] >= threshold)

    for row in rows:
        classification, reasons = _classify(row, threshold=threshold, high_memory=high_memory)
        row["classification"] = classification
        row["reasons"] = reasons

    if profile_limit > 0:
        candidates = [
            row
            for row in rows
            if row["classification"] in {"suspect", "known-bad"} and not row["predict_error"]
        ]
        candidates.sort(key=lambda r: int(r["memory"] or 0), reverse=True)
        for row in candidates[:profile_limit]:
            path = ROOT / row["path"] if not Path(row["path"]).is_absolute() else Path(row["path"])
            row.update(_profile_v2(path))
    return rows


def _write_outputs(rows: list[dict[str, str]], csv_path: Path, md_path: Path) -> None:
    def clean(value: object) -> str:
        text = str(value)
        return " ".join(text.split())

    fields = [
        "source",
        "task",
        "classification",
        "reasons",
        "path",
        "file_size",
        "sha256",
        "opset",
        "domains",
        "banned_ops",
        "op_count",
        "has_functions",
        "has_custom_domain",
        "static_error",
        "predict_error",
        "macs",
        "memory",
        "params",
        "cost",
        "points",
        "cascade_risk",
        "source_cumulative_memory",
        "source_cumulative_flag",
        "profile_status",
        "profile_macs",
        "profile_memory",
        "profile_params",
        "profile_cost",
        "mini_dry_run",
        "ops",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [{field: clean(row.get(field, "")) for field in fields} for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    counts = Counter(row["classification"] for row in csv_rows)
    source_counts = Counter(row["source"] for row in csv_rows)
    suspect_rows = [row for row in csv_rows if row["classification"] != "likely-safe"]
    suspect_rows.sort(key=lambda r: (r["classification"], -(int(r["memory"] or 0)), r["source"], r["task"]))

    md_lines = [
        "# External Cascade Audit",
        "",
        f"Rows: {len(rows)}",
        "",
        "## Classification Counts",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    for name in ("likely-safe", "suspect", "known-bad"):
        md_lines.append(f"| {name} | {counts.get(name, 0)} |")

    md_lines += [
        "",
        "## Sources",
        "",
        "| Source | ONNX files |",
        "|---|---:|",
    ]
    for source, count in sorted(source_counts.items()):
        md_lines.append(f"| {source} | {count} |")

    md_lines += [
        "",
        "## Non-Safe Rows",
        "",
        "| Source | Task | Class | Memory | Cost | Reasons |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in suspect_rows[:100]:
        md_lines.append(
            "| {source} | {task} | {classification} | {memory} | {cost} | {reasons} |".format(
                **row
            )
        )
    if len(suspect_rows) > 100:
        md_lines.append(f"| ... | ... | ... | ... | ... | {len(suspect_rows) - 100} additional rows in CSV |")

    md_lines += [
        "",
        "## Notes",
        "",
        "- `cost`, `points`, and `cascade_risk` use the current NeuroGolf scoring basis: `memory + params`. `macs` is retained as a separate audit field only.",
        "- `known-bad` means the model is statically rejected, over the file cap, or could not be predicted.",
        "- `suspect` means the model may still verify locally but deserves isolation because of near-zero cost, high memory, or source cumulative memory.",
        "- `mini_dry_run` is a planning column; Kaggle-side dry runs are not automated here.",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--externals-dir", action="append", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--high-memory", type=int, default=DEFAULT_HIGH_MEMORY)
    parser.add_argument("--profile-limit", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    roots = args.externals_dir or [DEFAULT_EXTERNALS_DIR]
    rows = build_audit(
        roots=[path.resolve() for path in roots],
        threshold=args.threshold,
        high_memory=args.high_memory,
        profile_limit=args.profile_limit,
        limit=args.limit,
    )
    if not rows:
        print("No task*.onnx files found. Provide --externals-dir inside this worktree.", file=sys.stderr)
        return 1
    _write_outputs(rows, args.out.resolve(), args.markdown.resolve())
    print(f"Wrote {args.out}")
    print(f"Wrote {args.markdown}")
    print(", ".join(f"{k}={v}" for k, v in sorted(Counter(row["classification"] for row in rows).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
