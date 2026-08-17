"""Shared helpers for the multi_task_sweeps suite.

Each sweep:
  1. Iterates every taskNNN.onnx in a canonical zip
  2. Applies a pattern detector + transformer
  3. Verifies candidate against LIVE via tools/verify_against_live.py
  4. Scores via tools/spread_score_isolated.py
  5. Returns a list of (task_id, original_cost, candidate_cost, candidate_onnx_bytes, sha256)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DEFAULT = REPO_ROOT / "submissions" / "_canonical_5911_36_backup.zip"
SCORE_SCRIPT = REPO_ROOT / "tools" / "spread_score_isolated.py"
VERIFY_SCRIPT = REPO_ROOT / "tools" / "verify_against_live.py"
LIVE_ZIP = REPO_ROOT / "submissions" / "submission.zip"
OUT_DIR = REPO_ROOT / "Logs" / "multi_task_sweeps"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = sys.executable


@dataclass
class SweepResult:
    task_id: str
    base_cost: int
    candidate_cost: int
    delta: int
    raw_log_e_gain: float
    sha256: str
    candidate_path: str
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def iter_canonical_tasks(zip_path: Path = CANONICAL_DEFAULT) -> Iterable[tuple[str, bytes]]:
    """Yield (task_id, onnx_bytes) for every taskNNN.onnx in the canonical zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for item in zf.infolist():
            name = item.filename
            if name.startswith("task") and name.endswith(".onnx"):
                tid = name[: -len(".onnx")]
                yield tid, zf.read(name)


def score_candidate(onnx_bytes: bytes) -> dict | None:
    """Return score_network_v2 dict, or None on error."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(onnx_bytes)
        path = f.name
    try:
        r = subprocess.run(
            [str(PYTHON_EXE), str(SCORE_SCRIPT), path],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        line = r.stdout.strip().split("\n")[0] if r.stdout else ""
        if not line:
            return None
        try:
            d = json.loads(line)
            if "error" in d:
                return None
            return d
        except json.JSONDecodeError:
            return None
    finally:
        try: os.unlink(path)
        except OSError: pass


def verify_candidate(task_id: str, onnx_bytes: bytes) -> tuple[bool, str]:
    """Run verify_against_live; return (passed, summary)."""
    tidnum = int(task_id.replace("task", ""))
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(onnx_bytes)
        path = f.name
    try:
        r = subprocess.run(
            [str(PYTHON_EXE), str(VERIFY_SCRIPT),
             "--proto", path, "--tid", str(tidnum), "--live-zip", str(LIVE_ZIP)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        out = (r.stdout or "") + (r.stderr or "")
        # Look for "OK Prototype matches LIVE" success line
        passed = "OK Prototype matches LIVE" in out
        # Extract the proto-vs-LIVE line
        summary = ""
        for line in out.split("\n"):
            if "Prototype  vs LIVE" in line:
                summary = line.strip()
                break
        return passed, summary
    finally:
        try: os.unlink(path)
        except OSError: pass


_BASE_COSTS = None


def load_base_costs() -> dict[str, int]:
    """Load baseline cost-per-task from task_state.csv.

    The canonical submission.zip's extracted ONNXs sometimes fail score_network_v2
    due to stripped value_info; task_state.csv has the authoritative current costs.
    """
    global _BASE_COSTS
    if _BASE_COSTS is not None:
        return _BASE_COSTS
    _BASE_COSTS = {}
    import csv
    path = REPO_ROOT / "The_400_Tasks" / "task_state.csv"
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("task", "")
            try:
                cost = int(row.get("current_cost", "") or 0)
            except ValueError:
                cost = 0
            if tid and cost > 0:
                _BASE_COSTS[tid] = cost
    return _BASE_COSTS


def sweep_task(task_id: str, base_onnx: bytes, transformer, sweep_name: str) -> SweepResult | None:
    """Apply transformer to base_onnx; verify; score; return SweepResult on success."""
    try:
        base_model = onnx.load_model_from_string(base_onnx)
    except Exception:
        return None
    try:
        new_model = transformer(base_model)
    except Exception:
        return None
    if new_model is None:
        return None
    try:
        new_bytes = new_model.SerializeToString()
    except Exception:
        return None
    # Score candidate (this works; base comes from task_state.csv)
    cand_score = score_candidate(new_bytes)
    if cand_score is None:
        return None
    base_costs = load_base_costs()
    base_cost = base_costs.get(task_id, 0)
    if base_cost == 0:
        return None  # Unknown base
    if cand_score["cost"] >= base_cost:
        return None  # No savings
    # Verify against LIVE
    passed, summary = verify_candidate(task_id, new_bytes)
    if not passed:
        return None
    delta = cand_score["cost"] - base_cost
    import math
    raw_gain = math.log(base_cost / cand_score["cost"]) if cand_score["cost"] > 0 else 0.0
    sha = hashlib.sha256(new_bytes).hexdigest()
    out_path = OUT_DIR / f"{sweep_name}_{task_id}_{sha[:8]}.onnx"
    out_path.write_bytes(new_bytes)
    return SweepResult(
        task_id=task_id,
        base_cost=base_cost,
        candidate_cost=cand_score["cost"],
        delta=delta,
        raw_log_e_gain=raw_gain,
        sha256=sha,
        candidate_path=str(out_path),
        notes=summary,
    )


def write_report(sweep_name: str, results: list[SweepResult]):
    """Write JSON + markdown summary to Logs/multi_task_sweeps/."""
    json_path = OUT_DIR / f"{sweep_name}_results.json"
    md_path = OUT_DIR / f"{sweep_name}_results.md"
    total_delta = sum(r.delta for r in results)
    total_gain = sum(r.raw_log_e_gain for r in results)
    json_path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    md_lines = [
        f"# {sweep_name} — {len(results)} winning candidates",
        "",
        f"**Total cost reduction**: {total_delta:,} bytes",
        f"**Total predicted LB gain (additive raw log_e)**: +{total_gain:.2f} LB",
        "",
        "| task | base | cand | delta | LB | sha |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in sorted(results, key=lambda x: x.raw_log_e_gain, reverse=True):
        md_lines.append(
            f"| {r.task_id} | {r.base_cost:,} | {r.candidate_cost:,} | "
            f"{r.delta:,} | +{r.raw_log_e_gain:.3f} | {r.sha256[:8]} |"
        )
    md_path.write_text("\n".join(md_lines))
    print(f"Report: {md_path}")
    print(f"Total: {len(results)} candidates, {total_delta:,} bytes saved, +{total_gain:.2f} LB predicted")
