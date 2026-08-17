"""batch_adversarial — run adversarial_test on multiple prototypes, produce shippability table.

Iterates over a list of (tid, prototype_path) pairs, runs adversarial test on
each, and produces a single table with:
  - prototype file
  - adversarial match %
  - cost
  - shippable verdict (100% match AND cost < live cost)

Usage:
    python tools/batch_adversarial.py
    python tools/batch_adversarial.py --tid 4,161,245
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
import tempfile
import math
from pathlib import Path
from zipfile import ZipFile


os.environ['ORT_LOGGING_LEVEL'] = '4'
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / 'tools'))


DEFAULT_LIVE_ZIP = Path(r"<NOGIT_ROOT>\submissions\alpha_b_t287v4_5653_67\submission.zip")
TASK_BY_TASK = REPO / "The_400_Tasks" / "Task_by_Task"


def find_prototypes(tids: list[int]) -> list[tuple[int, Path]]:
    """For each tid, find all .onnx prototypes in Task_by_Task/taskNNN/prototypes/."""
    out = []
    for tid in tids:
        proto_dir = TASK_BY_TASK / f"task{tid:03d}" / "prototypes"
        if not proto_dir.exists():
            continue
        for f in sorted(proto_dir.iterdir()):
            if f.suffix == ".onnx":
                out.append((tid, f))
    return out


def get_live_cost(tid: int, live_zip: Path) -> int | None:
    import _score_v2 as scv2
    import onnxruntime as ort
    ort.set_default_logger_severity(4)
    name = f"task{tid:03d}.onnx"
    with ZipFile(live_zip) as z:
        if name not in z.namelist():
            return None
        blob = z.read(name)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(blob); tmp.close()
    try:
        r = scv2.score_network_v2(tmp.name)
        if r:
            return (r[1] or 0) + (r[2] or 0)
    except Exception:
        pass
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass
    return None


def get_proto_cost(proto: Path) -> int | None:
    import _score_v2 as scv2
    import onnxruntime as ort
    ort.set_default_logger_severity(4)
    try:
        r = scv2.score_network_v2(str(proto))
        if r:
            return (r[1] or 0) + (r[2] or 0)
    except Exception:
        pass
    return None


def run_adversarial(proto: Path, tid: int) -> tuple[int, int]:
    """Return (match, total) on adversarial test."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "adversarial_test.py"),
         "--proto", str(proto), "--tid", str(tid), "--max-perturbations", "2"],
        capture_output=True, text=True, timeout=300,
    )
    # Parse "Aggregate:  X/Y (Z%)" line
    for line in proc.stdout.splitlines():
        if line.startswith("Aggregate:"):
            # "Aggregate:  X/Y (Z%)"
            parts = line.split()
            if len(parts) >= 2:
                a, b = parts[1].split('/')
                return int(a), int(b)
    return 0, 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tid", type=str, default="",
                   help="Comma-separated TIDs. Empty = all HIDDEN_BAD_TIDS.")
    p.add_argument("--live-zip", type=Path, default=DEFAULT_LIVE_ZIP)
    args = p.parse_args(argv)

    if args.tid:
        tids = [int(t.strip()) for t in args.tid.split(",") if t.strip()]
    else:
        tids = sorted([4, 22, 161, 245, 260, 270, 280, 383, 396])

    pairs = find_prototypes(tids)
    if not pairs:
        print(f"No prototypes found for tids {tids}")
        return 0

    live_costs = {tid: get_live_cost(tid, args.live_zip) for tid in tids}

    print(f"{'task':<8} {'prototype':<45} {'match':>11} {'cost':>10} {'live_cost':>10} {'verdict':<25}")
    for tid, proto in pairs:
        match, total = run_adversarial(proto, tid)
        rate_str = f"{match}/{total}" if total else "?"
        rate_pct = (match / total * 100) if total else 0.0
        cost = get_proto_cost(proto)
        cost_str = f"{cost:,}" if cost else "?"
        live = live_costs.get(tid)
        live_str = f"{live:,}" if live else "?"
        # Verdict
        if total and match == total:
            if cost and live and cost < live:
                verdict = f"SHIPPABLE (+{math.log(live/cost):.2f} pts)"
            elif cost and live:
                verdict = f"safe but worse cost"
            else:
                verdict = "100% match (cost?)"
        elif rate_pct >= 99:
            verdict = f"close ({rate_pct:.1f}%)"
        else:
            verdict = f"divergent ({rate_pct:.1f}%)"
        print(f"task{tid:03d}  {proto.name:<45} {rate_str:>11} {cost_str:>10} {live_str:>10} {verdict:<25}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
