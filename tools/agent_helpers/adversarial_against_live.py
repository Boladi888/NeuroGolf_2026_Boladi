"""Generate synthetic adversarial inputs + compare your candidate to LIVE.

This is the ANTI-silent-fail-trap check. visible+arc-gen verification is
not sufficient (see Algorithm_Library/rule_discovery_methodology.md step 5):
task008 numpy ref passed 266/266 visible but only 3/86 adversarial.

Usage:
    python tools/agent_helpers/adversarial_against_live.py \
        --candidate path/to/candidate.onnx \
        --tid 110 \
        --n 50
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_ZIP = REPO_ROOT / "submissions" / "submission.zip"


def encode_grid(grid):
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < 10 and r < 30 and c < 30:
                arr[0, color, r, c] = 1.0
    return arr


def synthetic_input(seed, h_range=(3, 25), w_range=(3, 25), bg_prob=0.5, n_colors=5):
    rng = np.random.RandomState(seed)
    h = rng.randint(*h_range)
    w = rng.randint(*w_range)
    colors = rng.choice(np.arange(1, 10), size=n_colors, replace=False)
    grid = []
    for r in range(h):
        row = []
        for c in range(w):
            if rng.random() < bg_prob:
                row.append(0)
            else:
                row.append(int(rng.choice(colors)))
        grid.append(row)
    return grid


def _task092_no_same_orientation_overlap(grid):
    a = np.array(grid, dtype=np.int64)
    h, w = a.shape
    pos = {}
    for r in range(h):
        for c in range(w):
            color = int(a[r, c])
            if color:
                pos.setdefault(color, []).append((r, c))
    hmask = np.zeros((h, w), dtype=np.int16)
    vmask = np.zeros((h, w), dtype=np.int16)
    for cells in pos.values():
        (r1, c1), (r2, c2) = cells
        if r1 == r2:
            hmask[r1, min(c1, c2):max(c1, c2) + 1] += 1
        elif c1 == c2:
            vmask[min(r1, r2):max(r1, r2) + 1, c1] += 1
    return hmask.max(initial=0) <= 1 and vmask.max(initial=0) <= 1


def synthetic_task092_input(seed, no_same_orientation_overlap=True):
    """Task092-shaped: five non-bg colors, each appears twice, aligned H/V.

    Official visible+arc-gen Task092 inputs also have no same-orientation line
    overlaps; keep that default so this is a task-shaped hidden proxy rather
    than a generic off-contract stress test.
    """
    rng = np.random.RandomState(seed)
    shapes = np.array([10, 20, 30])
    for _ in range(10000):
        h = int(rng.choice(shapes))
        w = int(rng.choice(shapes))
        grid = np.zeros((h, w), dtype=np.int64)
        occupied = set()
        colors = rng.choice(np.arange(1, 10), size=5, replace=False)
        ok = True
        for color in colors:
            placed = False
            for _try in range(1000):
                if rng.random_sample() < 0.5:
                    r = int(rng.randint(0, h))
                    c1, c2 = rng.choice(np.arange(w), size=2, replace=False)
                    cells = [(r, int(c1)), (r, int(c2))]
                else:
                    c = int(rng.randint(0, w))
                    r1, r2 = rng.choice(np.arange(h), size=2, replace=False)
                    cells = [(int(r1), c), (int(r2), c)]
                if all(cell not in occupied for cell in cells):
                    for r, c in cells:
                        occupied.add((r, c))
                        grid[r, c] = int(color)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and (not no_same_orientation_overlap or _task092_no_same_orientation_overlap(grid)):
            return grid.tolist()
    raise RuntimeError("failed to sample a task092-shaped grid")


def extract_live_onnx(live_zip, tid):
    with zipfile.ZipFile(live_zip, "r") as z:
        return z.read(f"{tid}.onnx")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True, help="Path to your candidate ONNX")
    p.add_argument("--tid", required=True, help="Task ID (e.g. 110 or task110)")
    p.add_argument("--live-zip", default=str(DEFAULT_LIVE_ZIP))
    p.add_argument("--n", type=int, default=50, help="Number of synthetic inputs")
    p.add_argument("--bg-prob", type=float, default=0.5, help="Background probability per cell")
    p.add_argument("--n-colors", type=int, default=5, help="Number of distinct fg colors per input")
    p.add_argument(
        "--sampler",
        choices=["generic", "task092-pairs", "task092-pairs-overlap"],
        default="generic",
        help="Synthetic input family. task092-pairs is the task-shaped hidden proxy for task092.",
    )
    args = p.parse_args()

    tid = args.tid if args.tid.startswith("task") else f"task{int(args.tid):03d}"

    # Load both models
    cand_sess = ort.InferenceSession(args.candidate, providers=["CPUExecutionProvider"])
    live_bytes = extract_live_onnx(args.live_zip, tid)
    live_path = Path(f"./_tmp_live_{tid}.onnx")
    live_path.write_bytes(live_bytes)
    live_sess = ort.InferenceSession(str(live_path), providers=["CPUExecutionProvider"])

    agree = disagree = candidate_only = live_only = both_fail = 0
    sample_disagreements = []

    print(f"Running {args.n} synthetic inputs against {tid}...")
    for seed in range(args.n):
        if args.sampler == "task092-pairs":
            grid = synthetic_task092_input(seed, no_same_orientation_overlap=True)
        elif args.sampler == "task092-pairs-overlap":
            grid = synthetic_task092_input(seed, no_same_orientation_overlap=False)
        else:
            grid = synthetic_input(seed, bg_prob=args.bg_prob, n_colors=args.n_colors)
        inp = encode_grid(grid)
        try:
            cand_out = cand_sess.run(["output"], {"input": inp})[0]
        except Exception:
            cand_out = None
        try:
            live_out = live_sess.run(["output"], {"input": inp})[0]
        except Exception:
            live_out = None

        if cand_out is None and live_out is None:
            both_fail += 1
            continue
        if cand_out is None:
            live_only += 1
            continue
        if live_out is None:
            candidate_only += 1
            continue

        # Apply (out > 0.0) to mimic NeuroGolf's run_network decoder
        cand_mask = (cand_out > 0.0).astype(np.float32)
        live_mask = (live_out > 0.0).astype(np.float32)

        if np.array_equal(cand_mask, live_mask):
            agree += 1
        else:
            disagree += 1
            if len(sample_disagreements) < 3:
                sample_disagreements.append({
                    "seed": seed,
                    "grid_shape": (len(grid), len(grid[0]) if grid else 0),
                    "cand_sum": float(cand_mask.sum()),
                    "live_sum": float(live_mask.sum()),
                    "diff_cells": int(np.abs(cand_mask - live_mask).sum()),
                })

    total = args.n
    print(f"\n=== Adversarial agreement: {agree}/{total} ({100*agree/total:.1f}%) ===")
    print(f"  agree: {agree}")
    print(f"  disagree: {disagree}")
    print(f"  candidate runs, live fails: {candidate_only}")
    print(f"  live runs, candidate fails: {live_only}")
    print(f"  both fail: {both_fail}")

    if sample_disagreements:
        print(f"\n=== Sample disagreements (top 3) ===")
        for s in sample_disagreements:
            print(f"  seed={s['seed']:>4}  grid={s['grid_shape']}  cand_sum={s['cand_sum']}  live_sum={s['live_sum']}  diff_cells={s['diff_cells']}")

    agreement = agree / total if total else 0
    print(f"\n=== Verdict ===")
    if agreement >= 0.95:
        print("PASS: >=95% agreement. Candidate is likely correct on hidden inputs.")
        rc = 0
    elif agreement >= 0.30:
        print(f"BORDERLINE: 30-95% agreement. Candidate is in the safe-to-ship band but investigate disagreements.")
        rc = 0
    else:
        print(f"FAIL: <30% agreement. Do NOT ship — on-contract behavior is at risk.")
        rc = 1

    live_path.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
