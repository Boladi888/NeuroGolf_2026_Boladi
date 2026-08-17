"""adversarial_test — generate synthetic edge-case inputs to detect
prototype-vs-live algorithm divergence.

The lesson from 2026-05-11: 100% visible+arc-gen verify is NOT sufficient for
hidden-correct. Algorithms can agree on visible inputs while diverging on
hidden ones. This tool synthesizes adversarial inputs (perturbations of
visible examples) and compares prototype output vs LIVE output. If they
diverge on ANY adversarial input, the prototype likely has a different
algorithm than the live and would be hidden-bad if shipped.

The LIVE ONNX is the source of truth — it's confirmed hidden-correct
(it's the one currently scoring well on Kaggle). We treat its outputs on
synthetic inputs as ground truth for adversarial testing.

Perturbations applied (per visible input):
  1. color_swap         — swap two non-bg colors in input
  2. color_permutation  — randomly permute the active color set
  3. rotate_90/180/270  — rotate the grid
  4. flip_h / flip_v    — reflect horizontally/vertically
  5. shift_h / shift_v  — shift the visible content by a small offset
  6. add_isolated_cell  — add a single cell of an existing color at an unused location
  7. shrink_grid        — clip the active region by 1 row/col
  8. empty_quadrant     — zero out one quadrant

Each perturbation produces a new input. Both prototype and live are run on
that input. Outputs are compared cell-by-cell on the inner grid.

Output:
  - For each perturbation type: prototype vs live match count
  - Aggregate: total adversarial cases / total matches
  - Worst perturbation type (highest divergence rate)

Usage:
    python tools/adversarial_test.py --proto path.onnx --tid 4
    python tools/adversarial_test.py --proto path.onnx --tid 4 --max-perturbations 50
    python tools/adversarial_test.py --proto path.onnx --tid 4 --first-fail
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

os.environ['ORT_LOGGING_LEVEL'] = '4'
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / 'tools'))
NOGIT_ROOT = Path(os.environ.get("NEUROGOLF_NOGIT_ROOT", REPO / ".notongit" / "NeuroGolf_nogit"))

import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(4)


DEFAULT_LIVE = NOGIT_ROOT / "submissions" / "alpha_b_t287v4_5653_67" / "submission.zip"
DEFAULT_DATA = Path(os.environ.get("NEUROGOLF_DATA_DIR", REPO / "data" / "competitions" / "neurogolf-2026"))


# ===== Encoding (matches the Kaggle scoring convention) =====

def encode(grid):
    a = np.array(grid, dtype=np.int64)
    h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    out[0, 0, h:, :] = 1.0
    out[0, 0, :h, w:] = 1.0
    return out


def decode(arr, h, w):
    # 2026-06-06 fix (task055 worker): threshold at >0 like the REAL decoder
    # (neurogolf_utils.run_network: (result > 0.0).astype(float)). The old
    # np.isclose(vec, 1.0) silently zeroed any model whose set-cells aren't
    # ~1.0 (BOOL True passed by luck; U8 bitmask / F16 staircase outputs
    # scored 0/all despite being 200k byte-equal under the true pipeline).
    grid = []
    for r in range(h):
        row = []
        for c in range(w):
            vec = arr[0, :, r, c]
            idxs = np.where(vec > 0.0)[0]
            if len(idxs) == 1:
                row.append(int(idxs[0]))
            elif len(idxs) == 0:
                row.append(0)
            else:
                row.append(11)
        grid.append(row)
    return grid


# ===== Perturbations =====
# Each takes (input_grid as np.ndarray) and returns a list of perturbed grids.

def perturb_color_swap(grid: np.ndarray) -> list[np.ndarray]:
    """Swap each pair of non-bg colors present in grid."""
    colors = sorted(set(grid.flatten().tolist()) - {0})
    results = []
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            new = grid.copy()
            new[grid == a] = b
            new[grid == b] = a
            results.append(new)
    return results


def perturb_color_permutation(grid: np.ndarray, seed: int = 42) -> list[np.ndarray]:
    """Apply a deterministic permutation of the non-bg colors."""
    rng = np.random.default_rng(seed)
    colors = sorted(set(grid.flatten().tolist()) - {0})
    if len(colors) < 2:
        return []
    # Try 2 permutations
    results = []
    for _ in range(2):
        perm = rng.permutation(colors)
        mapping = dict(zip(colors, perm))
        new = grid.copy()
        for c in colors:
            new[grid == c] = mapping[c]
        results.append(new)
    return results


def perturb_rotate(grid: np.ndarray) -> list[np.ndarray]:
    """Rotate by 90/180/270 degrees."""
    return [np.rot90(grid, k=k) for k in (1, 2, 3)]


def perturb_flip(grid: np.ndarray) -> list[np.ndarray]:
    return [np.flip(grid, axis=0), np.flip(grid, axis=1)]


def perturb_shift(grid: np.ndarray) -> list[np.ndarray]:
    """Shift content by ±1 row/col with bg fill."""
    results = []
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new = np.zeros_like(grid)
        H, W = grid.shape
        for r in range(H):
            for c in range(W):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    new[nr, nc] = grid[r, c]
        results.append(new)
    return results


def perturb_add_isolated_cell(grid: np.ndarray, seed: int = 42) -> list[np.ndarray]:
    """Add a single cell of an existing color at a currently-bg location."""
    colors = sorted(set(grid.flatten().tolist()) - {0})
    if not colors:
        return []
    bg_cells = np.argwhere(grid == 0)
    if len(bg_cells) == 0:
        return []
    rng = np.random.default_rng(seed)
    results = []
    # Pick up to 3 random bg cells, each with one of the existing colors
    for _ in range(min(3, len(colors))):
        idx = rng.integers(len(bg_cells))
        r, c = bg_cells[idx]
        color = rng.choice(colors)
        new = grid.copy()
        new[r, c] = color
        results.append(new)
    return results


def perturb_shrink_grid(grid: np.ndarray) -> list[np.ndarray]:
    """Clip the grid by 1 row/col on each side (if size > 4)."""
    H, W = grid.shape
    if H <= 4 or W <= 4:
        return []
    return [
        grid[1:, :],   # drop top row
        grid[:-1, :],  # drop bottom row
        grid[:, 1:],   # drop left col
        grid[:, :-1],  # drop right col
    ]


def perturb_empty_quadrant(grid: np.ndarray) -> list[np.ndarray]:
    """Zero out each of 4 quadrants."""
    H, W = grid.shape
    if H < 4 or W < 4:
        return []
    mh, mw = H // 2, W // 2
    results = []
    for q in range(4):
        new = grid.copy()
        r_slice = slice(0, mh) if q < 2 else slice(mh, H)
        c_slice = slice(0, mw) if q % 2 == 0 else slice(mw, W)
        new[r_slice, c_slice] = 0
        results.append(new)
    return results


PERTURBATIONS = {
    'color_swap': perturb_color_swap,
    'color_permutation': perturb_color_permutation,
    'rotate': perturb_rotate,
    'flip': perturb_flip,
    'shift': perturb_shift,
    'add_isolated_cell': perturb_add_isolated_cell,
    'shrink_grid': perturb_shrink_grid,
    'empty_quadrant': perturb_empty_quadrant,
}


# ===== Adversarial comparison loop =====

def run_session(sess, grid_np: np.ndarray) -> np.ndarray:
    """Run an ONNX session on a single grid; return DECODED output as np.array."""
    H, W = grid_np.shape
    inp = encode(grid_np.tolist())
    try:
        out = sess.run(['output'], {'input': inp})[0]
    except Exception:
        return None
    decoded = decode(out, H, W)
    return np.array(decoded)


def adversarial_test(proto_path: str, tid: int, live_zip: Path, data_dir: Path,
                     max_perturbations_per_pair: int | None = None,
                     first_fail: bool = False) -> dict:
    # Load live ONNX
    with ZipFile(live_zip) as z:
        live_blob = z.read(f"task{tid:03d}.onnx")
    live_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    live_tmp.write(live_blob)
    live_tmp.close()
    try:
        proto_sess = ort.InferenceSession(proto_path, providers=["CPUExecutionProvider"])
        live_sess = ort.InferenceSession(live_tmp.name, providers=["CPUExecutionProvider"])
    finally:
        try:
            os.unlink(live_tmp.name)
        except OSError:
            pass

    data = json.loads((data_dir / f"task{tid:03d}.json").read_text())
    pairs = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])

    results = {name: {"total": 0, "match": 0, "first_fail": None}
               for name in PERTURBATIONS}
    base_match = 0
    base_total = 0

    for pi, pair in enumerate(pairs):
        inp_g = np.array(pair["input"])
        if inp_g.shape[0] > 30 or inp_g.shape[1] > 30:
            continue

        # Base sanity (no perturbation)
        a = run_session(proto_sess, inp_g)
        b = run_session(live_sess, inp_g)
        base_total += 1
        if a is not None and b is not None and a.shape == b.shape and (a == b).all():
            base_match += 1

        # Now perturb
        for name, fn in PERTURBATIONS.items():
            perturbed_list = fn(inp_g)
            if max_perturbations_per_pair:
                perturbed_list = perturbed_list[:max_perturbations_per_pair]
            for j, pg in enumerate(perturbed_list):
                if pg.shape[0] > 30 or pg.shape[1] > 30:
                    continue
                a = run_session(proto_sess, pg)
                b = run_session(live_sess, pg)
                results[name]["total"] += 1
                if a is not None and b is not None and a.shape == b.shape and (a == b).all():
                    results[name]["match"] += 1
                elif first_fail and results[name]["first_fail"] is None:
                    results[name]["first_fail"] = {
                        "pair_idx": pi,
                        "perturb_idx": j,
                        "input_grid": pg.tolist(),
                        "proto_output": a.tolist() if a is not None else None,
                        "live_output": b.tolist() if b is not None else None,
                    }

    return {
        "tid": tid,
        "base_match": base_match,
        "base_total": base_total,
        "perturbations": results,
    }


def _run_batch(args) -> int:
    """Run adversarial_test on every prototype of each --batch tid; print a shippability table.

    Folds batch_adversarial.py — but uses the CURRENT default live-zip (--live-zip) instead of
    that tool's stale hardcoded path.
    """
    import zipfile
    import tempfile
    try:
        from _score_v2 import score_network_v2
    except Exception:
        from tools._score_v2 import score_network_v2
    tbt = Path(__file__).parent.parent / "The_400_Tasks" / "Task_by_Task"
    tids = [int(x) for x in str(args.batch).replace(" ", "").split(",") if x]
    print(f"{'prototype':<52} {'adv%':>6} {'cost':>9} {'live':>9}  verdict")
    print("-" * 96)
    with zipfile.ZipFile(args.live_zip) as z:
        names = set(z.namelist())
        for tid in tids:
            nm = f"task{tid:03d}.onnx"
            live_cost = None
            if nm in names:
                r = score_network_v2(z.extract(nm, tempfile.mkdtemp()))
                live_cost = (r[1] + r[2]) if r else None
            pdir = tbt / f"task{tid:03d}" / "prototypes"
            for proto in (sorted(pdir.glob("*.onnx")) if pdir.exists() else []):
                res = adversarial_test(str(proto), tid, args.live_zip, args.data_dir,
                                       max_perturbations_per_pair=args.max_perturbations)
                tot = sum(r["total"] for r in res["perturbations"].values())
                mat = sum(r["match"] for r in res["perturbations"].values())
                pct = (mat / tot * 100) if tot else 100.0
                sr = score_network_v2(str(proto))
                cost = (sr[1] + sr[2]) if sr else None
                ship = "SHIP" if (pct == 100.0 and cost is not None and live_cost is not None and cost < live_cost) else "-"
                print(f"{proto.name:<52} {pct:>5.1f}% {str(cost):>9} {str(live_cost):>9}  {ship}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--proto", type=Path, required=False)
    p.add_argument("--tid", type=int, required=False)
    p.add_argument("--live-zip", type=Path, default=DEFAULT_LIVE)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--max-perturbations", type=int, default=None,
                   help="Max perturbations per pair per type")
    p.add_argument("--first-fail", action="store_true",
                   help="Capture details of first failing perturbation")
    p.add_argument("--batch", type=str, default=None,
                   help="Comma-separated tids; run adversarial on every prototype in each "
                        "Task_by_Task/taskNNN/prototypes/ and print a shippability table. "
                        "Folds the old batch_adversarial.py.")
    args = p.parse_args(argv)

    if args.batch:
        return _run_batch(args)
    if args.proto is None or args.tid is None:
        p.error("--proto and --tid are required (or use --batch)")

    result = adversarial_test(
        str(args.proto), args.tid, args.live_zip, args.data_dir,
        max_perturbations_per_pair=args.max_perturbations,
        first_fail=args.first_fail,
    )

    print(f"Task: task{args.tid:03d}")
    print(f"Prototype: {args.proto}")
    print(f"Live: {args.live_zip.name} (task{args.tid:03d}.onnx)")
    print()
    print(f"Base (no perturbation):  {result['base_match']}/{result['base_total']}")
    print()
    print("Adversarial perturbations:")
    print(f"  {'Type':<22} {'Match':>10} {'Match%':>8}")
    total_total = 0
    total_match = 0
    worst = None
    worst_rate = 1.1
    for name, r in result["perturbations"].items():
        if r["total"] == 0:
            print(f"  {name:<22} {'-':>10} (no cases)")
            continue
        rate = r["match"] / r["total"]
        bar = f"{r['match']:>4}/{r['total']:<5}"
        pct = f"{rate * 100:>6.1f}%"
        mark = " OK" if rate == 1.0 else " !!"
        print(f"  {name:<22} {bar:>10} {pct:>8}{mark}")
        total_total += r["total"]
        total_match += r["match"]
        if rate < worst_rate:
            worst_rate = rate
            worst = name

    print()
    if total_total:
        rate = total_match / total_total * 100
        print(f"Aggregate:  {total_match}/{total_total} ({rate:.1f}%)")
    if worst:
        print(f"Worst perturbation: {worst} ({worst_rate*100:.1f}% match)")

    if args.first_fail:
        for name, r in result["perturbations"].items():
            if r["first_fail"]:
                ff = r["first_fail"]
                print()
                print(f"=== First failing {name} perturbation ===")
                print(f"  Pair {ff['pair_idx']}, perturbation #{ff['perturb_idx']}")
                print(f"  Input shape: {len(ff['input_grid'])}x{len(ff['input_grid'][0])}")
                # Show first 4 rows
                ig = ff['input_grid']
                print(f"  Input (first 4 rows):")
                for row in ig[:4]:
                    print(f"    {row}")
                if ff['proto_output'] and ff['live_output']:
                    po, lo = ff['proto_output'], ff['live_output']
                    print(f"  Proto output (first 4 rows):")
                    for row in po[:4]:
                        print(f"    {row}")
                    print(f"  Live output (first 4 rows):")
                    for row in lo[:4]:
                        print(f"    {row}")
                break

    # Exit code: 0 if no divergence detected, 1 otherwise
    return 0 if total_total > 0 and total_match == total_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
