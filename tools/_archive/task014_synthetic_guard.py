#!/usr/bin/env python3
"""Synthetic hidden-safety guard for task014 least-color bbox-crop candidates.

Task014's visible/arc-gen examples can be passed by candidates that find the
least-common nonzero color, crop its bounding box, and then keep only that color
plus background. The actual rule preserves every color inside that bbox. This
guard generates least-color bbox crops where other colors appear inside the
selected rectangle, catching that hidden-bad shortcut before upload.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class SyntheticCase:
    input_grid: list[list[int]]
    expected_grid: list[list[int]]
    target_color: int
    bbox: tuple[int, int, int, int]


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _make_case(rng: random.Random) -> SyntheticCase:
    h = rng.randint(6, 15)
    w = rng.randint(6, 15)
    target = rng.choice([2, 3, 4, 6, 7, 8, 9])
    other_colors = [c for c in (1, 2, 3, 4, 5, 6, 7, 8, 9) if c != target]
    rng.shuffle(other_colors)
    colors = other_colors[: rng.randint(2, 4)]

    grid = np.zeros((h, w), dtype=np.int8)
    base = colors[0]
    grid[:, :] = base

    # Make a target-color bbox with 2-5 target cells and at least one interior
    # non-target/non-background distractor cell. Counts for all other colors are
    # kept strictly higher than the target count to avoid picker tie ambiguity.
    bh = rng.randint(2, min(6, h))
    bw = rng.randint(2, min(6, w))
    r0 = rng.randint(0, h - bh)
    c0 = rng.randint(0, w - bw)
    r1 = r0 + bh - 1
    c1 = c0 + bw - 1
    target_count = rng.randint(2, min(5, bh * bw - 1))
    rect_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    target_cells = rng.sample(rect_cells, target_count)
    for r, c in target_cells:
        grid[r, c] = target

    distractor_cells = [cell for cell in rect_cells if cell not in set(target_cells)]
    rng.shuffle(distractor_cells)
    for idx, (r, c) in enumerate(distractor_cells[: max(1, min(4, len(distractor_cells)))]):
        grid[r, c] = 0 if idx % 2 else colors[min(1, len(colors) - 1)]

    for color in colors[1:]:
        needed = target_count + 1 - int((grid == color).sum())
        open_cells = [(r, c) for r in range(h) for c in range(w) if grid[r, c] != target]
        rng.shuffle(open_cells)
        for r, c in open_cells[: max(0, needed)]:
            grid[r, c] = color

    bbox = _bbox(target_cells)
    rr0, rr1, cc0, cc1 = bbox
    expected = grid[rr0 : rr1 + 1, cc0 : cc1 + 1].tolist()
    return SyntheticCase(grid.tolist(), expected, target, bbox)


def build_suite(count: int, seed: int) -> list[SyntheticCase]:
    rng = random.Random(seed)
    cases: list[SyntheticCase] = []
    attempts = 0
    while len(cases) < count and attempts < count * 50:
        attempts += 1
        case = _make_case(rng)
        arr = np.array(case.input_grid)
        nonzero_counts = {color: int((arr == color).sum()) for color in range(1, 10) if (arr == color).any()}
        if nonzero_counts.get(case.target_color) != min(nonzero_counts.values()):
            continue
        if list(nonzero_counts.values()).count(nonzero_counts[case.target_color]) != 1:
            continue
        expected = np.array(case.expected_grid)
        if not any(color not in (0, case.target_color) for color in np.unique(expected)):
            continue
        cases.append(case)
    if len(cases) != count:
        raise RuntimeError(f"could only generate {len(cases)} valid cases after {attempts} attempts")
    return cases


def _encode_input(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            arr[0, int(value), r, c] = 1.0
    return arr


def _expected_tensor(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            arr[0, int(value), r, c] = 1.0
    return arr


def _decode(tensor: np.ndarray) -> list[list[int]]:
    tensor = (tensor > 0.0).astype(np.float32)
    decoded = []
    for r in range(30):
        row = []
        for c in range(30):
            colors = [ch for ch in range(10) if tensor[0, ch, r, c] == 1]
            row.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
        while row and row[-1] == 10:
            row.pop()
        decoded.append(row)
    while decoded and not decoded[-1]:
        decoded.pop()
    return decoded


def check_model(path: Path, cases: list[SyntheticCase], max_failures: int) -> int:
    session = ort.InferenceSession(path.read_bytes())
    failures = []
    passed = 0
    for case in cases:
        actual = (session.run(["output"], {"input": _encode_input(case.input_grid)})[0] > 0.0).astype(np.float32)
        expected = _expected_tensor(case.expected_grid)
        ok = bool(np.array_equal(actual, expected))
        passed += int(ok)
        if not ok and len(failures) < max_failures:
            failures.append((case, _decode(actual)))

    failed = len(cases) - passed
    print(f"{path}: {passed}/{len(cases)} synthetic task014 cases passed; {failed} failed.")
    if failures:
        case, actual = failures[0]
        print("First mismatch:")
        print(f"  target_color={case.target_color} bbox={case.bbox}")
        print(f"  input={case.input_grid}")
        print(f"  expected={case.expected_grid}")
        print(f"  actual={actual}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", nargs="+", type=Path, help="task014 ONNX candidate(s) to check")
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--max-failures", type=int, default=1)
    args = parser.parse_args()

    cases = build_suite(args.cases, args.seed)
    print(f"Generated {len(cases)} synthetic task014 least-color bbox cases with seed {args.seed}.")
    total_failures = 0
    for path in args.onnx:
        total_failures += check_model(path, cases, max_failures=args.max_failures)
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
