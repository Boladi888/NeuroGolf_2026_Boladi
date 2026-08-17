#!/usr/bin/env python3
"""Synthetic hidden-safety guard for task153 two-blob tile-fit candidates.

The bundled task153 train/test/arc-gen examples cover only a narrow subset of
valid two-blob shape pairs. At least one cheap external candidate passed those
examples but scored zero on Kaggle public hidden data. This guard generates
task153-valid blob-shape pairs outside the bundled examples and compares a
candidate ONNX against the explicit tiling rule.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class SyntheticCase:
    area_combo: tuple[int, int]
    input_grid: list[list[int]]
    expected_grid: list[list[int]]
    mask_a: tuple[tuple[int, ...], ...]
    mask_b: tuple[tuple[int, ...], ...]
    pos_a: tuple[int, int]
    pos_b: tuple[int, int]


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return seen == cells


def _tight_connected_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for h in (1, 2, 3):
        for w in (1, 2, 3):
            for bits in range(1, 1 << (h * w)):
                cells = {(idx // w, idx % w) for idx in range(h * w) if bits >> idx & 1}
                if not _is_connected(cells):
                    continue
                rows = [r for r, _ in cells]
                cols = [c for _, c in cells]
                if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
                    continue
                mask = np.zeros((h, w), dtype=np.int8)
                for r, c in cells:
                    mask[r, c] = 1
                key = (h, w, tuple(int(v) for v in mask.flatten()))
                if key not in seen:
                    seen.add(key)
                    masks.append(mask)
    return masks


def _placements(mask: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = mask.shape
    out = []
    for ar in range(3 - h + 1):
        for ac in range(3 - w + 1):
            placed = np.zeros((3, 3), dtype=np.int8)
            placed[ar : ar + h, ac : ac + w] = mask * color
            out.append(placed)
    return out


def _unique_expected(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray | None:
    valid = []
    for placed_a in _placements(mask_a, 3):
        occupied_a = placed_a != 0
        for placed_b in _placements(mask_b, 7):
            occupied_b = placed_b != 0
            if np.logical_and(occupied_a, occupied_b).any():
                continue
            if int(occupied_a.sum()) + int(occupied_b.sum()) == 9:
                valid.append(placed_a + placed_b)
    return valid[0] if len(valid) == 1 else None


def _place_nonoverlap(mask_a: np.ndarray, mask_b: np.ndarray, seed: int) -> tuple[list[list[int]], tuple[int, int], tuple[int, int]]:
    h1, w1 = mask_a.shape
    h2, w2 = mask_b.shape
    pos_a = [(r, c) for r in range(10 - h1 + 1) for c in range(10 - w1 + 1)]
    pos_b = [(r, c) for r in range(10 - h2 + 1) for c in range(10 - w2 + 1)]
    rng = random.Random(seed)
    rng.shuffle(pos_a)
    rng.shuffle(pos_b)
    for ra, ca in pos_a:
        for rb, cb in pos_b:
            grid = np.zeros((10, 10), dtype=np.int8)
            for r, c in zip(*np.where(mask_a)):
                grid[ra + r, ca + c] = 3
            ok = True
            for r, c in zip(*np.where(mask_b)):
                if grid[rb + r, cb + c] != 0:
                    ok = False
                    break
                grid[rb + r, cb + c] = 7
            if ok:
                return grid.tolist(), (ra, ca), (rb, cb)
    raise RuntimeError("could not place non-overlapping masks")


def build_suite(area_combos: set[tuple[int, int]]) -> list[SyntheticCase]:
    masks = _tight_connected_masks()
    cases: list[SyntheticCase] = []
    for idx_a, mask_a in enumerate(masks):
        for idx_b, mask_b in enumerate(masks):
            combo = tuple(sorted((int(mask_a.sum()), int(mask_b.sum()))))
            if combo not in area_combos:
                continue
            if int(mask_a.sum()) + int(mask_b.sum()) != 9:
                continue
            expected = _unique_expected(mask_a, mask_b)
            if expected is None:
                continue
            grid, pos_a, pos_b = _place_nonoverlap(mask_a, mask_b, seed=idx_a * 1000 + idx_b)
            cases.append(
                SyntheticCase(
                    area_combo=combo,
                    input_grid=grid,
                    expected_grid=expected.tolist(),
                    mask_a=tuple(tuple(int(v) for v in row) for row in mask_a.tolist()),
                    mask_b=tuple(tuple(int(v) for v in row) for row in mask_b.tolist()),
                    pos_a=pos_a,
                    pos_b=pos_b,
                )
            )
    return cases


def _encode(grid: list[list[int]]) -> np.ndarray:
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


def _parse_area_combos(value: str) -> set[tuple[int, int]]:
    combos = set()
    for item in value.split(","):
        left, right = item.split(":")
        combos.add(tuple(sorted((int(left), int(right)))))
    return combos


def check_model(path: Path, cases: list[SyntheticCase], max_failures: int) -> int:
    session = ort.InferenceSession(path.read_bytes())
    failures = []
    by_area: Counter[tuple[tuple[int, int], str]] = Counter()
    for case in cases:
        actual = (session.run(["output"], {"input": _encode(case.input_grid)})[0] > 0.0).astype(np.float32)
        expected = _expected_tensor(case.expected_grid)
        ok = bool(np.array_equal(actual, expected))
        by_area[(case.area_combo, "pass" if ok else "fail")] += 1
        if not ok and len(failures) < max_failures:
            failures.append((case, _decode(actual)))

    passed = sum(count for (_, status), count in by_area.items() if status == "pass")
    failed = sum(count for (_, status), count in by_area.items() if status == "fail")
    print(f"{path}: {passed}/{len(cases)} synthetic task153 cases passed; {failed} failed.")
    for area in sorted({area for area, _ in by_area}):
        print(
            f"  area {area[0]}:{area[1]} pass={by_area[(area, 'pass')]} "
            f"fail={by_area[(area, 'fail')]}"
        )
    if failures:
        case, actual = failures[0]
        print("First mismatch:")
        print(f"  area={case.area_combo[0]}:{case.area_combo[1]} pos_a={case.pos_a} pos_b={case.pos_b}")
        print(f"  mask_a={case.mask_a}")
        print(f"  mask_b={case.mask_b}")
        print(f"  expected={case.expected_grid}")
        print(f"  actual={actual}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", nargs="+", type=Path, help="task153 ONNX candidate(s) to check")
    parser.add_argument(
        "--area-combos",
        default="3:6,4:5",
        help="Comma-separated area splits to generate. Default matches bundled task153 split families.",
    )
    parser.add_argument("--max-failures", type=int, default=1)
    args = parser.parse_args()

    cases = build_suite(_parse_area_combos(args.area_combos))
    print(f"Generated {len(cases)} synthetic task153 cases for area combos {args.area_combos}.")
    total_failures = 0
    for path in args.onnx:
        total_failures += check_model(path, cases, max_failures=args.max_failures)
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
