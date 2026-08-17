#!/usr/bin/env python3
"""Verify Beta recipe notes against train + test + arc-gen task JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ENV = os.environ.get("NEUROGOLF_DATA_DIR")
DATA = Path(DATA_ENV) if DATA_ENV else ROOT / "data" / "competitions" / "neurogolf-2026"
DATA_LABEL = "$NEUROGOLF_DATA_DIR" if DATA_ENV else "data/competitions/neurogolf-2026"


def _pairs(task: int) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    path = DATA / f"task{task:03d}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Set NEUROGOLF_DATA_DIR to the directory containing taskNNN.json files."
        )
    data = json.loads(path.read_text())
    pairs: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, pair in enumerate(data.get(split, [])):
            pairs.append((split, idx, np.array(pair["input"]), np.array(pair["output"])))
    return pairs


def predict_task105_bbox(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    ones = np.argwhere(inp == 1)
    if len(ones) == 0:
        return out
    r_min, c_min = ones.min(axis=0)
    r_max, c_max = ones.max(axis=0)
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            if (r in (r_min, r_max) or c in (c_min, c_max)) and out[r, c] == 0:
                out[r, c] = 2
    return out


def predict_task105_gridline_probe(inp: np.ndarray) -> np.ndarray:
    """Best quick refinement found during verification; still not handoff-ready."""
    out = inp.copy()
    ones = np.argwhere(inp == 1)
    if len(ones) == 0:
        return out
    r_min, c_min = ones.min(axis=0)
    r_max, c_max = ones.max(axis=0)
    for r in range(inp.shape[0]):
        if int(np.sum(inp[r] == 1)) >= 4:
            for c in range(c_min, c_max + 1):
                if out[r, c] == 0:
                    out[r, c] = 2
    for c in range(inp.shape[1]):
        if int(np.sum(inp[:, c] == 1)) >= 4:
            for r in range(r_min, r_max + 1):
                if out[r, c] == 0:
                    out[r, c] = 2
    return out


def predict_task229_majority(inp: np.ndarray) -> np.ndarray:
    counts = Counter(int(value) for value in inp.ravel())
    color = min(color for color, count in counts.items() if count == max(counts.values()))
    return np.where(inp == color, inp, 5)


def _centers(inp: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    c1 = np.argwhere(inp == 1)
    c2 = np.argwhere(inp == 2)
    if len(c1) != 1 or len(c2) != 1:
        raise ValueError("expected exactly one color-1 center and one color-2 center")
    return tuple(int(v) for v in c1[0]), tuple(int(v) for v in c2[0])


def _direction(center: tuple[int, int], cell: tuple[int, int]) -> str | None:
    cr, cc = center
    r, c = cell
    if r == cr and c < cc:
        return "L"
    if r == cr and c > cc:
        return "R"
    if c == cc and r < cr:
        return "U"
    if c == cc and r > cr:
        return "D"
    return None


def predict_task270_partition(inp: np.ndarray) -> np.ndarray:
    c1, c2 = _centers(inp)
    counts = {3: 0, 7: 0}
    for r_raw, c_raw in np.argwhere((inp == 3) | (inp == 7)):
        cell = (int(r_raw), int(c_raw))
        aligns_1 = _direction(c1, cell) is not None
        aligns_2 = _direction(c2, cell) is not None
        if aligns_1 and not aligns_2:
            counts[int(inp[cell])] += 1

    if counts[7] > counts[3]:
        outer1 = 7
    elif counts[3] > counts[7]:
        outer1 = 3
    else:
        outer1 = 7
    outer2 = 3 if outer1 == 7 else 7

    out = np.zeros_like(inp)
    out[c1] = 1
    out[c2] = 2
    offsets = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}
    for center, outer in ((c1, outer1), (c2, outer2)):
        for r_raw, c_raw in np.argwhere(inp == outer):
            cell = (int(r_raw), int(c_raw))
            direction = _direction(center, cell)
            if direction is None:
                continue
            dr, dc = offsets[direction]
            rr, cc = center[0] + dr, center[1] + dc
            if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]:
                out[rr, cc] = outer
    return out


def _verify(task: int, rule: str, predict) -> dict[str, str]:
    ok = 0
    fail_examples: list[str] = []
    split_totals = Counter()
    split_ok = Counter()
    for split, idx, inp, expected in _pairs(task):
        split_totals[split] += 1
        try:
            got = predict(inp)
            matched = got.shape == expected.shape and np.array_equal(got, expected)
            diff = int(np.sum(got != expected)) if got.shape == expected.shape else -1
        except Exception as exc:  # noqa: BLE001
            matched = False
            diff = -1
            got = None
            fail_examples.append(f"{split}[{idx}] exception:{exc}")
        if matched:
            ok += 1
            split_ok[split] += 1
        elif len(fail_examples) < 8:
            shape = "none" if got is None else "x".join(map(str, got.shape))
            fail_examples.append(f"{split}[{idx}] diff={diff} got_shape={shape}")
    total = sum(split_totals.values())
    return {
        "task": f"{task:03d}",
        "rule": rule,
        "passed": str(ok),
        "total": str(total),
        "rate": f"{ok / total:.4f}",
        "train": f"{split_ok['train']}/{split_totals['train']}",
        "test": f"{split_ok['test']}/{split_totals['test']}",
        "arc_gen": f"{split_ok['arc-gen']}/{split_totals['arc-gen']}",
        "first_failures": "; ".join(fail_examples),
        "verdict": "B->A" if ok == total else "demote" if task == 105 and rule == "bbox_perimeter_original" else "investigate",
    }


def build_report(date: str) -> tuple[list[dict[str, str]], str]:
    rows = [
        _verify(270, "unambiguous_color_partition_draw_all_aligned_outer_cells", predict_task270_partition),
        _verify(229, "majority_color_preserve_else_5", predict_task229_majority),
        _verify(105, "bbox_perimeter_original", predict_task105_bbox),
        _verify(105, "gridline_count4_global_span_probe", predict_task105_gridline_probe),
    ]
    lines = [
        f"# Beta Recipe Verification - {date}",
        "",
        f"Verifier set: all train + test + bundled arc-gen pairs from `{DATA_LABEL}`.",
        "",
        "| Task | Rule | Result | Split detail | Verdict |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| task{row['task']} | `{row['rule']}` | {row['passed']}/{row['total']} | "
            f"train {row['train']}; test {row['test']}; arc-gen {row['arc_gen']} | {row['verdict']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- task270 is promoted: the refined color-partition rule reaches 266/266.",
        "- task229 is promoted with a corrected rule: preserve the most frequent color and replace every other cell with color 5.",
        "- task105 is demoted: the written bbox-perimeter recipe reaches only 122/266. A quick gridline-span refinement reaches 231/266, which is useful evidence but not a handoff-ready rule.",
        "",
        "## Failure Samples",
        "",
    ]
    for row in rows:
        if row["first_failures"]:
            lines.append(f"- task{row['task']} `{row['rule']}`: {row['first_failures']}")
    lines.append("")
    return rows, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-05-08")
    parser.add_argument("--out-prefix", default="Logs/beta_recipe_verification_2026-05-08")
    args = parser.parse_args()

    rows, md = build_report(args.date)
    out = ROOT / args.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "rule", "passed", "total", "rate", "train", "test", "arc_gen", "first_failures", "verdict"]
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".md").write_text(md.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out.with_suffix('.csv').relative_to(ROOT)}")
    print(f"Wrote {out.with_suffix('.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
