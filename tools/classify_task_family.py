"""Heuristic/source classifier to fill task taxonomy columns in
NeuroGolf_Task_Status.xlsx.

Approach (combines three signal sources, in priority order):

  1. **Existing annotations** — never overwrite manual values from DeepSeek
     (the 20 hand-classified tasks). Always preserve any non-empty cell.

  2. **Research note filename** — descriptive slugs like
     `recipe-task017-periodic-hole-fill` map to (family, [transformations])
     via KEYWORD_MAP.

  3. **JSON feature heuristics** — analyze the task's input/output pairs for:
     - grid_resized (shape changes)
     - output is global flip of input (Reflection)
     - bg-cell-fill ratio (0 → non-0 cells → Filling Regions)
     - cell movement (sparse non-bg cells shift position → Translation/Gravity)
     - color vocab change (output uses different colors → Color Mapping)
     - mostly-input-preserved + marker fill (Symmetry Completion)

  4. **Karnak primitive labels** — fill blank `complexity_1_10` values from
     the local `arc_primitives.csv`. Existing values are preserved; the first
     20 repo values match this source exactly.

Outputs a classification per task. Writes back into the xlsx. Tasks that
remain ambiguous stay blank so they're easy to find and refine by hand.

Usage:
    python tools/classify_task_family.py [--xlsx PATH] [--data DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import PatternFill


REPO = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = REPO / "The_400_Tasks" / "NeuroGolf_Task_Status.xlsx"
DEFAULT_DATA_DIR = REPO / "data" / "competitions" / "neurogolf-2026"
DEFAULT_PRIMITIVES = REPO / "data" / "competitor_artifacts" / "karnak-task-transformation" / "arc_primitives.csv"
RESEARCH_NOTES = REPO / "Research_Archive" / "notes"
ORGANIC_PROTOTYPES = REPO / "Research_Archive" / "organic_prototypes"


# Families observed in user annotations (4):
FAMILIES = {
    "Pattern_Recognition",
    "Spatial_and_Geometric",
    "Object_Based",
    "Color_and_Logical",
}


# Transformations vocab observed in user annotations:
TRANSFORMATIONS = {
    "Tiling", "Translation", "Color Mapping", "Object Detection",
    "Filling Regions", "Sorting", "Gravity", "Cropping",
    "Symmetry Completion", "Reflection", "Shifting",
}


# Exact local notes override. task000 is absent from karnak's 400-row
# arc_primitives.csv but has a durable rule note in Task_by_Task/task000.md:
# fixed (+1,+1) drop-shadow from non-background objects.
TASK_TAXONOMY_OVERRIDES = {
    0: ("Object_Based", ["Object Detection", "Translation", "Filling Regions"]),
}

# Original hand-reviewed taxonomy seed rows. Do not overwrite these in normal
# or --rerun mode; later primitive-derived complexity backfill makes
# complexity_1_10 non-empty for almost all rows, so the old "complexity means
# trusted" heuristic would become too broad.
USER_TRUSTED_TIDS = set(range(1, 21))


# Recipe-filename keyword → (family, [transformations]).
# Ordered: first match wins.
KEYWORD_RULES = [
    # Highly specific patterns first
    (r"\bperiodic[\-_]?hole[\-_]?fill\b", "Pattern_Recognition", ["Filling Regions"]),
    (r"\bperiod[\-_]?extend\b", "Pattern_Recognition", ["Filling Regions", "Tiling"]),
    (r"\bsymmetry[\-_]?completion\b", "Pattern_Recognition", ["Symmetry Completion"]),
    (r"\bsymmetry[\-_]?denoise\b", "Pattern_Recognition", ["Symmetry Completion"]),
    (r"\bsymmetry\b", "Pattern_Recognition", ["Symmetry Completion"]),
    (r"\brotat\w*[\-_]?(symmetry|tile)\b", "Pattern_Recognition", ["Symmetry Completion", "Tiling"]),
    (r"\b4fold[\-_]?rotation\b", "Pattern_Recognition", ["Symmetry Completion", "Tiling"]),
    (r"\bbilateral[\-_]?symmetry\b", "Pattern_Recognition", ["Symmetry Completion"]),
    (r"\bfractal\b", "Spatial_and_Geometric", ["Tiling"]),
    (r"\bring[\-_]?(expansion|extend)\b", "Spatial_and_Geometric", ["Tiling"]),
    (r"\bwedge[\-_]?fill\b", "Pattern_Recognition", ["Filling Regions"]),
    (r"\bstaircase\b", "Pattern_Recognition", ["Filling Regions"]),
    (r"\bhorizontal[\-_]?tile\b", "Spatial_and_Geometric", ["Tiling"]),
    (r"\b(rectangle|interior)\b", "Object_Based", ["Object Detection", "Filling Regions"]),
    (r"\bbbox[\-_]?crop\b", "Object_Based", ["Cropping"]),
    (r"\bcolor[\-_]?filter[\-_]?bbox[\-_]?crop\b", "Object_Based", ["Cropping", "Color Mapping"]),
    (r"\bforeground[\-_]?bbox[\-_]?crop\b", "Object_Based", ["Cropping", "Object Detection"]),
    (r"\bodd[\-_]?one[\-_]?out\b", "Object_Based", ["Object Detection", "Color Mapping"]),
    (r"\bmarker[\-_]?(driven|aligned)[\-_]?shape\b", "Object_Based", ["Object Detection", "Translation"]),
    (r"\bmarker[\-_]?rectangle\b", "Object_Based", ["Object Detection", "Filling Regions"]),
    (r"\bshape[\-_]?(copy|recolor)\b", "Object_Based", ["Object Detection", "Translation", "Color Mapping"]),
    (r"\banchor[\-_]?cross\b", "Object_Based", ["Object Detection"]),
    (r"\baligned[\-_]?cells\b", "Object_Based", ["Object Detection"]),
    (r"\btwo[\-_]?blob[\-_]?tile\b", "Object_Based", ["Object Detection", "Tiling"]),
    (r"\bcross[\-_]?section\b", "Pattern_Recognition", ["Object Detection"]),
    (r"\bgrid[\-_]?label\b", "Pattern_Recognition", ["Object Detection"]),
    (r"\bgrid[\-_]?max[\-_]?count[\-_]?fill\b", "Object_Based", ["Object Detection", "Filling Regions"]),
    (r"\bgrid[\-_]?object[\-_]?summary\b", "Object_Based", ["Object Detection"]),
    (r"\bboustrophedon[\-_]?pack\b", "Spatial_and_Geometric", ["Tiling"]),
    (r"\bcompressed[\-_]?lookup\b", "Color_and_Logical", ["Color Mapping"]),
    (r"\bdiagonal[\-_]?marker[\-_]?ring\b", "Object_Based", ["Object Detection", "Filling Regions"]),
    (r"\btile[\-_]?with[\-_]?marked[\-_]?empties\b", "Spatial_and_Geometric", ["Tiling", "Filling Regions"]),
    (r"\beuclidean[\-_]?wedge\b", "Pattern_Recognition", ["Filling Regions"]),
    # Lower-confidence catch-alls
    (r"\bfill\b", "Pattern_Recognition", ["Filling Regions"]),
    (r"\btile\b", "Spatial_and_Geometric", ["Tiling"]),
    (r"\bcrop\b", "Object_Based", ["Cropping"]),
    (r"\brecolor\b", "Color_and_Logical", ["Color Mapping"]),
    (r"\bmarker\b", "Object_Based", ["Object Detection"]),
]


def gather_research_slugs(tid: int) -> list[str]:
    """Get all stems of Research_Archive/notes/*.md files mentioning this task."""
    out: list[str] = []
    pat = re.compile(rf"task[-_]?{tid:03d}\b")
    if RESEARCH_NOTES.exists():
        for f in RESEARCH_NOTES.iterdir():
            if not f.is_file():
                continue
            if pat.search(f.stem):
                out.append(f.stem)
    return out


def classify_from_slugs(slugs: list[str]) -> tuple[str | None, list[str]]:
    """Apply KEYWORD_RULES to recipe filename slugs."""
    family = None
    transformations: list[str] = []
    for slug in slugs:
        for pattern, fam, trans in KEYWORD_RULES:
            if re.search(pattern, slug, flags=re.IGNORECASE):
                if family is None:
                    family = fam
                for t in trans:
                    if t not in transformations:
                        transformations.append(t)
    return family, transformations


def load_primitives(path: Path) -> dict[int, dict[str, str]]:
    """Read karnak arc_primitives.csv as {task_id: row}."""
    if not path.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            task = (row.get("Task_ID") or "").strip()
            if not re.fullmatch(r"task\d{3}", task):
                continue
            out[int(task[4:])] = row
    return out


def grids_from_json(data: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
    out = []
    for p in pairs[:6]:  # sample 6 for speed
        inp = np.array(p["input"])
        out_ = np.array(p["output"])
        out.append((inp, out_))
    return out


def _is_marker_sym_fill(inp: np.ndarray, out: np.ndarray) -> bool:
    """True iff input has a "marker color" X with many cells, output has zero
    X-cells, and the filled values match input at H/V/HV/T/AT positions.

    Catches the task074 / task351 / task287 pattern.
    """
    if inp.shape != out.shape:
        return False
    # Find markers: colors with many cells in input but zero in output
    for X in range(1, 10):
        marker_cells = (inp == X)
        n_marker = int(marker_cells.sum())
        if n_marker < 3:
            continue
        if (out == X).any():
            continue
        # Now: where input was X, what are output values?
        H, W = inp.shape
        transforms = [inp[:, ::-1], inp[::-1, :], inp[::-1, ::-1]]
        if H == W:
            transforms.append(inp.T)
            transforms.append(inp[::-1, ::-1].T)
        matched = 0
        total = 0
        for r in range(H):
            for c in range(W):
                if marker_cells[r, c]:
                    total += 1
                    for t in transforms:
                        if t[r, c] == out[r, c]:
                            matched += 1
                            break
        if total >= 3 and matched / total >= 0.8:
            return True
    return False


def classify_from_features(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[str | None, list[str]]:
    """Pattern-detect features and infer family + transformations.

    Tries (in priority order):
      - Marker-symmetry fill (task074/task287/task351 pattern)
      - Pure global flip / reflection
      - Cropping (output is smaller than input)
      - Tiling (output is larger than input)
      - 0-cell fill (lots of 0s become non-bg)
      - Color mapping (cells stay in place but change color)
      - Cells moved (translation)
    """
    if not pairs:
        return None, []

    grid_resized_count = 0
    shrink_count = 0
    grow_count = 0
    is_flip = 0
    is_marker_sym = 0
    color_mapped = 0
    zero_filled = 0
    cells_moved = 0
    n = 0
    for inp, out in pairs:
        n += 1
        if inp.shape == out.shape:
            # Priority 1: marker-color sym fill (task074/287/351 pattern)
            if _is_marker_sym_fill(inp, out):
                is_marker_sym += 1
                continue
            # Priority 2: pure global flip
            if inp.shape[0] == inp.shape[1]:
                if (np.array_equal(out, inp[:, ::-1]) or np.array_equal(out, inp[::-1, :])
                    or np.array_equal(out, inp[::-1, ::-1]) or np.array_equal(out, inp.T)):
                    is_flip += 1
                    continue
            else:
                if (np.array_equal(out, inp[:, ::-1]) or np.array_equal(out, inp[::-1, :])
                    or np.array_equal(out, inp[::-1, ::-1])):
                    is_flip += 1
                    continue
            # 0-fill: bg cells become non-bg
            inp_zero = (inp == 0)
            out_zero = (out == 0)
            if inp_zero.sum() > out_zero.sum() and inp_zero.sum() - out_zero.sum() >= max(3, inp.size // 20):
                zero_filled += 1
                continue
            # Color mapping: same shape, non-bg cells stay but colors change
            inp_nonbg = (inp != 0)
            out_nonbg = (out != 0)
            if np.array_equal(inp_nonbg, out_nonbg) and not np.array_equal(inp, out):
                color_mapped += 1
                continue
            # Cells moved
            if inp_nonbg.sum() == out_nonbg.sum() and not np.array_equal(inp_nonbg, out_nonbg):
                cells_moved += 1
        else:
            grid_resized_count += 1
            if out.size < inp.size:
                shrink_count += 1
            else:
                grow_count += 1

    if n == 0:
        return None, []

    family: str | None = None
    trans: list[str] = []

    # Priority: marker-sym fill takes precedence
    if is_marker_sym >= n // 2:
        family = "Pattern_Recognition"
        trans = ["Symmetry Completion"]
    elif grid_resized_count >= n // 2:
        if shrink_count >= grow_count:
            family = "Object_Based"
            trans = ["Cropping", "Object Detection"]
        else:
            family = "Spatial_and_Geometric"
            trans = ["Tiling"]
    elif is_flip >= n // 2:
        family = "Spatial_and_Geometric"
        trans = ["Reflection"]
    elif zero_filled >= n // 2:
        family = "Pattern_Recognition"
        trans = ["Filling Regions"]
    elif color_mapped >= n // 2:
        family = "Color_and_Logical"
        trans = ["Color Mapping"]
    elif cells_moved >= n // 2:
        family = "Object_Based"
        trans = ["Translation", "Object Detection"]
    else:
        # Fallback: low-cell-change task. Inspect WHERE the changes are.
        # If most changed cells are bg→non-bg (some kind of fill / draw) → Pattern_Recognition.
        # If most are non-bg→non-bg (color change in place) → Object_Based + Color Mapping.
        total_diff = 0
        bg_to_nonbg = 0
        nonbg_to_nonbg = 0
        for inp, out in pairs:
            if inp.shape != out.shape:
                continue
            d = (inp != out)
            total_diff += int(d.sum())
            bg_to_nonbg += int(((inp == 0) & (out != 0))[d].sum())
            nonbg_to_nonbg += int(((inp != 0) & (out != 0))[d].sum())
        if total_diff > 0:
            if bg_to_nonbg > nonbg_to_nonbg:
                family = "Pattern_Recognition"
                trans = ["Filling Regions"]
            else:
                family = "Object_Based"
                trans = ["Color Mapping", "Object Detection"]
    return family, trans


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--primitives", type=Path, default=DEFAULT_PRIMITIVES,
                   help="karnak arc_primitives.csv; used only to fill blank complexity_1_10")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mark-low-confidence", action="store_true",
                   help="Color the family cell yellow for heuristic-inferred (not from recipe slug or user-annotated) classifications.")
    p.add_argument("--rerun", action="store_true",
                   help="Re-classify ALL rows, overwriting any prior heuristic-filled values. "
                        "(Still preserves the original 20 hand-annotated rows — they keep their highlight color.)")
    args = p.parse_args(argv)

    wb = openpyxl.load_workbook(args.xlsx)
    ws = wb["Tasks"]
    headers = [c.value for c in ws[1]]
    fam_idx = headers.index("family") + 1
    trans_idx = headers.index("transformations") + 1
    tid_idx = headers.index("task_id") + 1
    cx_idx = headers.index("complexity_1_10") + 1

    pristine_recipe = 0
    pristine_features = 0
    pristine_override = 0
    complexity_from_primitives = 0
    preserved_existing = 0
    unclassified = 0
    primitives = load_primitives(args.primitives)

    for r in range(2, ws.max_row + 1):
        tid = ws.cell(r, tid_idx).value
        if tid is None:
            continue
        tid = int(tid)
        if ws.cell(r, cx_idx).value in (None, ""):
            primitive_row = primitives.get(tid)
            if primitive_row and primitive_row.get("Estimated_Complexity"):
                ws.cell(r, cx_idx).value = int(primitive_row["Estimated_Complexity"])
                complexity_from_primitives += 1

        existing_fam = ws.cell(r, fam_idx).value
        existing_trans = ws.cell(r, trans_idx).value
        if tid in USER_TRUSTED_TIDS:
            preserved_existing += 1
            continue
        # If already populated AND --rerun not set, keep
        if (existing_fam or existing_trans) and not args.rerun:
            preserved_existing += 1
            continue
        # Otherwise: clear and re-classify
        if existing_fam or existing_trans:
            ws.cell(r, fam_idx).value = None
            ws.cell(r, trans_idx).value = None

        # 1. Exact overrides from durable per-task notes.
        if tid in TASK_TAXONOMY_OVERRIDES:
            family, transformations = TASK_TAXONOMY_OVERRIDES[tid]
            source = "override"
        else:
            # 2. Try recipe filename keywords.
            slugs = gather_research_slugs(tid)
            family, transformations = classify_from_slugs(slugs)
            source = "recipe"

        # 3. Fall back to JSON features
        if not family:
            json_path = args.data / f"task{tid:03d}.json"
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text())
                    pairs = grids_from_json(data)
                    family, transformations = classify_from_features(pairs)
                    source = "features"
                except Exception:
                    pass

        if family:
            ws.cell(r, fam_idx).value = family
            ws.cell(r, trans_idx).value = " | ".join(transformations) if transformations else None
            if source == "override":
                pristine_override += 1
            elif source == "recipe":
                pristine_recipe += 1
            else:
                pristine_features += 1
                if args.mark_low_confidence:
                    ws.cell(r, fam_idx).fill = PatternFill("solid", fgColor="FFF2CC")
                    ws.cell(r, trans_idx).fill = PatternFill("solid", fgColor="FFF2CC")
        else:
            unclassified += 1

    print(f"Preserved existing user-annotated: {preserved_existing}")
    print(f"Filled from exact overrides:       {pristine_override}")
    print(f"Filled from recipe filenames:      {pristine_recipe}")
    print(f"Filled from JSON feature heuristic: {pristine_features}")
    print(f"Filled complexity from primitives: {complexity_from_primitives}")
    print(f"Left unclassified:                 {unclassified}")

    if args.dry_run:
        print("(dry run; no save)")
    else:
        wb.save(args.xlsx)
        print(f"Wrote: {args.xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
