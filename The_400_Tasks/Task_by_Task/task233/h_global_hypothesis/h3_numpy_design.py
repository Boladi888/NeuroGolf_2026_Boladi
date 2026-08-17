"""H3 algorithm — numpy reference, designed to mirror what the ONNX will compute.

Compared to the existing `task233_numpy_reference.predict_task233` (which uses lists
of candidate tuples + Python sort), this version uses tensor operations that map
cleanly to ONNX ops (Conv, MatMul, ArgMax, ScatterElements).

Goal: validate the H3 design on seed 233245 AND 266 visible BEFORE writing ONNX.

Pipeline:
1. Detect main bbox (largest non-zero CC bounding rectangle of 2/0 region)
2. Detect up to 5 source 3x3 tiles outside main
3. For each source tile: determine decoration color, side relative to main, signed
   pattern (where 2s are)
4. For each (source, transform) pair: signed transformed pattern
5. Score each (slot, source, transform): signed dot product of crop_signed with
   transformed_source_signed. Perfect match = +9.
6. For each source: filter to match=9 slots, apply 3-tier preference (rot0 > interior >
   any) + side-relative tiebreak score (above→max r etc.).
7. Assign chosen slot per source via scatter.

If this matches predict_task233 on 266 visible + seed 233245, the ONNX port is
algorithmically sound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "The_400_Tasks/Task_by_Task/task233/prototypes"))

from task233_numpy_reference import _components, _inside, predict_task233  # noqa: E402


# Transform names in canonical D4 order matching the existing build script
TRANSFORMS = [
    "rot0",
    "rot0fliplr",
    "rot1",
    "rot1fliplr",
    "rot2",
    "rot2fliplr",
    "rot3",
    "rot3fliplr",
]


def apply_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    rest = transform[3:]
    flip = rest.endswith("fliplr")
    if flip:
        rest = rest[: -len("fliplr")]
    out = np.rot90(arr, int(rest))
    return np.fliplr(out) if flip else out


def transform_perm_matrix(transform: str) -> np.ndarray:
    """Return P [9, 9] s.t. transformed_flat = P @ original_flat (when both are 3x3 flattened row-major)."""
    base = np.arange(9).reshape(3, 3)
    transformed_indices = apply_transform(base, transform).flatten()
    P = np.zeros((9, 9), dtype=np.int8)
    for i, src in enumerate(transformed_indices):
        P[i, src] = 1
    return P


PERM_MATRICES = np.stack([transform_perm_matrix(t) for t in TRANSFORMS])  # [8, 9, 9]


def signed_score_h3(grid: np.ndarray) -> np.ndarray:
    """H3 prediction returning (h, w) output crop. Tensor-friendly logic."""
    grid = np.asarray(grid, dtype=np.int64)
    comps = _components(grid != 0)
    main = max(comps, key=lambda c: int(c["area"]))
    main_bbox = main["bbox"]
    r0, c0, r1, c1 = main_bbox
    crop = grid[r0:r1, c0:c1]
    h, w = crop.shape
    out = np.where(crop == 0, 2, crop).copy()

    # Collect source tiles (not inside main)
    source_tiles = []
    main_center = ((r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0)
    for comp in comps:
        bb = comp["bbox"]
        if _inside(bb, main_bbox):
            continue
        tr0, tc0, tr1, tc1 = bb
        tile = grid[tr0:tr1, tc0:tc1]
        if tile.shape != (3, 3):
            continue
        colors = sorted(int(x) for x in set(tile[tile != 0].tolist()))
        decoration = [c for c in colors if c != 2]
        if len(decoration) != 1:
            continue
        dec = decoration[0]
        # source's "is_2" pattern (3x3 binary)
        source_is_2 = (tile == 2).astype(np.int8)
        # side
        tile_center = ((tr0 + tr1 - 1) / 2.0, (tc0 + tc1 - 1) / 2.0)
        dr = tile_center[0] - main_center[0]
        dc = tile_center[1] - main_center[1]
        if abs(dr) > abs(dc):
            side = "below" if dr > 0 else "above"
        elif abs(dc) > abs(dr):
            side = "right" if dc > 0 else "left"
        else:
            side = "diag"
        source_tiles.append((dec, source_is_2, side, bb))

    if not source_tiles:
        return out.tolist()

    # Crop's "is_2" mask
    crop_is_2 = (crop == 2).astype(np.int8)  # [h, w]
    crop_signed = (1 - 2 * crop_is_2).astype(np.int32)  # +1 if hole(0), -1 if 2
    # Slot valid window: rows 0..h-3 (inclusive), cols 0..w-3 (inclusive)
    if h < 3 or w < 3:
        return out.tolist()
    nr = h - 2  # number of row slots
    nc = w - 2  # number of col slots

    # Build [nr*nc, 9] patches matrix of signed slot patterns
    patches = np.zeros((nr * nc, 9), dtype=np.int32)
    for r in range(nr):
        for c in range(nc):
            patches[r * nc + c] = crop_signed[r : r + 3, c : c + 3].flatten()

    # For each source, score against each transform, pick best
    for dec, source_is_2, side, bb in source_tiles:
        source_flat = source_is_2.flatten().astype(np.int32)  # [9] in {0, 1}
        # Apply 8 transforms via permutation matrices
        source_transformed = np.einsum("tij,j->ti", PERM_MATRICES, source_flat)  # [8, 9] in {0, 1}
        source_signed = (2 * source_transformed - 1).astype(np.int32)  # [8, 9] in {-1, +1}
        # Score matrix [8, nr*nc]
        score = source_signed @ patches.T  # [8, n_slots], range [-9, +9]
        # Match: score == 9
        match = (score == 9)  # [8, n_slots] bool

        if not match.any():
            continue  # no candidates for this source

        # Build per-slot side-relative tiebreak score [n_slots]
        rows = np.arange(nr).reshape(nr, 1).repeat(nc, axis=1).flatten()
        cols = np.arange(nc).reshape(1, nc).repeat(nr, axis=0).flatten()
        margin_r = np.minimum(rows, h - 3 - rows)
        margin_c = np.minimum(cols, w - 3 - cols)
        margin = np.minimum(margin_r, margin_c)
        interior = margin > 0

        if side == "above":
            side_score = rows.astype(np.int64) * 10000 + margin * 100 - cols
        elif side == "below":
            side_score = (-rows.astype(np.int64)) * 10000 + margin * 100 - cols
        elif side == "left":
            side_score = cols.astype(np.int64) * 10000 + margin * 100 - rows
        elif side == "right":
            side_score = (-cols.astype(np.int64)) * 10000 + margin * 100 - rows
        else:  # diag
            side_score = margin * 100

        # 3-tier preference: rot0 > interior > any
        # rot0 is transform index 0
        # Tier score [8, n_slots]:
        #   rot0 (g=0): tier_bonus = 2 * BIG
        #   non-rot0 + interior: tier_bonus = 1 * BIG
        #   non-rot0 + non-interior: tier_bonus = 0
        TIER_BIG = 10**8  # dominates side_score (max ~h*10000 < 200000)

        is_rot0 = np.zeros(8, dtype=np.int64)
        is_rot0[0] = 1  # only index 0 is "rot0"
        tier_per_g = is_rot0[:, None] * (2 * TIER_BIG)  # [8, 1]
        interior_bonus = interior.astype(np.int64) * TIER_BIG  # [n_slots]
        tier_score = tier_per_g + (1 - is_rot0[:, None]) * interior_bonus[None, :]  # [8, n_slots]
        # Wait: non-rot0 + interior → 0 * 2*BIG + 1 * BIG = BIG. Correct.
        # rot0 → 2*BIG + 0 (interior doesn't matter, only when no rot0)

        # But the reference's logic is: if any rot0 exists, use ONLY rot0. So among non-rot0 we
        # only consider them if NO rot0 candidate exists. Let me check via the highest-tier logic:
        # - rot0 candidate at score = 2*BIG + side_score (no interior boost)
        # - non-rot0 interior at BIG + side_score
        # - non-rot0 non-interior at 0 + side_score (or negative for match-failure)
        # ArgMax picks highest tier first, then side_score within tier. ✓

        composite = tier_score + side_score[None, :]  # [8, n_slots]
        # Mask out non-match
        composite = np.where(match, composite, -10**15)  # very negative if not a match

        best_flat = int(np.argmax(composite))
        best_g = best_flat // (nr * nc)
        best_slot = best_flat % (nr * nc)
        best_r = best_slot // nc
        best_c = best_slot % nc

        # Apply transform to source patch to get the COLORED stamp (with decoration where it is)
        # Actually we just need: the stamp shows the original source-tile colors (with 2 still as 2, decoration as decoration)
        # And we apply the transformed version at (best_r, best_c) in the crop
        source_colored = np.zeros((3, 3), dtype=np.int64)
        # source's actual colors: 2 where source_is_2, dec where not
        source_colored = np.where(source_is_2 == 1, 2, dec)
        transformed_stamp = apply_transform(source_colored, TRANSFORMS[best_g])
        out[best_r : best_r + 3, best_c : best_c + 3] = transformed_stamp

    return out.tolist()


def main():
    # Test on seed 233245 first
    sys.path.insert(0, str(REPO / "The_400_Tasks/Task_by_Task/task233/prototypes"))
    from task233_synthetic_gate import synthetic_case

    SEED = 233245
    example = synthetic_case(SEED, stress_dihedral=True)
    grid = np.asarray(example["input"])
    expected = np.asarray(example["output"])
    ref_pred = np.asarray(predict_task233(grid.tolist()))
    h3_pred = np.asarray(signed_score_h3(grid))

    print(f"Seed {SEED}:")
    print(f"  numpy_ref vs expected: {(ref_pred == expected).all()}")
    print(f"  h3 vs expected: {(h3_pred == expected).all()}")
    print(f"  h3 vs numpy_ref: {(h3_pred == ref_pred).all()}")
    if not (h3_pred == expected).all():
        diff = np.where(h3_pred != expected)
        print(f"  diff count: {len(diff[0])}")
        for r, c in zip(diff[0], diff[1]):
            print(f"    ({r}, {c}): expected={expected[r,c]} h3={h3_pred[r,c]}")

    # Now also test on a few seeds adjacent
    print("\nSweep seeds 233233..233253:")
    fails = []
    for i in range(21):
        s = 233233 + i
        try:
            ex = synthetic_case(s, stress_dihedral=True)
            g = np.asarray(ex["input"])
            exp = np.asarray(ex["output"])
            h3 = np.asarray(signed_score_h3(g))
            ok = (h3 == exp).all()
            if not ok:
                fails.append(s)
                print(f"  seed {s}: FAIL ({int((h3 != exp).sum())} cell diff)")
        except Exception as e:
            print(f"  seed {s}: error {e}")
    print(f"\nFailed seeds: {fails}")

    # Run on visible
    import json
    task_path = REPO / "data/competitions/neurogolf-2026/task233.json"
    if task_path.exists():
        task = json.loads(task_path.read_text())
        pairs = task["train"] + task["test"] + task.get("arc-gen", [])
        ok = 0
        for pair in pairs:
            try:
                pred = signed_score_h3(pair["input"])
                if pred == pair["output"]:
                    ok += 1
            except Exception:
                pass
        print(f"\nVisible: {ok}/{len(pairs)}")


if __name__ == "__main__":
    main()
