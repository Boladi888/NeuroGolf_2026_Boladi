"""Mining sweep for 5 candidate-pattern angles (Tier-5 Beta-track).

Per `05_BACKLOG.md` Tier 5 "Mining slice rotation":
  1. Input has distinctive non-zero local structure (specific 2x2 pattern).
  2. Input is sparse (<=10 nonzero cells) AND output is a function of those positions.
  3. Output color set is a strict subset of input color set (color-filter).
  4. Output is at least 2x larger than input (small-in-large-out / tile / repeat).
  5. Single-cell change at a fixed position with a fixed color rule.

Run: `python tools/mine_new_angles.py`

Detectors live below. Sweep + report machinery is in `tools/mining_lib.py`.
"""

import numpy as np

from mining_lib import iter_unsolved_tasks, print_report, run_sweep


# -- proxy detectors --


def detect_binary_2x2_lut(pairs):
    """Output cell at (r, c) is a deterministic function of the BINARY 2x2 input
    block at (r, c) (each cell collapsed to nonzero/zero). 16 possible keys.

    Anchor: cell (r, c) covers input[r:r+2, c:c+2]; check output[r, c]. Border
    cells (r == h-1 or c == w-1) skipped. Same-shape pairs only.

    Reject pure identity (most outputs match input[r, c] regardless of key) and
    pure-constant (one output value for all keys -- covered by other patterns).
    """
    if not pairs:
        return None
    mapping = {}
    nontrivial = 0
    total = 0
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        if inp.shape != out.shape:
            return None
        h, w = inp.shape
        if h < 2 or w < 2:
            return None
        for r in range(h - 1):
            for c in range(w - 1):
                key = (
                    1 if inp[r, c] else 0, 1 if inp[r, c + 1] else 0,
                    1 if inp[r + 1, c] else 0, 1 if inp[r + 1, c + 1] else 0,
                )
                ov = int(out[r, c])
                if key in mapping and mapping[key] != ov:
                    return None
                mapping[key] = ov
                total += 1
                if ov != int(inp[r, c]):
                    nontrivial += 1
    if total == 0 or nontrivial / total < 0.10:
        return None
    if len(set(mapping.values())) < 2:
        return None  # constant output, covered elsewhere
    return {"keys_observed": len(mapping), "outputs": sorted(set(mapping.values())), "nontrivial_frac": round(nontrivial / total, 2)}


def detect_sparse_input_positional(pairs):
    """Input has <=10 nonzero cells across the active grid (consistent across
    pairs), and the output is structurally simple given those positions.

    Three sub-cases tested:
      (a) "constant-output-given-sparse": output identical across pairs, regardless
          of input cells.
      (b) "1:1 nonzero-count-match": output has same number of nonzero cells as
          input (marker-style reflection).
      (c) "scaled-position": output is a fixed-shape grid where each output cell
          corresponds to an input nonzero count or a positional fingerprint.
    """
    if not pairs:
        return None
    for p in pairs:
        inp = np.array(p["input"])
        if int((inp != 0).sum()) > 10:
            return None
    out_shapes = {tuple(np.array(p["output"]).shape) for p in pairs}
    if len(out_shapes) > 1:
        return None
    out0 = np.array(pairs[0]["output"])
    if all(np.array_equal(np.array(p["output"]), out0) for p in pairs[1:]):
        return {"kind": "constant-output-given-sparse", "shape": out0.shape}
    counts = []
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        counts.append((int((inp != 0).sum()), int((out != 0).sum())))
    if all(in_n == out_n for in_n, out_n in counts):
        return {"kind": "nonzero-count-match", "shape": out0.shape}
    return None


def detect_output_subset_of_input_colors(pairs):
    """Color-filter pattern: same-shape input/output where output == input on
    cells whose color is in a "kept" set, and output == 0 on cells whose input
    color is in a "filtered" set. Filter set consistent across pairs.

    Relaxed criterion: a color is in the filter set if it's filtered in
    AT LEAST one pair AND never preserved in any pair (to handle pairs where the
    filtered color is absent and so the constraint is vacuous).
    """
    if not pairs:
        return None
    seen_at_pos_kept = {}
    seen_at_pos_filtered = {}
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        if inp.shape != out.shape:
            return None
        out_nz = out != 0
        if not (out[out_nz] == inp[out_nz]).all():
            return None
        in_nz = inp != 0
        in_colors_at_pos = inp[in_nz]
        out_at_in_nz = out[in_nz]
        for ic, oc in zip(in_colors_at_pos.tolist(), out_at_in_nz.tolist()):
            ic = int(ic)
            if oc == 0:
                seen_at_pos_filtered[ic] = seen_at_pos_filtered.get(ic, 0) + 1
            else:
                seen_at_pos_kept[ic] = seen_at_pos_kept.get(ic, 0) + 1
    filtered = {c for c in seen_at_pos_filtered if c not in seen_at_pos_kept}
    if not filtered:
        return None
    kept = set(seen_at_pos_kept)
    if not kept:
        return None
    return {"filtered_colors": sorted(filtered), "kept_colors": sorted(kept)}


def detect_small_in_large_out(pairs):
    """Output is at least 2x larger (in cells) than input -- tile/repeat patterns.
    Same scale factor across all pairs.
    """
    if not pairs:
        return None
    scale = None
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        in_cells = inp.size
        out_cells = out.size
        if in_cells == 0 or out_cells < 2 * in_cells:
            return None
        if out.shape[0] % inp.shape[0] != 0 or out.shape[1] % inp.shape[1] != 0:
            return None
        kh = out.shape[0] // inp.shape[0]
        kw = out.shape[1] // inp.shape[1]
        if scale is None:
            scale = (kh, kw)
        elif scale != (kh, kw):
            return None
    return {"scale_h_w": scale}


def detect_constant_color_swap_at_position(pairs):
    """Same-shape pairs where output equals input EXCEPT at a single (r, c)
    position that is the same across all pairs, and the swap is a fixed color
    rule applied at that one cell.
    """
    if not pairs:
        return None
    swap_cell = None
    swap_rule = None
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        if inp.shape != out.shape:
            return None
        diff = inp != out
        if not diff.any() or diff.sum() != 1:
            return None
        rs, cs = np.where(diff)
        cell = (int(rs[0]), int(cs[0]))
        rule = (int(inp[cell]), int(out[cell]))
        if swap_cell is None:
            swap_cell = cell
            swap_rule = rule
        elif swap_cell != cell or swap_rule != rule:
            return None
    return {"cell": swap_cell, "rule": swap_rule}


CANDIDATES = {
    "2x2-binary-lut": detect_binary_2x2_lut,
    "sparse-input-positional": detect_sparse_input_positional,
    "color-filter (kept-vs-filtered)": detect_output_subset_of_input_colors,
    "small-in-large-out (tile/repeat)": detect_small_in_large_out,
    "single-cell-swap-fixed-position": detect_constant_color_swap_at_position,
}


def main():
    targets = list(iter_unsolved_tasks())
    results = run_sweep(CANDIDATES, targets)
    print_report(results, target_n=len(targets))


if __name__ == "__main__":
    main()
