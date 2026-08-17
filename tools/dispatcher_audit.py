"""Verify the detect_pattern dispatcher picks the cheapest matching pattern for every SOLVED_TASK.

Builds every detector's matching network for each task in SOLVED_TASKS, scores them via
score_network, verifies functional correctness on the first 5 visible pairs, and asserts
the dispatcher's pick is within ε (default 0.5 pts) of the cheapest alternative.

Catches D1-class regressions (dispatcher routes to a more expensive matching pattern when
a cheaper one would also work). Run pre-PR after touching the dispatcher or any detector.

Usage:
    python tools/dispatcher_audit.py
    python tools/dispatcher_audit.py --tolerance 0.5
    python tools/dispatcher_audit.py --tasks 262,266,1

Exit code:
    0  every task's dispatcher pick is within tolerance of the cheapest alternative.
    1  at least one task has a cheaper alternative (worth fixing the dispatcher order).
    2  invocation / setup error.
"""

import argparse
import math
import os
import pathlib
import sys
import types

for _name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["IPython.display"].display = lambda *a, **kw: None
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026")))
SUBMISSIONS_DIR = NG_ROOT / "submissions"
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
sys.path.insert(0, str(NG_ROOT))
sys.path.insert(0, str(NG_ROOT / "tools"))

import neurogolf_utils as nu  # noqa: E402

nu._NEUROGOLF_DIR = f"{DATA_DIR.as_posix()}/"

import onnx  # noqa: E402
import onnxruntime  # noqa: E402

import build_submission as bs  # noqa: E402
from inspect_tasks import SOLVED_TASKS  # noqa: E402


def _candidate_factories():
    """Return list of (label, detector, network_factory_factory) covering every detector.

    Order mirrors `build_submission._collect_matches` so on-tie cost matches preserve the
    dispatcher's preferred candidate. When adding a new detector to `_collect_matches`,
    add the matching entry here too — otherwise this audit can't surface a cheaper
    alternative the dispatcher chose.
    """
    return [
        ("fixed-shape-spatial", bs._detect_fixed_shape_spatial,
         lambda r: bs._fixed_shape_network_factory(r)),
        ("translation", bs._detect_translation,
         lambda r: lambda b=1: bs.translation_network(r["h"], r["w"], r["dr"], r["dc"], batch=b)),
        ("slice-extract", bs._detect_slice_extraction,
         lambda r: lambda b=1: bs.slice_extract_network(r[0], r[1], r[2], r[3], batch=b)),
        ("marker-corners", bs._detect_marker_corners,
         lambda r: lambda b=1: bs.marker_corners_network(r[0], r[1], batch=b)),
        ("isolated-marker-recolor", bs._detect_isolated_marker_recolor,
         lambda r: lambda b=1: bs.isolated_marker_recolor_network(r[0], r[1], batch=b)),
        ("l-shape-missing-corner", bs._detect_l_shape_missing_corner,
         lambda r: lambda b=1: bs.l_shape_missing_corner_network(r[0], r[1], r[2], batch=b)),
        ("sandwich-fill", bs._detect_sandwich_fill,
         lambda r: lambda b=1: bs.sandwich_fill_network(r[0], r[1], r[2], r[3], r[4], batch=b)),
        ("aligned-marker-fill", bs._detect_aligned_marker_fill,
         lambda r: lambda b=1: bs.aligned_marker_fill_network(r[0], r[1], r[2], r[3], r[4], batch=b)),
        ("pi-project-bottom", bs._detect_pi_shape_project_to_bottom,
         lambda r: lambda b=1: bs.pi_shape_project_to_bottom_network(r, batch=b)),
        ("const-dominant-color", bs._detect_constant_output_dominant_color,
         lambda r: lambda b=1: bs.constant_output_dominant_color_network(r[0], r[1], batch=b)),
        ("outer-product-marker-tile", bs._detect_outer_product_marker_tile,
         lambda r: lambda b=1: bs.outer_product_marker_tile_network(r["mode"], marker_color=r.get("marker_color"), batch=b)),
        ("diagonal-shear", bs._detect_diagonal_shear,
         lambda r: lambda b=1: bs.diagonal_shear_network(batch=b)),
        ("symmetry-completion", bs._detect_symmetry_completion,
         lambda r: lambda b=1: bs.symmetry_completion_network(r["axis"], r["h"], r["w"], batch=b)),
        ("bbox-fill", bs._detect_bbox_fill,
         lambda r: lambda b=1: bs.bbox_fill_network(r["target_color"], batch=b)),
        ("gridline-adjacent-fill", bs._detect_gridline_adjacent_fill,
         lambda r: lambda b=1: bs.gridline_adjacent_fill_network(batch=b)),
        ("line-projection-to-full-line", bs._detect_line_projection_to_full_line,
         lambda r: lambda b=1: bs.line_projection_to_full_line_network(batch=b)),
        ("diagonal-zero-mask-projection", bs._detect_diagonal_zero_mask_projection,
         lambda r: lambda b=1: bs.diagonal_zero_mask_projection_network(batch=b)),
        ("marker-shape-copy", bs._detect_marker_shape_copy,
         lambda r: lambda b=1: bs.marker_shape_copy_network(batch=b)),
        ("block-grid-line-fill", bs._detect_block_grid_line_fill,
         lambda r: lambda b=1: bs.block_grid_line_fill_network(batch=b)),
        ("count-2x2-one-blocks", bs._detect_count_2x2_one_blocks,
         lambda r: lambda b=1: bs.count_2x2_one_blocks_network(batch=b)),
        ("same-color-neighbor-keep", bs._detect_same_color_neighbor_keep,
         lambda r: lambda b=1: bs.same_color_neighbor_keep_network(batch=b)),
        ("rectangle-interior-fill8", bs._detect_rectangle_interior_fill8,
         lambda r: lambda b=1: bs.rectangle_interior_fill8_network(r["h"], r["w"], batch=b)),
        ("border-distance-probe-recolor", bs._detect_border_distance_probe_recolor,
         lambda r: lambda b=1: bs.border_distance_probe_recolor_network(r["h"], r["w"], batch=b)),
        ("frame6-border3-fill4", bs._detect_frame6_border3_fill4,
         lambda r: lambda b=1: bs.frame6_border3_fill4_network(r["h"], r["w"], batch=b)),
        ("foreground-count-row", bs._detect_foreground_count_row,
         lambda r: lambda b=1: bs.foreground_count_row_network(batch=b)),
        ("edge-seed-bounce-ray", bs._detect_edge_seed_bounce_ray,
         lambda r: lambda b=1: bs.edge_seed_bounce_ray_network(batch=b)),
        ("outline-extract", bs._detect_outline_extract,
         lambda r: lambda b=1: bs.outline_extract_network(batch=b)),
        ("active-grid-frame", bs._detect_active_grid_frame,
         lambda r: lambda b=1: bs.active_grid_frame_network(r["border_color"], batch=b)),
        ("per-color-stamp", bs._detect_per_color_stamp,
         lambda r: lambda b=1: bs.per_color_stamp_network(r["rules"], batch=b)),
        ("marker-dilate-conv", bs._detect_marker_dilate_conv_safe,
         lambda r: lambda b=1: bs.marker_dilate_conv_network(r["marker_color"], r["target_color"], r["preserve_marker"], batch=b)),
        ("marker-dilate", bs._detect_marker_dilate,
         lambda r: lambda b=1: bs.marker_dilate_network(r["marker_color"], r["target_color"], r["preserve_marker"], batch=b)),
        ("color-perm", lambda p: bs._detect_color_perm(p, require_all_colors=False),
         lambda r: lambda b=1: bs.color_perm_network(r, batch=b)),
        ("color-replace", bs._detect_color_replace,
         lambda r: lambda b=1: bs.color_replace_network(r, batch=b)),
        ("enclosed-region-fill", bs._detect_enclosed_region_fill,
         lambda r: lambda b=1: bs.enclosed_region_fill_network(r, batch=b)),
        ("pixel-perm", bs._detect_pixel_perm,
         lambda r: lambda b=1: bs.pixel_perm_network(r["h"], r["w"], r["indices"], batch=b)),
        ("shape-changing-pixel-perm", bs._detect_shape_changing_pixel_perm,
         lambda r: lambda b=1: bs.shape_changing_pixel_perm_network(r["in_h"], r["in_w"], r["out_h"], r["out_w"], r["indices"], batch=b)),
        ("mirror-grid-2x2", bs._detect_mirror_grid_2x2,
         lambda r: lambda b=1: bs.mirror_grid_2x2_network(r[0], r[1], batch=b)),
        ("mirror-concat", bs._detect_mirror_concat,
         lambda r: lambda b=1: bs.mirror_concat_network(r[0], r[1], r[2], r[3], second_first=r[4], batch=b)),
        ("grid-pattern-to-color", bs._detect_grid_pattern_to_color,
         lambda r: lambda b=1: bs.grid_pattern_to_color_network(r[0], r[1], r[2], batch=b)),
        ("grid-pattern-to-grid", bs._detect_grid_pattern_to_grid,
         lambda r: lambda b=1: bs.grid_pattern_to_grid_network(r[0], r[1], r[2], r[3], r[4], batch=b)),
        ("grid-pattern-color-to-grid", bs._detect_grid_pattern_color_to_grid,
         lambda r: lambda b=1: bs.grid_pattern_color_to_grid_network(r[0], r[1], r[2], r[3], r[4], batch=b)),
        ("bin-nbr-lut", bs._detect_binary_neighborhood_lut,
         lambda r: lambda b=1: bs.binary_neighborhood_lut_network(r[0], r[1], batch=b)),
        ("foreground-bbox-crop", bs._detect_foreground_bbox_crop,
         lambda r: lambda b=1: bs.foreground_bbox_crop_network(batch=b)),
        ("frame-raycast-crop", bs._detect_frame_raycast_crop,
         lambda r: lambda b=1: bs.frame_raycast_crop_network(batch=b)),
        ("marker-rectangle-interior", bs._detect_marker_rectangle_interior,
         lambda r: lambda b=1: bs.marker_rectangle_interior_network(batch=b)),
        ("largest-rectangle-recolor-crop", bs._detect_largest_rectangle_recolor_crop,
         lambda r: lambda b=1: bs.largest_rectangle_recolor_crop_network(batch=b)),
        ("kron-block-up", bs._detect_kron_block_up,
         lambda r: lambda b=1: bs.kron_block_up_network(r["scale_h"], r["scale_w"], batch=b)),
        ("const-largest-bbox", bs._detect_constant_output_largest_bbox,
         lambda r: lambda b=1: bs.largest_bbox_constant_output_network(r[0], r[1], batch=b)),
        ("const-framed-cell", bs._detect_framed_cell_const_output,
         lambda r: lambda b=1: bs.framed_cell_const_output_network(r[0], r[1], batch=b)),
        ("marker-aligned-shape-recolor", bs._detect_marker_aligned_shape_recolor,
         lambda r: lambda b=1: bs.marker_aligned_shape_recolor_network(batch=b)),
        ("grid-division-count", bs._detect_grid_division_count,
         lambda r: lambda b=1: bs.grid_division_count_network(batch=b)),
        ("cross-section-grid-label", bs._detect_cross_section_grid_label,
         lambda r: lambda b=1: bs.cross_section_grid_label_network(batch=b)),
        ("grid-section-copy", bs._detect_grid_section_copy,
         lambda r: lambda b=1: bs.grid_section_copy_network(batch=b)),
        ("grid-max-count-fill", bs._detect_grid_max_count_fill,
         lambda r: lambda b=1: bs.grid_max_count_fill_network(batch=b)),
        ("grid-object-summary", bs._detect_grid_object_summary,
         lambda r: lambda b=1: bs.grid_object_summary_network(batch=b)),
        ("two-blob-tile-fit", bs._detect_two_blob_tile_fit,
         lambda r: lambda b=1: bs.two_blob_tile_fit_network(batch=b)),
        ("fourfold-rotation-tile", bs._detect_fourfold_rotation_tile,
         lambda r: lambda b=1: bs.fourfold_rotation_tile_network(batch=b)),
        ("diagonal-rays-from-line", bs._detect_diagonal_rays_from_line,
         lambda r: lambda b=1: bs.diagonal_rays_from_line_network(batch=b)),
        ("euclidean-wedge-fill", bs._detect_euclidean_wedge_fill,
         lambda r: lambda b=1: bs.euclidean_wedge_fill_network(batch=b)),
        ("bilateral-symmetry-denoise", bs._detect_bilateral_symmetry_denoise,
         lambda r: lambda b=1: bs.bilateral_symmetry_denoise_network(batch=b)),
        ("marker-section-extract", bs._detect_marker_section_extract,
         lambda r: lambda b=1: bs.marker_section_extract_network(batch=b)),
        ("marker-driven-shape-copy", bs._detect_marker_driven_shape_copy,
         lambda r: lambda b=1: bs.marker_driven_shape_copy_network(batch=b)),
        ("periodic-row-extension", bs._detect_periodic_row_extension,
         lambda r: lambda b=1: bs.periodic_row_extension_network(batch=b)),
        ("boolean-halves", bs._detect_boolean_halves,
         lambda r: lambda b=1: bs.boolean_halves_network(r["H"], r["W_in"], r["W_out"], r["check_channel"], r["out_color"], batch=b)),
        ("rank-vertical-bars", bs._detect_rank_vertical_bars,
         lambda r: lambda b=1: bs.rank_vertical_bars_network(batch=b)),
        ("vertical-halves", bs._detect_vertical_halves,
         lambda r: lambda b=1: bs.vertical_halves_network(r["H_in"], r["W"], r["H_half"], r["mode"], r["out_color"], sep_row=r["sep_row"], batch=b)),
        ("horizontal-or", bs._detect_horizontal_or,
         lambda r: lambda b=1: bs.horizontal_or_network(r["H"], r["W_half"], r["out_color"], batch=b)),
        ("cross-product", bs._detect_cross_product,
         lambda r: lambda b=1: bs.cross_product_network(r["N"], r["col_idx"], r["template_color"], r["out_color"], batch=b)),
        ("block-count-ge", bs._detect_block_count_ge,
         lambda r: lambda b=1: bs.block_count_ge_network(r["count_color"], r["threshold"], r["out_color"], r["N"], r["n"], r["block_size"], r["stride"], batch=b)),
        ("marker-recolor", bs._detect_marker_recolor,
         lambda r: lambda b=1: bs.marker_recolor_network(r["H"], r["W"], r["r_marker"], r["c_marker"], batch=b)),
        ("two-marker-cross", bs._detect_two_marker_cross,
         lambda r: lambda b=1: bs.two_marker_cross_network(r["color_A"], r["color_B"], r["color_inter"], r["H"], r["W"], batch=b)),
        ("distinct-count-pattern", bs._detect_distinct_count_pattern,
         lambda r: lambda b=1: bs.distinct_count_pattern_network(r["c1"], r["c2"], r["c3"], r["out_color"], batch=b)),
        ("uniform-row", bs._detect_uniform_row,
         lambda r: lambda b=1: bs.uniform_row_network(r["H"], r["W"], r["out_color"], batch=b)),
        ("cummax-down-column", bs._detect_cummax_down_column,
         lambda r: lambda b=1: bs.cummax_down_column_network(r["H"], r["W"], batch=b)),
        ("dominant-color-keep", bs._detect_dominant_color_keep,
         lambda r: lambda b=1: bs.dominant_color_keep_network(r["H"], r["W"], r["out_color"], batch=b)),
        ("color-to-pattern", bs._detect_color_to_pattern,
         lambda r: lambda b=1: bs.color_to_pattern_network(r["H_in"], r["W_in"], r["color_map"], r["out_color"], batch=b)),
        ("cross-line-fill", bs._detect_cross_line_fill,
         lambda r: lambda b=1: bs.cross_line_fill_network(r["H"], r["W"], r["color_col"], r["color_row"], r["color_inter"], batch=b)),
        ("rectangular-zero-rank-fill", bs._detect_rectangular_zero_rank_fill,
         lambda r: lambda b=1: bs.rectangular_zero_rank_fill_network(r["H"], r["W"], batch=b)),
        ("framed-zero-square-fill", bs._detect_framed_zero_square_fill,
         lambda r: lambda b=1: bs.framed_zero_square_fill_network(
             r["H"], r["W"], r["max_size"], r["frame_color"], r["fill_color"], batch=b)),
        ("bouncing-diagonal-ray", bs._detect_bouncing_diagonal_ray,
         lambda r: lambda b=1: bs.bouncing_diagonal_ray_network(r["max_w"], batch=b)),
        ("trirow-periodic-four-fill", bs._detect_trirow_periodic_four_fill,
         lambda r: lambda b=1: bs.trirow_periodic_four_fill_network(batch=b)),
        ("square-spiral-three", bs._detect_square_spiral_three,
         lambda r: lambda b=1: bs.square_spiral_three_network(r["max_n"], batch=b)),
        ("rotational-symmetry-completion", bs._detect_rotational_symmetry_completion,
         lambda r: lambda b=1: bs.rotational_symmetry_completion_network(batch=b)),
        ("fractal-ring-expansion", bs._detect_fractal_ring_expansion,
         lambda r: lambda b=1: bs.fractal_ring_expansion_network(batch=b)),
        ("color-filter-bbox-crop", lambda p: bs._detect_color_filter_bbox_crop(p, picker="least"),
         lambda r: lambda b=1: bs.color_filter_bbox_crop_network(picker=r["picker"], batch=b)),
        ("color-filter-bbox-crop-sb", lambda p: bs._detect_color_filter_bbox_crop(p, picker="smallest_bbox"),
         lambda r: lambda b=1: bs.color_filter_bbox_crop_network(picker=r["picker"], batch=b)),
        ("color-filter-bbox-crop-mc", lambda p: bs._detect_color_filter_bbox_crop(p, picker="most_count"),
         lambda r: lambda b=1: bs.color_filter_bbox_crop_network(picker=r["picker"], batch=b)),
        ("color-filter-bbox-crop-oo", lambda p: bs._detect_color_filter_bbox_crop(p, picker="odd_one_out"),
         lambda r: lambda b=1: bs.color_filter_bbox_crop_network(picker=r["picker"], batch=b)),
        ("color-filter-bbox-crop-lr", lambda p: bs._detect_color_filter_bbox_crop(p, picker="lr_symmetry"),
         lambda r: lambda b=1: bs.color_filter_bbox_crop_network(picker=r["picker"], batch=b)),
        ("color-filter-bbox-crop-md", lambda p: bs._detect_color_filter_bbox_crop(p, picker="max_density"),
         lambda r: lambda b=1: bs.color_filter_bbox_crop_network(picker=r["picker"], batch=b)),
        ("color-swap-5-stays", lambda p: True if bs._detect_color_swap_5_always_stays(p) else None,
         lambda r: lambda b=1: bs.color_swap_5_always_stays_network(batch=b)),
        ("seed-flood-fill", bs._detect_seed_flood_fill,
         lambda r: lambda b=1: bs.seed_flood_fill_network(r["fill_color"], batch=b)),
        ("two-by-two-zero-fill", bs._detect_two_by_two_zero_fill,
         lambda r: lambda b=1: bs.two_by_two_zero_fill_network(r["H"], r["W"], r["bg_color"], r["fill_color"], batch=b)),
        ("rotation-symmetry-fill", bs._detect_rotation_symmetry_fill,
         lambda r: lambda b=1: bs.rotation_symmetry_fill_network(r["H"], r["W"], r["fg_color"], r["fill_color"], batch=b)),
        ("diagonal-symmetry-fill", bs._detect_diagonal_symmetry_fill,
         lambda r: lambda b=1: bs.diagonal_symmetry_fill_network(r["H"], r["W"], r["fg_color"], r["fill_color"], batch=b)),
        ("frame-interior-fill", bs._detect_frame_interior_fill,
         lambda r: lambda b=1: bs.frame_interior_fill_network(r["frame_color"], r["fill_color"], batch=b)),
        ("marker-to-rectangle-line", bs._detect_marker_to_rectangle_line,
         lambda r: lambda b=1: bs.marker_to_rectangle_line_network(r["h"], r["w"], batch=b)),
        ("marker-adjacent-bbox-reflection", bs._detect_marker_adjacent_bbox_reflection,
         lambda r: lambda b=1: bs.marker_adjacent_bbox_reflection_network(batch=b)),
        ("top-marker-run-recolor", bs._detect_top_marker_run_recolor,
         lambda r: lambda b=1: bs.top_marker_run_recolor_network(batch=b)),
        ("empty-row-col-recolor2", bs._detect_empty_row_col_recolor2,
         lambda r: lambda b=1: bs.empty_row_col_recolor2_network(batch=b)),
        ("arch-marker-fall", bs._detect_arch_marker_fall,
         lambda r: lambda b=1: bs.arch_marker_fall_network(batch=b)),
        ("diagonal-pair-connect", bs._detect_diagonal_pair_connect,
         lambda r: lambda b=1: bs.diagonal_pair_connect_network(batch=b)),
        ("downshift-recolor8-to2", bs._detect_downshift_recolor8_to2,
         lambda r: lambda b=1: bs.downshift_recolor8_to2_network(batch=b)),
        ("staircase-bottom-half-recolor", bs._detect_task320,
         lambda r: lambda b=1: bs.staircase_bottom_half_recolor_network(batch=b)),
        ("horizontal-tile-2x", bs._detect_horizontal_tile_2x,
         lambda r: lambda b=1: bs.horizontal_tile_2x_network(batch=b)),
        ("period-extend-row-2x", bs._detect_period_extend_row_2x,
         lambda r: lambda b=1: bs.period_extend_row_2x_network(batch=b)),
        ("periodic-hole-fill", bs._detect_periodic_hole_fill,
         lambda r: lambda b=1: bs.periodic_hole_fill_network(batch=b)),
        ("zero-adjacency-label", bs._detect_zero_adjacency_label,
         lambda r: lambda b=1: bs.zero_adjacency_label_network(batch=b)),
        ("boustrophedon-pack-3x3", bs._detect_task316,
         lambda r: lambda b=1: bs.boustrophedon_pack_3x3_network(batch=b)),
        ("col-fill-2x2-tile", bs._detect_task388,
         lambda r: lambda b=1: bs.col_fill_2x2_tile_network(batch=b)),
        ("tile-diagonal-neighbor-mark", bs._detect_tile_diagonal_neighbor_mark,
         lambda r: lambda b=1: bs.tile_diagonal_neighbor_mark_network(batch=b)),
        ("color2-bbox-fill", bs._detect_color2_bbox_fill,
         lambda r: lambda b=1: bs.color2_bbox_fill_network(batch=b)),
        ("diagonal-marker-square-rings", bs._detect_diagonal_marker_square_rings,
         lambda r: lambda b=1: bs.diagonal_marker_square_rings_network(batch=b)),
        # ("anchor-cross-aligned-cells", bs._detect_anchor_cross_aligned_cells,
        #  lambda r: lambda b=1: bs.anchor_cross_aligned_cells_network(batch=b)),
        # task270 builder retained but NOT dispatched — see build_submission.py and task270 card.
        ("marker-count-cross-shift", bs._detect_marker_count_cross_shift,
         lambda r: lambda b=1: bs.marker_count_cross_shift_network(batch=b)),
        ("anchor-block-aligned-marker-line", bs._detect_anchor_block_aligned_marker_line,
         lambda r: lambda b=1: bs.anchor_block_aligned_marker_line_network(batch=b)),
        ("frame-inner-markers-to-outer-opposite", bs._detect_frame_inner_markers_to_outer_opposite,
         lambda r: lambda b=1: bs.frame_inner_markers_to_outer_opposite_network(batch=b)),
    ]


def _cost_of(network_fn, scratch_path):
    """Build, save, score the network. Return (cost, points) or None if rejected."""
    try:
        m = network_fn()
        onnx.save(m, str(scratch_path))
    except Exception:  # noqa: BLE001
        return None
    try:
        if not nu.check_network(str(scratch_path)):
            return None
        from _score_v2 import score_network_v2  # noqa: E402
        result = score_network_v2(str(scratch_path))
    except Exception:  # noqa: BLE001
        return None
    if result is None or result[0] is None:
        return None
    macs, mem, params = result
    # 2026-05-04: cost = memory + params (MACs removed from objective).
    cost = mem + params

    pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return cost, pts, cost


def _verifies(network_fn, pairs):
    """Build the network and check that it passes the first 5 visible pairs."""
    try:
        m = network_fn()
        sess = onnxruntime.InferenceSession(m.SerializeToString())
        right, wrong, _ = nu.verify_subset(sess, pairs[:5])
    except Exception:  # noqa: BLE001
        return False
    return wrong == 0


def audit(tasks, tolerance, verbose=True):
    """Return list of {task, current_label, current_cost, current_pts,
    cheaper_label, cheaper_cost, cheaper_pts, delta_pts}."""
    findings = []
    scratch = NG_ROOT / "submissions" / "_audit_scratch.onnx"
    scratch.parent.mkdir(exist_ok=True)
    candidates = _candidate_factories()

    if verbose:
        print(f"Auditing dispatcher across {len(tasks)} tasks "
              f"(tolerance: {tolerance:.2f} pts)")
        print()

    for tn in tasks:
        ex = nu.load_examples(tn)
        pairs = ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", [])
        cur_label, cur_fn = bs.detect_pattern(ex)
        if cur_fn is None:
            if verbose:
                print(f"  task{tn:03d}: dispatcher returns no match (skip)")
            continue
        cur_score = _cost_of(cur_fn, scratch)
        if cur_score is None:
            if verbose:
                print(f"  task{tn:03d}: dispatcher's pick {cur_label} failed scoring (skip)")
            continue
        cur_cost, cur_pts, cur_raw_cost = cur_score

        cheapest = (cur_label, cur_cost, cur_pts, cur_raw_cost)
        for name, det, factf in candidates:
            try:
                r = det(pairs)
            except Exception:  # noqa: BLE001
                continue
            if r is None:
                continue
            net_fn = factf(r)
            if net_fn is None:
                continue
            score = _cost_of(net_fn, scratch)
            if score is None:
                continue
            cost, pts, raw_cost = score
            if cost >= cheapest[1]:
                continue
            if not _verifies(net_fn, pairs):
                continue
            cheapest = (name, cost, pts, raw_cost)

        cheap_label, cheap_cost, cheap_pts, cheap_raw_cost = cheapest
        delta = cheap_pts - cur_pts
        
        risk_flag = ""
        if cur_raw_cost < 1000:
            risk_flag = " [CASCADE-RISK]"

        if delta > tolerance:
            findings.append({
                "task": tn,
                "current_label": cur_label,
                "current_cost": cur_cost,
                "current_pts": cur_pts,
                "cheaper_label": cheap_label,
                "cheaper_cost": cheap_cost,
                "cheaper_pts": cheap_pts,
                "delta_pts": delta,
            })
            if verbose:
                print(f"  task{tn:03d}  REGRESSION  {cur_label}{risk_flag} ({cur_cost:>8} / {cur_pts:.4f} pts) "
                      f"-> cheaper: {cheap_label} ({cheap_cost:>8} / {cheap_pts:.4f} pts) "
                      f"DELTA {delta:+.4f}")
        elif verbose:
            print(f"  task{tn:03d}      ok      {cur_label}{risk_flag} ({cur_cost:>8} / {cur_pts:.4f} pts)")

    try:
        scratch.unlink()
    except (FileNotFoundError, OSError):
        pass

    return findings


def _parse_tasks(arg):
    if not arg:
        return sorted(SOLVED_TASKS)
    return sorted({int(t) for t in arg.split(",") if t.strip()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated task ids (default: SOLVED_TASKS)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Allowable points-delta before flagging a regression (default 0.5)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)
    findings = audit(tasks, args.tolerance, verbose=not args.quiet)

    if not args.quiet:
        print()
        if findings:
            total = sum(f["delta_pts"] for f in findings)
            print(f"  {len(findings)}/{len(tasks)} tasks have a cheaper matching alternative "
                  f"(total recoverable: {total:+.4f} pts)")
            print(f"  Reorder dispatcher or implement cost-aware selection to capture this.")
        else:
            print(f"  All {len(tasks)} tasks: dispatcher picks the cheapest matching pattern.")

    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
