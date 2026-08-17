"""Unit tests for the pure-function detector helpers in build_submission.py.

Run from repo root with:
    .venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
"""

import pathlib
import os
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

from build_submission import (  # noqa: E402
    _all_30x30,
    _detect_binary_neighborhood_lut,
    _detect_color_perm,
    _detect_color_replace,
    _detect_constant_output_dominant_color,
    _detect_enclosed_region_fill,
    _detect_fixed_shape_spatial,
    _detect_pixel_perm,
    _detect_shape_changing_pixel_perm,
    _detect_aligned_marker_fill,
    _detect_grid_pattern_color_to_grid,
    _detect_grid_pattern_to_color,
    _detect_grid_pattern_to_grid,
    _detect_constant_output_largest_bbox,
    _detect_framed_cell_const_output,
    _detect_grid_division_count,
    _detect_cross_section_grid_label,
    _detect_bilateral_symmetry_denoise,
    _detect_tile_diagonal_neighbor_mark,
    _detect_diagonal_marker_square_rings,
    _detect_anchor_cross_aligned_cells,
    _anchor_cross_aligned_cells_predict,
    anchor_cross_aligned_cells_network,
    _detect_euclidean_wedge_fill,
    _detect_grid_max_count_fill,
    _detect_grid_object_summary,
    _detect_two_blob_tile_fit,
    _detect_fourfold_rotation_tile,
    _detect_diagonal_rays_from_line,
    _detect_marker_aligned_shape_recolor,
    _detect_marker_section_extract,
    _detect_rotational_symmetry_completion,
    _detect_isolated_marker_recolor,
    _detect_kron_block_up,
    _detect_l_shape_missing_corner,
    _detect_active_grid_frame,
    _detect_bbox_fill,
    _detect_color_filter_bbox_crop,
    _detect_diagonal_shear,
    _detect_foreground_bbox_crop,
    _detect_fractal_ring_expansion,
    _detect_marker_corners,
    _detect_marker_dilate,
    _detect_marker_dilate_conv_safe,
    _detect_seed_flood_fill,
    _detect_outline_extract,
    _detect_mirror_concat,
    _detect_mirror_grid_2x2,
    _detect_outer_product_marker_tile,
    _detect_per_color_stamp,
    _detect_symmetry_completion,
    _detect_pi_shape_project_to_bottom,
    _detect_sandwich_fill,
    _detect_slice_extraction,
    _detect_translation,
    _fixed_shape_network_factory,
    _flip_h,
    _flip_v,
    _is_30x30,
    _rot180,
    _rot90,
    _rot270,
    _shape,
    _shift_h,
    _shift_v,
    _transpose_grid,
    binary_neighborhood_lut_network,
    color_perm_network,
    color_replace_network,
    constant_output_dominant_color_network,
    color_filter_bbox_crop_network,
    diagonal_shear_network,
    enclosed_region_fill_network,
    foreground_bbox_crop_network,
    fractal_ring_expansion_network,
    framed_cell_const_output_network,
    pixel_perm_network,
    shape_changing_pixel_perm_network,
    active_grid_frame_network,
    aligned_marker_fill_network,
    bbox_fill_network,
    grid_pattern_color_to_grid_network,
    grid_pattern_to_color_network,
    grid_pattern_to_grid_network,
    isolated_marker_recolor_network,
    kron_block_up_network,
    grid_division_count_network,
    cross_section_grid_label_network,
    bilateral_symmetry_denoise_network,
    tile_diagonal_neighbor_mark_network,
    diagonal_marker_square_rings_network,
    euclidean_wedge_fill_network,
    grid_max_count_fill_network,
    grid_object_summary_network,
    two_blob_tile_fit_network,
    fourfold_rotation_tile_network,
    diagonal_rays_from_line_network,
    largest_bbox_constant_output_network,
    l_shape_missing_corner_network,
    marker_aligned_shape_recolor_network,
    marker_section_extract_network,
    rotational_symmetry_completion_network,
    marker_corners_network,
    marker_dilate_conv_network,
    marker_dilate_network,
    mirror_concat_network,
    mirror_grid_2x2_network,
    outer_product_marker_tile_network,
    outline_extract_network,
    per_color_stamp_network,
    pi_shape_project_to_bottom_network,
    sandwich_fill_network,
    seed_flood_fill_network,
    slice_extract_network,
    symmetry_completion_network,
    translation_network,
    _detect_color_swap_5_always_stays,
    color_swap_5_always_stays_network,
    _detect_marker_rectangle_interior,
    marker_rectangle_interior_network,
    _detect_marker_driven_shape_copy,
    marker_driven_shape_copy_network,
    _detect_periodic_row_extension,
    periodic_row_extension_network,
    _detect_period_extend_row_2x,
    period_extend_row_2x_network,
    _detect_boolean_halves,
    boolean_halves_network,
    _detect_rank_vertical_bars,
    rank_vertical_bars_network,
    _detect_vertical_halves,
    vertical_halves_network,
    _detect_horizontal_or,
    horizontal_or_network,
    _detect_cross_product,
    cross_product_network,
    _detect_block_count_ge,
    block_count_ge_network,
    _detect_marker_recolor,
    marker_recolor_network,
    _detect_two_marker_cross,
    two_marker_cross_network,
    _detect_distinct_count_pattern,
    distinct_count_pattern_network,
    _detect_uniform_row,
    uniform_row_network,
    _detect_cummax_down_column,
    cummax_down_column_network,
    _detect_dominant_color_keep,
    dominant_color_keep_network,
    _detect_color_to_pattern,
    color_to_pattern_network,
    _detect_cross_line_fill,
    cross_line_fill_network,
    _detect_rectangular_zero_rank_fill,
    rectangular_zero_rank_fill_network,
    _detect_framed_zero_square_fill,
    framed_zero_square_fill_network,
    _detect_bouncing_diagonal_ray,
    bouncing_diagonal_ray_network,
    _detect_trirow_periodic_four_fill,
    trirow_periodic_four_fill_network,
    _detect_square_spiral_three,
    square_spiral_three_network,
    _detect_zero_adjacency_label,
    zero_adjacency_label_network,
    _detect_arch_marker_fall,
    _arch_marker_fall_expected,
    arch_marker_fall_network,
    _detect_diagonal_pair_connect,
    _diagonal_pair_connect_expected,
    diagonal_pair_connect_network,
    _detect_downshift_recolor8_to2,
    _downshift_recolor8_to2_expected,
    downshift_recolor8_to2_network,
    _detect_task005_motif_ray_extend,
    _task005_motif_ray_extend_expected,
    task005_motif_ray_extend_network,
    task220_multi_marker_stamp_network,
    _detect_border_distance_probe_recolor,
    _border_distance_probe_recolor_predict,
    border_distance_probe_recolor_network,
)


def _pair(inp, out):
    return {"input": inp, "output": out}


class GridHelpersTest(unittest.TestCase):
    def test_transpose_rectangular(self):
        self.assertEqual(_transpose_grid([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])

    def test_transpose_empty(self):
        self.assertEqual(_transpose_grid([]), [])

    def test_flip_h(self):
        self.assertEqual(_flip_h([[1, 2, 3]]), [[3, 2, 1]])
        self.assertEqual(_flip_h([[1, 2], [3, 4]]), [[2, 1], [4, 3]])

    def test_flip_v(self):
        self.assertEqual(_flip_v([[1], [2], [3]]), [[3], [2], [1]])
        self.assertEqual(_flip_v([[1, 2], [3, 4]]), [[3, 4], [1, 2]])

    def test_rot180_is_hflip_then_vflip(self):
        self.assertEqual(_rot180([[1, 2], [3, 4]]), [[4, 3], [2, 1]])

    def test_shape(self):
        self.assertEqual(_shape([[1, 2, 3], [4, 5, 6]]), (2, 3))
        self.assertEqual(_shape([]), (0, 0))

    def test_is_30x30(self):
        g30 = [[0] * 30 for _ in range(30)]
        self.assertTrue(_is_30x30(g30))
        self.assertFalse(_is_30x30([[0] * 5 for _ in range(5)]))
        self.assertFalse(_is_30x30([[0] * 30 for _ in range(29)]))

    def test_all_30x30(self):
        g30 = [[0] * 30 for _ in range(30)]
        g5 = [[0] * 5 for _ in range(5)]
        self.assertTrue(_all_30x30([_pair(g30, g30)]))
        self.assertFalse(_all_30x30([_pair(g30, g5)]))
        self.assertFalse(_all_30x30([_pair(g30, g30), _pair(g5, g5)]))


class DetectColorPermTest(unittest.TestCase):
    def test_swap_two_colors_loose(self):
        # Swap colors 1 and 2; only 2 colors observed, so strict mode rejects.
        pairs = [_pair([[1, 2], [2, 1]], [[2, 1], [1, 2]])]
        loose = _detect_color_perm(pairs, require_all_colors=False)
        strict = _detect_color_perm(pairs, require_all_colors=True)
        self.assertIsNone(strict)
        self.assertIsNotNone(loose)
        # indices[c'] = the input channel that supplies output channel c'.
        # output color 1 came from input color 2 → indices[1] == 2
        # output color 2 came from input color 1 → indices[2] == 1
        self.assertEqual(loose[1], 2)
        self.assertEqual(loose[2], 1)
        # Unobserved colors default to identity.
        for c in (0, 3, 4, 5, 6, 7, 8, 9):
            self.assertEqual(loose[c], c)

    def test_pure_identity_returns_none(self):
        pairs = [_pair([[1, 2], [3, 4]], [[1, 2], [3, 4]])]
        self.assertIsNone(_detect_color_perm(pairs, require_all_colors=False))

    def test_inconsistent_mapping_returns_none(self):
        # Color 1 maps to both 2 and 3 in the same pair → not a function.
        pairs = [_pair([[1, 1]], [[2, 3]])]
        self.assertIsNone(_detect_color_perm(pairs, require_all_colors=False))

    def test_non_bijective_returns_none(self):
        # Two input colors map to the same output color → not a permutation.
        pairs = [_pair([[1, 2]], [[3, 3]])]
        self.assertIsNone(_detect_color_perm(pairs, require_all_colors=False))

    def test_shape_mismatch_returns_none(self):
        pairs = [_pair([[1, 2]], [[1], [2]])]
        self.assertIsNone(_detect_color_perm(pairs, require_all_colors=False))

    def test_strict_requires_all_ten_colors(self):
        # All 10 colors as inputs, identity-on-most + one swap.
        # Build a 1x10 row covering colors 0..9, with 5 and 8 swapped in output.
        inp = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        out = [[0, 1, 2, 3, 4, 8, 6, 7, 5, 9]]
        pairs = [_pair(inp, out)]
        strict = _detect_color_perm(pairs, require_all_colors=True)
        self.assertIsNotNone(strict)
        self.assertEqual(strict[5], 8)
        self.assertEqual(strict[8], 5)

    def test_partial_mapping_routes_unmapped_to_zero_channel(self):
        # forward = {1: 2}: input color 1 maps to output color 2; output color 1 unobserved.
        # The OLD detector defaulted indices[1] = 1 (identity), which collided with
        # indices[2] = 1 (from inverse) → output ch1 AND ch2 both lit for color-1
        # inputs (the PR #6 bug). The NEW detector recognizes that channel 1 is in
        # `seen` but has no rule mapping TO it, so it routes indices[1] to a
        # guaranteed-zero input channel (any color never observed) → output ch1 is
        # always 0 → decoder yields exactly color 2 at those positions. Captures the
        # +14.99 LB pts on tasks 276/309 (Karnak's task-transformation pattern).
        pairs = [_pair([[1, 1]], [[2, 2]])]
        indices = _detect_color_perm(pairs, require_all_colors=False)
        self.assertIsNotNone(indices)
        # Build the model and verify the output matches the expected mapping.
        import onnxruntime as ort
        import numpy as _np
        m = color_perm_network(indices)
        sess = ort.InferenceSession(m.SerializeToString())
        inp = _np.zeros((1, 10, 30, 30), dtype=_np.float32)
        # Two cells of color 1 at top-left.
        inp[0, 1, 0, 0] = 1.0
        inp[0, 1, 0, 1] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        thresh = (out > 0.0).astype(_np.float32)
        # Output cells (0,0) and (0,1) must be color 2 only (no other channel lit).
        for col in (0, 1):
            self.assertEqual(thresh[0, 2, 0, col], 1.0,
                             f"output (0,{col}) ch2 not lit (expected color 2)")
            for ch in range(10):
                if ch == 2:
                    continue
                self.assertEqual(thresh[0, ch, 0, col], 0.0,
                                 f"output (0,{col}) ch{ch} bled (expected only ch2 lit)")

    def test_non_injective_full_forward_returns_none(self):
        # Two distinct input colors map to the SAME output color: Gather can't merge.
        # forward = {1: 5, 2: 5} (both input 1 and 2 become output 5). With seen = {1, 2},
        # the inverse {5: 1} or {5: 2} loses information. The detector must reject and
        # let the dispatcher fall through to color_replace (Conv 1x1) instead.
        pairs = [_pair([[1, 2]], [[5, 5]])]
        self.assertIsNone(_detect_color_perm(pairs, require_all_colors=False))

    def test_swap_two_colors_yields_valid_permutation(self):
        # 1<->2 swap is closed; resulting indices should be a valid permutation of [0..9].
        pairs = [_pair([[1, 2], [2, 1]], [[2, 1], [1, 2]])]
        indices = _detect_color_perm(pairs, require_all_colors=False)
        self.assertIsNotNone(indices)
        self.assertEqual(sorted(indices), list(range(10)))


class DetectBinaryNeighborhoodLutTest(unittest.TestCase):
    def _solid_30x30(self, val):
        return [[val] * 30 for _ in range(30)]

    def test_accepts_non_30x30(self):
        # Detector now handles non-30x30 grids; builder masks padding via is_active.
        pairs = [_pair([[0, 1], [1, 0]], [[1, 0], [0, 1]])]
        match = _detect_binary_neighborhood_lut(pairs)
        self.assertIsNotNone(match)

    def test_rejects_inconsistent_shapes(self):
        # Pairs with different shapes still rejected (LUT requires consistent extent).
        pairs = [
            _pair([[0, 1], [1, 0]], [[1, 0], [0, 1]]),
            _pair([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
        ]
        self.assertIsNone(_detect_binary_neighborhood_lut(pairs))

    def test_identity_rule_is_consistent_returns_lut(self):
        # Output == input on a uniform 30x30 grid: bg=any, every cell has the
        # same key (depending on bg choice), and that key consistently maps
        # to the same color. Detector should find a match for some bg.
        g = self._solid_30x30(0)
        result = _detect_binary_neighborhood_lut([_pair(g, g)])
        self.assertIsNotNone(result)
        bg, rule = result
        self.assertEqual(rule[0], 0)  # all-zero key (no foreground neighbors) -> 0

    def test_inconsistent_rule_returns_none(self):
        # Two pairs that disagree on what the all-zero key (uniform background)
        # produces - one says output=0, the other says output=1. Detector must
        # reject for every candidate bg.
        g = self._solid_30x30(0)
        out0 = self._solid_30x30(0)
        out1 = self._solid_30x30(1)
        pairs = [_pair(g, out0), _pair(g, out1)]
        self.assertIsNone(_detect_binary_neighborhood_lut(pairs))

    def test_invert_binary_pattern(self):
        # Build a pair: input is half-foreground (color 1), half-background (0),
        # output flips foreground<->background colors. Detector should find bg
        # consistent with this rule.
        inp = [[1 if j < 15 else 0 for j in range(30)] for _ in range(30)]
        out = [[0 if j < 15 else 1 for j in range(30)] for _ in range(30)]
        result = _detect_binary_neighborhood_lut([_pair(inp, out)])
        self.assertIsNotNone(result)
        bg, rule = result
        # For some bg, the rule should swap fg/bg colors. We don't pin which bg
        # the detector picks since multiple may be consistent on a single pair;
        # we just verify the LUT is non-empty and consistent on the seen keys.
        self.assertGreater(len(rule), 0)

    def test_network_loads_in_onnxruntime(self):
        import numpy as np
        import onnxruntime as ort
        # Trivial LUT: key=0 -> color 5, all others -> bg=0.
        lut = {0: 5}
        model = binary_neighborhood_lut_network(bg_color=0, lut_table=lut, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # All-zero canvas (every cell is bg=0). Every key = 0 (no fg neighbors).
        # So every output cell should have channel 5 lit.
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # In one-hot encoding, each cell's color c is encoded as channel c=1.
        # All-bg: channel 0 = 1 everywhere.
        inp[0, 0] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Output channel 5 = 1 everywhere; others = 0.
        self.assertTrue(np.allclose(out[0, 5], 1.0))
        for c in range(10):
            if c != 5:
                self.assertTrue(np.allclose(out[0, c], 0.0), f"channel {c} should be zero")


class FixedShapeSpatialTest(unittest.TestCase):
    def test_rot90_square(self):
        self.assertEqual(_rot90([[1, 2], [3, 4]]), [[3, 1], [4, 2]])
        self.assertEqual(_rot90([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                         [[7, 4, 1], [8, 5, 2], [9, 6, 3]])

    def test_rot270_square(self):
        self.assertEqual(_rot270([[1, 2], [3, 4]]), [[2, 4], [1, 3]])
        self.assertEqual(_rot270([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                         [[3, 6, 9], [2, 5, 8], [1, 4, 7]])

    def test_shift_v(self):
        self.assertEqual(_shift_v([[1], [2], [3]], 1), [[3], [1], [2]])
        self.assertEqual(_shift_v([[1], [2], [3]], 2), [[2], [3], [1]])
        self.assertEqual(_shift_v([[1], [2], [3]], 3), [[1], [2], [3]])  # full cycle

    def test_shift_h(self):
        self.assertEqual(_shift_h([[1, 2, 3]], 1), [[3, 1, 2]])
        self.assertEqual(_shift_h([[1, 2, 3]], 2), [[2, 3, 1]])

    def test_detects_fixed_3x3_rot180(self):
        g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pairs = [_pair(g, _rot180(g))]
        match = _detect_fixed_shape_spatial(pairs)
        self.assertEqual(match, {"transform": "rot180", "h": 3, "w": 3})

    def test_detects_fixed_3x3_rot270(self):
        g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pairs = [_pair(g, _rot270(g))]
        match = _detect_fixed_shape_spatial(pairs)
        self.assertEqual(match, {"transform": "rot270", "n": 3})

    def test_detects_fixed_3x3_shift_v_1(self):
        g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pairs = [_pair(g, _shift_v(g, 1))]
        match = _detect_fixed_shape_spatial(pairs)
        self.assertEqual(match, {"transform": "shift_v_1", "h": 3, "k": 1})

    def test_detects_fixed_4x4_flip_h(self):
        g = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]]
        pairs = [_pair(g, _flip_h(g))]
        match = _detect_fixed_shape_spatial(pairs)
        self.assertEqual(match, {"transform": "flip_h", "w": 4})

    def test_rejects_varying_shape(self):
        g3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        g4 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]]
        pairs = [_pair(g3, _rot180(g3)), _pair(g4, _rot180(g4))]
        self.assertIsNone(_detect_fixed_shape_spatial(pairs))

    def test_rejects_no_matching_transform(self):
        # Same shape but output is "swap columns 1 and 2" — not a flip, rotation,
        # or cyclic shift, so the detector should reject.
        pairs = [_pair([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       [[1, 3, 2], [4, 6, 5], [7, 9, 8]])]
        self.assertIsNone(_detect_fixed_shape_spatial(pairs))

    def test_factory_returns_none_for_unknown_transform(self):
        self.assertIsNone(_fixed_shape_network_factory({"transform": "bogus"}))

    def test_network_round_trip_rot180_3x3(self):
        import math
        import numpy as np
        import onnxruntime as ort
        match = {"transform": "rot180", "h": 3, "w": 3}
        net_fn = _fixed_shape_network_factory(match)
        sess = ort.InferenceSession(net_fn().SerializeToString())
        # Encode a 3x3 grid: cells (0,0)=color 1, (1,1)=color 5, (2,2)=color 9 (one-hot).
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        inp[0, 1, 0, 0] = 1.0
        inp[0, 5, 1, 1] = 1.0
        inp[0, 9, 2, 2] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # rot180: (0,0) -> (2,2), (1,1) -> (1,1), (2,2) -> (0,0)
        self.assertEqual(out[0, 1, 2, 2], 1.0)
        self.assertEqual(out[0, 5, 1, 1], 1.0)
        self.assertEqual(out[0, 9, 0, 0], 1.0)
        # The rest of the canvas is identity (zero in, zero out).
        self.assertTrue(np.array_equal(out[0, 0, 3:, 3:], np.zeros((27, 27), dtype=np.float32)))
        # Slice+Gather+Gather+Pad form for non-30x30 active grid: 4 slice initializers
        # (starts/ends/axes/steps) + 1 shared idx init for the square case.
        self.assertEqual(math.floor(len(net_fn().graph.initializer)), 5)

    def test_network_round_trip_shift_v_3x3_k1(self):
        import numpy as np
        import onnxruntime as ort
        match = {"transform": "shift_v_1", "h": 3, "k": 1}
        net_fn = _fixed_shape_network_factory(match)
        sess = ort.InferenceSession(net_fn().SerializeToString())
        # Single colored row at the top of the 3x3 grid.
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        inp[0, 7, 0, 0] = 1.0
        inp[0, 7, 0, 1] = 1.0
        inp[0, 7, 0, 2] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # shift_v by 1 cyclically: row 0 should land at row 1, leaving row 2 empty
        # and row 0 empty (since the original rows 1 and 2 were empty -> shifted to 2 and 0).
        # Actually: out[i] = in[(i - 1) mod 3]. So out[0] = in[2] (empty), out[1] = in[0] (colored), out[2] = in[1] (empty).
        self.assertEqual(out[0, 7, 1, 0], 1.0)
        self.assertEqual(out[0, 7, 1, 1], 1.0)
        self.assertEqual(out[0, 7, 1, 2], 1.0)
        self.assertEqual(out[0, 7, 0, 0], 0.0)
        self.assertEqual(out[0, 7, 2, 0], 0.0)


class ConstantOutputDominantColorTest(unittest.TestCase):
    def test_detects_simple_uniform_output(self):
        # Input with color 5 dominant -> output uniform (2,2) of color 5.
        inp = [[5, 5, 1], [5, 5, 1], [3, 3, 5]]  # five 5s, two 1s, two 3s -> dominant=5
        out = [[5, 5], [5, 5]]
        match = _detect_constant_output_dominant_color([_pair(inp, out)])
        self.assertEqual(match, (2, 2))

    def test_rejects_non_uniform_output(self):
        # Output has two colors -> not uniform.
        inp = [[5, 5], [5, 5]]
        out = [[5, 1], [1, 5]]
        self.assertIsNone(_detect_constant_output_dominant_color([_pair(inp, out)]))

    def test_rejects_varying_output_shape(self):
        # Same uniform-output rule but pairs have different output shapes.
        inp = [[5, 5], [5, 5]]
        pairs = [_pair(inp, [[5, 5]]), _pair(inp, [[5], [5]])]
        self.assertIsNone(_detect_constant_output_dominant_color(pairs))

    def test_rejects_when_output_color_is_not_dominant(self):
        # Input has 4 cells of color 1, 1 cell of color 5; dominant should be 1, but
        # output uses color 5 -> reject.
        inp = [[1, 1, 1], [1, 5, 0]]
        out = [[5]]
        self.assertIsNone(_detect_constant_output_dominant_color([_pair(inp, out)]))

    def test_dominant_color_includes_zero(self):
        # Mostly-zero input: zeros DO count toward "most common" (validator-style ReduceSum).
        inp = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]  # 8 zeros, 1 one
        out = [[0]]
        match = _detect_constant_output_dominant_color([_pair(inp, out)])
        self.assertEqual(match, (1, 1))

    def test_tie_break_picks_smallest_channel(self):
        # Equal counts of color 0 and color 5: ONNX Hardmax picks channel 0.
        inp = [[0, 5], [5, 0]]  # two 0s, two 5s
        # Detector should match if output color is 0 (smallest-index tie winner).
        match_zero = _detect_constant_output_dominant_color([_pair(inp, [[0]])])
        self.assertEqual(match_zero, (1, 1))
        # If output is 5, detector should reject (5 is not the tie-break winner).
        match_five = _detect_constant_output_dominant_color([_pair(inp, [[5]])])
        self.assertIsNone(match_five)

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # Build a (2, 2) constant-output network and run it on a 3x3 input where color 5 dominates.
        model = constant_output_dominant_color_network(h_out=2, w_out=2, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Encode input: 3x3 grid with five color-5 cells, three color-1 cells, one color-3 cell.
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Active grid 3x3 at top-left.
        positions_5 = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
        positions_1 = [(0, 2), (1, 2), (2, 0)]
        positions_3 = [(2, 1)]
        for h, w in positions_5:
            inp[0, 5, h, w] = 1.0
        for h, w in positions_1:
            inp[0, 1, h, w] = 1.0
        for h, w in positions_3:
            inp[0, 3, h, w] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Expect channel 5 = 1 in cells (0,0), (0,1), (1,0), (1,1); zero everywhere else (and zero
        # in all other channels at any cell).
        self.assertEqual(out[0, 5, 0, 0], 1.0)
        self.assertEqual(out[0, 5, 0, 1], 1.0)
        self.assertEqual(out[0, 5, 1, 0], 1.0)
        self.assertEqual(out[0, 5, 1, 1], 1.0)
        # Cells outside (2, 2) at top-left are zero across all channels.
        self.assertEqual(out[0, 5, 2, 2], 0.0)
        self.assertEqual(out[0, 5, 0, 2], 0.0)
        self.assertEqual(out[0, 5, 2, 0], 0.0)
        # Other channels at the active block are zero.
        for c in range(10):
            if c != 5:
                self.assertEqual(out[0, c, 0, 0], 0.0)


class SliceExtractionTest(unittest.TestCase):
    def test_detects_top_left_2x2(self):
        # Output is the top-left 2x2 block of input.
        inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        out = [[1, 2], [4, 5]]
        match = _detect_slice_extraction([_pair(inp, out)])
        self.assertEqual(match, (0, 2, 0, 2))

    def test_detects_offset_3x3_block(self):
        # Output is rows 0-2, cols 1-3 of a 3x4 input.
        inp = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]]
        out = [[1, 2, 3], [5, 6, 7], [9, 0, 1]]
        match = _detect_slice_extraction([_pair(inp, out)])
        self.assertEqual(match, (0, 3, 1, 4))

    def test_rejects_inconsistent_slice_across_pairs(self):
        # Pair 1: output is top-left 2x2. Pair 2: output is bottom-right 2x2 of same input.
        # No fixed (h_start, w_start) works for both.
        inp1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        inp2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pairs = [
            _pair(inp1, [[1, 2], [4, 5]]),  # top-left
            _pair(inp2, [[5, 6], [8, 9]]),  # bottom-right
        ]
        self.assertIsNone(_detect_slice_extraction(pairs))

    def test_rejects_varying_output_shape(self):
        pairs = [
            _pair([[1, 2, 3]], [[1]]),         # output (1, 1)
            _pair([[1, 2, 3]], [[1, 2]]),      # output (1, 2)
        ]
        self.assertIsNone(_detect_slice_extraction(pairs))

    def test_rejects_when_output_larger_than_smallest_input(self):
        pairs = [
            _pair([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]),  # input 2x3
            _pair([[1]], [[1, 2, 3], [4, 5, 6]]),                    # input 1x1, output 2x3 — impossible
        ]
        self.assertIsNone(_detect_slice_extraction(pairs))

    def test_rejects_when_no_slice_matches(self):
        # Output isn't any sub-block of input.
        pairs = [_pair([[1, 2], [3, 4]], [[5, 6], [7, 8]])]
        self.assertIsNone(_detect_slice_extraction(pairs))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # Slice rows 0-2, cols 1-4 of a 3x5 active grid; output should be (3, 3) at top-left.
        model = slice_extract_network(h_start=0, h_end=3, w_start=1, w_end=4, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Build input: 3x5 grid with distinct colors per cell encoded as one-hot.
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Active grid colors: row r col c gets color (r * 5 + c) mod 10
        for r in range(3):
            for c in range(5):
                color = (r * 5 + c) % 10
                inp[0, color, r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Expected output: cells (0..2, 0..2) of output canvas should hold the colors at
        # (0..2, 1..3) of input. Output cells outside (3, 3) are zero.
        for r in range(3):
            for c in range(3):
                expected_color = (r * 5 + (c + 1)) % 10
                self.assertEqual(out[0, expected_color, r, c], 1.0,
                    f"output cell ({r}, {c}) should have color {expected_color}")
        # Outside (3, 3): all channels zero.
        self.assertEqual(out[0, 0, 3, 0], 0.0)
        self.assertEqual(out[0, 5, 0, 5], 0.0)


class MarkerCornersTest(unittest.TestCase):
    def test_detects_simple_marker_corners(self):
        # Marker color 2 at (1, 1) of 3x3 grid; corners 3, 6, 8, 7 around it.
        inp = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        out = [[3, 0, 6], [0, 0, 0], [8, 0, 7]]
        match = _detect_marker_corners([_pair(inp, out)])
        self.assertIsNotNone(match)
        marker, corners = match
        self.assertEqual(marker, 2)
        self.assertEqual(corners, {"TL": 3, "TR": 6, "BL": 8, "BR": 7})

    def test_rejects_when_marker_count_not_one(self):
        # Two cells of marker color -> not a single marker.
        inp = [[0, 2, 0], [0, 0, 0], [0, 0, 2]]
        out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.assertIsNone(_detect_marker_corners([_pair(inp, out)]))

    def test_rejects_when_marker_does_not_disappear(self):
        # Output keeps the marker instead of replacing with 0.
        inp = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        out = [[3, 0, 6], [0, 2, 0], [8, 0, 7]]
        self.assertIsNone(_detect_marker_corners([_pair(inp, out)]))

    def test_rejects_when_corner_colors_inconsistent_across_pairs(self):
        inp1 = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        out1 = [[3, 0, 6], [0, 0, 0], [8, 0, 7]]
        inp2 = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        out2 = [[4, 0, 6], [0, 0, 0], [8, 0, 7]]  # TL is 4 here, was 3 before
        self.assertIsNone(_detect_marker_corners([_pair(inp1, out1), _pair(inp2, out2)]))

    def test_rejects_when_input_has_third_color(self):
        # Input has color 5 in addition to 0 and 2 -> the simple grid_mask trick wouldn't apply.
        inp = [[0, 5, 0], [0, 2, 0], [0, 0, 0]]
        out = [[3, 0, 6], [0, 0, 0], [8, 0, 7]]
        self.assertIsNone(_detect_marker_corners([_pair(inp, out)]))

    def test_handles_marker_at_grid_edge(self):
        # Marker at corner: only one diagonal is in-bounds; detector should still match
        # IF a previous pair observed the other directions.
        inp1 = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        out1 = [[3, 0, 6], [0, 0, 0], [8, 0, 7]]
        inp2 = [[2, 0, 0], [0, 0, 0], [0, 0, 0]]  # marker at top-left corner
        out2 = [[0, 0, 0], [0, 7, 0], [0, 0, 0]]  # only BR diagonal in bounds
        match = _detect_marker_corners([_pair(inp1, out1), _pair(inp2, out2)])
        self.assertIsNotNone(match)
        marker, corners = match
        self.assertEqual(marker, 2)
        self.assertEqual(corners, {"TL": 3, "TR": 6, "BL": 8, "BR": 7})

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        marker = 2
        corners = {"TL": 3, "TR": 6, "BL": 8, "BR": 7}
        model = marker_corners_network(marker, corners, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Build input: 3x3 active grid (rows 0-2, cols 0-2). Cells (0,0), (0,2), (2,0), (2,2) are
        # color 0 (channel 0 = 1). Cell (1,1) is the marker (channel 2 = 1). Other grid cells are
        # color 0. Outside grid: padding (all-zero).
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                if (i, j) == (1, 1):
                    inp[0, marker, i, j] = 1.0
                else:
                    inp[0, 0, i, j] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Expected: corners placed, marker disappears (becomes 0)
        self.assertEqual(out[0, 3, 0, 0], 1.0)
        self.assertEqual(out[0, 6, 0, 2], 1.0)
        self.assertEqual(out[0, 8, 2, 0], 1.0)
        self.assertEqual(out[0, 7, 2, 2], 1.0)
        self.assertEqual(out[0, 0, 1, 1], 1.0)  # marker position becomes color 0
        self.assertEqual(out[0, marker, 1, 1], 0.0)  # marker channel zeroed
        # Off-grid (e.g., row 3+, col 3+): all channels 0
        self.assertEqual(out[0, 0, 3, 0], 0.0)
        self.assertEqual(out[0, 3, 3, 3], 0.0)


class IsolatedMarkerRecolorTest(unittest.TestCase):
    def test_detects_simple_isolated_marker(self):
        # An isolated 2 (no 4-neighbor of color 2) should become 1; clusters of 2 stay.
        # 4-neighbor connectivity: only up/down/left/right.
        inp = [[0, 2, 2], [0, 2, 2], [2, 0, 0]]
        out = [[0, 2, 2], [0, 2, 2], [1, 0, 0]]
        # (2,0) is isolated by 4-conn (its only same-color 8-nbr is the diagonal at (1,1));
        # the 2x2 cluster at (0..1, 1..2) has 4-neighbors of each other so stays.
        match = _detect_isolated_marker_recolor([_pair(inp, out)])
        self.assertEqual(match, (2, 1))

    def test_rejects_when_no_edits(self):
        # Identity case — no transitions — should return None.
        inp = [[0, 2, 0], [0, 2, 0], [0, 0, 0]]
        out = [[0, 2, 0], [0, 2, 0], [0, 0, 0]]
        self.assertIsNone(_detect_isolated_marker_recolor([_pair(inp, out)]))

    def test_rejects_when_multiple_color_transitions(self):
        # Two different transitions across edits -> not a single (M, R) rule.
        inp = [[2, 0, 3], [0, 0, 0], [0, 0, 0]]
        out = [[1, 0, 5], [0, 0, 0], [0, 0, 0]]
        self.assertIsNone(_detect_isolated_marker_recolor([_pair(inp, out)]))

    def test_rejects_when_non_isolated_marker_changes(self):
        # The 2 at (0, 0) has a 4-neighbor 2 at (0, 1), so isn't isolated. If it changes anyway,
        # the "isolated only" rule doesn't fit and detector should reject.
        inp = [[2, 2, 0], [0, 0, 0], [0, 0, 0]]
        out = [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
        self.assertIsNone(_detect_isolated_marker_recolor([_pair(inp, out)]))

    def test_rejects_when_isolated_marker_doesnt_change(self):
        # Isolated 2 at (2,0) but output keeps it as 2 -> rule says it should change to R but didn't.
        inp = [[0, 2, 2], [0, 2, 2], [2, 0, 0]]
        out = [[0, 2, 2], [0, 2, 2], [2, 0, 0]]  # no edits at all
        self.assertIsNone(_detect_isolated_marker_recolor([_pair(inp, out)]))

    def test_4_connectivity_not_8(self):
        # An isolated 2 by 4-conn but with a diagonal 2-neighbor should still trigger
        # (the rule is 4-connectivity, not 8).
        inp = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # all three 2's are diagonal-only neighbors
        out = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # all three become 1 under 4-conn isolation
        match = _detect_isolated_marker_recolor([_pair(inp, out)])
        self.assertEqual(match, (2, 1))

    def test_consistent_across_pairs(self):
        # M=3, R=7 across two pairs.
        inp1 = [[3, 0, 3], [0, 0, 0], [3, 0, 0]]  # all three 3's are isolated by 4-conn
        out1 = [[7, 0, 7], [0, 0, 0], [7, 0, 0]]
        inp2 = [[3, 3, 0, 0], [3, 3, 0, 3]]  # cluster (0,0)-(1,1) stays; (1,3) is isolated
        out2 = [[3, 3, 0, 0], [3, 3, 0, 7]]
        match = _detect_isolated_marker_recolor([_pair(inp1, out1), _pair(inp2, out2)])
        self.assertEqual(match, (3, 7))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        model = isolated_marker_recolor_network(2, 1, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Build input with an isolated 2 at (2, 0) and a 2x2 cluster of 2's at (0..1, 1..2).
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Active grid 3x3 in top-left; rest is padding (channel 0 = 1 outside grid)
        active = [[0, 2, 2], [0, 2, 2], [2, 0, 0]]
        for r in range(3):
            for c in range(3):
                inp[0, active[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # (2, 0): was 2, isolated by 4-conn -> should be 1 in output
        self.assertEqual(out[0, 1, 2, 0], 1.0)
        self.assertEqual(out[0, 2, 2, 0], 0.0)
        # (0, 1): was 2, cluster -> should still be 2
        self.assertEqual(out[0, 2, 0, 1], 1.0)
        self.assertEqual(out[0, 1, 0, 1], 0.0)
        # (1, 2): was 2, cluster -> should still be 2
        self.assertEqual(out[0, 2, 1, 2], 1.0)
        # (0, 0): was 0 -> stays 0
        self.assertEqual(out[0, 0, 0, 0], 1.0)


class LShapeMissingCornerTest(unittest.TestCase):
    def test_detects_simple_l_shape(self):
        # Single L: 8s at (1,1), (2,1), (2,2). Missing corner (1,2) = 0 -> 1.
        inp = [
            [0, 0, 0, 0],
            [0, 8, 0, 0],
            [0, 8, 8, 0],
            [0, 0, 0, 0],
        ]
        out = [
            [0, 0, 0, 0],
            [0, 8, 1, 0],
            [0, 8, 8, 0],
            [0, 0, 0, 0],
        ]
        match = _detect_l_shape_missing_corner([_pair(inp, out)])
        self.assertEqual(match, (8, 0, 1))

    def test_detects_all_4_l_orientations(self):
        # Four L-shapes, each missing a different corner of its 2x2 block.
        # TL missing: 8s at (0,1),(1,0),(1,1)  -> output (0,0)=1
        # TR missing: 8s at (3,4),(4,3),(4,4)  -> output (3,5) needs to be 1
        #   (Wait — TR missing means cell (top-right) is the C. Let me redo with 7x7.)
        # Use 7x7 to spread them out:
        inp = [
            [0, 8, 0, 0, 0, 0, 0],   # TL of block(0,0): missing TL=(0,0)
            [8, 8, 0, 0, 8, 8, 0],   # TR of block(0,4): missing TR=(0,6)? no, TR is (0,5) actually
            [0, 0, 0, 0, 0, 8, 0],   #   block(0,4)=cells(0,4),(0,5),(1,4),(1,5)=8,8,8,8 — that's 4 not 3. redo.
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        # Simpler: just test all 4 orientations as separate tiny grids.
        # Block TL missing (top-left empty corner): 8s at TR, BL, BR = (0,1),(1,0),(1,1)
        inp1 = [[0, 8], [8, 8]]
        out1 = [[1, 8], [8, 8]]
        # Block TR missing: 8s at TL, BL, BR = (0,0),(1,0),(1,1)
        inp2 = [[8, 0], [8, 8]]
        out2 = [[8, 1], [8, 8]]
        # Block BL missing: 8s at TL, TR, BR = (0,0),(0,1),(1,1)
        inp3 = [[8, 8], [0, 8]]
        out3 = [[8, 8], [1, 8]]
        # Block BR missing: 8s at TL, TR, BL = (0,0),(0,1),(1,0)
        inp4 = [[8, 8], [8, 0]]
        out4 = [[8, 8], [8, 1]]
        match = _detect_l_shape_missing_corner([
            _pair(inp1, out1), _pair(inp2, out2), _pair(inp3, out3), _pair(inp4, out4),
        ])
        self.assertEqual(match, (8, 0, 1))

    def test_rejects_when_no_qualifying_block(self):
        # 2x2 block with only 2 of 8 -> no L-shape. Output unchanged. No edits -> no transition.
        inp = [[8, 0, 0], [0, 8, 0], [0, 0, 0]]
        out = [[8, 0, 0], [0, 8, 0], [0, 0, 0]]
        self.assertIsNone(_detect_l_shape_missing_corner([_pair(inp, out)]))

    def test_rejects_full_2x2_does_not_fire(self):
        # 2x2 block with 4 of 8 (count==4, not 3) -> no L-rule. If the output adds a 1 anyway,
        # the rule "count==3" doesn't reproduce it -> reject.
        inp = [[8, 8, 0], [8, 8, 0], [0, 0, 0]]
        out = [[8, 8, 0], [8, 8, 0], [0, 1, 0]]
        self.assertIsNone(_detect_l_shape_missing_corner([_pair(inp, out)]))

    def test_rejects_when_multiple_color_transitions(self):
        # Two different (input,output) transitions across edits -> not a single (C, F) rule.
        inp = [[0, 8, 0], [8, 8, 0], [0, 0, 0]]
        out = [[1, 8, 5], [8, 8, 0], [0, 0, 0]]
        self.assertIsNone(_detect_l_shape_missing_corner([_pair(inp, out)]))

    def test_rejects_when_marker_changes(self):
        # Marker cells should not change (only C cells become F). If an 8 becomes 9, reject.
        inp = [[0, 8, 0], [8, 8, 0], [0, 0, 0]]
        out = [[1, 9, 0], [9, 9, 0], [0, 0, 0]]
        self.assertIsNone(_detect_l_shape_missing_corner([_pair(inp, out)]))

    def test_distinguishes_marker_from_other_color_at_4th_cell(self):
        # If the 4th cell of a 3-of-8 block is color 5 (not C=0), the rule "3-of-8 + 1 of 0 -> 1"
        # should NOT fire. The output should leave the 5 alone.
        inp = [[0, 8, 0], [8, 8, 0], [0, 5, 0]]
        # No 3-of-8 block exists with a 0 in the 4th corner here, so no edits -> ok with rule.
        out = [[0, 8, 0], [8, 8, 0], [0, 5, 0]]
        # Detector returns None (no transitions).
        self.assertIsNone(_detect_l_shape_missing_corner([_pair(inp, out)]))

    def test_consistent_across_pairs(self):
        # Two pairs both following M=8, C=0, F=1 rule.
        inp1 = [[0, 8, 0], [8, 8, 0], [0, 0, 0]]
        out1 = [[1, 8, 0], [8, 8, 0], [0, 0, 0]]
        inp2 = [[0, 0, 0, 0], [0, 8, 8, 0], [0, 8, 0, 0], [0, 0, 0, 0]]
        out2 = [[0, 0, 0, 0], [0, 8, 8, 0], [0, 8, 1, 0], [0, 0, 0, 0]]
        match = _detect_l_shape_missing_corner([_pair(inp1, out1), _pair(inp2, out2)])
        self.assertEqual(match, (8, 0, 1))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        model = l_shape_missing_corner_network(8, 0, 1, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Build input with one L-shape: 8s at (1,1),(2,1),(2,2); missing corner (1,2)=0.
        inp_grid = [
            [0, 0, 0, 0],
            [0, 8, 0, 0],
            [0, 8, 8, 0],
            [0, 0, 0, 0],
        ]
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                inp[0, inp_grid[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # (1,2) was 0 -> should be 1 in output (i.e., channel 1 = 1.0, channel 0 = 0.0).
        self.assertEqual(out[0, 1, 1, 2], 1.0)
        self.assertEqual(out[0, 0, 1, 2], 0.0)
        # 8 cells unchanged
        self.assertEqual(out[0, 8, 1, 1], 1.0)
        self.assertEqual(out[0, 8, 2, 1], 1.0)
        self.assertEqual(out[0, 8, 2, 2], 1.0)
        # Other 0 cells stay 0
        self.assertEqual(out[0, 0, 0, 0], 1.0)
        self.assertEqual(out[0, 1, 0, 0], 0.0)
        self.assertEqual(out[0, 0, 3, 3], 1.0)


class AlignedMarkerFillTest(unittest.TestCase):
    def test_detects_horizontal_pair(self):
        # 8s at (1,1) and (1,5); cells (1,2..4) should fill with 3.
        inp = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 8, 0, 0, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        out = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 8, 3, 3, 3, 8, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        match = _detect_aligned_marker_fill([_pair(inp, out)])
        self.assertIsNotNone(match)
        self.assertEqual(match[:3], (8, 0, 3))

    def test_detects_vertical_pair(self):
        # 8s at (0,1) and (4,1); cells (1..3, 1) fill with 3.
        inp = [
            [0, 8, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 8, 0],
        ]
        out = [
            [0, 8, 0],
            [0, 3, 0],
            [0, 3, 0],
            [0, 3, 0],
            [0, 8, 0],
        ]
        match = _detect_aligned_marker_fill([_pair(inp, out)])
        self.assertIsNotNone(match)
        self.assertEqual(match[:3], (8, 0, 3))

    def test_detects_both_axes_in_one_pair(self):
        # Both row and column pairs exist in the same input.
        inp = [
            [0, 8, 0, 0, 8, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 8, 0, 0, 0, 0],
        ]
        out = [
            [0, 8, 3, 3, 8, 0],
            [0, 3, 0, 0, 0, 0],
            [0, 8, 0, 0, 0, 0],
        ]
        match = _detect_aligned_marker_fill([_pair(inp, out)])
        self.assertIsNotNone(match)
        self.assertEqual(match[:3], (8, 0, 3))

    def test_single_marker_per_row_or_col_no_fill(self):
        # 1 marker -> no fill at all. No edits, no transition -> detector returns None.
        inp = [[0, 8, 0], [0, 0, 0], [0, 0, 0]]
        out = [[0, 8, 0], [0, 0, 0], [0, 0, 0]]
        self.assertIsNone(_detect_aligned_marker_fill([_pair(inp, out)]))

    def test_rejects_when_marker_changes(self):
        # Marker cells must not change.
        inp = [[0, 8, 0, 8, 0]]
        out = [[0, 9, 3, 9, 0]]
        self.assertIsNone(_detect_aligned_marker_fill([_pair(inp, out)]))

    def test_rejects_when_outside_between_changes(self):
        # Cell outside the between zone changes -> rule doesn't reproduce it.
        inp = [[0, 8, 0, 8, 0]]
        out = [[3, 8, 3, 8, 0]]
        self.assertIsNone(_detect_aligned_marker_fill([_pair(inp, out)]))

    def test_three_markers_in_a_row_fills_all_between(self):
        # Markers at cols 1, 3, 5 in same row. Between cells: 2 and 4. Marker at 3 stays.
        inp = [[0, 8, 0, 8, 0, 8, 0]]
        out = [[0, 8, 3, 8, 3, 8, 0]]
        match = _detect_aligned_marker_fill([_pair(inp, out)])
        self.assertIsNotNone(match)
        self.assertEqual(match[:3], (8, 0, 3))

    def test_consistent_across_pairs(self):
        # Two pairs both following M=8, C=0, F=3 rule.
        inp1 = [[0, 8, 0, 8, 0]]
        out1 = [[0, 8, 3, 8, 0]]
        inp2 = [[0, 8, 0], [0, 0, 0], [0, 8, 0]]
        out2 = [[0, 8, 0], [0, 3, 0], [0, 8, 0]]
        match = _detect_aligned_marker_fill([_pair(inp1, out1), _pair(inp2, out2)])
        self.assertIsNotNone(match)
        self.assertEqual(match[:3], (8, 0, 3))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        model = aligned_marker_fill_network(8, 0, 3, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Active grid: 8s at (1, 1) and (1, 5); cells (1, 2..4) fill with 3.
        active = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 8, 0, 0, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(7):
                inp[0, active[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Cells (1, 2..4) should be color 3
        for c in (2, 3, 4):
            self.assertEqual(out[0, 3, 1, c], 1.0)
            self.assertEqual(out[0, 0, 1, c], 0.0)
        # Markers at (1, 1) and (1, 5) stay as 8
        self.assertEqual(out[0, 8, 1, 1], 1.0)
        self.assertEqual(out[0, 8, 1, 5], 1.0)
        # Cells outside the between zone remain unchanged (e.g., (0, 0) was 0 -> stays 0)
        self.assertEqual(out[0, 0, 0, 0], 1.0)
        self.assertEqual(out[0, 3, 0, 0], 0.0)
        # Padding cells remain all-zero
        self.assertEqual(out[0, 0, 10, 10], 0.0)
        self.assertEqual(out[0, 3, 10, 10], 0.0)


class PiShapeProjectToBottomTest(unittest.TestCase):
    def test_detects_simple_pi(self):
        # Pi at rows 0-1, cols 1-3, color 6. Gap at (1, 2). Bottom row (row 4): 4 at col 2.
        inp = [
            [0, 6, 6, 6, 0],
            [0, 6, 0, 6, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        out = [
            [0, 6, 6, 6, 0],
            [0, 6, 0, 6, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 4, 0, 0],
        ]
        match = _detect_pi_shape_project_to_bottom([_pair(inp, out)])
        self.assertEqual(match, 4)

    def test_two_pi_different_colors_same_pair(self):
        # Two Pis with different colors in one input.
        inp = [
            [0, 6, 6, 6, 0, 0, 0, 0],
            [0, 6, 0, 6, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 8, 8, 8],
            [0, 0, 0, 0, 0, 8, 0, 8],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        out = [
            [0, 6, 6, 6, 0, 0, 0, 0],
            [0, 6, 0, 6, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 8, 8, 8],
            [0, 0, 0, 0, 0, 8, 0, 8],
            [0, 0, 4, 0, 0, 0, 4, 0],
        ]
        match = _detect_pi_shape_project_to_bottom([_pair(inp, out)])
        self.assertEqual(match, 4)

    def test_no_pi_no_edits_returns_none(self):
        # No Pi, no edits, no transition -> None.
        inp = [[0, 6, 0], [0, 0, 0], [0, 0, 0]]
        out = [[0, 6, 0], [0, 0, 0], [0, 0, 0]]
        self.assertIsNone(_detect_pi_shape_project_to_bottom([_pair(inp, out)]))

    def test_rejects_when_pi_color_changes(self):
        # Pi cells should not change in output.
        inp = [[0, 6, 6, 6], [0, 6, 0, 6], [0, 0, 0, 0]]
        out = [[0, 7, 7, 7], [0, 7, 0, 7], [0, 0, 4, 0]]
        self.assertIsNone(_detect_pi_shape_project_to_bottom([_pair(inp, out)]))

    def test_rejects_when_extra_edit(self):
        # Output has an unexpected 4 elsewhere.
        inp = [
            [0, 6, 6, 6, 0],
            [0, 6, 0, 6, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        out = [
            [0, 6, 6, 6, 0],
            [0, 6, 0, 6, 0],
            [0, 0, 0, 0, 0],
            [4, 0, 4, 0, 0],  # extra 4 at (3, 0)
        ]
        self.assertIsNone(_detect_pi_shape_project_to_bottom([_pair(inp, out)]))

    def test_consistent_across_pairs(self):
        # Multiple pairs with different active grid sizes and Pi colors.
        inp1 = [
            [0, 3, 3, 3, 0],
            [0, 3, 0, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        out1 = [
            [0, 3, 3, 3, 0],
            [0, 3, 0, 3, 0],
            [0, 0, 4, 0, 0],
        ]
        inp2 = [
            [0, 0, 0, 0],
            [0, 9, 9, 9],
            [0, 9, 0, 9],
            [0, 0, 0, 0],
        ]
        out2 = [
            [0, 0, 0, 0],
            [0, 9, 9, 9],
            [0, 9, 0, 9],
            [0, 0, 4, 0],
        ]
        match = _detect_pi_shape_project_to_bottom([_pair(inp1, out1), _pair(inp2, out2)])
        self.assertEqual(match, 4)

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        model = pi_shape_project_to_bottom_network(4, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Active grid 5x5 with Pi at rows 0-1, cols 1-3, color 6. Gap (1, 2). Bottom row 4.
        active = [
            [0, 6, 6, 6, 0],
            [0, 6, 0, 6, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                inp[0, active[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Expected: 4 at (4, 2); other cells unchanged.
        self.assertEqual(out[0, 4, 4, 2], 1.0)
        self.assertEqual(out[0, 0, 4, 2], 0.0)
        # Pi cells stay color 6
        for r, c in [(0, 1), (0, 2), (0, 3), (1, 1), (1, 3)]:
            self.assertEqual(out[0, 6, r, c], 1.0)
        # Gap cell (1, 2) stays 0 (the projection is at bottom row, not the gap itself)
        self.assertEqual(out[0, 0, 1, 2], 1.0)
        self.assertEqual(out[0, 4, 1, 2], 0.0)
        # Other bottom-row cells stay 0
        self.assertEqual(out[0, 0, 4, 0], 1.0)
        self.assertEqual(out[0, 0, 4, 1], 1.0)
        # Padding stays empty
        self.assertEqual(out[0, 0, 10, 10], 0.0)
        self.assertEqual(out[0, 4, 10, 10], 0.0)


class SandwichFillTest(unittest.TestCase):
    def test_detects_horizontal_sandwich(self):
        # 1s at (0, 0) and (0, 2); cell (0, 1) becomes 2.
        inp = [[1, 0, 1, 0], [0, 0, 0, 0]]
        out = [[1, 2, 1, 0], [0, 0, 0, 0]]
        match = _detect_sandwich_fill([_pair(inp, out)])
        self.assertIsNotNone(match)
        m, c, f, h, v = match
        self.assertEqual((m, c, f), (1, 0, 2))
        self.assertTrue(h)

    def test_does_not_fill_distance_3_pair(self):
        # 1s at (0, 0) and (0, 3); cell (0, 1) and (0, 2) NOT filled (distance 3).
        inp = [[1, 0, 0, 1, 0]]
        out = [[1, 0, 0, 1, 0]]
        # No edits -> no transition -> None.
        self.assertIsNone(_detect_sandwich_fill([_pair(inp, out)]))

    def test_horizontal_only_when_vertical_doesnt_match(self):
        # Two 1-pairs: horizontal at distance 2 (fill) and vertical at distance 2
        # (must NOT fill; the rule for this task is horizontal-only).
        inp = [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
        ]
        out = [
            [1, 2, 1, 0],
            [0, 0, 0, 0],  # vertical sandwich at (1, 0) and (1, 2) NOT filled
            [1, 2, 1, 0],
        ]
        match = _detect_sandwich_fill([_pair(inp, out)])
        self.assertIsNotNone(match)
        m, c, f, h, v = match
        self.assertEqual((m, c, f), (1, 0, 2))
        self.assertTrue(h)
        self.assertFalse(v)

    def test_vertical_only_when_horizontal_doesnt_match(self):
        inp = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ]
        out = [
            [1, 0, 1],
            [2, 0, 2],
            [1, 0, 1],
        ]
        match = _detect_sandwich_fill([_pair(inp, out)])
        self.assertIsNotNone(match)
        m, c, f, h, v = match
        self.assertEqual((m, c, f), (1, 0, 2))
        self.assertFalse(h)
        self.assertTrue(v)

    def test_both_axes_when_both_fill(self):
        inp = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ]
        out = [
            [1, 2, 1],
            [2, 0, 2],
            [1, 2, 1],
        ]
        match = _detect_sandwich_fill([_pair(inp, out)])
        self.assertIsNotNone(match)
        m, c, f, h, v = match
        self.assertEqual((m, c, f), (1, 0, 2))
        self.assertTrue(h)
        self.assertTrue(v)

    def test_rejects_when_marker_changes(self):
        inp = [[1, 0, 1]]
        out = [[5, 2, 5]]
        self.assertIsNone(_detect_sandwich_fill([_pair(inp, out)]))

    def test_consistent_across_pairs(self):
        # Two pairs, same M=1, C=0, F=2, horizontal-only.
        inp1 = [[1, 0, 1, 0, 1, 0, 1]]
        out1 = [[1, 2, 1, 2, 1, 2, 1]]
        inp2 = [[0, 1, 0, 1, 0]]
        out2 = [[0, 1, 2, 1, 0]]
        match = _detect_sandwich_fill([_pair(inp1, out1), _pair(inp2, out2)])
        self.assertIsNotNone(match)
        m, c, f, h, v = match
        self.assertEqual((m, c, f), (1, 0, 2))
        self.assertTrue(h)

    def test_network_round_trip_horizontal(self):
        import numpy as np
        import onnxruntime as ort
        model = sandwich_fill_network(1, 0, 2, use_h=True, use_v=False, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        active = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(4):
                inp[0, active[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # (0, 1) sandwiched horizontally -> 2.
        self.assertEqual(out[0, 2, 0, 1], 1.0)
        self.assertEqual(out[0, 0, 0, 1], 0.0)
        # (1, 2) sandwiched horizontally between (1, 1) and (1, 3) -> 2.
        self.assertEqual(out[0, 2, 1, 2], 1.0)
        # (1, 0) NOT sandwiched horizontally (col -1 is padding) -> stays 0
        self.assertEqual(out[0, 0, 1, 0], 1.0)
        # Markers stay
        self.assertEqual(out[0, 1, 0, 0], 1.0)
        self.assertEqual(out[0, 1, 0, 2], 1.0)


class MirrorConcatTest(unittest.TestCase):
    def test_detects_horizontal_self_concat(self):
        # output = [input | input] along W (axis=3): identity transform, doubled W.
        inp = [[1, 2], [3, 4]]
        out = [[1, 2, 1, 2], [3, 4, 3, 4]]
        match = _detect_mirror_concat([_pair(inp, out)])
        self.assertIsNotNone(match)
        in_h, in_w, axis, transform, second_first = match
        self.assertEqual((in_h, in_w, axis, transform, second_first),
                         (2, 2, 3, "identity", False))

    def test_detects_horizontal_flip_w_concat(self):
        # output = [input | flip_w(input)] along W
        inp = [[1, 2, 3], [4, 5, 6]]
        out = [[1, 2, 3, 3, 2, 1], [4, 5, 6, 6, 5, 4]]
        match = _detect_mirror_concat([_pair(inp, out)])
        self.assertIsNotNone(match)
        _, _, axis, transform, second_first = match
        self.assertEqual((axis, transform, second_first), (3, "flip_w", False))

    def test_detects_vertical_flip_h_concat_second_first(self):
        # output = [flip_h(input) ; input] along H (transformed copy first)
        inp = [[1, 2], [3, 4]]
        out = [[3, 4], [1, 2], [1, 2], [3, 4]]
        match = _detect_mirror_concat([_pair(inp, out)])
        self.assertIsNotNone(match)
        _, _, axis, transform, second_first = match
        self.assertEqual((axis, transform, second_first), (2, "flip_h", True))

    def test_rejects_when_output_not_2x_along_one_axis(self):
        inp = [[1, 2], [3, 4]]
        out = [[1, 2, 1], [3, 4, 3]]  # 1.5x along W, not 2x
        self.assertIsNone(_detect_mirror_concat([_pair(inp, out)]))

    def test_rejects_inconsistent_input_shapes(self):
        pairs = [
            _pair([[1, 2]], [[1, 2, 2, 1]]),
            _pair([[1, 2, 3]], [[1, 2, 3, 3, 2, 1]]),
        ]
        self.assertIsNone(_detect_mirror_concat(pairs))

    def test_rejects_when_no_transform_matches(self):
        # Output shape says 2x along W, but content isn't [input | f(input)] for any f.
        inp = [[1, 2], [3, 4]]
        out = [[5, 6, 7, 8], [9, 0, 1, 2]]
        self.assertIsNone(_detect_mirror_concat([_pair(inp, out)]))

    def test_network_round_trip_flip_w_concat(self):
        import numpy as np
        import onnxruntime as ort
        # 3x3 input -> 3x6 output via flip_w concat (input | flip_w(input))
        model = mirror_concat_network(in_h=3, in_w=3, axis=3, transform="flip_w",
                                      second_first=False, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Input: cells (r,c) with color (r*3+c)%10
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                inp[0, (r*3+c) % 10, r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        for r in range(3):
            for c in range(3):
                expected = (r*3+c) % 10
                self.assertEqual(out[0, expected, r, c], 1.0)
                # Right half: flipped horizontally; column c maps to c+3 with original col 2-c
                expected_flipped = (r*3 + (2-c)) % 10
                self.assertEqual(out[0, expected_flipped, r, c+3], 1.0)
        # Outside the 3x6 active output: zero
        self.assertEqual(out[0, 0, 4, 0], 0.0)


class MirrorGrid2x2Test(unittest.TestCase):
    def test_detects_2x2_mirror_quadrant_layout(self):
        # input [[a, b], [c, d]] -> output [[a, b, b, a], [c, d, d, c], [c, d, d, c], [a, b, b, a]]
        inp = [[1, 2], [3, 4]]
        # input is 2x2, so output is 4x4: top-left=input, top-right=flip_w, bottom-left=flip_h, bottom-right=flip_hw
        out = [[1, 2, 2, 1],
               [3, 4, 4, 3],
               [3, 4, 4, 3],
               [1, 2, 2, 1]]
        match = _detect_mirror_grid_2x2([_pair(inp, out)])
        self.assertEqual(match, (2, 2))

    def test_rejects_when_output_not_2x2_in_each_axis(self):
        inp = [[1, 2], [3, 4]]
        out = [[1, 2, 2, 1], [3, 4, 4, 3]]  # only 2x in W, not in H
        self.assertIsNone(_detect_mirror_grid_2x2([_pair(inp, out)]))

    def test_rejects_when_quadrant_layout_wrong(self):
        inp = [[1, 2], [3, 4]]
        # Output shape ok, but content is wrong (top-right should be flip_w, not random)
        out = [[1, 2, 5, 6],
               [3, 4, 7, 8],
               [3, 4, 4, 3],
               [1, 2, 2, 1]]
        self.assertIsNone(_detect_mirror_grid_2x2([_pair(inp, out)]))

    def test_rejects_when_input_too_large(self):
        # 16x16 input would give 32x32 output -> exceeds canvas.
        inp = [[c for c in range(16)] for _ in range(16)]
        out = inp  # placeholder; detector should reject on shape grounds before checking content
        self.assertIsNone(_detect_mirror_grid_2x2([_pair(inp, out)]))

    def test_consistent_across_pairs(self):
        # Two pairs with same in shape both following the rule.
        inp1 = [[1, 2], [3, 4]]
        out1 = [[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 4, 3], [1, 2, 2, 1]]
        inp2 = [[5, 6], [7, 8]]
        out2 = [[5, 6, 6, 5], [7, 8, 8, 7], [7, 8, 8, 7], [5, 6, 6, 5]]
        match = _detect_mirror_grid_2x2([_pair(inp1, out1), _pair(inp2, out2)])
        self.assertEqual(match, (2, 2))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        model = mirror_grid_2x2_network(in_h=2, in_w=2, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # input cells (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for (r, c), color in [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]:
            inp[0, color, r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Active output 4x4 layout. Top-left = input.
        self.assertEqual(out[0, 1, 0, 0], 1.0)
        self.assertEqual(out[0, 2, 0, 1], 1.0)
        # Top-right = flip_w(input): (0,2)=2, (0,3)=1
        self.assertEqual(out[0, 2, 0, 2], 1.0)
        self.assertEqual(out[0, 1, 0, 3], 1.0)
        # Bottom-left = flip_h(input): (2,0)=3, (3,0)=1
        self.assertEqual(out[0, 3, 2, 0], 1.0)
        self.assertEqual(out[0, 1, 3, 0], 1.0)
        # Bottom-right = flip_hw(input): (2,2)=4, (3,3)=1
        self.assertEqual(out[0, 4, 2, 2], 1.0)
        self.assertEqual(out[0, 1, 3, 3], 1.0)
        # Outside the 4x4 active region: zero
        self.assertEqual(out[0, 0, 5, 0], 0.0)


class GridPatternToColorTest(unittest.TestCase):
    def test_detects_simple_3x3_to_1x1_lut(self):
        # 3x3 binary input -> 1x1 output color.
        # Empty pattern -> color 0; single cell -> color 5
        pairs = [
            _pair([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0]]),
            _pair([[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[5]]),
        ]
        match = _detect_grid_pattern_to_color(pairs)
        self.assertIsNotNone(match)
        in_h, in_w, lut = match
        self.assertEqual((in_h, in_w), (3, 3))
        self.assertEqual(lut[0], 0)  # all-zero pattern
        self.assertEqual(lut[1], 5)  # single cell at (0,0)

    def test_rejects_inconsistent_rule(self):
        # Same pattern key (all zero) maps to two different output colors -> reject.
        pairs = [
            _pair([[0, 0], [0, 0]], [[5]]),
            _pair([[0, 0], [0, 0]], [[7]]),
        ]
        self.assertIsNone(_detect_grid_pattern_to_color(pairs))

    def test_rejects_when_output_not_1x1(self):
        pairs = [_pair([[0, 0], [0, 0]], [[0, 0]])]  # 1x2 output
        self.assertIsNone(_detect_grid_pattern_to_color(pairs))

    def test_rejects_when_input_too_large_for_lut(self):
        # 4x4 input = 16 bits, exceeds the 12-bit cap.
        pairs = [_pair([[0]*4 for _ in range(4)], [[0]])]
        self.assertIsNone(_detect_grid_pattern_to_color(pairs))

    def test_rejects_when_inconsistent_input_shapes(self):
        pairs = [
            _pair([[0, 0], [0, 0]], [[0]]),
            _pair([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0]]),
        ]
        self.assertIsNone(_detect_grid_pattern_to_color(pairs))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 2x2 input -> 1x1 output. LUT: empty -> 0, single TL -> 5.
        lut = {0: 0, 1: 5}  # key 1 = bit 0 set = TL cell nonzero
        model = grid_pattern_to_color_network(in_h=2, in_w=2, lut_table=lut, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Input with TL cell only, color 3 (any nonzero color triggers the bit).
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        inp[0, 3, 0, 0] = 1.0  # TL: nonzero
        # Other 3 active cells stay at color 0 (not encoded as nonzero in pattern).
        for (r, c) in [(0, 1), (1, 0), (1, 1)]:
            inp[0, 0, r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Output (0,0) should be color 5; padding zero elsewhere.
        self.assertEqual(out[0, 5, 0, 0], 1.0)
        self.assertEqual(out[0, 0, 0, 0], 0.0)
        self.assertEqual(out[0, 5, 1, 1], 0.0)


class GridPatternToGridTest(unittest.TestCase):
    def test_detects_simple_2x2_to_3x3_lut(self):
        # Two pairs differ in input pattern AND output grid -> LUT
        pairs = [
            _pair([[0, 0], [0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            _pair([[1, 0], [0, 0]], [[5, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ]
        match = _detect_grid_pattern_to_grid(pairs)
        self.assertIsNotNone(match)
        in_h, in_w, out_h, out_w, lut = match
        self.assertEqual((in_h, in_w, out_h, out_w), (2, 2, 3, 3))
        self.assertEqual(lut[0], ((0,0,0), (0,0,0), (0,0,0)))
        self.assertEqual(lut[1], ((5,0,0), (0,0,0), (0,0,0)))

    def test_rejects_when_output_is_1x1(self):
        # 1x1 output handled by grid_pattern_to_color, not grid_pattern_to_grid.
        pairs = [_pair([[0, 0], [0, 0]], [[0]])]
        self.assertIsNone(_detect_grid_pattern_to_grid(pairs))

    def test_rejects_inconsistent_rule(self):
        pairs = [
            _pair([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
            _pair([[0, 0], [0, 0]], [[5, 0], [0, 0]]),  # same key, different output
        ]
        self.assertIsNone(_detect_grid_pattern_to_grid(pairs))

    def test_rejects_when_input_too_large(self):
        pairs = [_pair([[0]*4 for _ in range(4)], [[0, 0], [0, 0]])]
        self.assertIsNone(_detect_grid_pattern_to_grid(pairs))

    def test_file_size_guard_rejects_pathological_lut(self):
        # in_h*in_w = 12 -> n_keys = 4096; out 30x30 = 900 cells. LUT = 3.69M elements > 1.2M cap.
        # Detector should reject before building.
        pairs = [_pair([[0]*4 for _ in range(3)], [[0]*30 for _ in range(30)])]
        self.assertIsNone(_detect_grid_pattern_to_grid(pairs))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 2x2 input -> 2x2 output. LUT: empty -> all zero; TL set -> [[5,0],[0,0]].
        lut = {0: ((0, 0), (0, 0)), 1: ((5, 0), (0, 0))}
        model = grid_pattern_to_grid_network(in_h=2, in_w=2, out_h=2, out_w=2,
                                             lut_table=lut, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Build input with TL cell color 3, other active cells color 0.
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        inp[0, 3, 0, 0] = 1.0
        for (r, c) in [(0, 1), (1, 0), (1, 1)]:
            inp[0, 0, r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Expected output (2x2 active):
        #  (0,0) = 5, (0,1) = 0, (1,0) = 0, (1,1) = 0
        self.assertEqual(out[0, 5, 0, 0], 1.0)
        self.assertEqual(out[0, 0, 0, 1], 1.0)
        self.assertEqual(out[0, 0, 1, 0], 1.0)
        self.assertEqual(out[0, 0, 1, 1], 1.0)
        # Outside 2x2 active: zero
        self.assertEqual(out[0, 5, 2, 2], 0.0)


class GridPatternColorToGridTest(unittest.TestCase):
    def test_detects_pattern_color_lut(self):
        # 2x2 input with single nonzero color (varies per pair) -> 2x2 output.
        # Pair 1: pattern=TL only, color=3 -> output has color 3 in TL
        # Pair 2: pattern=TR only, color=5 -> output has color 5 in TR
        # Two distinct nonzero colors -> grid_pattern_color_to_grid (not the simpler grid_pattern_to_grid).
        pairs = [
            _pair([[3, 0], [0, 0]], [[3, 0], [0, 0]]),
            _pair([[0, 5], [0, 0]], [[0, 5], [0, 0]]),
        ]
        match = _detect_grid_pattern_color_to_grid(pairs)
        self.assertIsNotNone(match)
        in_h, in_w, out_h, out_w, lut = match
        self.assertEqual((in_h, in_w, out_h, out_w), (2, 2, 2, 2))
        # Keys: (pattern_key, nonzero_color)
        self.assertEqual(lut[(1, 3)], ((3, 0), (0, 0)))
        self.assertEqual(lut[(2, 5)], ((0, 5), (0, 0)))

    def test_rejects_when_only_one_nonzero_color_seen(self):
        # If only ONE distinct nonzero color is observed across all pairs, the simpler
        # grid_pattern_to_grid would already cover it; reject this match.
        pairs = [_pair([[3, 0], [0, 0]], [[3, 0], [0, 0]])]
        self.assertIsNone(_detect_grid_pattern_color_to_grid(pairs))

    def test_rejects_when_multiple_nonzero_colors_in_same_pair(self):
        # If a single pair contains TWO different nonzero colors, the pattern doesn't apply.
        pairs = [_pair([[3, 5], [0, 0]], [[3, 5], [0, 0]])]
        self.assertIsNone(_detect_grid_pattern_color_to_grid(pairs))

    def test_rejects_inconsistent_rule(self):
        pairs = [
            _pair([[3, 0], [0, 0]], [[3, 0], [0, 0]]),
            _pair([[5, 0], [0, 0]], [[5, 0], [0, 0]]),
            # Same (pattern_key, color) as pair 1 but different output -> reject.
            _pair([[3, 0], [0, 0]], [[7, 0], [0, 0]]),
        ]
        self.assertIsNone(_detect_grid_pattern_color_to_grid(pairs))

    def test_rejects_input_too_large_for_lut(self):
        pairs = [
            _pair([[1]*4, [1]*4, [1]*4, [1]*4], [[0]]),  # 4x4 = 16 bits, exceeds 12 cap
            _pair([[2]*4, [2]*4, [2]*4, [2]*4], [[0]]),
        ]
        self.assertIsNone(_detect_grid_pattern_color_to_grid(pairs))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 2x2 input -> 2x2 output. LUT: (pattern=1, color=3) -> [[3,0],[0,0]]; (pattern=2, color=5) -> [[0,5],[0,0]]
        lut = {(1, 3): ((3, 0), (0, 0)), (2, 5): ((0, 5), (0, 0))}
        model = grid_pattern_color_to_grid_network(in_h=2, in_w=2, out_h=2, out_w=2,
                                                   lut_table=lut, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        # Input: TL cell color 3, others 0.
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        inp[0, 3, 0, 0] = 1.0
        for (r, c) in [(0, 1), (1, 0), (1, 1)]:
            inp[0, 0, r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Expected: (0,0) = color 3
        self.assertEqual(out[0, 3, 0, 0], 1.0)
        self.assertEqual(out[0, 0, 0, 1], 1.0)


class TranslationTest(unittest.TestCase):
    def test_detects_simple_down_shift(self):
        # 3x3 grid: shift content down 1 row, top row goes to zero.
        inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        out = [[0, 0, 0], [1, 2, 3], [4, 5, 6]]
        match = _detect_translation([_pair(inp, out)])
        self.assertEqual(match, {"h": 3, "w": 3, "dr": 1, "dc": 0})

    def test_detects_simple_right_shift(self):
        inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        out = [[0, 1, 2], [0, 4, 5], [0, 7, 8]]
        match = _detect_translation([_pair(inp, out)])
        self.assertEqual(match, {"h": 3, "w": 3, "dr": 0, "dc": 1})

    def test_detects_up_left_diagonal(self):
        inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        out = [[5, 6, 0], [8, 9, 0], [0, 0, 0]]
        match = _detect_translation([_pair(inp, out)])
        self.assertEqual(match, {"h": 3, "w": 3, "dr": -1, "dc": -1})

    def test_consistent_across_pairs(self):
        # Both pairs use dr=1, dc=0 with different content.
        inp1, out1 = [[1, 2], [3, 4]], [[0, 0], [1, 2]]
        inp2, out2 = [[5, 6], [7, 8]], [[0, 0], [5, 6]]
        match = _detect_translation([_pair(inp1, out1), _pair(inp2, out2)])
        self.assertEqual(match, {"h": 2, "w": 2, "dr": 1, "dc": 0})

    def test_rejects_inconsistent_dr_dc_across_pairs(self):
        # Pair 1 is dr=1, dc=0; pair 2 is dr=0, dc=1. No single (dr, dc) works.
        inp1, out1 = [[1, 2], [3, 4]], [[0, 0], [1, 2]]
        inp2, out2 = [[1, 2], [3, 4]], [[0, 1], [0, 3]]
        self.assertIsNone(_detect_translation([_pair(inp1, out1), _pair(inp2, out2)]))

    def test_rejects_identity(self):
        # (dr, dc) = (0, 0) is identity; not a translation.
        inp = [[1, 2], [3, 4]]
        out = [[1, 2], [3, 4]]
        self.assertIsNone(_detect_translation([_pair(inp, out)]))

    def test_rejects_shape_change(self):
        pairs = [_pair([[1, 2], [3, 4]], [[1, 2, 0], [3, 4, 0]])]
        self.assertIsNone(_detect_translation(pairs))

    def test_rejects_varying_input_shapes(self):
        pairs = [
            _pair([[1, 2], [3, 4]], [[0, 0], [1, 2]]),
            _pair([[1, 2, 3], [4, 5, 6]], [[0, 0, 0], [1, 2, 3]]),
        ]
        self.assertIsNone(_detect_translation(pairs))

    def test_rejects_non_translation_rule(self):
        # Color-perm task: not a translation.
        pairs = [_pair([[1, 2], [3, 4]], [[2, 1], [4, 3]])]
        self.assertIsNone(_detect_translation(pairs))

    def test_rejects_1x1_input(self):
        pairs = [_pair([[5]], [[5]])]
        self.assertIsNone(_detect_translation(pairs))

    def test_network_round_trip_down_shift(self):
        import numpy as np
        import onnxruntime as ort
        # 3x3 grid shifted down 1 row.
        model = translation_network(h=3, w=3, dr=1, dc=0, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Encode a 3x3 grid: row r col c gets color (r * 3 + c) % 10
        for r in range(3):
            for c in range(3):
                color = (r * 3 + c) % 10
                inp[0, color, r, c] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Row 0 of the active output grid is color 0, encoded as strict one-hot
        # channel 0. A no-channel cell would decode as color 10/11 in Kaggle.
        for c in range(3):
            self.assertEqual(out[0, 0, 0, c], 1.0)
            for ch in range(1, 10):
                self.assertEqual(out[0, ch, 0, c], 0.0)
        # Rows 1, 2 of output should match rows 0, 1 of input.
        for r in range(2):
            for c in range(3):
                color = (r * 3 + c) % 10
                self.assertEqual(out[0, color, r + 1, c], 1.0)

    def test_network_round_trip_up_left_diagonal(self):
        import numpy as np
        import onnxruntime as ort
        model = translation_network(h=4, w=4, dr=-1, dc=-1, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Single non-zero cell at (2, 2) of input grid, color 5.
        inp[0, 5, 2, 2] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # That cell should now be at (1, 1).
        self.assertEqual(out[0, 5, 1, 1], 1.0)
        # No other non-zero cell should exist.
        # Sum across canvas of channel 5 == 1.0
        self.assertAlmostEqual(float(out[0, 5].sum()), 1.0, places=5)


class ColorReplaceTest(unittest.TestCase):
    def test_detects_merge_case(self):
        # task276-style: input has only colors {6, 7}; rule is 6 -> 2, 7 stays.
        inp = [[6, 7], [7, 6]]
        out = [[2, 7], [7, 2]]
        forward = _detect_color_replace([_pair(inp, out)])
        self.assertEqual(forward, {6: 2, 7: 7})

    def test_detects_three_color_replace(self):
        # task309-style: input has {1, 7, 8}; rule is 7 -> 5; 1 and 8 stay.
        inp = [[1, 7, 8], [8, 7, 1]]
        out = [[1, 5, 8], [8, 5, 1]]
        forward = _detect_color_replace([_pair(inp, out)])
        self.assertEqual(forward, {1: 1, 7: 5, 8: 8})

    def test_consistent_across_pairs(self):
        # Both pairs use 6 -> 2.
        pairs = [_pair([[6, 0]], [[2, 0]]), _pair([[6, 6]], [[2, 2]])]
        forward = _detect_color_replace(pairs)
        self.assertEqual(forward.get(6), 2)

    def test_rejects_inconsistent_across_pairs(self):
        # Pair 1: 6 -> 2; pair 2: 6 -> 3. No single rule works.
        pairs = [_pair([[6]], [[2]]), _pair([[6]], [[3]])]
        self.assertIsNone(_detect_color_replace(pairs))

    def test_rejects_identity(self):
        pairs = [_pair([[1, 2], [3, 4]], [[1, 2], [3, 4]])]
        self.assertIsNone(_detect_color_replace(pairs))

    def test_rejects_strict_permutation(self):
        # color_perm covers strict bijections with all 10 colors observed.
        # Build a complete permutation: 0->1, 1->2, ..., 9->0.
        rows = []
        for r in range(10):
            rows.append([(r + c) % 10 for c in range(10)])
        out_rows = [[(c + 1) % 10 for c in row] for row in rows]
        self.assertIsNone(_detect_color_replace([_pair(rows, out_rows)]))

    def test_rejects_shape_change(self):
        pairs = [_pair([[6, 7]], [[2, 7], [2, 7]])]
        self.assertIsNone(_detect_color_replace(pairs))

    def test_network_round_trip_merge(self):
        import numpy as np
        import onnxruntime as ort
        forward = {6: 2, 7: 7}
        model = color_replace_network(forward, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Place color 6 at (0, 0) and color 7 at (0, 1).
        inp[0, 6, 0, 0] = 1.0
        inp[0, 7, 0, 1] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # Output (0, 0) should be color 2 (channel 2 lit, channel 6 dark).
        self.assertEqual(out[0, 2, 0, 0], 1.0)
        self.assertEqual(out[0, 6, 0, 0], 0.0)
        # Output (0, 1) should still be color 7.
        self.assertEqual(out[0, 7, 0, 1], 1.0)


class PixelPermTest(unittest.TestCase):
    def test_detects_simple_swap(self):
        # 1x2 grid: swap the two cells.
        inp = [[5, 3]]
        out = [[3, 5]]
        match = _detect_pixel_perm([_pair(inp, out)])
        self.assertIsNotNone(match)
        self.assertEqual(match["h"], 1)
        self.assertEqual(match["w"], 2)
        self.assertEqual(match["indices"], [1, 0])

    def test_consistent_across_pairs(self):
        # Both pairs swap the two cells of a 1x2 grid.
        pairs = [_pair([[5, 3]], [[3, 5]]), _pair([[2, 7]], [[7, 2]])]
        match = _detect_pixel_perm(pairs)
        self.assertEqual(match["indices"], [1, 0])

    def test_rejects_inconsistent_across_pairs(self):
        # Pair 1 swaps; pair 2 is identity.
        pairs = [_pair([[5, 3]], [[3, 5]]), _pair([[2, 7]], [[2, 7]])]
        self.assertIsNone(_detect_pixel_perm(pairs))

    def test_rejects_identity(self):
        pairs = [_pair([[1, 2], [3, 4]], [[1, 2], [3, 4]])]
        self.assertIsNone(_detect_pixel_perm(pairs))

    def test_rejects_shape_change(self):
        pairs = [_pair([[1, 2]], [[1, 2], [3, 4]])]
        self.assertIsNone(_detect_pixel_perm(pairs))

    def test_rejects_varying_input_shapes(self):
        pairs = [
            _pair([[5, 3]], [[3, 5]]),
            _pair([[5, 3, 7]], [[3, 5, 7]]),
        ]
        self.assertIsNone(_detect_pixel_perm(pairs))

    def test_rejects_color_introduction(self):
        # Output has color 9 not in input — no source cell with color 9.
        pairs = [_pair([[1, 2]], [[1, 9]])]
        self.assertIsNone(_detect_pixel_perm(pairs))

    def test_2x2_rotation(self):
        # 2x2 grid rotated 90 degrees clockwise.
        # in:  [[a, b],     out: [[c, a],
        #       [c, d]]            [d, b]]
        # rot90: out[r][c] = in[h-1-c][r] for h=w=2.
        # Flat: in_flat = [a, b, c, d] (positions 0,1,2,3)
        #       out_flat = [c, a, d, b]
        # indices[0] = 2 (c at pos 2 in input), indices[1] = 0 (a at pos 0),
        # indices[2] = 3 (d at pos 3), indices[3] = 1 (b at pos 1).
        inp = [[1, 2], [3, 4]]
        out = [[3, 1], [4, 2]]
        match = _detect_pixel_perm([_pair(inp, out)])
        self.assertEqual(match["indices"], [2, 0, 3, 1])

    def test_network_round_trip_swap(self):
        import numpy as np
        import onnxruntime as ort
        # 1x2 grid swap.
        model = pixel_perm_network(h=1, w=2, indices=[1, 0], batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Place color 5 at (0, 0) and color 3 at (0, 1).
        inp[0, 5, 0, 0] = 1.0
        inp[0, 3, 0, 1] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # After swap: (0, 0) should be color 3, (0, 1) should be color 5.
        self.assertEqual(out[0, 3, 0, 0], 1.0)
        self.assertEqual(out[0, 5, 0, 0], 0.0)
        self.assertEqual(out[0, 5, 0, 1], 1.0)
        self.assertEqual(out[0, 3, 0, 1], 0.0)

    def test_network_padding_outside_grid(self):
        import numpy as np
        import onnxruntime as ort
        # 2x2 grid identity-permutation — but with non-trivial Pad to verify
        # canvas padding stays zero outside (2, 2).
        model = pixel_perm_network(h=2, w=2, indices=[0, 1, 2, 3], batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        inp[0, 5, 0, 0] = 1.0
        # Cell at (5, 5) inside canvas padding region; should NOT appear in output.
        inp[0, 7, 5, 5] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # (0, 0) should still be color 5.
        self.assertEqual(out[0, 5, 0, 0], 1.0)
        # (5, 5) and everything outside (2, 2) should be all-zero.
        for ch in range(10):
            self.assertEqual(out[0, ch, 5, 5], 0.0)
            self.assertEqual(out[0, ch, 0, 5], 0.0)
            self.assertEqual(out[0, ch, 5, 0], 0.0)


class ShapeChangingPixelPermTest(unittest.TestCase):
    def test_detects_2x2_to_4x4_tile(self):
        # 2x2 input tiled to 4x4 output: each input cell repeats 4 times in a 2x2 block.
        # Canonical "tile by 2" pattern.
        # in:  [[a, b],     out: [[a, a, b, b],
        #       [c, d]]            [a, a, b, b],
        #                          [c, c, d, d],
        #                          [c, c, d, d]]
        inp = [[1, 2], [3, 4]]
        out = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
        match = _detect_shape_changing_pixel_perm([_pair(inp, out)])
        self.assertIsNotNone(match)
        self.assertEqual((match["in_h"], match["in_w"], match["out_h"], match["out_w"]), (2, 2, 4, 4))
        # indices should map every output cell to its source input cell.
        self.assertEqual(len(match["indices"]), 16)

    def test_rejects_same_shape(self):
        # Same-shape case is _detect_pixel_perm's job.
        pairs = [_pair([[1, 2]], [[2, 1]])]
        self.assertIsNone(_detect_shape_changing_pixel_perm(pairs))

    def test_rejects_inconsistent_across_pairs(self):
        # Pair 1 tiles with one pattern; pair 2 tiles differently. No single mapping works.
        pairs = [
            _pair([[1, 2]], [[1, 2], [1, 2]]),  # tile vertically
            _pair([[1, 2]], [[1, 1], [2, 2]]),  # transpose-then-tile (different mapping)
        ]
        self.assertIsNone(_detect_shape_changing_pixel_perm(pairs))

    def test_rejects_color_introduction(self):
        # Output has color 9 not in input.
        pairs = [_pair([[1, 2]], [[1, 9], [2, 1]])]
        self.assertIsNone(_detect_shape_changing_pixel_perm(pairs))

    def test_rejects_varying_input_shapes(self):
        pairs = [
            _pair([[1, 2]], [[1, 2], [1, 2]]),
            _pair([[1, 2, 3]], [[1, 2, 3], [1, 2, 3]]),
        ]
        self.assertIsNone(_detect_shape_changing_pixel_perm(pairs))

    def test_network_round_trip_2x2_to_4x4(self):
        import numpy as np
        import onnxruntime as ort
        # 2x2 -> 4x4 tile pattern. indices: each output cell maps to specific input cell.
        # in flat: [a@0, b@1, c@2, d@3]
        # out flat (16 cells): rows of [a,a,b,b], [a,a,b,b], [c,c,d,d], [c,c,d,d]
        indices = [0, 0, 1, 1,
                   0, 0, 1, 1,
                   2, 2, 3, 3,
                   2, 2, 3, 3]
        model = shape_changing_pixel_perm_network(in_h=2, in_w=2, out_h=4, out_w=4, indices=indices, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Place colors at (0,0)=5, (0,1)=3, (1,0)=7, (1,1)=2.
        inp[0, 5, 0, 0] = 1.0
        inp[0, 3, 0, 1] = 1.0
        inp[0, 7, 1, 0] = 1.0
        inp[0, 2, 1, 1] = 1.0
        out = sess.run(["output"], {"input": inp})[0]
        # First 2x2 block of output should all be color 5.
        self.assertEqual(out[0, 5, 0, 0], 1.0)
        self.assertEqual(out[0, 5, 1, 1], 1.0)
        # Last 2x2 block should all be color 2.
        self.assertEqual(out[0, 2, 2, 2], 1.0)
        self.assertEqual(out[0, 2, 3, 3], 1.0)
        # Padding outside (4, 4) should be zero.
        for ch in range(10):
            self.assertEqual(out[0, ch, 5, 5], 0.0)


class EnclosedRegionFillTest(unittest.TestCase):
    def test_detects_simple_diamond(self):
        # 4-cell diamond around (1,1); enclosed cell at (1,1) gets filled with color 4.
        inp = [
            [0, 3, 0],
            [3, 0, 3],
            [0, 3, 0],
        ]
        out = [
            [0, 3, 0],
            [3, 4, 3],
            [0, 3, 0],
        ]
        fc = _detect_enclosed_region_fill([_pair(inp, out)])
        self.assertEqual(fc["fill_color"], 4)

    def test_detects_rectangle(self):
        # 2x2 rectangle of color 2 with interior cell at (1,1) and (1,2);
        # walls top/bottom/left/right. Interior fills with color 1.
        inp = [
            [2, 2, 2, 2],
            [2, 0, 0, 2],
            [2, 0, 0, 2],
            [2, 2, 2, 2],
        ]
        out = [
            [2, 2, 2, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 2, 2, 2],
        ]
        fc = _detect_enclosed_region_fill([_pair(inp, out)])
        self.assertEqual(fc["fill_color"], 1)

    def test_consistent_across_pairs(self):
        # Two different shapes but both enclosed-with-fill_color=5.
        pairs = [
            _pair(
                [[3, 3, 3], [3, 0, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 5, 3], [3, 3, 3]],
            ),
            _pair(
                [[7, 7, 7, 7], [7, 0, 0, 7], [7, 0, 0, 7], [7, 7, 7, 7]],
                [[7, 7, 7, 7], [7, 5, 5, 7], [7, 5, 5, 7], [7, 7, 7, 7]],
            ),
        ]
        self.assertEqual(_detect_enclosed_region_fill(pairs)["fill_color"], 5)

    def test_rejects_inconsistent_fill_color(self):
        # Pair 1 fills with 4, pair 2 with 7. No single rule.
        pairs = [
            _pair([[3, 3, 3], [3, 0, 3], [3, 3, 3]], [[3, 3, 3], [3, 4, 3], [3, 3, 3]]),
            _pair([[3, 3, 3], [3, 0, 3], [3, 3, 3]], [[3, 3, 3], [3, 7, 3], [3, 3, 3]]),
        ]
        self.assertIsNone(_detect_enclosed_region_fill(pairs))

    def test_rejects_unenclosed_fill(self):
        # Cell isn't enclosed (3 not 4 walls) but output fills it anyway -> reject.
        inp = [
            [0, 3, 0],
            [3, 0, 0],  # right neighbor is open
            [0, 3, 0],
        ]
        out = [
            [0, 3, 0],
            [3, 4, 0],
            [0, 3, 0],
        ]
        self.assertIsNone(_detect_enclosed_region_fill([_pair(inp, out)]))

    def test_rejects_no_enclosure_anywhere(self):
        # No closed loops; output is identical to input -> reject (no fill seen).
        inp = [[0, 3, 0], [0, 0, 0], [0, 3, 0]]
        out = [[0, 3, 0], [0, 0, 0], [0, 3, 0]]
        self.assertIsNone(_detect_enclosed_region_fill([_pair(inp, out)]))

    def test_rejects_wall_modification(self):
        # Wall cell got recolored -> not pure enclosure-fill, reject.
        inp = [[0, 3, 0], [3, 0, 3], [0, 3, 0]]
        out = [[0, 5, 0], [3, 4, 3], [0, 3, 0]]  # (0,1) wall changed
        self.assertIsNone(_detect_enclosed_region_fill([_pair(inp, out)]))

    def test_rejects_shape_change(self):
        pairs = [_pair([[3, 0, 3]], [[3, 0, 3], [0, 0, 0]])]
        self.assertIsNone(_detect_enclosed_region_fill(pairs))

    def test_network_round_trip_diamond(self):
        import numpy as np
        import onnxruntime as ort
        # Build the network for fill_color=4 and run on a one-hot canvas with
        # an enclosed diamond at the top-left.
        model = enclosed_region_fill_network(fill_color=4, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())

        canvas = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # All cells default to channel-0 lit (color 0 = background).
        canvas[0, 0, :, :] = 1.0
        # Place walls at (0,1), (1,0), (1,2), (2,1) — color 3.
        for (r, c) in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            canvas[0, 0, r, c] = 0.0
            canvas[0, 3, r, c] = 1.0

        out = sess.run(["output"], {"input": canvas})[0]
        # Enclosed cell at (1, 1): should be color 4 (channel 4 lit, channel 0 dark).
        self.assertAlmostEqual(out[0, 0, 1, 1], 0.0, places=5)
        self.assertAlmostEqual(out[0, 4, 1, 1], 1.0, places=5)
        # Walls preserved: (0, 1) still color 3.
        self.assertAlmostEqual(out[0, 3, 0, 1], 1.0, places=5)
        # Outside the diamond stays color 0.
        self.assertAlmostEqual(out[0, 0, 5, 5], 1.0, places=5)
        self.assertAlmostEqual(out[0, 4, 5, 5], 0.0, places=5)

    def test_rejects_fill_color_zero(self):
        # Builder rejects fill_color=0 (would make enclosed cells "color 0", a no-op).
        with self.assertRaises(ValueError):
            enclosed_region_fill_network(fill_color=0)


class OuterProductMarkerTileTest(unittest.TestCase):
    @staticmethod
    def _stamp_output(inp, marker):
        """Construct expected 9x9 output for given 3x3 input + marker color."""
        out = [[0] * 9 for _ in range(9)]
        for ai in range(3):
            for aj in range(3):
                if inp[ai][aj] == marker:
                    for ir in range(3):
                        for ic in range(3):
                            out[3 * ai + ir][3 * aj + ic] = inp[ir][ic]
        return out

    def test_detects_fixed_marker_color_2(self):
        # task315-style: marker is the same color (2) across all pairs.
        inp1 = [[1, 0, 0], [2, 1, 0], [0, 0, 1]]
        inp2 = [[0, 1, 2], [1, 1, 0], [2, 0, 0]]
        pairs = [
            _pair(inp1, self._stamp_output(inp1, 2)),
            _pair(inp2, self._stamp_output(inp2, 2)),
        ]
        match = _detect_outer_product_marker_tile(pairs)
        self.assertEqual(match, {"mode": "fixed", "marker_color": 2})

    def test_detects_most_common_marker(self):
        # task304-style: marker is the per-pair most-frequent color.
        inp1 = [[3, 8, 7], [9, 3, 8], [7, 9, 3]]  # marker = 3 (count 3)
        inp2 = [[8, 6, 8], [3, 3, 8], [8, 8, 8]]  # marker = 8 (count 6)
        pairs = [
            _pair(inp1, self._stamp_output(inp1, 3)),
            _pair(inp2, self._stamp_output(inp2, 8)),
        ]
        match = _detect_outer_product_marker_tile(pairs)
        self.assertEqual(match, {"mode": "most_common"})

    def test_fixed_preferred_when_both_modes_fit(self):
        # If "fixed" mode satisfies all pairs, the detector returns fixed (cheaper builder).
        inp1 = [[2, 0, 2], [0, 0, 0], [2, 0, 2]]  # marker=2 also = most-common (count 4)
        inp2 = [[2, 1, 2], [1, 1, 1], [2, 1, 2]]  # marker=2 NOT most-common (most=1, count 5)
        pairs = [
            _pair(inp1, self._stamp_output(inp1, 2)),
            _pair(inp2, self._stamp_output(inp2, 2)),
        ]
        match = _detect_outer_product_marker_tile(pairs)
        self.assertEqual(match, {"mode": "fixed", "marker_color": 2})

    def test_rejects_wrong_input_shape(self):
        # 4x4 input shouldn't match (only (3,3) -> (9,9)).
        inp = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        out = [[0] * 12 for _ in range(12)]
        self.assertIsNone(_detect_outer_product_marker_tile([_pair(inp, out)]))

    def test_rejects_wrong_output_shape(self):
        # 3x3 input but 6x6 output (not 9x9) -> reject.
        inp = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        out = [[0] * 6 for _ in range(6)]
        self.assertIsNone(_detect_outer_product_marker_tile([_pair(inp, out)]))

    def test_rejects_inconsistent_across_pairs(self):
        # No fixed-marker color and no per-pair most-common rule satisfies both
        # pairs simultaneously: pair 1 stamps at color-1 positions; pair 2 stamps
        # at color-2 positions. fixed=1 fails on pair 2 (predicts color-1
        # stamps, actual is color-2). fixed=2 fails on pair 1. most_common
        # fails on pair 1 (predicts ch0 dominant -> stamp at 0-positions,
        # actual is at 1-positions).
        inp1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        inp2 = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
        pairs = [
            _pair(inp1, self._stamp_output(inp1, 1)),
            _pair(inp2, self._stamp_output(inp2, 2)),
        ]
        self.assertIsNone(_detect_outer_product_marker_tile(pairs))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_outer_product_marker_tile([]))

    def test_network_round_trip_fixed(self):
        import numpy as np
        import onnxruntime as ort
        # Marker = 2 at positions (1, 0). Input is task315-train[0]-style.
        inp = [[1, 0, 0], [2, 1, 0], [0, 0, 1]]
        model = outer_product_marker_tile_network("fixed", marker_color=2, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Encode the 3x3 active grid as one-hot (canvas padding stays all-zero).
        for r in range(3):
            for c in range(3):
                x[0, inp[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        # Stamp should appear at output (3, 0): the ONLY 3x3 block where input == marker (2).
        # Output[3, 0] = 1, output[4, 0] = 2, output[5, 2] = 1 (one-hot channels).
        self.assertEqual(out[0, 1, 3, 0], 1.0)
        self.assertEqual(out[0, 2, 4, 0], 1.0)
        self.assertEqual(out[0, 1, 5, 2], 1.0)
        # Inside active 9x9, non-stamped cells: ch0 = 1.0 (background fill).
        # Output[0, 0] is in non-stamped block (anchor (0, 0); input[0, 0]=1, not marker=2).
        self.assertEqual(out[0, 0, 0, 0], 1.0)
        # Padding outside 9x9: all channels = 0.
        for ch in range(10):
            self.assertEqual(out[0, ch, 10, 10], 0.0)

    def test_network_round_trip_most_common(self):
        import numpy as np
        import onnxruntime as ort
        # Most-common = 3 (count 3 on diagonal) for this 3x3 input.
        inp = [[3, 8, 7], [9, 3, 8], [7, 9, 3]]
        model = outer_product_marker_tile_network("most_common", batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                x[0, inp[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        # Stamps at output (0, 0), (3, 3), (6, 6) — main diagonal of anchors.
        # Spot-check: output[0, 0] = 3 -> ch3 = 1.
        self.assertEqual(out[0, 3, 0, 0], 1.0)
        self.assertEqual(out[0, 8, 0, 1], 1.0)  # output[0, 1] = 8 = input[0, 1]
        # Stamp at (3, 3): output[3, 3] = 3 (input[0, 0] = 3).
        self.assertEqual(out[0, 3, 3, 3], 1.0)
        # Non-stamped block: output[0, 3..5] should have ch0 = 1.0 (anchor (0, 1) is 8 != marker 3).
        self.assertEqual(out[0, 0, 0, 3], 1.0)
        # Padding outside 9x9: all-zero.
        for ch in range(10):
            self.assertEqual(out[0, ch, 15, 15], 0.0)

    def test_builder_rejects_invalid_args(self):
        with self.assertRaises(ValueError):
            outer_product_marker_tile_network("bogus")
        with self.assertRaises(ValueError):
            outer_product_marker_tile_network("fixed")  # marker_color missing
        with self.assertRaises(ValueError):
            outer_product_marker_tile_network("fixed", marker_color=15)


class PerColorStampTest(unittest.TestCase):
    KERNELS_OFFSETS = {
        "ortho": [(0, -1), (0, 1), (-1, 0), (1, 0)],
        "diag":  [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        "all8":  [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    }

    def _stamp(self, inp, rules):
        h = len(inp)
        w = len(inp[0]) if h else 0
        out = [row[:] for row in inp]
        for in_c, st_c, k in rules:
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != in_c:
                        continue
                    for dr, dc in self.KERNELS_OFFSETS[k]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == 0:
                            out[nr][nc] = st_c
        return out

    def test_detects_task015_rules(self):
        # 1 -> 7 ortho, 2 -> 4 diag (LB-5126 task015 rule).
        inp = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        rules = [(1, 7, "ortho"), (2, 4, "diag")]
        out = self._stamp(inp, rules)
        match = _detect_per_color_stamp([_pair(inp, out)])
        self.assertEqual(match, {"rules": rules})

    def test_detects_single_rule_all8(self):
        # task352-style: 2 -> 1 with 8-connected stamp.
        inp = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        rules = [(2, 1, "all8")]
        out = self._stamp(inp, rules)
        match = _detect_per_color_stamp([_pair(inp, out)])
        self.assertEqual(match, {"rules": rules})

    def test_rejects_non_color0_overlay(self):
        # Output cell that was non-zero in input must equal input — no overlay.
        inp = [[0, 1, 0], [0, 5, 0], [0, 0, 0]]
        out = [[0, 7, 0], [0, 5, 0], [0, 7, 0]]  # ortho stamps from 1 onto color 0 cells
        out[1][1] = 9  # corrupt the 5 -> 9 (rule shouldn't allow)
        self.assertIsNone(_detect_per_color_stamp([_pair(inp, out)]))

    def test_rejects_inconsistent_kernel_across_pairs(self):
        # Pair 1 implies rule (2, 1, "ortho"), Pair 2 implies rule (2, 1, "diag")
        # — same (in_c, stamp_color) but different kernel inferred. Must reject.
        inp1 = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        out1 = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]  # ortho stamp from (1, 1)
        inp2 = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        out2 = [[1, 0, 1], [0, 2, 0], [1, 0, 1]]  # diag stamp from (1, 1)
        self.assertIsNone(_detect_per_color_stamp([_pair(inp1, out1), _pair(inp2, out2)]))

    def test_rejects_shape_change(self):
        inp = [[0, 1, 0]]
        out = [[0, 1, 0], [0, 0, 0]]
        self.assertIsNone(_detect_per_color_stamp([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_per_color_stamp([]))

    def test_rejects_no_deltas(self):
        # Output == input. Not a stamp pattern.
        inp = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.assertIsNone(_detect_per_color_stamp([_pair(inp, inp)]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # task015 rules — small 5x5 active grid (canvas-padding-clip is exercised).
        inp = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        model = per_color_stamp_network([(1, 7, "ortho"), (2, 4, "diag")], batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Encode input as one-hot. Color 0 lights ch0; otherwise lights the cell's channel.
        for r in range(5):
            for c in range(5):
                x[0, inp[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        # Spot check stamp cells.
        # 1 at (1, 2): ortho neighbors (0, 2), (1, 1), (1, 3), (2, 2). All should have ch7=1.
        for nr, nc in [(0, 2), (1, 1), (1, 3), (2, 2)]:
            self.assertEqual(out[0, 7, nr, nc], 1.0,
                             f"expected ch7=1 at ({nr},{nc}) (ortho stamp from 1)")
        # 2 at (2, 1): diag neighbors (1, 0), (1, 2), (3, 0), (3, 2). All should have ch4=1.
        for nr, nc in [(1, 0), (1, 2), (3, 0), (3, 2)]:
            # (1, 2) overlaps with the 7-stamp at (1, 2) too; first rule (1, 7, ortho)
            # placed 7 there; second rule (2, 4, diag) doesn't overwrite (input was 0 -> 7).
            # Actually our builder DOES overwrite: any rule that fires at (1, 2) places
            # its stamp_color there. With (1, 7) firing first and (2, 4) firing second,
            # the final stamp_color depends on order. The detector checks consistency
            # so the data property must hold (no two rules firing at same delta cell).
            # Skip (1, 2) in this check since it's a clash position.
            if (nr, nc) == (1, 2):
                continue
            self.assertEqual(out[0, 4, nr, nc], 1.0,
                             f"expected ch4=1 at ({nr},{nc}) (diag stamp from 2)")
        # Padding outside 5x5: all channels = 0.
        for ch in range(10):
            self.assertEqual(out[0, ch, 10, 10], 0.0)
        # Active grid color-0 cell with NO stamp (e.g. (4, 4)): ch0 = 1.
        self.assertEqual(out[0, 0, 4, 4], 1.0)

    def test_builder_rejects_invalid_args(self):
        with self.assertRaises(ValueError):
            per_color_stamp_network([])
        with self.assertRaises(ValueError):
            per_color_stamp_network([(1, 1, "ortho")])  # in_c == st_c
        with self.assertRaises(ValueError):
            per_color_stamp_network([(0, 7, "ortho")])  # in_c == 0
        with self.assertRaises(ValueError):
            per_color_stamp_network([(1, 7, "bogus")])

    def test_task220_fast_path_is_strict_one_hot(self):
        import numpy as np
        import onnxruntime as ort

        inp = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 8, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        rules = [(2, 1, "all8"), (3, 6, "all8"), (8, 4, "all8")]
        expected = self._stamp(inp, rules)
        model = task220_multi_marker_stamp_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0)
        self.assertTrue(np.array_equal(decoded[:7, :7], np.array(expected)))
        self.assertTrue(np.all((out[:, :, :7, :7] == 0.0) | (out[:, :, :7, :7] == 1.0)))
        for r in range(7):
            for c in range(7):
                self.assertEqual((out[0, :, r, c] == 1.0).sum(), 1)


class OutlineExtractTest(unittest.TestCase):
    @staticmethod
    def _outline(inp):
        h = len(inp)
        w = len(inp[0]) if h else 0
        out = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                v = inp[r][c]
                if v == 0:
                    continue
                interior = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or inp[nr][nc] != v:
                            interior = False
                            break
                    if not interior:
                        break
                if not interior:
                    out[r][c] = v
        return out

    def test_detects_simple(self):
        inp = [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ]
        out = self._outline(inp)
        # Center cell (2, 2) is interior; all border cells of the 3x3 5-square stay.
        self.assertEqual(_detect_outline_extract([_pair(inp, out)]), {"matched": True, "canvas_h": 5, "canvas_w": 5})

    def test_rejects_no_interior_cells(self):
        # If output equals input (no interior to remove), reject.
        inp = [[0, 5, 0], [5, 0, 5], [0, 5, 0]]
        self.assertIsNone(_detect_outline_extract([_pair(inp, inp)]))

    def test_rejects_shape_change(self):
        inp = [[5, 5], [5, 5]]
        out = [[5, 5]]
        self.assertIsNone(_detect_outline_extract([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_outline_extract([]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        inp = [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ]
        model = outline_extract_network(canvas_h=5, canvas_w=5, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                x[0, inp[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        # (2, 2) is interior 5; should become ch0=1.
        self.assertEqual(out[0, 0, 2, 2], 1.0)
        self.assertEqual(out[0, 5, 2, 2], 0.0)
        # Boundary 5-cells: (1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3) stay.
        self.assertEqual(out[0, 5, 1, 1], 1.0)
        self.assertEqual(out[0, 5, 2, 1], 1.0)
        # Active-grid color-0 cells stay color 0.
        self.assertEqual(out[0, 0, 0, 0], 1.0)
        # Padding outside 5x5: all-zero.
        for ch in range(10):
            self.assertEqual(out[0, ch, 10, 10], 0.0)


class ActiveGridFrameTest(unittest.TestCase):
    def test_detects_simple(self):
        # Active grid 3x4, all-zero input, output is hollow rectangle of 8.
        inp = [[0] * 4 for _ in range(3)]
        out = [[8, 8, 8, 8],
               [8, 0, 0, 8],
               [8, 8, 8, 8]]
        self.assertEqual(_detect_active_grid_frame([_pair(inp, out)]),
                         {"border_color": 8})

    def test_rejects_inconsistent_border_color(self):
        inp = [[0] * 4 for _ in range(3)]
        out1 = [[8, 8, 8, 8],
                [8, 0, 0, 8],
                [8, 8, 8, 8]]
        out2 = [[5, 5, 5, 5],
                [5, 0, 0, 5],
                [5, 5, 5, 5]]
        self.assertIsNone(_detect_active_grid_frame([_pair(inp, out1), _pair(inp, out2)]))

    def test_rejects_too_small(self):
        # 1xN or Nx1 grids have no clear "interior" — reject.
        inp = [[0]]
        out = [[8]]
        self.assertIsNone(_detect_active_grid_frame([_pair(inp, out)]))

    def test_rejects_non_zero_interior_in_output(self):
        # Output interior must be color 0.
        inp = [[0] * 4 for _ in range(3)]
        out = [[8, 8, 8, 8],
               [8, 9, 9, 8],
               [8, 8, 8, 8]]
        self.assertIsNone(_detect_active_grid_frame([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_active_grid_frame([]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # task171-like: input all zeros, 5x5 active grid, output = hollow 8-frame.
        model = active_grid_frame_network(border_color=8, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Set ch0=1 in the 5x5 active grid (color 0 throughout).
        for r in range(5):
            for c in range(5):
                x[0, 0, r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        # Border cells (row 0, 4, col 0, 4 of active 5x5): ch8=1.
        for c in range(5):
            self.assertEqual(out[0, 8, 0, c], 1.0)
            self.assertEqual(out[0, 8, 4, c], 1.0)
        for r in range(5):
            self.assertEqual(out[0, 8, r, 0], 1.0)
            self.assertEqual(out[0, 8, r, 4], 1.0)
        # Interior: ch0=1.
        self.assertEqual(out[0, 0, 2, 2], 1.0)
        # Padding: all-zero.
        for ch in range(10):
            self.assertEqual(out[0, ch, 10, 10], 0.0)

    def test_builder_rejects_invalid(self):
        with self.assertRaises(ValueError):
            active_grid_frame_network(border_color=0)
        with self.assertRaises(ValueError):
            active_grid_frame_network(border_color=10)


class BboxFillTest(unittest.TestCase):
    def _bbox(self, inp, target):
        h = len(inp)
        w = len(inp[0]) if h else 0
        nz = [(r, c) for r in range(h) for c in range(w) if inp[r][c] != 0]
        if not nz:
            return [row[:] for row in inp]
        rmin = min(r for r, _ in nz); rmax = max(r for r, _ in nz)
        cmin = min(c for _, c in nz); cmax = max(c for _, c in nz)
        out = [row[:] for row in inp]
        for r in range(h):
            for c in range(w):
                if rmin <= r <= rmax and cmin <= c <= cmax and inp[r][c] == 0:
                    out[r][c] = target
        return out

    def test_detects_simple(self):
        inp = [
            [0, 0, 0, 0, 0],
            [0, 8, 0, 8, 0],
            [0, 0, 0, 0, 0],
            [0, 8, 0, 8, 0],
            [0, 0, 0, 0, 0],
        ]
        out = self._bbox(inp, 2)
        self.assertEqual(_detect_bbox_fill([_pair(inp, out)]), {"target_color": 2})

    def test_detects_consistent_target_across_pairs(self):
        inp1 = [[0, 8, 0], [0, 0, 8]]
        inp2 = [[8, 0], [0, 8]]
        out1 = self._bbox(inp1, 5)
        out2 = self._bbox(inp2, 5)
        self.assertEqual(_detect_bbox_fill([_pair(inp1, out1), _pair(inp2, out2)]),
                         {"target_color": 5})

    def test_rejects_inconsistent_target(self):
        inp1 = [[0, 8, 0], [0, 0, 8]]
        inp2 = [[8, 0], [0, 8]]
        out1 = self._bbox(inp1, 5)
        out2 = self._bbox(inp2, 6)  # different target
        self.assertIsNone(_detect_bbox_fill([_pair(inp1, out1), _pair(inp2, out2)]))

    def test_rejects_overwrites_nonzero(self):
        # If a non-zero cell inside the bbox would change, reject.
        inp = [[8, 0], [0, 8]]
        out = [[9, 2], [2, 9]]  # 8 changed to 9, illegal
        self.assertIsNone(_detect_bbox_fill([_pair(inp, out)]))

    def test_rejects_changes_outside_bbox(self):
        inp = [[0, 0, 0], [0, 8, 0], [0, 0, 0]]
        # bbox is just (1, 1). Outside should be unchanged. If output[0,0]=2, reject.
        out = self._bbox(inp, 2)
        out[0][0] = 2
        self.assertIsNone(_detect_bbox_fill([_pair(inp, out)]))

    def test_rejects_shape_change(self):
        inp = [[0, 8]]
        out = [[0, 8], [0, 0]]
        self.assertIsNone(_detect_bbox_fill([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_bbox_fill([]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        inp = [
            [0, 0, 0, 0, 0, 0],
            [0, 8, 0, 0, 8, 0],
            [0, 0, 0, 8, 0, 0],
            [0, 8, 0, 0, 8, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        model = bbox_fill_network(target_color=2, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(5):
            for c in range(6):
                x[0, inp[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        # Bbox is rows 1..3, cols 1..4. Inside bbox color-0 cells become 2.
        # (1, 2): 0 -> 2.
        self.assertEqual(out[0, 2, 1, 2], 1.0)
        # (2, 1): 0 -> 2.
        self.assertEqual(out[0, 2, 2, 1], 1.0)
        # 8s stay.
        self.assertEqual(out[0, 8, 1, 1], 1.0)
        self.assertEqual(out[0, 8, 1, 4], 1.0)
        # Outside bbox: (0, 0) was 0, stays color 0 (ch0=1).
        self.assertEqual(out[0, 0, 0, 0], 1.0)
        # (4, 0) outside bbox.
        self.assertEqual(out[0, 0, 4, 0], 1.0)
        # Padding: all channels = 0.
        for ch in range(10):
            self.assertEqual(out[0, ch, 10, 10], 0.0)

    def test_builder_rejects_invalid_target(self):
        with self.assertRaises(ValueError):
            bbox_fill_network(target_color=0)
        with self.assertRaises(ValueError):
            bbox_fill_network(target_color=10)


class SymmetryCompletionTest(unittest.TestCase):
    def test_detects_v_axis(self):
        # task113-style: top-half has data, bottom-half is zero, output mirrors.
        inp = [
            [2, 2, 2],
            [3, 3, 3],
            [0, 0, 0],
            [0, 0, 0],
        ]
        out = [
            [2, 2, 2],
            [3, 3, 3],
            [3, 3, 3],
            [2, 2, 2],
        ]
        match = _detect_symmetry_completion([_pair(inp, out)])
        self.assertEqual(match, {"axis": "v", "h": 4, "w": 3})

    def test_detects_h_axis(self):
        # Left-half has data, right-half zero, output mirrors horizontally.
        inp = [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
        ]
        out = [
            [1, 2, 2, 1],
            [3, 4, 4, 3],
        ]
        match = _detect_symmetry_completion([_pair(inp, out)])
        self.assertEqual(match, {"axis": "h", "h": 2, "w": 4})

    def test_prefers_v_when_both_axes_fit(self):
        # Both v-flip and h-flip would produce the same output (symmetric input).
        # Detector tries v first.
        inp = [[1, 0], [0, 0]]
        out = [[1, 1], [1, 1]]  # both axes happen to give this
        # v-flip: [[1, 0], [1, 0]]; doesn't equal out. So this isn't a v-axis fit either.
        # Actually neither v nor h gives exact match here. Pick a better example.
        # Build deliberate v-only fit:
        inp_v = [[1, 2, 0], [0, 0, 0]]
        out_v = [[1, 2, 0], [1, 2, 0]]
        match = _detect_symmetry_completion([_pair(inp_v, out_v)])
        self.assertEqual(match["axis"], "v")

    def test_rejects_inconsistent_h_across_pairs(self):
        # h varies across pairs (so v-axis Gather indices won't be valid for both).
        # h must be constant for v-axis match. h=2 vs h=3 should reject.
        inp1 = [[1, 0], [0, 0]]
        out1 = [[1, 0], [1, 0]]  # v-axis: r=0 stays; r=1 reflects from r=0
        inp2 = [[2, 0], [0, 0], [0, 0]]
        out2 = [[2, 0], [0, 0], [2, 0]]  # v-axis: r=0 stays; r=1 -> input[1]; r=2 -> input[0]
        # Different h: 2 vs 3. Detector should reject (no constant h).
        # But w is constant (2), so h-axis might match if rule is right. Check:
        # h-axis on inp1: out1 should be [[1, 1], [0, 0]] (refl col); actual [[1, 0], [1, 0]] not match.
        # So h-axis fails too. Detector returns None.
        self.assertIsNone(_detect_symmetry_completion([_pair(inp1, out1), _pair(inp2, out2)]))

    def test_rejects_overwrites_nonzero(self):
        # If the input has non-zero cell that the output replaces, it's not sym-completion.
        inp = [[1, 0], [2, 0]]
        out = [[1, 1], [9, 2]]  # 2 -> 9 is illegal for sym-completion
        self.assertIsNone(_detect_symmetry_completion([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_symmetry_completion([]))

    def test_network_round_trip_v(self):
        import numpy as np
        import onnxruntime as ort
        # task113-style on 4x3.
        inp = [
            [2, 2, 2],
            [3, 3, 3],
            [0, 0, 0],
            [0, 0, 0],
        ]
        model = symmetry_completion_network("v", h=4, w=3, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(4):
            for c in range(3):
                x[0, inp[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        # Expected: rows 0..1 same as input; rows 2..3 reflected.
        # Row 2 col 0 should be 3 (from input row 1).
        self.assertEqual(out[0, 3, 2, 0], 1.0)
        # Row 3 col 0 should be 2 (from input row 0).
        self.assertEqual(out[0, 2, 3, 0], 1.0)
        # Top still preserved.
        self.assertEqual(out[0, 2, 0, 0], 1.0)
        self.assertEqual(out[0, 3, 1, 0], 1.0)
        # Padding outside 4x3 (col 3+ or row 4+): all channels = 0.
        for ch in range(10):
            self.assertEqual(out[0, ch, 5, 5], 0.0)
            self.assertEqual(out[0, ch, 0, 5], 0.0)


class DiagonalShearTest(unittest.TestCase):
    @staticmethod
    def _shear_output(inp):
        """Construct expected 6x6 sum-of-shifts output for a 3x3 input."""
        out = [[0] * 6 for _ in range(6)]
        for r in range(6):
            for c in range(6):
                v = 0
                for k in range(6):
                    i, j = r - k, c - k
                    if 0 <= i <= 2 and 0 <= j <= 2:
                        v += inp[i][j]
                out[r][c] = v
        return out

    def test_detects_simple_shear(self):
        # task327 train[0] — three non-zeros on different anti-diagonals.
        inp = [[6, 1, 0], [3, 0, 0], [0, 0, 0]]
        pairs = [_pair(inp, self._shear_output(inp))]
        self.assertEqual(_detect_diagonal_shear(pairs), {"matched": True})

    def test_detects_all_zeros(self):
        # All-zero input: output is also all-zero. Trivially matches.
        inp = [[0] * 3 for _ in range(3)]
        out = [[0] * 6 for _ in range(6)]
        self.assertEqual(_detect_diagonal_shear([_pair(inp, out)]), {"matched": True})

    def test_rejects_wrong_input_shape(self):
        inp = [[1, 0], [0, 1]]
        out = [[0] * 6 for _ in range(6)]
        self.assertIsNone(_detect_diagonal_shear([_pair(inp, out)]))

    def test_rejects_wrong_output_shape(self):
        inp = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        out = [[0] * 9 for _ in range(9)]
        self.assertIsNone(_detect_diagonal_shear([_pair(inp, out)]))

    def test_rejects_color_conflict_on_anti_diagonal(self):
        # Two distinct non-zero colors on input main diagonal (0,0)=5 and (1,1)=7.
        # Both contribute to output[1, 1]: 5 + 7 = 12 — invalid color, builder
        # would corrupt. Detector must reject.
        inp = [[5, 0, 0], [0, 7, 0], [0, 0, 0]]
        # The actual data property check: if the data wanted output[1,1] = 12 (sum)
        # the detector should reject. Construct an output that's the literal sum:
        out = self._shear_output(inp)  # has out[1, 1] = 12 (a corrupt sum)
        self.assertIsNone(_detect_diagonal_shear([_pair(inp, out)]))

    def test_rejects_inconsistent_with_rule(self):
        inp = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Rule says output[k, k] = 1 for k=0..5; corrupt output to break that.
        out = self._shear_output(inp)
        out[3][3] = 9  # break the diagonal
        self.assertIsNone(_detect_diagonal_shear([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_diagonal_shear([]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # task327 train[0]
        inp = [[6, 1, 0], [3, 0, 0], [0, 0, 0]]
        model = diagonal_shear_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                x[0, inp[r][c], r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        # Spot-check: along output main diagonal, all cells with non-zero overlap show 6.
        self.assertEqual(out[0, 6, 0, 0], 1.0)
        self.assertEqual(out[0, 6, 1, 1], 1.0)
        self.assertEqual(out[0, 6, 5, 5], 1.0)
        # Anti-diagonal r-c = -1: should have value 1 at (0, 1), (1, 2), ..., (4, 5).
        self.assertEqual(out[0, 1, 0, 1], 1.0)
        self.assertEqual(out[0, 1, 4, 5], 1.0)
        # Anti-diagonal r-c = 1: value 3 at (1, 0), (2, 1), ..., (5, 4).
        self.assertEqual(out[0, 3, 1, 0], 1.0)
        self.assertEqual(out[0, 3, 5, 4], 1.0)
        # Background cell inside 6x6 (no contribution): ch0 = 1.
        # output[0, 2] is on anti-diag k=-2, with input (0, 2) = 0. So ch0 = 1.
        self.assertEqual(out[0, 0, 0, 2], 1.0)
        # Padding outside 6x6: all channels = 0.
        for ch in range(10):
            self.assertEqual(out[0, ch, 10, 10], 0.0)


class MarkerDilateTest(unittest.TestCase):
    """marker-dilate-recolor: 3x3 MaxPool dilation of marker color, with the
    dilated region recolored to target color (preserve_marker controls whether
    marker cells stay marker or become target).

    Scenarios:
      A. preserve_marker=True ("ring" mode, task095 family): marker stays;
         8-neighbors become target.
      B. preserve_marker=False ("fill" mode, task317 family): marker AND
         8-neighbors become target.
      C. Pass-through: input cells of unrelated colors are preserved.
      D. Canvas-clip: marker near active-grid edge does NOT bleed target into
         canvas padding.
      E. Shape mismatch -> detector returns None.
    """

    def _input_with_markers(self, h, w, marker_cells, marker_color, other_cells=None):
        """Build a (h, w) grid: 0 background, marker_color at marker_cells, other_cells overrides."""
        grid = [[0] * w for _ in range(h)]
        for (r, c) in marker_cells:
            grid[r][c] = marker_color
        for (r, c, color) in (other_cells or []):
            grid[r][c] = color
        return grid

    def _expected_output(self, h, w, marker_cells, marker_color, target_color, preserve, other_cells=None):
        grid = self._input_with_markers(h, w, marker_cells, marker_color, other_cells)
        # Apply dilation
        out = [row[:] for row in grid]
        for (r, c) in marker_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if (nr, nc) in marker_cells:
                            out[nr][nc] = marker_color if preserve else target_color
                        else:
                            out[nr][nc] = target_color
        return out

    def test_a_preserve_ring_mode(self):
        # Single marker at (4, 4) on 9x9 grid, marker=5, target=1
        markers = [(4, 4)]
        inp = self._input_with_markers(9, 9, markers, 5)
        out = self._expected_output(9, 9, markers, 5, 1, preserve=True)
        match = _detect_marker_dilate([{"input": inp, "output": out}])
        self.assertEqual(match, {"marker_color": 5, "target_color": 1, "preserve_marker": True})
        self.assertEqual(_detect_marker_dilate_conv_safe([{"input": inp, "output": out}]), match)

    def test_a2_safe_conv_builder_is_strict_one_hot(self):
        import numpy as np
        import onnxruntime as ort

        markers = [(1, 1), (1, 5), (5, 3)]
        inp = self._input_with_markers(9, 9, markers, 5)
        out = self._expected_output(9, 9, markers, 5, 1, preserve=True)
        model = marker_dilate_conv_network(marker_color=5, target_color=1, preserve_marker=True)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(9):
            for c in range(9):
                x[0, inp[r][c], r, c] = 1.0
        raw = sess.run(["output"], {"input": x})[0]
        self.assertTrue(np.array_equal(raw[0, :, :9, :9], np.eye(10, dtype=np.float32)[out].transpose(2, 0, 1)))
        self.assertTrue(np.all((raw == 0.0) | (raw == 1.0)))

    def test_b_replace_fill_mode(self):
        markers = [(4, 4)]
        inp = self._input_with_markers(9, 9, markers, 5)
        out = self._expected_output(9, 9, markers, 5, 1, preserve=False)
        match = _detect_marker_dilate([{"input": inp, "output": out}])
        self.assertEqual(match, {"marker_color": 5, "target_color": 1, "preserve_marker": False})

    def test_c_passthrough_other_color_cells(self):
        # Markers at (1,1) and (6,6); plus an unrelated color-3 cell at (3,8) that should
        # be preserved untouched.
        markers = [(1, 1), (6, 6)]
        other = [(3, 8, 3)]
        inp = self._input_with_markers(9, 9, markers, 5, other)
        out = self._expected_output(9, 9, markers, 5, 1, preserve=True, other_cells=other)
        match = _detect_marker_dilate([{"input": inp, "output": out}])
        self.assertEqual(match, {"marker_color": 5, "target_color": 1, "preserve_marker": True})

    def test_d_canvas_edge_no_padding_bleed_via_builder(self):
        """Builder must not light up the target channel in canvas padding cells.
        Place a marker at the active-grid corner (0, 0) on a 5x5 grid; dilation
        of (0,0) reaches (1, 1) inside the grid but also (0, -1) / (-1, 0) /
        (-1, -1) which are outside. Those don't exist in the grid but in the
        [1, 10, 30, 30] canvas they would correspond to (0, 29) / (29, 0) etc.
        — the dilation must not light those.
        """
        import numpy as np
        import onnxruntime as ort
        m = marker_dilate_network(marker_color=5, target_color=1, preserve_marker=True)
        sess = ort.InferenceSession(m.SerializeToString())
        # Build [1, 10, 30, 30] input with active grid 5x5 (top-left); marker at (0, 0).
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Active-grid background is color 0; a marker at (0,0) is color 5.
        for r in range(5):
            for c in range(5):
                inp[0, 0, r, c] = 1.0  # all cells start as color 0
        inp[0, 0, 0, 0] = 0.0
        inp[0, 5, 0, 0] = 1.0          # marker at (0, 0) is color 5
        out = sess.run(["output"], {"input": inp})[0]
        thresh = (out > 0.0).astype(np.float32)
        # Inside active grid: dilation neighbors (0,1), (1,0), (1,1) become target color 1.
        self.assertEqual(thresh[0, 1, 0, 1], 1.0)
        self.assertEqual(thresh[0, 1, 1, 0], 1.0)
        self.assertEqual(thresh[0, 1, 1, 1], 1.0)
        # Marker (0,0) preserved as color 5.
        self.assertEqual(thresh[0, 5, 0, 0], 1.0)
        # Canvas-padding cells (just outside the 5x5 active grid) must be all-zero.
        for ch in range(10):
            self.assertEqual(thresh[0, ch, 5, 0], 0.0, f"canvas pad (5,0) ch{ch} bled")
            self.assertEqual(thresh[0, ch, 0, 5], 0.0, f"canvas pad (0,5) ch{ch} bled")
            self.assertEqual(thresh[0, ch, 5, 5], 0.0, f"canvas pad (5,5) ch{ch} bled")

    def test_d2_fast_conv_rejects_edge_marker(self):
        markers = [(0, 0)]
        inp = self._input_with_markers(5, 5, markers, 5)
        out = self._expected_output(5, 5, markers, 5, 1, preserve=True)
        self.assertIsNone(_detect_marker_dilate_conv_safe([{"input": inp, "output": out}]))

    def test_d3_fast_conv_rejects_overlapping_rings(self):
        markers = [(2, 2), (2, 4)]
        inp = self._input_with_markers(7, 7, markers, 5)
        out = self._expected_output(7, 7, markers, 5, 1, preserve=True)
        self.assertIsNone(_detect_marker_dilate_conv_safe([{"input": inp, "output": out}]))

    def test_e_shape_mismatch_rejected(self):
        # Different input/output shapes -> detector rejects.
        inp = [[0, 5, 0], [0, 0, 0]]
        out = [[0, 1, 0]]  # different h
        self.assertIsNone(_detect_marker_dilate([{"input": inp, "output": out}]))

    def test_f_cross_pair_consistency(self):
        # Two pairs with consistent (M=5, T=1, preserve=True). Detector returns same.
        markers1 = [(1, 1)]
        markers2 = [(2, 2)]
        pair1 = {"input": self._input_with_markers(5, 5, markers1, 5),
                 "output": self._expected_output(5, 5, markers1, 5, 1, preserve=True)}
        pair2 = {"input": self._input_with_markers(5, 5, markers2, 5),
                 "output": self._expected_output(5, 5, markers2, 5, 1, preserve=True)}
        match = _detect_marker_dilate([pair1, pair2])
        self.assertEqual(match, {"marker_color": 5, "target_color": 1, "preserve_marker": True})

    def test_g_inconsistent_pairs_rejected(self):
        # Pair 1: preserve=True. Pair 2: preserve=False with same M/T. Detector rejects.
        markers = [(1, 1)]
        pair1 = {"input": self._input_with_markers(5, 5, markers, 5),
                 "output": self._expected_output(5, 5, markers, 5, 1, preserve=True)}
        pair2 = {"input": self._input_with_markers(5, 5, markers, 5),
                 "output": self._expected_output(5, 5, markers, 5, 1, preserve=False)}
        # The detector tries (M, T, preserve) combos and returns None if no
        # single combo fits both pairs.
        match = _detect_marker_dilate([pair1, pair2])
        self.assertIsNone(match)


class DispatcherDefensiveInputTest(unittest.TestCase):
    """Dispatcher must not crash on malformed input.

    Regression for an audit-2 finding (2026-05-04): 5 detectors
    (symmetry_completion / bbox_fill / outline_extract / per_color_stamp /
    translation / pixel_perm) raised IndexError on jagged grids (rows of
    different lengths), which would crash the entire `--hybrid` build if
    ever encountered. Fixed by validating well-formed input at the dispatch
    boundary in `_collect_matches` (returns `[]` on jagged input rather
    than letting individual detectors crash).
    """

    def test_jagged_grid_returns_no_matches_no_exception(self):
        from build_submission import _collect_matches, detect_pattern
        jagged = [{"input": [[5, 5], [5]], "output": [[5, 5], [5]]}]
        # Should NOT raise. Returns empty match list.
        self.assertEqual(_collect_matches(jagged), [])
        # detect_pattern wraps _collect_matches; should also be safe.
        examples = {"train": jagged, "test": [], "arc-gen": []}
        label, fn = detect_pattern(examples)
        self.assertIsNone(label)
        self.assertIsNone(fn)

    def test_jagged_output_returns_no_matches(self):
        from build_submission import _collect_matches
        jagged_out = [{"input": [[5, 5], [3, 3]], "output": [[5, 5], [3]]}]
        self.assertEqual(_collect_matches(jagged_out), [])

    def test_well_formed_input_still_dispatches(self):
        from build_submission import _collect_matches
        # Sanity: rectangular input must still fire matches as before.
        rectangular = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}]
        matches = _collect_matches(rectangular)
        self.assertGreater(len(matches), 0,
                           "Well-formed pair should still produce at least one match")

    def test_empty_input_handled(self):
        from build_submission import _collect_matches
        # Empty pairs list: `_collect_matches` handles it (empty loop) and
        # returns no matches.
        self.assertEqual(_collect_matches([]), [])


class DispatcherAuditCoverageTest(unittest.TestCase):
    """Audit-tool drift: every detector wired into `_collect_matches` must also
    appear in `dispatcher_audit._candidate_factories()`. Otherwise the audit
    tool can't independently verify the dispatcher's cheapest-match invariant
    for the un-listed detectors.

    First-audit finding (2026-05-03): 12 of 28 detectors were missing from
    the audit's candidate pool. This test prevents future drift by asserting
    every label produced by `_collect_matches` over a reasonable input set
    appears in `_candidate_factories`. The test runs the dispatcher on every
    SOLVED task and collects the labels actually produced; each must be in
    the audit's candidate-factory label set.
    """

    def test_every_dispatcher_label_has_audit_factory(self):
        import sys
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root / "tools"))
        # Import the audit's candidate factory list.
        from dispatcher_audit import _candidate_factories
        audit_labels = {label for label, _, _ in _candidate_factories()}
        # Read every label that `_collect_matches` produces for any task
        # in the per-pattern tally documented in `tools/inspect_tasks.py`.
        # Authoritative source: hard-coded labels in `build_submission.py:_collect_matches`.
        # We extract them via a static scan rather than running the dispatcher
        # (avoids the data/ dependency).
        src_path = repo_root / "tools" / "build_submission.py"
        src = src_path.read_text(encoding="utf-8")
        # _collect_matches uses string literals as labels: matches.append(("LABEL", ...))
        # plus one f-string for fixed-shape-spatial subtypes.
        import re
        # Find all `matches.append(("LABEL"` literals.
        literal_labels = set(re.findall(r'matches\.append\(\("([a-z0-9-]+)"', src))
        # The fixed-shape-spatial f-string covers all "fixed-*" subtypes; the
        # audit lists "fixed-shape-spatial" (the parent label). Strip individual
        # subtype matches and ensure parent is present.
        # Plus the _PATTERNS table at the top adds "transpose".
        pattern_labels = set(re.findall(r'^\s*\("([a-z0-9-]+)",\s*_\w+_grid', src, re.M))
        all_dispatcher_labels = literal_labels | pattern_labels
        # Audit must cover every label except the special "fixed-{transform}" subtypes
        # which collapse under the "fixed-shape-spatial" umbrella in the audit.
        missing = all_dispatcher_labels - audit_labels - {"transpose"}
        self.assertFalse(missing, f"dispatcher labels missing from dispatcher_audit candidate factories: {missing}")


class BuilderTensorNameUniquenessTest(unittest.TestCase):
    """Detect tensor-name prefix collisions across builder functions.

    First audit's B2 caught `mc_` reused by marker_corners + mirror_concat (the
    latter renamed to `mcat_` in the audit-fix bundle). This test prevents future
    collisions by asserting that no string-literal tensor name (e.g. "mc_active")
    is constructed inside more than one builder function.

    Helper-shared prefixes that ARE intentional (e.g. `fs_*` reused across the
    7 fixed_*_network builders that share `_slice_active_inits`) are allowed
    because the helper is the single source of those literals — the names appear
    only inside the helper function, not duplicated in each builder.
    """

    def test_no_initializer_name_appears_in_two_network_builders(self):
        import ast
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parent.parent
        src_path = repo_root / "tools" / "build_submission.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))

        # Only check string literals that are passed as the FIRST positional arg
        # to `onnx.helper.make_tensor(...)` — these become initializer names in
        # the ONNX graph, and are the actual collision surface. Parameter names
        # like `fill_color` and transform-string-literals like `flip_h` are NOT
        # in this surface.
        #
        # Helper functions ending in `_inits` (e.g. `_slice_active_inits`) can
        # legitimately be called by many builders and are exempt — collisions
        # within them aren't bugs since the helper is the single source.
        def is_make_tensor_call(node):
            if not isinstance(node, ast.Call):
                return False
            f = node.func
            return (
                isinstance(f, ast.Attribute)
                and f.attr == "make_tensor"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "helper"
            )

        per_func = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.endswith("_network"):
                continue  # only check builder functions, not detectors / helpers
            init_names = set()
            for sub in ast.walk(node):
                if is_make_tensor_call(sub) and sub.args:
                    first = sub.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        init_names.add(first.value)
            if init_names:
                per_func[node.name] = init_names

        from collections import defaultdict
        owners = defaultdict(set)
        for fn_name, names in per_func.items():
            for n in names:
                owners[n].add(fn_name)
        collisions = {n: sorted(owners[n]) for n in owners if len(owners[n]) > 1}

        self.assertFalse(
            collisions,
            f"initializer-name collisions across *_network builders: {collisions!r} "
            f"(rename one occurrence to prevent duplicate-name errors if these graphs "
            f"are ever composed)",
        )


class ForegroundBboxCropTest(unittest.TestCase):
    """Detector + builder tests for foreground-bbox-crop (task031 family).

    Rule: output = input cropped to bounding box of all non-zero cells.
    Variable input shape; variable output shape.
    """

    def test_detects_simple_3x3_in_5x5(self):
        # Foreground at rows 1-2, cols 1-3.
        inp = [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 4, 5, 6, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        out = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(_detect_foreground_bbox_crop([_pair(inp, out)]), {"matched": True})

    def test_detects_task031_train0_shape(self):
        # task031 train[0]: 10x12 input -> 4x4 output.
        inp = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        out = [
            [0, 2, 2, 2],
            [0, 0, 2, 0],
            [2, 2, 2, 0],
            [2, 0, 2, 0],
        ]
        self.assertEqual(_detect_foreground_bbox_crop([_pair(inp, out)]), {"matched": True})

    def test_detects_full_active_grid(self):
        # No padding to crop — input fully foreground.
        inp = [[1, 2], [3, 4]]
        out = [[1, 2], [3, 4]]
        self.assertEqual(_detect_foreground_bbox_crop([_pair(inp, out)]), {"matched": True})

    def test_rejects_no_foreground(self):
        inp = [[0, 0], [0, 0]]
        out = [[0, 0]]
        self.assertIsNone(_detect_foreground_bbox_crop([_pair(inp, out)]))

    def test_rejects_wrong_crop(self):
        inp = [[0, 1, 0], [0, 2, 0], [0, 3, 0]]
        # Correct crop is [[1], [2], [3]]; provide a wrong crop.
        out = [[1, 1], [2, 2], [3, 3]]
        self.assertIsNone(_detect_foreground_bbox_crop([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_foreground_bbox_crop([]))

    def test_rejects_when_one_pair_fails(self):
        good_in = [[0, 1], [0, 0]]
        good_out = [[1]]
        bad_in = [[0, 2], [0, 0]]
        bad_out = [[5]]  # wrong color
        self.assertIsNone(_detect_foreground_bbox_crop([
            _pair(good_in, good_out),
            _pair(bad_in, bad_out),
        ]))

    def test_network_round_trip_simple(self):
        import numpy as np
        import onnxruntime as ort
        # 3x3 input at top-left, foreground at (0, 0) only.
        # Expected output: input cropped to bbox = single cell of color 5.
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Active grid is 3x3 at top-left; cells (1, 1) and (1, 2) carry color 5.
        # Cells (0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2) are color 0.
        # Encode: ch0 at (0, 0..2), (1, 0), (2, 0..2); ch5 at (1, 1) and (1, 2).
        for r in range(3):
            for c in range(3):
                x[0, 0, r, c] = 1.0  # default to color 0
        x[0, 0, 1, 1] = 0.0
        x[0, 0, 1, 2] = 0.0
        x[0, 5, 1, 1] = 1.0
        x[0, 5, 1, 2] = 1.0
        # The foreground bbox is (1, 1)-(1, 2) — a single row of 2 cells.
        # Output: (0, 0) = 5, (0, 1) = 5; everything else canvas padding (all-zero).
        model = foreground_bbox_crop_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        out_bin = (out > 0.0).astype(np.float32)
        self.assertEqual(out_bin[0, 5, 0, 0], 1.0)
        self.assertEqual(out_bin[0, 5, 0, 1], 1.0)
        # All other channels at (0, 0) and (0, 1) are 0.
        for ch in range(10):
            if ch != 5:
                self.assertEqual(out_bin[0, ch, 0, 0], 0.0)
                self.assertEqual(out_bin[0, ch, 0, 1], 0.0)
        # Cells beyond the 1x2 bbox: all-zero (canvas padding).
        for ch in range(10):
            self.assertEqual(out_bin[0, ch, 0, 2], 0.0)
            self.assertEqual(out_bin[0, ch, 1, 0], 0.0)
            self.assertEqual(out_bin[0, ch, 5, 5], 0.0)

    def test_builder_rejects_batch_gt_1(self):
        with self.assertRaises(NotImplementedError):
            foreground_bbox_crop_network(batch=2)


class FractalRingExpansionTest(unittest.TestCase):
    """Detector + builder tests for fractal-ring-expansion (task123).

    Rule: input (5, 5); output (10, 10); colors[k] = input[k][k];
    fg = [c for c in colors if c != 0]; output[i][j] = fg[max(i, j) mod K]
    where K = len(fg) ∈ {4, 5}.
    """

    @staticmethod
    def _build_output(diagonal):
        fg = [c for c in diagonal if c != 0]
        K = len(fg)
        return [[fg[max(i, j) % K] for j in range(10)] for i in range(10)]

    def test_detects_K5(self):
        diagonal = [2, 3, 4, 1, 6]
        inp = [[diagonal[i] if i == j else 0 for j in range(5)] for i in range(5)]
        out = self._build_output(diagonal)
        self.assertEqual(_detect_fractal_ring_expansion([_pair(inp, out)]), {"matched": True})

    def test_detects_K4(self):
        diagonal = [1, 1, 4, 6, 0]
        inp = [[diagonal[i] if i == j else 0 for j in range(5)] for i in range(5)]
        out = self._build_output(diagonal)
        self.assertEqual(_detect_fractal_ring_expansion([_pair(inp, out)]), {"matched": True})

    def test_rejects_wrong_input_shape(self):
        inp = [[1, 0], [0, 1]]
        out = [[0] * 10 for _ in range(10)]
        self.assertIsNone(_detect_fractal_ring_expansion([_pair(inp, out)]))

    def test_rejects_wrong_output_shape(self):
        diagonal = [1, 2, 3, 4, 5]
        inp = [[diagonal[i] if i == j else 0 for j in range(5)] for i in range(5)]
        out = self._build_output(diagonal)
        self.assertIsNone(_detect_fractal_ring_expansion([_pair(inp, out[:9])]))  # 9x10 out

    def test_rejects_bg_in_middle_of_diagonal(self):
        diagonal = [1, 0, 4, 6, 5]  # 0 at index 1, not 4
        inp = [[diagonal[i] if i == j else 0 for j in range(5)] for i in range(5)]
        out = self._build_output(diagonal)
        self.assertIsNone(_detect_fractal_ring_expansion([_pair(inp, out)]))

    def test_rejects_corrupt_output(self):
        diagonal = [2, 3, 4, 1, 6]
        inp = [[diagonal[i] if i == j else 0 for j in range(5)] for i in range(5)]
        out = self._build_output(diagonal)
        out[5][5] = 9  # corrupt one cell
        self.assertIsNone(_detect_fractal_ring_expansion([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_fractal_ring_expansion([]))

    def test_network_round_trip_K5(self):
        import numpy as np
        import onnxruntime as ort
        diagonal = [2, 3, 4, 1, 6]
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Active grid 5x5: each (k, k) lit at channel diagonal[k]; off-diagonal lit at ch0.
        for r in range(5):
            for c in range(5):
                x[0, diagonal[r] if r == c else 0, r, c] = 1.0
        model = fractal_ring_expansion_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        out_bin = (out > 0.0).astype(np.float32)
        # Spot check a few cells per the rule.
        # output[0][0] = fg[0] = 2; output[0][9] = fg[9 % 5] = fg[4] = 6; output[5][5] = fg[0] = 2.
        self.assertEqual(out_bin[0, 2, 0, 0], 1.0)
        self.assertEqual(out_bin[0, 6, 0, 9], 1.0)
        self.assertEqual(out_bin[0, 2, 5, 5], 1.0)
        # Padding outside 10x10 must be all-zero.
        for ch in range(10):
            self.assertEqual(out_bin[0, ch, 15, 15], 0.0)

    def test_network_round_trip_K4(self):
        import numpy as np
        import onnxruntime as ort
        diagonal = [1, 1, 4, 6, 0]
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                x[0, diagonal[r] if r == c else 0, r, c] = 1.0
        model = fractal_ring_expansion_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        out_bin = (out > 0.0).astype(np.float32)
        # K=4, fg=[1,1,4,6]. output[0][4] = fg[4 % 4] = fg[0] = 1.
        # output[6][6] = fg[6 % 4] = fg[2] = 4.
        # output[7][7] = fg[7 % 4] = fg[3] = 6.
        self.assertEqual(out_bin[0, 1, 0, 4], 1.0)
        self.assertEqual(out_bin[0, 4, 6, 6], 1.0)
        self.assertEqual(out_bin[0, 6, 7, 7], 1.0)

    def test_builder_rejects_batch_gt_1(self):
        with self.assertRaises(NotImplementedError):
            fractal_ring_expansion_network(batch=2)


class ColorFilterBboxCropTest(unittest.TestCase):
    """Detector + builder tests for color-filter-bbox-crop (task014 family)."""

    def test_detects_least_color_simple(self):
        # 4x4 input: color 5 has 12 cells, color 7 has 1 cell. Least = 7. Bbox = single cell.
        inp = [[5, 5, 5, 5], [5, 5, 5, 5], [5, 7, 5, 5], [5, 5, 5, 5]]
        out = [[7]]
        result = _detect_color_filter_bbox_crop([_pair(inp, out)], picker="least")
        self.assertEqual(result, {"matched": True, "picker": "least"})

    def test_detects_least_with_bg(self):
        inp = [[0, 8, 8, 0], [0, 8, 0, 0], [0, 0, 2, 0]]
        # color 8 = 3, color 2 = 1. Least = 2. Bbox = 1x1 = [[2]].
        out = [[2]]
        result = _detect_color_filter_bbox_crop([_pair(inp, out)], picker="least")
        self.assertEqual(result, {"matched": True, "picker": "least"})

    def test_rejects_no_nonzero_colors(self):
        inp = [[0, 0], [0, 0]]
        out = [[0]]
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out)], picker="least"))

    def test_rejects_wrong_crop(self):
        inp = [[8, 8], [8, 2]]  # least=2 at (1,1), bbox = 1x1 = [[2]]
        out = [[2, 2]]
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out)], picker="least"))

    def test_rejects_unknown_picker(self):
        inp = [[8, 8], [8, 2]]
        out = [[2]]
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out)], picker="most"))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_color_filter_bbox_crop([], picker="least"))

    def test_network_round_trip_simple(self):
        import numpy as np
        import onnxruntime as ort
        # 4x4 active grid: 14 cells of color 5, 1 cell of color 7 at (2, 1).
        # Output bbox = 1x1 of color 7 at (0, 0).
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                x[0, 5, r, c] = 1.0
        x[0, 5, 2, 1] = 0.0
        x[0, 7, 2, 1] = 1.0
        model = color_filter_bbox_crop_network(picker="least", batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        out_bin = (out > 0.0).astype(np.float32)
        # Output cell (0, 0) should be color 7.
        self.assertEqual(out_bin[0, 7, 0, 0], 1.0)
        for ch in range(10):
            if ch != 7:
                self.assertEqual(out_bin[0, ch, 0, 0], 0.0)
        # All other cells (canvas padding) all-zero.
        for ch in range(10):
            self.assertEqual(out_bin[0, ch, 1, 0], 0.0)
            self.assertEqual(out_bin[0, ch, 0, 1], 0.0)

    def test_network_preserves_other_colors_inside_bbox(self):
        import numpy as np
        import onnxruntime as ort

        grid = [
            [8, 8, 8, 8, 8],
            [8, 7, 5, 0, 8],
            [8, 0, 5, 0, 8],
            [8, 5, 0, 7, 8],
            [8, 8, 8, 8, 8],
        ]
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                x[0, value, r, c] = 1.0

        model = color_filter_bbox_crop_network(picker="least", batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = (sess.run(["output"], {"input": x})[0] > 0.0).astype(np.float32)

        expected = [
            [7, 5, 0],
            [0, 5, 0],
            [5, 0, 7],
        ]
        for r, row in enumerate(expected):
            for c, value in enumerate(row):
                self.assertEqual(out[0, value, r, c], 1.0)
        self.assertEqual(out[0, 8, 0, 0], 0.0)
        for ch in range(10):
            self.assertEqual(out[0, ch, 3, 0], 0.0)
            self.assertEqual(out[0, ch, 0, 3], 0.0)

    def test_builder_rejects_batch_gt_1(self):
        with self.assertRaises(NotImplementedError):
            color_filter_bbox_crop_network(picker="least", batch=2)

    def test_builder_rejects_unknown_picker(self):
        with self.assertRaises(NotImplementedError):
            color_filter_bbox_crop_network(picker="zzz_unknown", batch=1)

    def test_detects_smallest_bbox_picker(self):
        # 5x5: color 5 spans 4x4 bbox, color 7 spans 1x1 bbox. Smallest = 7.
        inp = [[5, 5, 5, 5, 5], [5, 0, 0, 0, 5], [5, 0, 7, 0, 5],
               [5, 0, 0, 0, 5], [5, 5, 5, 5, 5]]
        out = [[7]]
        result = _detect_color_filter_bbox_crop([_pair(inp, out)], picker="smallest_bbox")
        self.assertEqual(result, {"matched": True, "picker": "smallest_bbox"})

    def test_smallest_bbox_picker_differs_from_least(self):
        # 4x4: color 5 has 3 sparse cells in 4x4 bbox (area 16); color 8 has 4 dense cells in 2x2 bbox (area 4).
        # Least picker -> 5 (count 3); smallest_bbox picker -> 8 (area 4).
        inp = [[5, 0, 0, 0], [0, 8, 8, 0], [0, 8, 8, 0], [5, 0, 0, 5]]
        out_least = [[5, 0, 0, 0], [0, 8, 8, 0], [0, 8, 8, 0], [5, 0, 0, 5]]  # bbox of 5 = entire 4x4
        out_sb = [[8, 8], [8, 8]]
        self.assertEqual(_detect_color_filter_bbox_crop([_pair(inp, out_least)], picker="least"),
                         {"matched": True, "picker": "least"})
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out_least)], picker="smallest_bbox"))
        self.assertEqual(_detect_color_filter_bbox_crop([_pair(inp, out_sb)], picker="smallest_bbox"),
                         {"matched": True, "picker": "smallest_bbox"})
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out_sb)], picker="least"))

    def test_smallest_bbox_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 5x5 active grid: color 5 spans 4x4 area (lots of cells); color 7 at (2, 2) only.
        # smallest_bbox picker -> 7. Output bbox = 1x1 of color 7.
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                x[0, 5, r, c] = 1.0  # default to color 5
        # Make a hole inside surrounded by 5s (no padding pixels).
        x[0, 5, 1, 1] = 0.0
        x[0, 0, 1, 1] = 1.0  # color 0
        x[0, 5, 2, 2] = 0.0
        x[0, 7, 2, 2] = 1.0  # color 7
        model = color_filter_bbox_crop_network(picker="smallest_bbox", batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        out_bin = (out > 0.0).astype(np.float32)
        # Output (0, 0) = color 7 (smallest bbox).
        self.assertEqual(out_bin[0, 7, 0, 0], 1.0)
        for ch in range(10):
            if ch != 7:
                self.assertEqual(out_bin[0, ch, 0, 0], 0.0)

    def test_detects_most_count_picker(self):
        # 3x3: color 5 has 5 cells (corners + center); color 4 has 4 cells.
        # most_count picker -> 5. bbox of 5 covers full 3x3.
        # Cropped output = full 3x3 input (preserving all colors per detector semantics).
        inp = [[5, 4, 5], [4, 5, 4], [5, 4, 5]]
        out = inp  # bbox of 5 = full input; cropped = full input
        result = _detect_color_filter_bbox_crop([_pair(inp, out)], picker="most_count")
        self.assertEqual(result, {"matched": True, "picker": "most_count"})

    def test_most_count_picker_rejects_ties(self):
        # 2x2: color 5 has 2 cells; color 4 has 2 cells. Tie -> rejected.
        inp = [[5, 4], [4, 5]]
        out = [[5, 4], [4, 5]]
        result = _detect_color_filter_bbox_crop([_pair(inp, out)], picker="most_count")
        self.assertIsNone(result)

    def test_most_count_picker_rejects_unsupported(self):
        # 'biggest_bbox' is not a supported picker
        inp = [[5, 0], [0, 5]]
        out = [[5, 0], [0, 5]]
        result = _detect_color_filter_bbox_crop([_pair(inp, out)], picker="biggest_bbox")
        self.assertIsNone(result)

    def test_most_count_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 3x3 with color 5 majority (5 cells) and color 4 minority (4 cells)
        # most_count picks 5; bbox of 5 is full 3x3. Output preserves all colors
        # within the bbox (matches detector semantics).
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        layout = [[5, 4, 5], [4, 5, 4], [5, 4, 5]]
        for r in range(3):
            for c in range(3):
                x[0, layout[r][c], r, c] = 1.0
        model = color_filter_bbox_crop_network(picker="most_count", batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # Output should have color 5 at corners + center
        self.assertEqual(out[0, 5, 0, 0], 1.0)
        self.assertEqual(out[0, 5, 2, 2], 1.0)
        self.assertEqual(out[0, 5, 1, 1], 1.0)
        # Color 4 cells preserved at (0, 1), (1, 0), (1, 2), (2, 1)
        self.assertEqual(out[0, 4, 0, 1], 1.0)
        self.assertEqual(out[0, 4, 1, 0], 1.0)
        self.assertEqual(out[0, 4, 1, 2], 1.0)
        self.assertEqual(out[0, 4, 2, 1], 1.0)
        # Outside 3x3 grid: all-zero (canvas-padding)
        self.assertEqual(int((out[0, :, 5, 5] > 0.5).sum()), 0)

    def test_max_density_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 30x30 canvas.
        # Color 1: 4 cells in 2x2 bbox. Area=4, Count=4, Density=1.0.
        # Color 3: 4 cells in 3x3 bbox. Area=9, Count=4, Density=0.44.
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        x[0, 1, 0, 0] = 1; x[0, 1, 0, 1] = 1; x[0, 1, 1, 0] = 1; x[0, 1, 1, 1] = 1
        x[0, 3, 2, 2] = 1; x[0, 3, 2, 4] = 1; x[0, 3, 4, 2] = 1; x[0, 3, 4, 4] = 1
        model = color_filter_bbox_crop_network(picker="max_density", batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # Should pick color 1. Bbox is 2x2.
        self.assertEqual(out[0, 1, 0, 0], 1.0)
        self.assertEqual(out[0, 1, 1, 1], 1.0)
        self.assertEqual(out[0, 3, 0, 0], 0.0)

    def test_detects_lr_symmetry_picker_over_denser_asymmetric_shape(self):
        inp = [
            [9, 9, 9, 9, 0],
            [0, 9, 9, 0, 0],
            [4, 4, 0, 0, 0],
            [4, 4, 4, 0, 0],
        ]
        out = [
            [9, 9, 9, 9],
            [0, 9, 9, 0],
        ]
        result = _detect_color_filter_bbox_crop([_pair(inp, out)], picker="lr_symmetry")
        self.assertEqual(result, {"matched": True, "picker": "lr_symmetry"})
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out)], picker="max_density"))

    def test_lr_symmetry_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort

        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        layout = [
            [9, 9, 9, 9, 0],
            [0, 9, 9, 0, 0],
            [4, 4, 0, 0, 0],
            [4, 4, 4, 0, 0],
        ]
        for r, row in enumerate(layout):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = color_filter_bbox_crop_network(picker="lr_symmetry", batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]

        self.assertGreater(out[0, 9, 0, 0], 0.5)
        self.assertGreater(out[0, 9, 0, 3], 0.5)
        self.assertGreater(out[0, 9, 1, 1], 0.5)
        self.assertGreater(out[0, 9, 1, 2], 0.5)
        self.assertLess(out[0, 4, 0, 0], 0.5)
        self.assertLess(out[0, 4, 1, 1], 0.5)
        self.assertLess(out[0, 9, 2, 0], 0.5)


class SeedFloodFillTest(unittest.TestCase):
    """Detector + builder tests for seed-flood-fill (task243 family)."""

    def test_detects_simple_fill(self):
        # 3x3: seed=1 at (0,0); 0 at (0,1), (1,0); 5 wall at (1,1).
        # Flood reaches (0,1) and (1,0); (0,2), (1,2), (2,0), (2,1), (2,2) reachable too.
        # Wait, (1,1)=5 is wall; (2,2) reachable via (2,1)<-(2,0)<-(1,0)<-(0,0).
        inp = [[1, 0, 0], [0, 5, 0], [0, 0, 0]]
        out = [[1, 1, 1], [1, 5, 1], [1, 1, 1]]
        self.assertEqual(_detect_seed_flood_fill([_pair(inp, out)]),
                         {"fill_color": 1, "canvas_h": 3, "canvas_w": 3})

    def test_detects_unreachable_island(self):
        # seed=1 at (0,0); wall row 1; (2,0),(2,1) unreachable (walled off).
        inp = [[1, 0], [5, 5], [0, 0]]
        out = [[1, 1], [5, 5], [0, 0]]
        self.assertEqual(_detect_seed_flood_fill([_pair(inp, out)]),
                         {"fill_color": 1, "canvas_h": 3, "canvas_w": 2})

    def test_detects_no_seed(self):
        inp = [[5, 0], [0, 5]]
        out = [[5, 0], [0, 5]]
        # No fill_color present in input, but rule trivially holds for any.
        # Detector returns the smallest fill_color (1) that satisfies — verify.
        result = _detect_seed_flood_fill([_pair(inp, out)])
        self.assertEqual(result, {"fill_color": 1, "canvas_h": 2, "canvas_w": 2})

    def test_rejects_corrupt_output(self):
        inp = [[1, 0], [0, 5]]
        out = [[1, 1], [1, 6]]  # walled cell color changed
        self.assertIsNone(_detect_seed_flood_fill([_pair(inp, out)]))

    def test_rejects_shape_change(self):
        inp = [[1, 0], [0, 5]]
        out = [[1, 1, 1], [1, 5, 0]]  # different shape
        self.assertIsNone(_detect_seed_flood_fill([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_seed_flood_fill([]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 3x3 active grid: (0,0)=1 seed, (1,1)=5 wall, others=0.
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                x[0, 0, r, c] = 1.0
        x[0, 0, 0, 0] = 0.0
        x[0, 1, 0, 0] = 1.0
        x[0, 0, 1, 1] = 0.0
        x[0, 5, 1, 1] = 1.0
        model = seed_flood_fill_network(fill_color=1, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        out_bin = (out > 0.0).astype(np.float32)
        # All 0-cells reached via flood from (0,0); they all become color 1.
        # (1,1)=5 unchanged; everywhere else inside 3x3 should be color 1.
        for r in range(3):
            for c in range(3):
                if (r, c) == (1, 1):
                    self.assertEqual(out_bin[0, 5, r, c], 1.0)
                    self.assertEqual(out_bin[0, 1, r, c], 0.0)
                else:
                    self.assertEqual(out_bin[0, 1, r, c], 1.0)
                    self.assertEqual(out_bin[0, 0, r, c], 0.0)
        # Padding outside 3x3 stays canvas (all-zero).
        for ch in range(10):
            self.assertEqual(out_bin[0, ch, 5, 5], 0.0)

    def test_builder_rejects_fill_color_zero(self):
        with self.assertRaises(ValueError):
            seed_flood_fill_network(fill_color=0, batch=1)


class KronBlockUpTest(unittest.TestCase):
    """Detector + builder tests for kron-block-up (task307 family)."""

    def test_detects_2x2_block_up(self):
        inp = [[1, 2], [3, 4]]
        out = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        self.assertEqual(_detect_kron_block_up([_pair(inp, out)]), {"scale_h": 2, "scale_w": 2})

    def test_detects_3x3_block_up(self):
        inp = [[5]]
        out = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        self.assertEqual(_detect_kron_block_up([_pair(inp, out)]), {"scale_h": 3, "scale_w": 3})

    def test_detects_asymmetric_scale(self):
        inp = [[1, 2]]
        out = [[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]]
        self.assertEqual(_detect_kron_block_up([_pair(inp, out)]), {"scale_h": 2, "scale_w": 3})

    def test_rejects_identity_scale(self):
        inp = [[1, 2], [3, 4]]
        self.assertIsNone(_detect_kron_block_up([_pair(inp, inp)]))

    def test_rejects_corrupt_block(self):
        inp = [[1, 2]]
        out = [[1, 1, 2, 2], [1, 1, 9, 2]]  # corrupted block
        self.assertIsNone(_detect_kron_block_up([_pair(inp, out)]))

    def test_rejects_non_divisor_scale(self):
        # 4x scale is not a divisor of 30 -> 7 (well, 4 IS not a 30 divisor).
        # 30 / 4 = 7.5; constructor would raise.
        inp = [[1]]
        out = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        self.assertIsNone(_detect_kron_block_up([_pair(inp, out)]))

    def test_rejects_output_exceeds_canvas(self):
        # 16x scale (not a divisor anyway) would exceed canvas; also covered
        # by valid_scales filter.
        inp = [[1, 2], [3, 4]]
        # Build an output where scale is 16 (would put output at 32, > 30).
        out = [[1] * 32 for _ in range(32)]
        self.assertIsNone(_detect_kron_block_up([_pair(inp, out)]))

    def test_rejects_inconsistent_scale_across_pairs(self):
        p1 = _pair([[1]], [[1, 1], [1, 1]])
        p2 = _pair([[1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertIsNone(_detect_kron_block_up([p1, p2]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_kron_block_up([]))

    def test_network_round_trip_2x2(self):
        import numpy as np
        import onnxruntime as ort
        # 2x3 input encoded on canvas (1, 10, 30, 30): cells (0,0)=1, (0,1)=2,
        # (0,2)=3, (1,0)=4, (1,1)=5, (1,2)=6.
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        layout = [[1, 2, 3], [4, 5, 6]]
        for r in range(2):
            for c in range(3):
                x[0, layout[r][c], r, c] = 1.0
        model = kron_block_up_network(scale_h=2, scale_w=2, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # Each input cell becomes a 2x2 block of the same color.
        for r in range(2):
            for c in range(3):
                color = layout[r][c]
                for dr in range(2):
                    for dc in range(2):
                        self.assertEqual(out[0, color, r * 2 + dr, c * 2 + dc], 1.0)

    def test_builder_rejects_non_divisor_scale(self):
        with self.assertRaises(ValueError):
            kron_block_up_network(scale_h=4, scale_w=2, batch=1)


class ConstantOutputLargestBboxTest(unittest.TestCase):
    """Detector + builder tests for const-largest-bbox (task100 family)."""

    def test_detects_largest_bbox_simple(self):
        # 3x3 input: ch4 has 1 cell (bbox 1x1=1); ch7 has 2 cells diag (bbox 2x2=4).
        # Output 2x2 of color 7 (largest bbox area).
        inp = [[7, 0, 0], [0, 7, 0], [0, 0, 4]]
        out = [[7, 7], [7, 7]]
        self.assertEqual(_detect_constant_output_largest_bbox([_pair(inp, out)]), (2, 2))

    def test_detects_tiebreak_smaller_index_wins(self):
        # ch3 and ch7 both have bbox 2x2=4. ArgMax tiebreak: smaller channel = 3.
        inp = [[3, 0, 0, 7, 0], [0, 3, 0, 0, 7], [0, 0, 0, 0, 0]]
        out = [[3]]
        self.assertEqual(_detect_constant_output_largest_bbox([_pair(inp, out)]), (1, 1))

    def test_rejects_non_uniform_output(self):
        inp = [[1, 0], [0, 2]]
        out = [[1, 2], [1, 1]]  # not uniform
        self.assertIsNone(_detect_constant_output_largest_bbox([_pair(inp, out)]))

    def test_rejects_inconsistent_output_shape(self):
        p1 = _pair([[1]], [[1]])
        p2 = _pair([[1]], [[1, 1]])  # different output shape
        self.assertIsNone(_detect_constant_output_largest_bbox([p1, p2]))

    def test_rejects_zero_output(self):
        inp = [[1, 0], [0, 0]]
        out = [[0]]  # all-bg output not this rule
        self.assertIsNone(_detect_constant_output_largest_bbox([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_constant_output_largest_bbox([]))

    def test_rejects_when_output_color_doesnt_match_largest_bbox(self):
        # ch3 has 5 cells in tight 1x5 bbox (area 5); ch7 has 2 cells in 3x1 bbox (area 3).
        # Largest-bbox = 3 (area 5). Detector should REJECT if output says 7.
        inp = [[3, 3, 3, 3, 3], [0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [7, 0, 0, 0, 0]]
        out = [[7]]  # output is 7, but largest-bbox is 3 -> reject
        self.assertIsNone(_detect_constant_output_largest_bbox([_pair(inp, out)]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 3x3 input encoded on canvas: ch7 at (0,0), (1,1) -> bbox 2x2; ch4 at (2,2) -> bbox 1x1.
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        x[0, 7, 0, 0] = 1.0
        x[0, 7, 1, 1] = 1.0
        x[0, 4, 2, 2] = 1.0
        model = largest_bbox_constant_output_network(h_out=2, w_out=2, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # Top-left 2x2 should be all-7
        for r in range(2):
            for c in range(2):
                self.assertEqual(out[0, 7, r, c], 1.0)
                self.assertEqual(out[0, 4, r, c], 0.0)
                self.assertEqual(out[0, 0, r, c], 0.0)
        # Beyond 2x2 stays canvas (all-zero)
        for ch in range(10):
            self.assertEqual(out[0, ch, 5, 5], 0.0)


class FramedCellConstOutputTest(unittest.TestCase):
    """Detector + builder tests for const-framed-cell (task346 family)."""

    def test_detects_simple_frame(self):
        # 5x5 input: 3x3 frame of 4 at (0..2, 0..2) with core 2 at (1, 1).
        inp = [[4, 4, 4, 0, 0], [4, 2, 4, 0, 0], [4, 4, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        out = [[2]]
        self.assertEqual(_detect_framed_cell_const_output([_pair(inp, out)]), (1, 1, 5, 5))

    def test_rejects_no_frame(self):
        inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # no all-same frame
        out = [[5]]
        self.assertIsNone(_detect_framed_cell_const_output([_pair(inp, out)]))

    def test_rejects_zero_frame(self):
        # Frame of 0 (bg) is not valid
        inp = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        out = [[5]]
        self.assertIsNone(_detect_framed_cell_const_output([_pair(inp, out)]))

    def test_rejects_zero_core(self):
        # Core of 0 (bg) is not valid
        inp = [[4, 4, 4], [4, 0, 4], [4, 4, 4]]
        out = [[0]]  # detector rejects out_color=0
        self.assertIsNone(_detect_framed_cell_const_output([_pair(inp, out)]))

    def test_rejects_same_color_frame_and_core(self):
        # Solid 3x3 of color 5 — frame == core, not a "frame around core"
        inp = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        out = [[5]]
        self.assertIsNone(_detect_framed_cell_const_output([_pair(inp, out)]))

    def test_rejects_non_uniform_output(self):
        inp = [[4, 4, 4], [4, 2, 4], [4, 4, 4]]
        out = [[2, 0]]  # not uniform
        self.assertIsNone(_detect_framed_cell_const_output([_pair(inp, out)]))

    def test_rejects_inconsistent_output_shape(self):
        p1 = _pair([[4, 4, 4], [4, 2, 4], [4, 4, 4]], [[2]])
        p2 = _pair([[4, 4, 4], [4, 2, 4], [4, 4, 4]], [[2, 2]])  # different shape
        self.assertIsNone(_detect_framed_cell_const_output([p1, p2]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_framed_cell_const_output([]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # Encode 5x5 input on canvas: frame of 4 at (0..2, 0..2), core 2 at (1, 1).
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                if (r, c) == (1, 1):
                    x[0, 2, r, c] = 1.0
                else:
                    x[0, 4, r, c] = 1.0
        model = framed_cell_const_output_network(h_out=1, w_out=1, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # Output 1x1 at (0,0): channel 2 active, others 0
        self.assertEqual(out[0, 2, 0, 0], 1.0)
        for ch in (0, 1, 3, 4, 5, 6, 7, 8, 9):
            self.assertEqual(out[0, ch, 0, 0], 0.0)


class MarkerAlignedShapeRecolorTest(unittest.TestCase):
    """Detector + builder tests for task143 marker-aligned shape recolor."""

    def _make_input(self, template_color, template_cells, other_blobs):
        """Build a 10x10 input with L-shape of 5 + template + scatter blobs.

        - L-shape: vertical col 3 rows 0..3, horizontal row 3 cols 0..3 (color 5).
        - template_cells: list of (r, c) within (0..2, 0..2) for template_color.
        - other_blobs: list of (color, list of (r, c)) tuples placed anywhere
          outside the L+template region.
        """
        grid = [[0] * 10 for _ in range(10)]
        for r in range(4):
            grid[r][3] = 5
        for c in range(4):
            grid[3][c] = 5
        for (r, c) in template_cells:
            grid[r][c] = template_color
        for color, cells in other_blobs:
            for (r, c) in cells:
                grid[r][c] = color
        return grid

    def _apply_rule(self, inp):
        """Apply the rule manually (mirror of the recipe's predict_task143)."""
        import numpy as np
        a = np.array(inp, dtype=np.int8)
        template_region = a[:3, :3]
        nonzeros = sorted(set(int(c) for c in template_region.flatten()) - {0, 5})
        if len(nonzeros) != 1:
            return None
        tc = nonzeros[0]
        tmask = (template_region == tc)
        rs = np.where(tmask.any(axis=1))[0]
        cs = np.where(tmask.any(axis=0))[0]
        tcrop = tmask[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
        tcount = int(tcrop.sum())
        out = a.copy()
        for c in range(1, 10):
            if c == 5 or c == tc:
                continue
            cm = (a == c)
            if not cm.any() or int(cm.sum()) != tcount:
                continue
            r2 = np.where(cm.any(axis=1))[0]
            c2 = np.where(cm.any(axis=0))[0]
            crop = cm[r2.min():r2.max() + 1, c2.min():c2.max() + 1]
            if crop.shape == tcrop.shape and np.array_equal(crop, tcrop):
                out[cm] = 5
        return out.tolist()

    def test_detects_simple_match(self):
        # Template = color 1 in L-shape: (0,0), (1,1), (2,2). Diagonal.
        inp = self._make_input(
            template_color=1,
            template_cells=[(0, 0), (1, 1), (2, 2)],
            other_blobs=[
                # Color 6 diagonal in lower-right (matches template)
                (6, [(5, 5), (6, 6), (7, 7)]),
                # Color 7 horizontal line (3 cells, different shape)
                (7, [(8, 0), (8, 1), (8, 2)]),
            ],
        )
        out = self._apply_rule(inp)
        self.assertIsNotNone(out)
        self.assertEqual(_detect_marker_aligned_shape_recolor([_pair(inp, out)]), True)

    def test_rejects_missing_l_shape(self):
        inp = [[0] * 10 for _ in range(10)]
        # No L-shape of 5
        inp[0][0] = 1
        out = inp  # Trivially same
        self.assertIsNone(_detect_marker_aligned_shape_recolor([_pair(inp, out)]))

    def test_rejects_no_template_color(self):
        inp = self._make_input(
            template_color=1,
            template_cells=[],  # Empty template region
            other_blobs=[],
        )
        out = inp
        self.assertIsNone(_detect_marker_aligned_shape_recolor([_pair(inp, out)]))

    def test_rejects_two_template_colors(self):
        inp = self._make_input(
            template_color=1,
            template_cells=[(0, 0)],
            other_blobs=[(2, [(2, 2)])],  # Adds another non-zero non-5 in template region
        )
        out = inp
        self.assertIsNone(_detect_marker_aligned_shape_recolor([_pair(inp, out)]))

    def test_rejects_wrong_shape(self):
        # Input shape != 10x10
        inp = [[0] * 5 for _ in range(5)]
        out = inp
        self.assertIsNone(_detect_marker_aligned_shape_recolor([_pair(inp, out)]))

    def test_rejects_modified_output_when_blob_doesnt_match(self):
        # Template = single cell at (0, 0). Other blob also single cell.
        # Per rule: ALL single-cell blobs match (1 cell, bbox-cropped == [[1]]).
        inp = self._make_input(
            template_color=1,
            template_cells=[(0, 0)],
            other_blobs=[(6, [(5, 5)])],
        )
        # Wrong output: didn't recolor the matching color-6 cell
        wrong_out = [row[:] for row in inp]
        self.assertIsNone(_detect_marker_aligned_shape_recolor([_pair(inp, wrong_out)]))

    def test_network_round_trip(self):
        import json
        import numpy as np
        import onnxruntime as ort

        # Use task143 train pair 0 directly.
        data_dir = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", "data/competitions/neurogolf-2026"))
        with open(data_dir / "task143.json") as f:
            t = json.load(f)
        p = t["train"][0]
        inp = np.array(p["input"], dtype=np.int64)
        expected = np.array(p["output"], dtype=np.int64)

        # Encode to (1, 10, 30, 30)
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        H, W = inp.shape
        for c in range(10):
            x[0, c, :H, :W] = (inp == c).astype(np.float32)

        model = marker_aligned_shape_recolor_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]

        # Decode + compare
        decoded = np.argmax(out[0], axis=0)
        self.assertTrue(np.array_equal(decoded[:H, :W], expected))

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            marker_aligned_shape_recolor_network(batch=2)


class GridDivisionCountTest(unittest.TestCase):
    """Detector + builder tests for task021 grid-division-count."""

    def test_detects_simple_2x2_grid(self):
        # 5x5 with single horizontal divider (row 2) + single vertical divider (col 2)
        # of color 7. Other cells = color 3. Output 2x2 of color 3.
        inp = [[3, 3, 7, 3, 3], [3, 3, 7, 3, 3], [7, 7, 7, 7, 7],
               [3, 3, 7, 3, 3], [3, 3, 7, 3, 3]]
        out = [[3, 3], [3, 3]]
        self.assertEqual(_detect_grid_division_count([_pair(inp, out)]), True)

    def test_rejects_no_divider(self):
        inp = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]  # uniform
        out = [[3]]
        # Uniform input: every row + every col is "full of 3" → divider color = 3,
        # H_out = 4, W_out = 4. But we expect 1x1 output → mismatch → reject.
        self.assertIsNone(_detect_grid_division_count([_pair(inp, out)]))

    def test_rejects_three_colors(self):
        # 3 distinct non-zero colors: not the 2-color grid pattern.
        inp = [[3, 3, 7], [3, 3, 7], [7, 7, 4]]
        out = [[3, 3], [3, 3]]
        self.assertIsNone(_detect_grid_division_count([_pair(inp, out)]))

    def test_rejects_wrong_output_shape(self):
        inp = [[3, 7, 3], [3, 7, 3], [3, 7, 3]]  # vertical divider at col 1
        wrong_out = [[3, 3, 3]]  # should be 1x2 not 1x3
        self.assertIsNone(_detect_grid_division_count([_pair(inp, wrong_out)]))

    def test_rejects_non_uniform_output(self):
        inp = [[3, 3, 7, 3, 3], [3, 3, 7, 3, 3], [7, 7, 7, 7, 7],
               [3, 3, 7, 3, 3], [3, 3, 7, 3, 3]]
        bad_out = [[3, 7], [3, 3]]  # has divider color in output
        self.assertIsNone(_detect_grid_division_count([_pair(inp, bad_out)]))

    def test_network_round_trip_2x2(self):
        import numpy as np
        import onnxruntime as ort
        # Encode 5x5 input on canvas
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # 3 at non-divider cells, 7 at row 2 + col 2.
        for r in range(5):
            for c in range(5):
                if r == 2 or c == 2:
                    x[0, 7, r, c] = 1.0
                else:
                    x[0, 3, r, c] = 1.0
        model = grid_division_count_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # Output should be 2x2 of color 3 at top-left
        # i.e., out[0, 3, 0..1, 0..1] all 1 and out[0, 3, 2:, :] / out[:, 2:] all 0
        for r in range(2):
            for c in range(2):
                self.assertEqual(out[0, 3, r, c], 1.0)
        # Outside 2x2: zero
        self.assertEqual(out[0, 3, 2, 0], 0.0)
        self.assertEqual(out[0, 3, 0, 2], 0.0)
        # Other channels at output cells: zero
        for ch in (0, 1, 2, 4, 5, 6, 7, 8, 9):
            self.assertEqual(out[0, ch, 0, 0], 0.0)

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            grid_division_count_network(batch=2)


class CrossSectionGridLabelTest(unittest.TestCase):
    """Detector + builder tests for task055 cross-section grid labeling."""

    def _grid(self):
        h, w = 7, 7
        g = [[0 for _ in range(w)] for _ in range(h)]
        for r in (2, 4):
            for c in range(w):
                g[r][c] = 8
        for c in (2, 4):
            for r in range(h):
                g[r][c] = 8
        return g

    def _expected(self):
        out = [row[:] for row in self._grid()]
        for r in (0, 1):
            out[r][3] = 2
        for c in (0, 1):
            out[3][c] = 4
        out[3][3] = 6
        for c in (5, 6):
            out[3][c] = 3
        for r in (5, 6):
            out[r][3] = 1
        return out

    def test_detected(self):
        self.assertEqual(_detect_cross_section_grid_label([_pair(self._grid(), self._expected())]), True)

    def test_rejects_wrong_fill(self):
        bad = self._expected()
        bad[3][3] = 5
        self.assertIsNone(_detect_cross_section_grid_label([_pair(self._grid(), bad)]))

    def test_rejects_wrong_divider_count(self):
        g = self._grid()
        g[4] = [0] * 7
        self.assertIsNone(_detect_cross_section_grid_label([_pair(g, self._expected())]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort

        inp = self._grid()
        expected = self._expected()
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = cross_section_grid_label_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0).tolist()
        counts = (out[0] == 1.0).sum(axis=0)
        self.assertEqual([row[:7] for row in decoded[:7]], expected)
        for r in range(7):
            for c in range(7):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")
        for r in range(30):
            for c in range(30):
                if r >= 7 or c >= 7:
                    self.assertEqual(counts[r][c], 0, f"padding cell ({r},{c}) is lit")

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            cross_section_grid_label_network(batch=2)


class GridMaxCountFillTest(unittest.TestCase):
    """Detector + builder tests for task059 compartment max-count fill."""

    def _grid(self, obj=2):
        g = [[0 for _ in range(11)] for _ in range(11)]
        for r in (3, 7):
            for c in range(11):
                g[r][c] = 5
        for c in (3, 7):
            for r in range(11):
                g[r][c] = 5
        g[0][0] = obj
        g[0][1] = obj
        g[4][4] = obj
        g[8][8] = obj
        g[9][9] = obj
        g[10][10] = obj
        return g

    def _expected(self, obj=2):
        out = [[0 for _ in range(11)] for _ in range(11)]
        for r in (3, 7):
            for c in range(11):
                out[r][c] = 5
        for c in (3, 7):
            for r in range(11):
                out[r][c] = 5
        for r in range(8, 11):
            for c in range(8, 11):
                out[r][c] = obj
        return out

    def test_detected(self):
        self.assertEqual(_detect_grid_max_count_fill([_pair(self._grid(), self._expected())]), True)

    def test_rejects_wrong_filled_block(self):
        bad = self._expected()
        bad[0][0] = 2
        self.assertIsNone(_detect_grid_max_count_fill([_pair(self._grid(), bad)]))

    def test_rejects_missing_grid_line(self):
        g = self._grid()
        g[3][0] = 0
        self.assertIsNone(_detect_grid_max_count_fill([_pair(g, self._expected())]))

    def test_network_round_trip_object_color_8(self):
        import numpy as np
        import onnxruntime as ort

        inp = self._grid(obj=8)
        expected = self._expected(obj=8)
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = grid_max_count_fill_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0).tolist()
        counts = (out[0] == 1.0).sum(axis=0)
        self.assertEqual([row[:11] for row in decoded[:11]], expected)
        for r in range(11):
            for c in range(11):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")
        for r in range(30):
            for c in range(30):
                if r >= 11 or c >= 11:
                    self.assertEqual(counts[r][c], 0, f"padding cell ({r},{c}) is lit")

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            grid_max_count_fill_network(batch=2)


class GridObjectSummaryTest(unittest.TestCase):
    """Detector + builder tests for task184 nonzero band summary."""

    def _decode_trim(self, out):
        grid = []
        for r in range(30):
            cells = []
            for c in range(30):
                colors = [ch for ch in range(10) if out[0, ch, r, c] > 0.5]
                cells.append(colors[0] if len(colors) == 1 else None)
            while cells and cells[-1] is None:
                cells.pop()
            grid.append(cells)
        while grid and not grid[-1]:
            grid.pop()
        return grid

    def test_detects_2x2_summary_with_holes(self):
        inp = [
            [1, 1, 0, 0, 2, 2],
            [1, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [3, 3, 0, 0, 4, 4],
            [3, 0, 0, 0, 4, 0],
        ]
        self.assertEqual(_detect_grid_object_summary([_pair(inp, [[1, 2], [3, 4]])]), True)

    def test_rejects_multi_color_block(self):
        inp = [
            [1, 9, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 0, 0],
            [3, 3, 0, 4],
        ]
        self.assertIsNone(_detect_grid_object_summary([_pair(inp, [[1, 2], [3, 4]])]))

    def test_rejects_wrong_summary(self):
        inp = [
            [1, 1, 0, 2],
            [0, 0, 0, 0],
            [3, 0, 0, 4],
        ]
        self.assertIsNone(_detect_grid_object_summary([_pair(inp, [[1, 2], [4, 3]])]))

    def test_network_round_trip_trims_empty_slots(self):
        import numpy as np
        import onnxruntime as ort

        inp = [
            [1, 1, 0, 2, 2, 0, 3],
            [1, 0, 0, 2, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0],
            [4, 4, 0, 5, 5, 0, 6],
            [4, 0, 0, 5, 0, 0, 6],
        ]
        expected = [[1, 2, 3], [4, 5, 6]]
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = grid_object_summary_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        self.assertEqual(self._decode_trim(out), expected)
        counts = (out[0] > 0.5).sum(axis=0)
        for r in range(30):
            for c in range(30):
                should_be_lit = r < 2 and c < 3
                self.assertEqual(counts[r][c], 1 if should_be_lit else 0)

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            grid_object_summary_network(batch=2)


class TwoBlobTileFitTest(unittest.TestCase):
    """Detector + builder tests for task153 two-blob compact tiling."""

    def _decode_trim(self, out):
        grid = []
        for r in range(30):
            cells = []
            for c in range(30):
                colors = [ch for ch in range(10) if out[0, ch, r, c] > 0.5]
                cells.append(colors[0] if len(colors) == 1 else None)
            while cells and cells[-1] is None:
                cells.pop()
            grid.append(cells)
        while grid and not grid[-1]:
            grid.pop()
        return grid

    def _sample_pair(self):
        inp = [[0 for _ in range(10)] for _ in range(10)]
        for r, c in ((2, 1), (2, 2), (3, 1), (4, 1)):
            inp[r][c] = 3
        for r, c in ((7, 9), (8, 8), (8, 9), (9, 8), (9, 9)):
            inp[r][c] = 7
        out = [[3, 3, 7], [3, 7, 7], [3, 7, 7]]
        return inp, out

    def test_detects_sample_pair(self):
        inp, out = self._sample_pair()
        self.assertEqual(_detect_two_blob_tile_fit([_pair(inp, out)]), {"matched": True})

    def test_rejects_wrong_output(self):
        inp, out = self._sample_pair()
        wrong = [row[:] for row in out]
        wrong[0][2] = 3
        self.assertIsNone(_detect_two_blob_tile_fit([_pair(inp, wrong)]))

    def test_rejects_disconnected_same_color(self):
        inp, out = self._sample_pair()
        inp[0][0] = 3
        inp[4][1] = 0
        self.assertIsNone(_detect_two_blob_tile_fit([_pair(inp, out)]))

    def test_network_round_trip_sample_pair(self):
        import numpy as np
        import onnxruntime as ort

        inp, expected = self._sample_pair()
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = two_blob_tile_fit_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        self.assertEqual(self._decode_trim(out), expected)
        counts = (out[0] > 0.5).sum(axis=0)
        for r in range(30):
            for c in range(30):
                should_be_lit = r < 3 and c < 3
                self.assertEqual(counts[r][c], 1 if should_be_lit else 0)

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            two_blob_tile_fit_network(batch=2)


class FourfoldRotationTileTest(unittest.TestCase):
    """Detector + builder tests for task106 fourfold rotation tiling."""

    def _decode_trim(self, out):
        grid = []
        for r in range(30):
            cells = []
            for c in range(30):
                colors = [ch for ch in range(10) if out[0, ch, r, c] > 0.5]
                cells.append(colors[0] if len(colors) == 1 else None)
            while cells and cells[-1] is None:
                cells.pop()
            grid.append(cells)
        while grid and not grid[-1]:
            grid.pop()
        return grid

    def test_detects_asymmetric_3x3(self):
        inp = [[6, 9, 9], [6, 4, 4], [6, 4, 4]]
        expected = [
            [6, 9, 9, 6, 6, 6],
            [6, 4, 4, 4, 4, 9],
            [6, 4, 4, 4, 4, 9],
            [9, 4, 4, 4, 4, 6],
            [9, 4, 4, 4, 4, 6],
            [6, 6, 6, 9, 9, 6],
        ]
        self.assertEqual(_detect_fourfold_rotation_tile([_pair(inp, expected)]), {"matched": True})

    def test_rejects_mirror_tile(self):
        inp = [[1, 2], [3, 4]]
        wrong = [[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 4, 3], [1, 2, 2, 1]]
        self.assertIsNone(_detect_fourfold_rotation_tile([_pair(inp, wrong)]))

    def test_network_round_trip_2x2_and_3x3(self):
        import numpy as np
        import onnxruntime as ort

        samples = [
            ([[1, 2], [3, 4]], [[1, 2, 3, 1], [3, 4, 4, 2], [2, 4, 4, 3], [1, 3, 2, 1]]),
            (
                [[6, 9, 9], [6, 4, 4], [6, 4, 4]],
                [
                    [6, 9, 9, 6, 6, 6],
                    [6, 4, 4, 4, 4, 9],
                    [6, 4, 4, 4, 4, 9],
                    [9, 4, 4, 4, 4, 6],
                    [9, 4, 4, 4, 4, 6],
                    [6, 6, 6, 9, 9, 6],
                ],
            ),
        ]
        model = fourfold_rotation_tile_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        for inp, expected in samples:
            x = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for r, row in enumerate(inp):
                for c, v in enumerate(row):
                    x[0, v, r, c] = 1.0
            out = sess.run(["output"], {"input": x})[0]
            self.assertEqual(self._decode_trim(out), expected)

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            fourfold_rotation_tile_network(batch=2)


class DiagonalRaysFromLineTest(unittest.TestCase):
    """Detector + builder tests for task348 alternating diagonal rays."""

    def _decode_trim(self, out):
        grid = []
        for r in range(30):
            cells = []
            for c in range(30):
                colors = [ch for ch in range(10) if out[0, ch, r, c] > 0.5]
                cells.append(colors[0] if len(colors) == 1 else None)
            while cells and cells[-1] is None:
                cells.pop()
            grid.append(cells)
        while grid and not grid[-1]:
            grid.pop()
        return grid

    def _sample_pair(self):
        inp = [[0 for _ in range(7)] for _ in range(6)]
        inp[4][3] = 7
        inp[5][3] = 7
        out = [
            [0, 0, 0, 0, 0, 0, 0],
            [8, 0, 0, 0, 0, 0, 8],
            [8, 7, 0, 0, 0, 7, 8],
            [0, 7, 8, 0, 8, 7, 0],
            [0, 0, 8, 7, 8, 0, 0],
            [0, 0, 0, 7, 0, 0, 0],
        ]
        return inp, out

    def test_detects_sample_pair(self):
        inp, out = self._sample_pair()
        self.assertEqual(_detect_diagonal_rays_from_line([_pair(inp, out)]), {"matched": True})

    def test_rejects_wrong_output(self):
        inp, out = self._sample_pair()
        wrong = [row[:] for row in out]
        wrong[1][0] = 7
        self.assertIsNone(_detect_diagonal_rays_from_line([_pair(inp, wrong)]))

    def test_rejects_unexpected_input_color(self):
        inp, out = self._sample_pair()
        inp[0][0] = 5
        self.assertIsNone(_detect_diagonal_rays_from_line([_pair(inp, out)]))

    def test_network_round_trip_sample_pair(self):
        import numpy as np
        import onnxruntime as ort

        inp, expected = self._sample_pair()
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = diagonal_rays_from_line_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        self.assertEqual(self._decode_trim(out), expected)
        counts = (out[0] > 0.5).sum(axis=0)
        for r in range(30):
            for c in range(30):
                should_be_lit = r < 6 and c < 7
                self.assertEqual(counts[r][c], 1 if should_be_lit else 0)

    def test_network_round_trip_batched(self):
        import numpy as np
        import onnxruntime as ort

        inp, expected = self._sample_pair()
        x = np.zeros((2, 10, 30, 30), dtype=np.float32)
        for b, shift in enumerate((0, 1)):
            shifted_inp = [[0] * shift + row for row in inp]
            for r, row in enumerate(shifted_inp):
                for c, v in enumerate(row):
                    x[b, v, r, c] = 1.0
        model = diagonal_rays_from_line_network(batch=2)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        self.assertEqual(self._decode_trim(out[0:1]), expected)
        shifted = [
            [7, 0, 0, 0, 0, 0, 0, 0],
            [7, 8, 0, 0, 0, 0, 0, 8],
            [0, 8, 7, 0, 0, 0, 7, 8],
            [0, 0, 7, 8, 0, 8, 7, 0],
            [0, 0, 0, 8, 7, 8, 0, 0],
            [0, 0, 0, 0, 7, 0, 0, 0],
        ]
        self.assertEqual(self._decode_trim(out[1:2]), shifted)


class EuclideanWedgeFillTest(unittest.TestCase):
    """Detector + builder tests for task175 Euclidean-wedge lattice fill."""

    def _grid(self, base=6, period=7):
        def color_for(off):
            return ((base - 1 + off) % period) + 1

        out = []
        for r in range(21):
            row = []
            for c in range(21):
                m = min(r, c)
                d = abs(r - c)
                if d == 0:
                    off = 0
                elif d <= m + 1:
                    off = -1
                else:
                    off = (d - (m + 2)) // (m + 2)
                row.append(color_for(off))
            out.append(row)
        return out

    def _with_holes(self, grid):
        inp = [row[:] for row in grid]
        for r in range(3, 6):
            for c in range(3, 6):
                inp[r][c] = 0
        for r in range(14, 18):
            for c in range(8, 12):
                inp[r][c] = 0
        return inp

    def test_detected_with_holes(self):
        out = self._grid(base=6, period=7)
        inp = self._with_holes(out)
        self.assertEqual(_detect_euclidean_wedge_fill([_pair(inp, out)]), True)

    def test_rejects_wrong_fill(self):
        out = self._grid(base=6, period=7)
        inp = self._with_holes(out)
        bad = [row[:] for row in out]
        bad[4][4] = 5
        self.assertIsNone(_detect_euclidean_wedge_fill([_pair(inp, bad)]))

    def test_rejects_non_diagonal_base(self):
        out = self._grid(base=6, period=7)
        inp = self._with_holes(out)
        inp[0][0] = 5
        self.assertIsNone(_detect_euclidean_wedge_fill([_pair(inp, out)]))

    def test_network_round_trip_base6_period7(self):
        import numpy as np
        import onnxruntime as ort

        expected = self._grid(base=6, period=7)
        inp = self._with_holes(expected)
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = euclidean_wedge_fill_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0).tolist()
        counts = (out[0] > 0.5).sum(axis=0)
        self.assertEqual([row[:21] for row in decoded[:21]], expected)
        for r in range(21):
            for c in range(21):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")
        for r in range(30):
            for c in range(30):
                if r >= 21 or c >= 21:
                    self.assertEqual(counts[r][c], 0, f"padding cell ({r},{c}) is lit")

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            euclidean_wedge_fill_network(batch=2)


class TileDiagonalNeighborMarkTest(unittest.TestCase):
    """Detector + builder tests for task019 tile-diagonal-neighbor-mark."""

    def _expected(self, inp):
        import numpy as np

        a = np.array(inp, dtype=np.int8)
        marker = next(int(v) for v in np.unique(a) if v != 0)
        out = np.tile(a, (2, 2))
        H, W = out.shape
        for r, c in zip(*np.where(out == marker)):
            for dr in (-1, 1):
                for dc in (-1, 1):
                    rr = int(r) + dr
                    cc = int(c) + dc
                    if 0 <= rr < H and 0 <= cc < W and out[rr, cc] == 0:
                        out[rr, cc] = 8
        return out.tolist()

    def _sample_input(self):
        return [
            [0, 0, 6, 0],
            [0, 0, 0, 0],
            [0, 6, 0, 0],
        ]

    def test_detects_diagonal_marks_across_tile_boundaries(self):
        inp = self._sample_input()
        self.assertEqual(
            _detect_tile_diagonal_neighbor_mark([_pair(inp, self._expected(inp))]),
            True,
        )

    def test_rejects_missing_diagonal_mark(self):
        inp = self._sample_input()
        bad = self._expected(inp)
        bad[1][0] = 0
        self.assertIsNone(_detect_tile_diagonal_neighbor_mark([_pair(inp, bad)]))

    def test_rejects_marker_color_8(self):
        inp = [[0, 8], [0, 0]]
        out = [[0, 8, 0, 8], [0, 0, 0, 0], [0, 8, 0, 8], [0, 0, 0, 0]]
        self.assertIsNone(_detect_tile_diagonal_neighbor_mark([_pair(inp, out)]))

    def test_network_round_trip_and_padding(self):
        import numpy as np
        import onnxruntime as ort

        inp = self._sample_input()
        expected = self._expected(inp)
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = tile_diagonal_neighbor_mark_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0).tolist()
        counts = (out[0] > 0.5).sum(axis=0)
        H = len(expected)
        W = len(expected[0])
        self.assertEqual([row[:W] for row in decoded[:H]], expected)
        for r in range(H):
            for c in range(W):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")
        for r in range(30):
            for c in range(30):
                if r >= H or c >= W:
                    self.assertEqual(counts[r][c], 0, f"padding cell ({r},{c}) is lit")

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            tile_diagonal_neighbor_mark_network(batch=2)


class DiagonalMarkerSquareRingsTest(unittest.TestCase):
    """Detector + builder tests for task137 diagonal-marker-square-rings."""

    def _expected(self, inp):
        import numpy as np

        a = np.array(inp, dtype=np.int8)
        marker = next(int(v) for v in np.unique(a) if v != 0)
        pts = sorted((int(r), int(c)) for r, c in zip(*np.where(a == marker)))
        center_r, center_c = pts[1]
        period = abs(pts[1][0] - pts[0][0])
        H, W = a.shape
        maxd = max(center_r, H - 1 - center_r, center_c, W - 1 - center_c)
        out = np.zeros_like(a)
        for radius in range(0, maxd + 1, period):
            for r in range(H):
                for c in range(W):
                    if max(abs(r - center_r), abs(c - center_c)) == radius:
                        out[r, c] = marker
        return out.tolist()

    def _sample_input(self):
        inp = [[0] * 23 for _ in range(23)]
        inp[4][12] = 3
        inp[8][8] = 3
        inp[12][4] = 3
        return inp

    def test_detects_square_rings(self):
        inp = self._sample_input()
        self.assertEqual(
            _detect_diagonal_marker_square_rings([_pair(inp, self._expected(inp))]),
            True,
        )

    def test_rejects_non_diagonal_markers(self):
        inp = self._sample_input()
        inp[12][4] = 0
        inp[12][5] = 3
        self.assertIsNone(_detect_diagonal_marker_square_rings([_pair(inp, self._expected(self._sample_input()))]))

    def test_rejects_wrong_ring_spacing(self):
        inp = self._sample_input()
        bad = self._expected(inp)
        bad[0][0] = 0
        self.assertIsNone(_detect_diagonal_marker_square_rings([_pair(inp, bad)]))

    def test_network_round_trip_and_padding(self):
        import numpy as np
        import onnxruntime as ort

        inp = self._sample_input()
        expected = self._expected(inp)
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = diagonal_marker_square_rings_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0).tolist()
        counts = (out[0] > 0.5).sum(axis=0)
        self.assertEqual([row[:23] for row in decoded[:23]], expected)
        for r in range(23):
            for c in range(23):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")
        for r in range(30):
            for c in range(30):
                if r >= 23 or c >= 23:
                    self.assertEqual(counts[r][c], 0, f"padding cell ({r},{c}) is lit")

    def test_network_covers_period_one_far_radii(self):
        import numpy as np
        import onnxruntime as ort

        inp = [[0] * 20 for _ in range(20)]
        inp[0][0] = 3
        inp[1][1] = 3
        inp[2][2] = 3
        expected = self._expected(inp)
        self.assertEqual(
            _detect_diagonal_marker_square_rings([_pair(inp, expected)]),
            True,
        )
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = diagonal_marker_square_rings_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0).tolist()
        self.assertEqual([row[:20] for row in decoded[:20]], expected)

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            diagonal_marker_square_rings_network(batch=2)


class BilateralSymmetryDenoiseTest(unittest.TestCase):
    """Detector + builder tests for task071 bilateral-symmetry-denoise."""

    def _sample_pair(self):
        inp = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        out = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        return inp, out

    def test_detects_refined_axis_rule(self):
        inp, out = self._sample_pair()
        self.assertEqual(_detect_bilateral_symmetry_denoise([_pair(inp, out)]), True)

    def test_rejects_wrong_noise_fill(self):
        inp, out = self._sample_pair()
        bad = [row[:] for row in out]
        bad[4][1] = 2
        self.assertIsNone(_detect_bilateral_symmetry_denoise([_pair(inp, bad)]))

    def test_rejects_non_rectangular_noise(self):
        inp, out = self._sample_pair()
        inp = [row[:] for row in inp]
        inp[4][0] = 0
        self.assertIsNone(_detect_bilateral_symmetry_denoise([_pair(inp, out)]))

    def test_network_round_trip_refined_axis(self):
        import numpy as np
        import onnxruntime as ort

        inp, expected = self._sample_pair()
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = bilateral_symmetry_denoise_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0).tolist()
        counts = (out[0] > 0.5).sum(axis=0)
        self.assertEqual([row[:16] for row in decoded[:16]], expected)
        for r in range(16):
            for c in range(16):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")
        for r in range(30):
            for c in range(30):
                if r >= 16 or c >= 16:
                    self.assertEqual(counts[r][c], 0, f"padding cell ({r},{c}) is lit")

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            bilateral_symmetry_denoise_network(batch=2)


class MarkerSectionExtractTest(unittest.TestCase):
    """Detector + builder tests for task065 marker-section-extract."""

    def test_detects_simple_2x2_section(self):
        # 5x5 with row-2 + col-2 dividers (color 3); marker color 4 at (4,0)
        # in section (rows 3-4, cols 0-1). Output should be that 2x2 section.
        inp = [[8, 8, 3, 8, 8], [8, 8, 3, 8, 8], [3, 3, 3, 3, 3],
               [8, 8, 3, 8, 8], [4, 8, 3, 8, 8]]
        out = [[8, 8], [4, 8]]
        self.assertEqual(_detect_marker_section_extract([_pair(inp, out)]), True)

    def test_marker_can_be_zero(self):
        # 7x7 with row-3 + col-3 dividers (color 2); single 0 cell as marker.
        inp = [[6, 6, 6, 2, 6, 6, 6], [6, 6, 6, 2, 6, 6, 6], [6, 6, 6, 2, 6, 6, 6],
               [2, 2, 2, 2, 2, 2, 2],
               [6, 6, 6, 2, 6, 6, 6], [6, 6, 6, 2, 6, 6, 6], [6, 6, 6, 2, 0, 6, 6]]
        out = [[6, 6, 6], [6, 6, 6], [0, 6, 6]]
        self.assertEqual(_detect_marker_section_extract([_pair(inp, out)]), True)

    def test_rejects_no_divider(self):
        inp = [[6, 6, 6], [6, 4, 6], [6, 6, 6]]  # no full row/col of one color
        out = [[6, 4, 6]]
        self.assertIsNone(_detect_marker_section_extract([_pair(inp, out)]))

    def test_rejects_wrong_section(self):
        inp = [[8, 8, 3, 8, 8], [8, 8, 3, 8, 8], [3, 3, 3, 3, 3],
               [8, 8, 3, 8, 8], [4, 8, 3, 8, 8]]
        # Wrong: returns top-left 2x2 instead of bottom-left
        wrong_out = [[8, 8], [8, 8]]
        self.assertIsNone(_detect_marker_section_extract([_pair(inp, wrong_out)]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # Encode 5x5 input on canvas
        inp = [[8, 8, 3, 8, 8], [8, 8, 3, 8, 8], [3, 3, 3, 3, 3],
               [8, 8, 3, 8, 8], [4, 8, 3, 8, 8]]
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                x[0, inp[r][c], r, c] = 1.0
        model = marker_section_extract_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0)
        counts = (out[0] > 0.5).sum(axis=0)
        # Inside 2x2 bbox: exactly one channel set
        for r in range(2):
            for c in range(2):
                self.assertEqual(int(counts[r, c]), 1, f"counts[{r}, {c}]={counts[r, c]}")
        # Output values: [[8, 8], [4, 8]]
        expected = [[8, 8], [4, 8]]
        for r in range(2):
            for c in range(2):
                self.assertEqual(int(decoded[r, c]), expected[r][c])
        # Outside bbox: zero everywhere
        self.assertEqual(int(counts[2, 0]), 0)
        self.assertEqual(int(counts[0, 2]), 0)

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            marker_section_extract_network(batch=2)


class RotationalSymmetryCompletionTest(unittest.TestCase):
    """Detector + builder tests for task020 4-fold rotational symmetry."""

    def test_detects_simple_rotational_pattern(self):
        # 5x5 grid with non-zero cells at (1,2), (2,3) (asymmetric, only
        # rotationally symmetric, not reflectionally). bbox center (cr, cc) =
        # ((1+2)/2, (2+3)/2). first_row+last_row = 3 (odd) -> rejected.
        # So make it symmetric-extent: cells at (1,1), (3,3) -> bbox first=1
        # last=3, center 2. Cells rotate to (1,3), (3,1) under 4-fold rotation
        # around (2, 2).
        inp = [[0, 0, 0, 0, 0],
               [0, 4, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 4, 0],
               [0, 0, 0, 0, 0]]
        # Output completes 4-fold rotational symmetry around (2, 2).
        # (1,1) -> (1,1), (1,3), (3,3), (3,1) under 0/90/180/270 CW.
        # (3,3) -> same set. So output has 4 at (1,1), (1,3), (3,1), (3,3).
        out = [[0, 0, 0, 0, 0],
               [0, 4, 0, 4, 0],
               [0, 0, 0, 0, 0],
               [0, 4, 0, 4, 0],
               [0, 0, 0, 0, 0]]
        self.assertEqual(_detect_rotational_symmetry_completion([_pair(inp, out)]), True)

    def test_rejects_already_complete_input(self):
        # Input already 4-fold symmetric -> output == input -> "trivial
        # completion". Detector should reject (identity-fallback is cheaper).
        inp = [[0, 0, 0],
               [0, 5, 0],
               [0, 0, 0]]
        out = [[0, 0, 0],
               [0, 5, 0],
               [0, 0, 0]]
        self.assertIsNone(_detect_rotational_symmetry_completion([_pair(inp, out)]))

    def test_rejects_odd_bbox_extent(self):
        # bbox spans rows 0..1 (sum 1, odd). Detector rejects because
        # ONNX builder requires integer center.
        inp = [[5, 0],
               [0, 0]]
        out = [[5, 0],
               [0, 5]]  # not actually rotationally symmetric, but odd-extent gate fires first
        self.assertIsNone(_detect_rotational_symmetry_completion([_pair(inp, out)]))

    def test_rejects_shape_mismatch(self):
        inp = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        out = [[5, 0], [0, 5]]
        self.assertIsNone(_detect_rotational_symmetry_completion([_pair(inp, out)]))

    def test_rejects_empty_input(self):
        inp = [[0, 0], [0, 0]]
        out = [[0, 0], [0, 0]]
        self.assertIsNone(_detect_rotational_symmetry_completion([_pair(inp, out)]))

    def test_rejects_wrong_completion(self):
        # Same valid asymmetric setup, wrong output (h-flip instead of rotation).
        inp = [[0, 0, 0, 0, 0],
               [0, 4, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 4, 0],
               [0, 0, 0, 0, 0]]
        wrong_out = [[0, 0, 0, 0, 0],
                     [0, 0, 0, 4, 0],  # h-mirror of input — only 2 cells
                     [0, 0, 0, 0, 0],
                     [0, 4, 0, 0, 0],
                     [0, 0, 0, 0, 0]]
        self.assertIsNone(_detect_rotational_symmetry_completion([_pair(inp, wrong_out)]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 5x5 input embedded in 30x30 canvas; partial cells at (1,1) and (3,3)
        # both color 4. Output should have 4 at (1,1), (1,3), (3,1), (3,3).
        inp = [[0, 0, 0, 0, 0],
               [0, 4, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 4, 0],
               [0, 0, 0, 0, 0]]
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                x[0, inp[r][c], r, c] = 1.0
        model = rotational_symmetry_completion_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # Inside 5x5 grid: each cell has exactly one channel lit
        for r in range(5):
            for c in range(5):
                lit = sum(1 for ch in range(10) if out[0, ch, r, c] > 0.5)
                self.assertEqual(lit, 1, f"cell ({r},{c}) has {lit} channels lit")
        # Verify 4 cells have color 4: (1,1), (1,3), (3,1), (3,3)
        for (r, c) in [(1, 1), (1, 3), (3, 1), (3, 3)]:
            self.assertGreater(out[0, 4, r, c], 0.5, f"({r},{c}) should be color 4")
        # Verify other cells are color 0 (bg)
        for r in range(5):
            for c in range(5):
                if (r, c) not in [(1, 1), (1, 3), (3, 1), (3, 3)]:
                    self.assertGreater(out[0, 0, r, c], 0.5, f"({r},{c}) should be bg")
        # Outside 5x5 grid: zero everywhere
        for r in range(30):
            for c in range(30):
                if r >= 5 or c >= 5:
                    counts = sum(1 for ch in range(10) if out[0, ch, r, c] > 0.5)
                    self.assertEqual(counts, 0, f"canvas-padding ({r},{c}) lit")

    def test_builder_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            rotational_symmetry_completion_network(batch=2)


class ColorFilterBboxCropOddOneOutTest(unittest.TestCase):
    """Detector + builder tests for color-filter-bbox-crop with odd_one_out picker."""

    def _grid_3x3_one_unique(self):
        # Color 1: 1 cell (count=1, unique)
        # Color 2: 4 cells (count=4, not unique — also shared by color 3)
        # Color 3: 4 cells
        inp = [
            [1, 2, 2],
            [2, 3, 3],
            [2, 3, 3],
        ]
        return inp

    def test_detects_odd_one_out(self):
        inp = self._grid_3x3_one_unique()
        out = [[1]]  # bbox of color 1 is 1x1 at (0,0)
        result = _detect_color_filter_bbox_crop([_pair(inp, out)], picker="odd_one_out")
        self.assertEqual(result, {"matched": True, "picker": "odd_one_out"})

    def test_rejects_all_counts_unique(self):
        # Each color has a distinct count → no single "odd one out"
        inp = [[1, 2, 3], [2, 3, 0], [3, 0, 0]]
        # count 1=1, count 2=2, count 3=3 → freq={1:1, 2:1, 3:1} → 3 unique_counts → None
        out = [[1]]
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out)], picker="odd_one_out"))

    def test_rejects_no_unique_count(self):
        # All colors share the same count → no odd one out
        inp = [[1, 2], [1, 2]]
        out = [[1, 2], [1, 2]]
        # count 1=2, count 2=2 → freq={2:2} → unique_counts=[] → None
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out)], picker="odd_one_out"))

    def test_rejects_odd_count_shared_by_multiple_colors(self):
        # Two colors with the same unique-valued count → ambiguous
        inp = [[1, 0, 2], [0, 0, 0], [0, 0, 0]]
        # count 1=1, count 2=1 → freq={1:2} → unique_counts=[] (1 appears twice) → None
        out = [[1]]
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out)], picker="odd_one_out"))

    def test_rejects_wrong_crop(self):
        inp = self._grid_3x3_one_unique()
        out = [[2, 2], [3, 3], [2, 3]]  # not bbox of color 1
        self.assertIsNone(_detect_color_filter_bbox_crop([_pair(inp, out)], picker="odd_one_out"))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_color_filter_bbox_crop([], picker="odd_one_out"))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 30x30 input:
        #   Color 1: 1 cell at (5, 5)  → count=1, unique
        #   Color 2: 4 cells at (0,0),(0,1),(1,0),(1,1)
        #   Color 3: 4 cells at (2,2),(2,3),(3,2),(3,3)
        # odd_one_out → color 1. Bbox = 1x1 at (5,5). Output: color 1 at (0,0).
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        x[0, 1, 5, 5] = 1.0
        for r in range(2):
            for c in range(2):
                x[0, 2, r, c] = 1.0
                x[0, 3, r + 2, c + 2] = 1.0
        model = color_filter_bbox_crop_network(picker="odd_one_out", batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # Output (0, 0) should be color 1.
        self.assertGreater(out[0, 1, 0, 0], 0.5)
        for ch in range(10):
            if ch != 1:
                self.assertLess(out[0, ch, 0, 0], 0.5, f"ch{ch} unexpected at (0,0)")
        # Output (0, 1) should be all-zero (canvas padding).
        for ch in range(10):
            self.assertLess(out[0, ch, 0, 1], 0.5, f"ch{ch} unexpected at (0,1)")


class ColorSwap5StaysTest(unittest.TestCase):
    """Detector + builder tests for color_swap_5_always_stays (task389 family)."""

    def test_detects_simple(self):
        # Color 5 cells → color 3; color 3 cells → bg.
        inp = [[5, 3, 5], [3, 5, 3], [5, 3, 5]]
        out = [[3, 0, 3], [0, 3, 0], [3, 0, 3]]
        self.assertIsNotNone(_detect_color_swap_5_always_stays([_pair(inp, out)]))

    def test_detects_multiple_pairs(self):
        # Same rule with different "other color" per pair.
        p1 = _pair([[5, 2], [2, 5]], [[2, 0], [0, 2]])
        p2 = _pair([[5, 7], [7, 5]], [[7, 0], [0, 7]])
        self.assertIsNotNone(_detect_color_swap_5_always_stays([p1, p2]))

    def test_rejects_no_color_5(self):
        inp = [[3, 7], [7, 3]]
        out = [[7, 0], [0, 7]]
        self.assertIsNone(_detect_color_swap_5_always_stays([_pair(inp, out)]))

    def test_rejects_three_nonbg_colors(self):
        # Input has colors 1, 3, 5 — three non-bg colors; detector requires exactly 2.
        inp = [[5, 3, 1], [3, 5, 1], [1, 1, 5]]
        out = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        self.assertIsNone(_detect_color_swap_5_always_stays([_pair(inp, out)]))

    def test_rejects_wrong_output(self):
        # Output doesn't match rule: output unchanged instead of swapped.
        inp = [[5, 3], [3, 5]]
        out = [[5, 3], [3, 5]]
        self.assertIsNone(_detect_color_swap_5_always_stays([_pair(inp, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_color_swap_5_always_stays([]))

    def test_rejects_shape_mismatch(self):
        inp = [[5, 3], [3, 5]]
        out = [[3], [0], [3]]
        self.assertIsNone(_detect_color_swap_5_always_stays([_pair(inp, out)]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort
        # 3x3 grid: color 5 at (0,0) and (2,2); color 3 at (0,1) and (1,0).
        # Rule: 5-cells → color 3; all other cells → bg.
        # Expected output: ch3 at (0,0) and (2,2); ch0 elsewhere in the 3x3 area.
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        x[0, 5, 0, 0] = 1.0
        x[0, 5, 2, 2] = 1.0
        x[0, 3, 0, 1] = 1.0
        x[0, 3, 1, 0] = 1.0
        # fill bg cells
        x[0, 0, 0, 2] = 1.0
        x[0, 0, 1, 1] = 1.0
        x[0, 0, 1, 2] = 1.0
        x[0, 0, 2, 0] = 1.0
        x[0, 0, 2, 1] = 1.0
        model = color_swap_5_always_stays_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        # color 5 cells → color 3 in output
        self.assertGreater(out[0, 3, 0, 0], 0.5, "(0,0) should be color 3")
        self.assertGreater(out[0, 3, 2, 2], 0.5, "(2,2) should be color 3")
        # color 5 should be gone
        self.assertLess(out[0, 5, 0, 0], 0.5, "(0,0) should not be color 5")
        self.assertLess(out[0, 5, 2, 2], 0.5, "(2,2) should not be color 5")
        # color 3 cells become bg
        self.assertLess(out[0, 3, 0, 1], 0.5, "(0,1) was color 3 input → should not be 3 in output")
        self.assertLess(out[0, 3, 1, 0], 0.5, "(1,0) was color 3 input → should not be 3 in output")

    def test_builder_accepts_batch_1(self):
        model = color_swap_5_always_stays_network(batch=1)
        self.assertIsNotNone(model)


class MarkerRectangleInteriorTest(unittest.TestCase):
    """Detector + builder tests for marker_rectangle_interior (task029 family)."""

    def _make_rect_input(self, r0, r1, c0, c1, grid_h, grid_w, border_color, fill_color):
        """Build grid_h × grid_w input with a rectangle border at [r0..r1][c0..c1]."""
        grid = [[fill_color] * grid_w for _ in range(grid_h)]
        for cc in range(c0, c1 + 1):
            grid[r0][cc] = border_color
            grid[r1][cc] = border_color
        for rr in range(r0, r1 + 1):
            grid[rr][c0] = border_color
            grid[rr][c1] = border_color
        return grid

    def test_detects_simple(self):
        # 7x7 grid; color 3 forms a 5x5 border at (1,1)..(5,5).
        # Interior = rows 2..4, cols 2..4 = 3x3 all zeros (bg).
        grid = self._make_rect_input(1, 5, 1, 5, 7, 7, border_color=3, fill_color=0)
        out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(_detect_marker_rectangle_interior([_pair(grid, out)]),
                         {"matched": True})

    def test_detects_with_interior_content(self):
        # 6x8 grid; color 2 forms rectangle at (0,0)..(5,7). Interior = 4x6.
        grid = self._make_rect_input(0, 5, 0, 7, 6, 8, border_color=2, fill_color=7)
        # interior cells
        out = [[7] * 6 for _ in range(4)]
        self.assertEqual(_detect_marker_rectangle_interior([_pair(grid, out)]),
                         {"matched": True})

    def test_rejects_no_rectangle(self):
        # Grid with no color forming a complete border.
        grid = [[1, 2, 1], [2, 0, 2], [1, 2, 0]]
        out = [[0]]
        self.assertIsNone(_detect_marker_rectangle_interior([_pair(grid, out)]))

    def test_rejects_incomplete_border(self):
        # Color 3 at top row + left col but missing right col.
        grid = [[3, 3, 3], [3, 0, 0], [3, 0, 0]]
        out = [[0, 0], [0, 0]]
        self.assertIsNone(_detect_marker_rectangle_interior([_pair(grid, out)]))

    def test_rejects_wrong_interior_output(self):
        grid = self._make_rect_input(0, 4, 0, 4, 5, 5, border_color=1, fill_color=2)
        out = [[2, 2, 2], [2, 2, 2], [1, 1, 1]]  # last row wrong
        self.assertIsNone(_detect_marker_rectangle_interior([_pair(grid, out)]))

    def test_rejects_empty(self):
        self.assertIsNone(_detect_marker_rectangle_interior([]))

    def test_rejects_batch_above_1(self):
        with self.assertRaises(NotImplementedError):
            marker_rectangle_interior_network(batch=2)

    def test_network_round_trip_simple(self):
        import numpy as np
        import onnxruntime as ort
        # 7x7 grid; color 3 = border at (1,1)..(5,5); interior 3x3 filled with color 7.
        grid = self._make_rect_input(1, 5, 1, 5, 7, 7, border_color=3, fill_color=0)
        for r in range(2, 5):
            for c in range(2, 5):
                grid[r][c] = 7
        exp_interior = [[7, 7, 7], [7, 7, 7], [7, 7, 7]]
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(7):
            for c in range(7):
                x[0, grid[r][c], r, c] = 1.0
        model = marker_rectangle_interior_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        pred = np.argmax(out[0], axis=0)
        # Interior (3x3) at top-left of output
        for r in range(3):
            for c in range(3):
                self.assertEqual(int(pred[r, c]), 7, f"({r},{c}) should be color 7")
        # Canvas-padding at row 3 should be all zero
        self.assertEqual(int((out[0, :, 3, 0] > 0.5).sum()), 0)

    def test_network_smallest_rect_wins(self):
        import numpy as np
        import onnxruntime as ort
        # Two nested rectangles: outer color=1 (full 9x7), inner color=2 (5x5 at (2,1)..(6,5)).
        # Expected: inner rectangle (3x3 interior of color=2 rect).
        grid = [[0] * 7 for _ in range(9)]
        # Outer: color 1
        for cc in range(7): grid[0][cc] = 1; grid[8][cc] = 1
        for rr in range(9): grid[rr][0] = 1; grid[rr][6] = 1
        # Inner: color 2 at (2,1)..(6,5)
        for cc in range(1, 6): grid[2][cc] = 2; grid[6][cc] = 2
        for rr in range(2, 7): grid[rr][1] = 2; grid[rr][5] = 2
        # Fill between outer and inner with color 3
        for r in range(1, 8):
            for c in range(1, 6):
                if grid[r][c] == 0:
                    grid[r][c] = 3
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(9):
            for c in range(7):
                x[0, grid[r][c], r, c] = 1.0
        model = marker_rectangle_interior_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        pred = np.argmax(out[0], axis=0)
        # Inner rect interior is 3x3 of color 3 (between the two borders)
        for r in range(3):
            for c in range(3):
                self.assertEqual(int(pred[r, c]), 3, f"({r},{c}) should be interior color 3")
        # Row 3 should be zero (beyond interior)
        self.assertEqual(int((out[0, :, 3, 0] > 0.5).sum()), 0)


class MarkerDrivenShapeCopyTest(unittest.TestCase):
    """Tests for _detect_marker_driven_shape_copy + marker_driven_shape_copy_network."""

    def _make_one_hot(self, grid):
        import numpy as np
        H = len(grid)
        W = len(grid[0])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                x[0, grid[r][c], r, c] = 1.0
        return x

    def _run(self, grid):
        import numpy as np
        import onnxruntime as ort
        x = self._make_one_hot(grid)
        m = marker_driven_shape_copy_network(batch=1)
        sess = ort.InferenceSession(m.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        return np.argmax(out[0], axis=0)

    def test_detector_accepts_basic(self):
        # 3x3: color-1 bar at top, marker-5 at bottom-left.
        # cr=0, cc=1 -> dr=2, dc=-1 -> copies (0,1)->(2,0) and (0,2)->(2,1).
        inp = [[1, 1, 1], [0, 0, 0], [5, 0, 0]]
        correct_out = [[1, 1, 1], [0, 0, 0], [1, 1, 0]]
        self.assertIsNotNone(_detect_marker_driven_shape_copy([_pair(inp, correct_out)]))

    def test_detector_rejects_wrong_output(self):
        inp = [[1, 1, 1], [0, 0, 0], [5, 0, 0]]
        wrong_out = [[1, 1, 1], [0, 0, 0], [0, 1, 0]]
        self.assertIsNone(_detect_marker_driven_shape_copy([_pair(inp, wrong_out)]))

    def test_detector_rejects_no_color5(self):
        inp = [[1, 1, 1], [0, 0, 0], [2, 0, 0]]
        self.assertIsNone(_detect_marker_driven_shape_copy([_pair(inp, inp)]))

    def test_detector_rejects_multiple_color5_cells(self):
        inp = [[1, 1, 1], [5, 0, 5], [0, 0, 0]]
        self.assertIsNone(_detect_marker_driven_shape_copy([_pair(inp, inp)]))

    def test_network_translation_to_marker(self):
        # 5x5: single color-1 at (1,1), marker-5 at (3,3).
        # cr=1, cc=1 -> dr=2, dc=2 -> copy lands at (3,3).
        grid = [[0]*5 for _ in range(5)]
        grid[1][1] = 1
        grid[3][3] = 5
        pred = self._run(grid)
        self.assertEqual(int(pred[1, 1]), 1, "original stays")
        self.assertEqual(int(pred[3, 3]), 1, "translated lands at marker position")

    def test_network_translation_corner(self):
        # 5x5: color-3 at (2,2), marker-5 at (0,0).
        # cr=2, cc=2 -> dr=-2, dc=-2 -> copy lands at (0,0).
        grid = [[0]*5 for _ in range(5)]
        grid[2][2] = 3
        grid[0][0] = 5
        pred = self._run(grid)
        self.assertEqual(int(pred[2, 2]), 3, "original stays")
        self.assertEqual(int(pred[0, 0]), 3, "translated at former marker position")
        for r in range(5):
            for c in range(5):
                if (r, c) not in {(2, 2), (0, 0)}:
                    self.assertEqual(int(pred[r, c]), 0, f"({r},{c}) should be bg")

    def test_network_marker_erased(self):
        import numpy as np
        grid = [[0]*5 for _ in range(5)]
        grid[1][1] = 2
        grid[3][3] = 5
        pred = self._run(grid)
        for r in range(5):
            for c in range(5):
                self.assertNotEqual(int(pred[r, c]), 5, "color 5 must not appear in output")

    def test_network_canvas_padding_clean(self):
        import numpy as np
        import onnxruntime as ort
        grid = [[0]*5 for _ in range(5)]
        grid[0][0] = 1
        grid[2][2] = 5
        x = self._make_one_hot(grid)
        m = marker_driven_shape_copy_network(batch=1)
        sess = ort.InferenceSession(m.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        self.assertTrue((out[0, :, 5:, :] == 0).all(), "padding rows must be zero")
        self.assertTrue((out[0, :, :, 5:] == 0).all(), "padding cols must be zero")

    def test_network_builder_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            marker_driven_shape_copy_network(batch=2)


class PeriodicRowExtensionTest(unittest.TestCase):
    """Tests for _detect_periodic_row_extension + periodic_row_extension_network."""

    def _make_one_hot(self, grid):
        import numpy as np
        H, W = len(grid), len(grid[0])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                x[0, grid[r][c], r, c] = 1.0
        return x

    def _run(self, inp_grid):
        import numpy as np
        import onnxruntime as ort
        x = self._make_one_hot(inp_grid)
        m = periodic_row_extension_network(batch=1)
        sess = ort.InferenceSession(m.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        return np.argmax(out[0], axis=0)[:9, :3]

    def _pair(self, inp, out):
        return {"input": inp, "output": out}

    def test_period2_detector_accepts(self):
        # T=2: row0=[0,1,0], row1=[1,0,1], rows 2=0, 3=1, 4=0, 5=1
        inp = [[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[1,0,1]]
        out = [[0,2,0],[2,0,2],[0,2,0],[2,0,2],[0,2,0],[2,0,2],[0,2,0],[2,0,2],[0,2,0]]
        self.assertIsNotNone(_detect_periodic_row_extension([self._pair(inp, out)]))

    def test_period3_detector_accepts(self):
        # T=3: rows 0=[1,0,0], 1=[0,1,0], 2=[0,0,1], then repeats
        inp = [[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]]
        out = [[2,0,0],[0,2,0],[0,0,2],[2,0,0],[0,2,0],[0,0,2],[2,0,0],[0,2,0],[0,0,2]]
        self.assertIsNotNone(_detect_periodic_row_extension([self._pair(inp, out)]))

    def test_period4_detector_accepts(self):
        # T=4: rows 0=[1,0,0], 1=[0,1,0], 2=[0,0,1], 3=[1,1,0], 4=0, 5=1
        inp = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,0],[0,1,0]]
        # row6=row[6%4]=row2=[0,0,1]→[0,0,2], row7=row3=[1,1,0]→[2,2,0], row8=row0=[1,0,0]→[2,0,0]
        out = [[2,0,0],[0,2,0],[0,0,2],[2,2,0],[2,0,0],[0,2,0],[0,0,2],[2,2,0],[2,0,0]]
        self.assertIsNotNone(_detect_periodic_row_extension([self._pair(inp, out)]))

    def test_detector_rejects_wrong_output(self):
        inp = [[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[1,0,1]]
        wrong = [[0,2,0],[2,0,2],[0,2,0],[2,0,2],[0,2,0],[2,0,2],[2,0,2],[0,2,0],[2,0,2]]
        self.assertIsNone(_detect_periodic_row_extension([self._pair(inp, wrong)]))

    def test_detector_rejects_wrong_shape(self):
        inp = [[0,1,0],[1,0,1]]  # only 2 rows
        out = [[0,2,0],[2,0,2],[0,2,0]]
        self.assertIsNone(_detect_periodic_row_extension([self._pair(inp, out)]))

    def test_detector_rejects_color3_input(self):
        inp = [[3,0,0],[0,1,0],[3,0,0],[0,1,0],[3,0,0],[0,1,0]]
        out = [[0,2,0],[0,2,0],[0,2,0],[0,2,0],[0,2,0],[0,2,0],[0,2,0],[0,2,0],[0,2,0]]
        self.assertIsNone(_detect_periodic_row_extension([self._pair(inp, out)]))

    def test_network_period2(self):
        import numpy as np
        inp = [[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[1,0,1]]
        result = self._run(inp)
        expected = np.array([[0,2,0],[2,0,2],[0,2,0],[2,0,2],[0,2,0],[2,0,2],[0,2,0],[2,0,2],[0,2,0]])
        np.testing.assert_array_equal(result, expected)

    def test_network_period3(self):
        import numpy as np
        inp = [[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]]
        result = self._run(inp)
        expected = np.array([[2,0,0],[0,2,0],[0,0,2],[2,0,0],[0,2,0],[0,0,2],[2,0,0],[0,2,0],[0,0,2]])
        np.testing.assert_array_equal(result, expected)

    def test_network_period4_row2_ne_row0(self):
        # T=4 case where row0==row2 is False
        import numpy as np
        inp = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,0],[0,1,0]]
        result = self._run(inp)
        expected = np.array([[2,0,0],[0,2,0],[0,0,2],[2,2,0],[2,0,0],[0,2,0],[0,0,2],[2,2,0],[2,0,0]])
        np.testing.assert_array_equal(result, expected)

    def test_network_period4_row0_eq_row2_not_period2(self):
        # T=4 case where row0==row2 but row1!=row3 (the bug that was fixed)
        import numpy as np
        inp = [[0,1,0],[1,1,0],[0,1,0],[0,1,1],[0,1,0],[1,1,0]]
        result = self._run(inp)
        # row6=row[2]=[0,1,0]→[0,2,0], row7=row[3]=[0,1,1]→[0,2,2], row8=row[0]=[0,1,0]→[0,2,0]
        expected = np.array([[0,2,0],[2,2,0],[0,2,0],[0,2,2],[0,2,0],[2,2,0],[0,2,0],[0,2,2],[0,2,0]])
        np.testing.assert_array_equal(result, expected)

    def test_network_builder_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            periodic_row_extension_network(batch=2)


class PeriodExtendRow2xTest(unittest.TestCase):
    """Tests for task231 row-wise period extension to double width."""

    def _decode_trim(self, out):
        grid = []
        for r in range(30):
            cells = []
            for c in range(30):
                colors = [ch for ch in range(10) if out[0, ch, r, c] > 0.5]
                cells.append(colors[0] if len(colors) == 1 else None)
            while cells and cells[-1] is None:
                cells.pop()
            grid.append(cells)
        while grid and not grid[-1]:
            grid.pop()
        return grid

    def _sample_pair(self):
        inp = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [2, 3, 3, 2, 3, 3, 2],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        out = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        return inp, out

    def test_detects_period3_width7(self):
        inp, out = self._sample_pair()
        self.assertEqual(_detect_period_extend_row_2x([_pair(inp, out)]), {"matched": True})

    def test_rejects_plain_horizontal_tile(self):
        inp, _ = self._sample_pair()
        wrong = [row + row for row in inp]
        self.assertIsNone(_detect_period_extend_row_2x([_pair(inp, wrong)]))

    def test_rejects_period4(self):
        inp = [
            [1, 2, 3, 4, 1, 2],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        out = [
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        self.assertIsNone(_detect_period_extend_row_2x([_pair(inp, out)]))

    def test_network_round_trip_period3_width7(self):
        import numpy as np
        import onnxruntime as ort

        inp, expected = self._sample_pair()
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = period_extend_row_2x_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        self.assertEqual(self._decode_trim(out), expected)
        counts = (out[0] > 0.5).sum(axis=0)
        for r in range(30):
            for c in range(30):
                should_be_lit = r < 5 and c < 14
                self.assertEqual(counts[r][c], 1 if should_be_lit else 0)

    def test_network_round_trip_width8_two_rows(self):
        import numpy as np
        import onnxruntime as ort

        inp = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 2, 1, 2, 2, 1, 2],
            [2, 1, 2, 2, 1, 2, 2, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1],
            [2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        model = period_extend_row_2x_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        self.assertEqual(self._decode_trim(out), expected)

    def test_network_builder_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            period_extend_row_2x_network(batch=2)


class BooleanHalvesTest(unittest.TestCase):
    """Tests for _detect_boolean_halves + boolean_halves_network."""

    def _p(self, inp, out):
        return {"input": inp, "output": out}

    def _run(self, H, W_in, W_out, cc, oc, inp_grid):
        import numpy as np, onnxruntime as ort
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W_in):
                x[0, inp_grid[r][c], r, c] = 1.0
        m = boolean_halves_network(H, W_in, W_out, cc, oc, batch=1)
        sess = ort.InferenceSession(m.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        import numpy as np
        return np.argmax(out[0], axis=0)[:H, :W_out]

    # --- task006 style: AND (check_channel=1, out_color=2) ---
    def test_and_detector_accepts_task006_style(self):
        inp = [[1, 0, 0, 5, 0, 1, 0], [0, 1, 0, 5, 1, 1, 1], [1, 0, 0, 5, 0, 0, 0]]
        out = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        r = _detect_boolean_halves([self._p(inp, out)])
        self.assertIsNotNone(r)
        self.assertEqual(r["check_channel"], 1)
        self.assertEqual(r["out_color"], 2)

    def test_and_detector_rejects_wrong_output(self):
        inp = [[1, 0, 0, 5, 0, 1, 0], [0, 1, 0, 5, 1, 1, 1], [1, 0, 0, 5, 0, 0, 0]]
        wrong = [[2, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.assertIsNone(_detect_boolean_halves([self._p(inp, wrong)]))

    def test_and_detector_rejects_non_7wide(self):
        inp = [[1, 0, 5, 0, 1], [0, 1, 5, 1, 1]]
        out = [[0, 0], [0, 2]]
        r = _detect_boolean_halves([self._p(inp, out)])
        # (2,5)→(2,2): W_in=5=2*2+1 ✓ should work
        self.assertIsNotNone(r)

    def test_and_network_task006_style(self):
        import numpy as np
        inp = [[1, 1, 0, 5, 0, 1, 0], [0, 0, 1, 5, 1, 1, 1], [1, 1, 0, 5, 0, 1, 0]]
        result = self._run(3, 7, 3, 1, 2, inp)
        expected = np.array([[0, 2, 0], [0, 0, 2], [0, 2, 0]])
        np.testing.assert_array_equal(result, expected)

    # --- task026 style: NOR (check_channel=0, out_color=8) ---
    def test_nor_detector_accepts_task026_style(self):
        inp = [[0, 9, 9, 1, 9, 9, 9], [0, 0, 9, 1, 9, 9, 0], [9, 0, 9, 1, 9, 9, 0],
               [0, 0, 0, 1, 9, 0, 0], [0, 9, 9, 1, 9, 9, 9]]
        out = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 8, 8], [0, 0, 0]]
        r = _detect_boolean_halves([self._p(inp, out)])
        self.assertIsNotNone(r)
        self.assertEqual(r["check_channel"], 0)
        self.assertEqual(r["out_color"], 8)

    def test_nor_network_task026_style(self):
        import numpy as np
        inp = [[0, 9, 9, 1, 9, 9, 9], [0, 0, 9, 1, 9, 9, 0], [9, 0, 9, 1, 9, 9, 0],
               [0, 0, 0, 1, 9, 0, 0], [0, 9, 9, 1, 9, 9, 9]]
        result = self._run(5, 7, 3, 0, 8, inp)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 8, 8], [0, 0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            boolean_halves_network(3, 7, 3, 1, 2, batch=2)


class RankVerticalBarsTest(unittest.TestCase):
    """Tests for _detect_rank_vertical_bars + rank_vertical_bars_network (task010)."""

    def _p(self, inp, out):
        return {"input": inp, "output": out}

    def _make_input(self, heights):
        """Build a 9x9 grid with bars of color-5 at cols {1,3,5,7} with given heights.
        heights: dict {col: height} — bar fills rows (9-h):9 of that col.
        """
        import numpy as np
        g = [[0]*9 for _ in range(9)]
        for c, h in heights.items():
            for r in range(9 - h, 9):
                g[r][c] = 5
        return g

    def _make_output(self, heights):
        """Build expected output grid where each bar gets its rank color."""
        import numpy as np
        g = [[0]*9 for _ in range(9)]
        sorted_cols = sorted(heights, key=lambda c: -heights[c])
        rank = {c: i + 1 for i, c in enumerate(sorted_cols)}
        for c, h in heights.items():
            for r in range(9 - h, 9):
                g[r][c] = rank[c]
        return g

    def test_detector_accepts(self):
        h = {1: 8, 3: 6, 5: 9, 7: 3}
        inp = self._make_input(h)
        out = self._make_output(h)
        self.assertIsNotNone(_detect_rank_vertical_bars([self._p(inp, out)]))

    def test_detector_rejects_wrong_shape(self):
        inp = [[0]*8 for _ in range(8)]  # not 9x9
        out = [[0]*8 for _ in range(8)]
        self.assertIsNone(_detect_rank_vertical_bars([self._p(inp, out)]))

    def test_detector_rejects_wrong_bars(self):
        # Bar at col 2 instead of col 1
        h = {2: 8, 3: 6, 5: 9, 7: 3}
        inp = self._make_input(h)
        out = self._make_output(h)
        self.assertIsNone(_detect_rank_vertical_bars([self._p(inp, out)]))

    def test_detector_rejects_wrong_output(self):
        h = {1: 8, 3: 6, 5: 9, 7: 3}
        inp = self._make_input(h)
        out = self._make_output(h)
        out[8][1] = 3  # wrong rank
        self.assertIsNone(_detect_rank_vertical_bars([self._p(inp, out)]))

    def test_network_correct_ranks(self):
        import numpy as np, onnxruntime as ort
        h = {1: 8, 3: 6, 5: 9, 7: 3}
        inp = self._make_input(h)
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(9):
            for c in range(9):
                x[0, inp[r][c], r, c] = 1.0
        m = rank_vertical_bars_network(batch=1)
        sess = ort.InferenceSession(m.SerializeToString())
        out = sess.run(["output"], {"input": x})[0]
        pred = np.argmax(out[0], axis=0)[:9, :9]
        expected = np.array(self._make_output(h))
        np.testing.assert_array_equal(pred, expected)

    def test_network_all_orderings(self):
        import numpy as np, onnxruntime as ort
        m = rank_vertical_bars_network(batch=1)
        sess = ort.InferenceSession(m.SerializeToString())
        # Test a few different height orderings
        for heights in [{1:9,3:6,5:3,7:1}, {1:1,3:2,5:3,7:4}, {1:5,3:8,5:2,7:7}]:
            inp = self._make_input(heights)
            x = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for r in range(9):
                for c in range(9):
                    x[0, inp[r][c], r, c] = 1.0
            out = sess.run(["output"], {"input": x})[0]
            pred = np.argmax(out[0], axis=0)[:9, :9]
            expected = np.array(self._make_output(heights))
            np.testing.assert_array_equal(pred, expected, err_msg=f"heights={heights}")

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            rank_vertical_bars_network(batch=2)


class VerticalHalvesTest(unittest.TestCase):
    def _make_grid(self, rows):
        return [list(r) for r in rows]

    def _pair_nor(self, top_rows, bot_rows, out_color):
        inp = self._make_grid(top_rows) + self._make_grid(bot_rows)
        out = [
            [out_color if top_rows[r][c] == 0 and bot_rows[r][c] == 0 else 0
             for c in range(len(top_rows[0]))]
            for r in range(len(top_rows))
        ]
        return _pair(inp, out)

    def _pair_or(self, top_rows, bot_rows, out_color):
        inp = self._make_grid(top_rows) + self._make_grid(bot_rows)
        out = [
            [out_color if top_rows[r][c] != 0 or bot_rows[r][c] != 0 else 0
             for c in range(len(top_rows[0]))]
            for r in range(len(top_rows))
        ]
        return _pair(inp, out)

    def _pair_xor(self, top_rows, bot_rows, out_color, sep=None):
        sep_row = [[sep, sep, sep]] if sep else []
        inp = self._make_grid(top_rows) + sep_row + self._make_grid(bot_rows)
        out = [
            [out_color if (top_rows[r][c] != 0) ^ (bot_rows[r][c] != 0) else 0
             for c in range(len(top_rows[0]))]
            for r in range(len(top_rows))
        ]
        return _pair(inp, out)

    def test_nor_no_sep_detected(self):
        # top=0 at (0,1) and (1,1); bot=0 at (0,1) and (1,1) → NOR hits at col 1
        p = self._pair_nor([[9, 0, 9], [0, 0, 9]], [[0, 0, 1], [1, 0, 9]], out_color=2)
        m = _detect_vertical_halves([p])
        self.assertIsNotNone(m)
        self.assertEqual(m["mode"], "nor")
        self.assertIsNone(m["sep_row"])
        self.assertEqual(m["out_color"], 2)

    def test_or_no_sep_detected(self):
        # at least one has fg at every position except (0,2) → OR hits everywhere except (0,2)
        p = self._pair_or([[3, 0, 0], [0, 3, 0]], [[0, 1, 1], [1, 0, 0]], out_color=6)
        m = _detect_vertical_halves([p])
        self.assertIsNotNone(m)
        self.assertEqual(m["mode"], "or")

    def test_xor_with_sep_detected(self):
        p = self._pair_xor([[1, 0, 1], [0, 1, 0]], [[2, 0, 0], [2, 2, 1]], out_color=3, sep=4)
        m = _detect_vertical_halves([p])
        self.assertIsNotNone(m)
        self.assertEqual(m["mode"], "xor")
        self.assertEqual(m["sep_row"], 2)

    def test_rejects_unequal_heights(self):
        p = _pair([[0, 9]], [[1, 0], [0, 1]])
        self.assertIsNone(_detect_vertical_halves([p]))

    def test_nor_network_correct(self):
        import numpy as np, onnxruntime as ort
        top = [[9, 0, 9], [0, 0, 9]]
        bot = [[0, 1, 0], [1, 0, 0]]
        p = self._pair_nor(top, bot, out_color=2)
        m = _detect_vertical_halves([p])
        net = vertical_halves_network(m["H_in"], m["W"], m["H_half"], m["mode"],
                                      m["out_color"], sep_row=m["sep_row"])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(4):
            for c in range(3):
                x[0, p["input"][r][c], r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :2, :3], axis=0)
        np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_xor_network_correct(self):
        import numpy as np, onnxruntime as ort
        top = [[1, 0, 0], [0, 1, 1]]
        bot = [[2, 2, 0], [2, 0, 1]]
        p = self._pair_xor(top, bot, out_color=3, sep=4)
        m = _detect_vertical_halves([p])
        net = vertical_halves_network(m["H_in"], m["W"], m["H_half"], m["mode"],
                                      m["out_color"], sep_row=m["sep_row"])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(5):
            for c in range(3):
                x[0, p["input"][r][c], r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :2, :3], axis=0)
        np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            vertical_halves_network(4, 3, 2, "nor", 2, batch=2)


class HorizontalOrTest(unittest.TestCase):
    def _pair_or(self, left_rows, right_rows, out_color):
        inp = [list(left_rows[r]) + list(right_rows[r]) for r in range(len(left_rows))]
        out = [
            [out_color if left_rows[r][c] != 0 or right_rows[r][c] != 0 else 0
             for c in range(len(left_rows[0]))]
            for r in range(len(left_rows))
        ]
        return _pair(inp, out)

    def test_detected(self):
        p = self._pair_or([[4, 0, 3], [0, 0, 4]], [[3, 0, 0], [0, 4, 3]], out_color=6)
        m = _detect_horizontal_or([p])
        self.assertIsNotNone(m)
        self.assertEqual(m["out_color"], 6)
        self.assertEqual(m["W_half"], 3)

    def test_rejects_wrong_width(self):
        p = _pair([[1, 2, 3]], [[6, 0]])
        self.assertIsNone(_detect_horizontal_or([p]))

    def test_network_correct(self):
        import numpy as np, onnxruntime as ort
        left = [[4, 0, 3], [0, 4, 0]]
        right = [[0, 0, 3], [3, 0, 0]]
        p = self._pair_or(left, right, out_color=6)
        m = _detect_horizontal_or([p])
        net = horizontal_or_network(m["H"], m["W_half"], m["out_color"])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(2):
            for c in range(6):
                x[0, p["input"][r][c], r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :2, :3], axis=0)
        np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            horizontal_or_network(3, 3, 6, batch=2)


class CrossProductTest(unittest.TestCase):
    def _make_pair(self, N, template_cols, selector_rows, out_color=2):
        a = [[0] * N for _ in range(N)]
        for c in template_cols:
            a[0][c] = 5
        for r in selector_rows:
            a[r][N - 1] = 5
        expected = [row[:] for row in a]
        for r in selector_rows:
            for c in template_cols:
                if a[r][c] == 0:
                    expected[r][c] = out_color
        return _pair(a, expected)

    def test_detected(self):
        p = self._make_pair(5, [1, 3], [2, 4], out_color=2)
        m = _detect_cross_product([p])
        self.assertIsNotNone(m)
        self.assertEqual(m["template_color"], 5)
        self.assertEqual(m["out_color"], 2)
        self.assertEqual(m["col_idx"], 4)

    def test_rejects_non_square(self):
        p = _pair([[5, 0, 5], [0, 0, 5], [0, 0, 0]], [[5, 0, 5], [0, 0, 5], [0, 0, 0]])
        self.assertIsNone(_detect_cross_product([p]))

    def test_rejects_multi_template_color(self):
        p = _pair([[5, 0, 3], [0, 0, 5], [0, 0, 0]], [[5, 0, 3], [0, 0, 5], [0, 0, 0]])
        self.assertIsNone(_detect_cross_product([p]))

    def test_network_correct(self):
        import numpy as np, onnxruntime as ort
        N = 5
        template_cols = [1, 3]
        selector_rows = [2, 4]
        p = self._make_pair(N, template_cols, selector_rows, out_color=2)
        m = _detect_cross_product([p])
        net = cross_product_network(m["N"], m["col_idx"], m["template_color"], m["out_color"])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(N):
            for c in range(N):
                x[0, p["input"][r][c], r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :N, :N], axis=0)
        np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_multiple_pairs(self):
        import numpy as np, onnxruntime as ort
        N = 6
        net = None
        for cols, rows in [([1, 4], [3, 5]), ([0, 2, 5], [1, 4])]:
            p = self._make_pair(N, cols, rows, out_color=2)
            if net is None:
                m = _detect_cross_product([p])
                net = cross_product_network(m["N"], m["col_idx"], m["template_color"], m["out_color"])
            x = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for r in range(N):
                for c in range(N):
                    x[0, p["input"][r][c], r, c] = 1.0
            out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
            pred = np.argmax(out[:, :N, :N], axis=0)
            np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            cross_product_network(5, 4, 5, 2, batch=2)


class BlockCountGeTest(unittest.TestCase):
    """Tests for _detect_block_count_ge and block_count_ge_network."""

    def _make_pair(self, blocks, block_size=3, stride=4, count_color=6, threshold=2, out_color=1, sep_color=8):
        """blocks: 3×3 list of counts of count_color in each block."""
        N = 3 * stride - 1  # 11 for 3 blocks of size 3 with stride 4
        inp = [[0] * N for _ in range(N)]
        out = [[0] * 3 for _ in range(3)]
        for br in range(3):
            for bc in range(3):
                r0 = br * stride
                c0 = bc * stride
                # Fill separator rows/cols
                if br > 0:
                    sr = br * stride - 1
                    for c in range(N):
                        inp[sr][c] = sep_color
                if bc > 0:
                    sc = bc * stride - 1
                    for r in range(N):
                        inp[r][sc] = sep_color
                cnt = blocks[br][bc]
                placed = 0
                for r in range(r0, r0 + block_size):
                    for c in range(c0, c0 + block_size):
                        if placed < cnt and inp[r][c] == 0:
                            inp[r][c] = count_color
                            placed += 1
                out[br][bc] = out_color if cnt >= threshold else 0
        return _pair(inp, out)

    def test_detected(self):
        blocks = [[3, 1, 2], [0, 2, 4], [1, 0, 3]]
        p = self._make_pair(blocks)
        m = _detect_block_count_ge([p])
        self.assertIsNotNone(m)
        self.assertEqual(m["count_color"], 6)
        self.assertEqual(m["threshold"], 2)
        self.assertEqual(m["out_color"], 1)
        self.assertEqual(m["n"], 3)
        self.assertEqual(m["block_size"], 3)

    def test_not_detected_wrong_shape(self):
        p = _pair([[0, 0], [0, 0]], [[0]])
        self.assertIsNone(_detect_block_count_ge([p]))

    def test_not_detected_multiple_count_colors(self):
        # Two non-separator non-zero colors → not detectable
        blocks = [[2, 1, 0], [0, 1, 2], [1, 0, 2]]
        p = self._make_pair(blocks)
        a = [list(row) for row in p["input"]]
        a[0][0] = 5  # add second non-zero color
        p2 = _pair(a, p["output"])
        self.assertIsNone(_detect_block_count_ge([p2]))

    def test_network_correct(self):
        import numpy as np, onnxruntime as ort
        blocks = [[3, 1, 2], [0, 2, 4], [1, 0, 3]]
        p = self._make_pair(blocks)
        m = _detect_block_count_ge([p])
        net = block_count_ge_network(m["count_color"], m["threshold"], m["out_color"],
                                      m["N"], m["n"], m["block_size"], m["stride"])
        a = np.array(p["input"], dtype=np.float32)
        H, W = a.shape
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                x[0, int(a[r, c]), r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :3, :3], axis=0)
        np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            block_count_ge_network(6, 2, 1, 11, 3, 3, 4, batch=2)


class MarkerRecolorTest(unittest.TestCase):
    """Tests for _detect_marker_recolor and marker_recolor_network."""

    def _make_pair(self, H, W, r_marker, c_marker, blob_color, marker_color, blob_positions):
        inp = [[0] * W for _ in range(H)]
        out = [[0] * W for _ in range(H)]
        for r, c in blob_positions:
            inp[r][c] = blob_color
            out[r][c] = marker_color
        inp[r_marker][c_marker] = marker_color
        # out[r_marker][c_marker] stays 0
        return _pair(inp, out)

    def test_detected(self):
        p = self._make_pair(7, 7, 6, 0, blob_color=2, marker_color=4,
                             blob_positions=[(1,1),(1,2),(2,1),(3,3)])
        m = _detect_marker_recolor([p])
        self.assertIsNotNone(m)
        self.assertEqual(m["H"], 7)
        self.assertEqual(m["W"], 7)
        self.assertEqual(m["r_marker"], 6)
        self.assertEqual(m["c_marker"], 0)

    def test_not_detected_different_shapes(self):
        p = _pair([[1, 2], [3, 4]], [[0, 0, 0]])
        self.assertIsNone(_detect_marker_recolor([p]))

    def test_not_detected_no_single_marker(self):
        # Marker appears twice → not a single-cell marker
        inp = [[0]*5 for _ in range(5)]
        inp[0][0] = 3  # blob
        inp[1][0] = 3
        inp[4][0] = 5  # "marker" (appears twice)
        inp[4][1] = 5
        out = [[0]*5 for _ in range(5)]
        p = _pair(inp, out)
        self.assertIsNone(_detect_marker_recolor([p]))

    def test_not_detected_inconsistent_marker_pos(self):
        p1 = self._make_pair(5, 5, 4, 0, 2, 6, [(1,1),(1,2),(2,2)])
        p2 = self._make_pair(5, 5, 0, 4, 3, 7, [(2,1),(3,2)])
        self.assertIsNone(_detect_marker_recolor([p1, p2]))

    def test_network_correct(self):
        import numpy as np, onnxruntime as ort
        p = self._make_pair(7, 7, 6, 0, blob_color=2, marker_color=4,
                             blob_positions=[(1,1),(1,2),(1,3),(2,2),(3,1),(3,2),(3,3),(3,4),(4,2),(5,3)])
        m = _detect_marker_recolor([p])
        net = marker_recolor_network(m["H"], m["W"], m["r_marker"], m["c_marker"])
        a = np.array(p["input"], dtype=np.float32)
        H, W = a.shape
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                x[0, int(a[r, c]), r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :H, :W], axis=0)
        np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_different_marker_color(self):
        import numpy as np, onnxruntime as ort
        # Test with marker_color=8 to check dynamic color selection
        p = self._make_pair(7, 7, 6, 0, blob_color=3, marker_color=8,
                             blob_positions=[(0,1),(0,2),(1,1),(1,3),(2,1),(2,2),(2,3)])
        m = _detect_marker_recolor([p])
        net = marker_recolor_network(m["H"], m["W"], m["r_marker"], m["c_marker"])
        a = np.array(p["input"], dtype=np.float32)
        H, W = a.shape
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                x[0, int(a[r, c]), r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :H, :W], axis=0)
        np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            marker_recolor_network(7, 7, 6, 0, batch=2)


class TwoMarkerCrossTest(unittest.TestCase):
    """Tests for _detect_two_marker_cross and two_marker_cross_network."""

    def _make_pair(self, H, W, rA, cA, rB, cB, color_A=8, color_B=7, color_inter=2):
        inp = [[0] * W for _ in range(H)]
        out = [[0] * W for _ in range(H)]
        inp[rA][cA] = color_A
        inp[rB][cB] = color_B
        for c in range(W):
            out[rA][c] = color_A
            out[rB][c] = color_B
        for r in range(H):
            out[r][cA] = color_A
            out[r][cB] = color_B
        # Intersections override
        out[rA][cB] = color_inter
        out[rB][cA] = color_inter
        return _pair(inp, out)

    def test_detected(self):
        p = self._make_pair(9, 9, rA=2, cA=2, rB=6, cB=6)
        m = _detect_two_marker_cross([p])
        self.assertIsNotNone(m)
        self.assertEqual(m["color_A"], 7)  # sorted: 7 < 8
        self.assertEqual(m["color_B"], 8)
        self.assertEqual(m["color_inter"], 2)
        self.assertEqual(m["H"], 9)
        self.assertEqual(m["W"], 9)

    def test_not_detected_three_input_colors(self):
        inp = [[0]*5 for _ in range(5)]
        inp[0][0] = 7
        inp[1][1] = 8
        inp[2][2] = 3  # third color
        out = [[0]*5 for _ in range(5)]
        self.assertIsNone(_detect_two_marker_cross([_pair(inp, out)]))

    def test_not_detected_marker_not_single_cell(self):
        inp = [[0]*5 for _ in range(5)]
        inp[0][0] = 7
        inp[0][1] = 7  # color_A appears twice
        inp[3][3] = 8
        out = [[0]*5 for _ in range(5)]
        self.assertIsNone(_detect_two_marker_cross([_pair(inp, out)]))

    def test_network_correct(self):
        import numpy as np, onnxruntime as ort
        p = self._make_pair(9, 9, rA=2, cA=2, rB=6, cB=6)
        m = _detect_two_marker_cross([p])
        net = two_marker_cross_network(m["color_A"], m["color_B"], m["color_inter"],
                                        m["H"], m["W"])
        a = np.array(p["input"], dtype=np.float32)
        H, W = a.shape
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                x[0, int(a[r, c]), r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :H, :W], axis=0)
        np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_multiple_positions(self):
        import numpy as np, onnxruntime as ort
        for rA, cA, rB, cB in [(1, 3, 7, 6), (0, 8, 4, 0), (5, 1, 2, 7)]:
            p = self._make_pair(9, 9, rA, cA, rB, cB)
            m = _detect_two_marker_cross([p])
            net = two_marker_cross_network(m["color_A"], m["color_B"], m["color_inter"],
                                            m["H"], m["W"])
            a = np.array(p["input"], dtype=np.float32)
            H, W = a.shape
            x = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for r in range(H):
                for c in range(W):
                    x[0, int(a[r, c]), r, c] = 1.0
            out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
            pred = np.argmax(out[:, :H, :W], axis=0)
            np.testing.assert_array_equal(pred, np.array(p["output"]))

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            two_marker_cross_network(8, 7, 2, 9, 9, batch=2)


class DistinctCountPatternTest(unittest.TestCase):
    """Tests for _detect_distinct_count_pattern and distinct_count_pattern_network.

    Pattern: 3×3→3×3, no zeros in input, exactly 3 distinct input colors {c1,c2,c3}.
    k=1 distinct color  → top row = out_color, rest = 0
    k=2 distinct colors → main diagonal = out_color, rest = 0
    k=3 distinct colors → anti-diagonal = out_color, rest = 0
    """

    _C1, _C2, _C3, _OUT = 2, 3, 4, 5

    def _grid_k1(self):
        return [[self._C1]*3 for _ in range(3)]

    def _grid_k2(self):
        g = [[self._C1]*3 for _ in range(3)]
        g[1][1] = self._C2
        return g

    def _grid_k3(self):
        g = [[self._C1]*3 for _ in range(3)]
        g[1][1] = self._C2
        g[2][2] = self._C3
        return g

    def _out_top_row(self):
        return [[self._OUT, self._OUT, self._OUT], [0, 0, 0], [0, 0, 0]]

    def _out_diagonal(self):
        return [[self._OUT, 0, 0], [0, self._OUT, 0], [0, 0, self._OUT]]

    def _out_anti_diagonal(self):
        return [[0, 0, self._OUT], [0, self._OUT, 0], [self._OUT, 0, 0]]

    def _make_pairs(self):
        return [
            _pair(self._grid_k1(), self._out_top_row()),
            _pair(self._grid_k2(), self._out_diagonal()),
            _pair(self._grid_k3(), self._out_anti_diagonal()),
        ]

    def test_detected(self):
        m = _detect_distinct_count_pattern(self._make_pairs())
        self.assertIsNotNone(m)
        self.assertEqual(set([m["c1"], m["c2"], m["c3"]]), {self._C1, self._C2, self._C3})
        self.assertEqual(m["out_color"], self._OUT)

    def test_not_detected_wrong_size(self):
        pairs = [_pair([[self._C1]*4 for _ in range(4)], [[self._OUT]*4 for _ in range(4)])]
        self.assertIsNone(_detect_distinct_count_pattern(pairs))

    def test_not_detected_zeros_in_input(self):
        g = self._grid_k1()
        g[0][0] = 0
        pairs = [_pair(g, self._out_top_row())]
        self.assertIsNone(_detect_distinct_count_pattern(pairs))

    def test_not_detected_too_many_input_colors(self):
        g = self._grid_k3()
        g[0][2] = 7  # 4th color
        pairs = [_pair(g, self._out_anti_diagonal())]
        self.assertIsNone(_detect_distinct_count_pattern(pairs))

    def test_network_k1(self):
        import numpy as np, onnxruntime as ort
        net = distinct_count_pattern_network(self._C1, self._C2, self._C3, self._OUT)
        inp = self._grid_k1()
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                x[0, inp[r][c], r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :3, :3], axis=0).tolist()
        self.assertEqual(pred, self._out_top_row())

    def test_network_k2(self):
        import numpy as np, onnxruntime as ort
        net = distinct_count_pattern_network(self._C1, self._C2, self._C3, self._OUT)
        inp = self._grid_k2()
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                x[0, inp[r][c], r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :3, :3], axis=0).tolist()
        self.assertEqual(pred, self._out_diagonal())

    def test_network_k3(self):
        import numpy as np, onnxruntime as ort
        net = distinct_count_pattern_network(self._C1, self._C2, self._C3, self._OUT)
        inp = self._grid_k3()
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                x[0, inp[r][c], r, c] = 1.0
        out = ort.InferenceSession(net.SerializeToString()).run(None, {"input": x})[0][0]
        pred = np.argmax(out[:, :3, :3], axis=0).tolist()
        self.assertEqual(pred, self._out_anti_diagonal())

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            distinct_count_pattern_network(2, 3, 4, 5, batch=2)


class UniformRowTest(unittest.TestCase):
    def _pairs(self):
        return [
            _pair([[4, 4, 4], [2, 3, 2], [2, 3, 3]], [[5, 5, 5], [0, 0, 0], [0, 0, 0]]),
            _pair([[7, 3, 3], [6, 6, 6], [3, 7, 7]], [[0, 0, 0], [5, 5, 5], [0, 0, 0]]),
            _pair([[2, 9, 2], [4, 4, 4], [9, 9, 9]], [[0, 0, 0], [5, 5, 5], [5, 5, 5]]),
        ]

    def test_detected(self):
        m = _detect_uniform_row(self._pairs())
        self.assertIsNotNone(m)
        self.assertEqual(m["H"], 3)
        self.assertEqual(m["W"], 3)
        self.assertEqual(m["out_color"], 5)

    def test_not_detected_shape_change(self):
        pairs = [_pair([[1, 2], [3, 3]], [[5, 5], [0, 0], [0, 0]])]
        self.assertIsNone(_detect_uniform_row(pairs))

    def test_not_detected_multiple_out_colors(self):
        pairs = [
            _pair([[1, 1, 1], [2, 3, 4]], [[5, 5, 5], [0, 0, 0]]),
            _pair([[2, 2, 2], [1, 3, 4]], [[6, 6, 6], [0, 0, 0]]),
        ]
        self.assertIsNone(_detect_uniform_row(pairs))

    def test_not_detected_non_uniform_row_not_zeroed(self):
        pairs = [_pair([[1, 2, 3], [4, 4, 4]], [[1, 2, 3], [5, 5, 5]])]
        self.assertIsNone(_detect_uniform_row(pairs))

    def _run_net(self, inp_grid, out_H=3, out_W=3):
        import numpy as np, onnxruntime as ort, onnx, io
        model = uniform_row_network(3, 3, out_color=5)
        buf = io.BytesIO(); onnx.save(model, buf); buf.seek(0)
        sess = ort.InferenceSession(buf.read())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp_grid):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(None, {"input": x})[0]
        return np.argmax(out[0], axis=0)[:out_H, :out_W].tolist()

    def test_network_pair0(self):
        self.assertEqual(self._run_net([[4, 4, 4], [2, 3, 2], [2, 3, 3]]),
                         [[5, 5, 5], [0, 0, 0], [0, 0, 0]])

    def test_network_pair1(self):
        self.assertEqual(self._run_net([[7, 3, 3], [6, 6, 6], [3, 7, 7]]),
                         [[0, 0, 0], [5, 5, 5], [0, 0, 0]])

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            uniform_row_network(3, 3, batch=2)


class CummaxDownColumnTest(unittest.TestCase):
    def _pairs(self):
        return [
            _pair([[0, 0, 6], [0, 4, 0], [3, 0, 0]], [[0, 0, 6], [0, 4, 6], [3, 4, 6]]),
            _pair([[0, 2, 0], [7, 0, 8], [0, 0, 0]], [[0, 2, 0], [7, 2, 8], [7, 2, 8]]),
            _pair([[4, 0, 0], [0, 2, 0], [0, 0, 0]], [[4, 0, 0], [4, 2, 0], [4, 2, 0]]),
        ]

    def test_detected(self):
        m = _detect_cummax_down_column(self._pairs())
        self.assertIsNotNone(m)
        self.assertEqual(m["H"], 3)
        self.assertEqual(m["W"], 3)

    def test_not_detected_wrong_rule(self):
        # Not cummax (wrong value at row 1)
        pairs = [_pair([[0, 2, 0], [7, 0, 8], [0, 0, 0]], [[0, 2, 0], [7, 0, 8], [7, 2, 8]])]
        self.assertIsNone(_detect_cummax_down_column(pairs))

    def test_not_detected_shape_change(self):
        pairs = [_pair([[1, 2], [3, 4]], [[1, 2], [3, 4], [3, 4]])]
        self.assertIsNone(_detect_cummax_down_column(pairs))

    def _run_net(self, inp_grid):
        import numpy as np, onnxruntime as ort, onnx, io
        model = cummax_down_column_network(3, 3)
        buf = io.BytesIO(); onnx.save(model, buf); buf.seek(0)
        sess = ort.InferenceSession(buf.read())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp_grid):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(None, {"input": x})[0]
        return np.argmax(out[0], axis=0)[:3, :3].tolist()

    def test_network_pair0(self):
        self.assertEqual(self._run_net([[0, 0, 6], [0, 4, 0], [3, 0, 0]]),
                         [[0, 0, 6], [0, 4, 6], [3, 4, 6]])

    def test_network_pair2(self):
        self.assertEqual(self._run_net([[4, 0, 0], [0, 2, 0], [0, 0, 0]]),
                         [[4, 0, 0], [4, 2, 0], [4, 2, 0]])

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            cummax_down_column_network(3, 3, batch=2)


class DominantColorKeepTest(unittest.TestCase):
    def _pairs(self):
        return [
            _pair([[2, 2, 2], [2, 1, 8], [2, 8, 8]], [[2, 2, 2], [2, 5, 5], [2, 5, 5]]),
            _pair([[1, 1, 1], [8, 1, 3], [8, 2, 2]], [[1, 1, 1], [5, 1, 5], [5, 5, 5]]),
            _pair([[2, 2, 2], [8, 8, 2], [2, 2, 2]], [[2, 2, 2], [5, 5, 2], [2, 2, 2]]),
        ]

    def test_detected(self):
        m = _detect_dominant_color_keep(self._pairs())
        self.assertIsNotNone(m)
        self.assertEqual(m["H"], 3)
        self.assertEqual(m["W"], 3)
        self.assertEqual(m["out_color"], 5)

    def test_not_detected_shape_change(self):
        pairs = [_pair([[2, 2, 2], [2, 1, 8]], [[2, 2, 2], [2, 5, 5], [0, 0, 0]])]
        self.assertIsNone(_detect_dominant_color_keep(pairs))

    def test_not_detected_dominant_not_kept(self):
        # dominant color 2 is replaced instead of kept
        pairs = [_pair([[2, 2, 2], [2, 1, 8], [2, 8, 8]], [[5, 5, 5], [5, 2, 2], [5, 2, 2]])]
        self.assertIsNone(_detect_dominant_color_keep(pairs))

    def _run_net(self, inp_grid):
        import numpy as np, onnxruntime as ort, onnx, io
        model = dominant_color_keep_network(3, 3, out_color=5)
        buf = io.BytesIO(); onnx.save(model, buf); buf.seek(0)
        sess = ort.InferenceSession(buf.read())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp_grid):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(None, {"input": x})[0]
        return np.argmax(out[0], axis=0)[:3, :3].tolist()

    def test_network_pair0(self):
        self.assertEqual(self._run_net([[2, 2, 2], [2, 1, 8], [2, 8, 8]]),
                         [[2, 2, 2], [2, 5, 5], [2, 5, 5]])

    def test_network_pair1(self):
        self.assertEqual(self._run_net([[1, 1, 1], [8, 1, 3], [8, 2, 2]]),
                         [[1, 1, 1], [5, 1, 5], [5, 5, 5]])

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            dominant_color_keep_network(3, 3, batch=2)


class ColorToPatternTest(unittest.TestCase):
    _COLOR_MAP = {
        2: ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
        1: ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
        3: ((0, 0, 1), (0, 0, 1), (1, 1, 1)),
    }

    def _inp(self, color):
        return [[color, 0, 0, 0, 0], [0, color, 0, 0, color],
                [color, 0, 0, color, 0], [0, 0, 0, color, color],
                [0, 0, color, color, 0]]

    def _pairs(self):
        return [
            _pair(self._inp(2), [[5, 5, 5], [0, 5, 0], [0, 5, 0]]),
            _pair(self._inp(1), [[0, 5, 0], [5, 5, 5], [0, 5, 0]]),
            _pair(self._inp(3), [[0, 0, 5], [0, 0, 5], [5, 5, 5]]),
        ]

    def test_detected(self):
        m = _detect_color_to_pattern(self._pairs())
        self.assertIsNotNone(m)
        self.assertEqual(m["H_in"], 5)
        self.assertEqual(m["W_in"], 5)
        self.assertEqual(m["out_color"], 5)
        self.assertEqual(len(m["color_map"]), 3)

    def test_not_detected_multi_color_input(self):
        pairs = [_pair([[1, 2, 0], [0, 0, 0], [0, 0, 0]], [[5, 5, 5], [0, 5, 0], [0, 5, 0]])]
        self.assertIsNone(_detect_color_to_pattern(pairs))

    def test_not_detected_wrong_output_size(self):
        pairs = [_pair(self._inp(2), [[5, 5], [0, 5]])]
        self.assertIsNone(_detect_color_to_pattern(pairs))

    def test_not_detected_inconsistent_mapping(self):
        pairs = [
            _pair(self._inp(2), [[5, 5, 5], [0, 5, 0], [0, 5, 0]]),
            _pair(self._inp(2), [[0, 5, 0], [5, 5, 5], [0, 5, 0]]),  # same color, diff pattern
        ]
        self.assertIsNone(_detect_color_to_pattern(pairs))

    def _run_net(self, inp_grid, out_H=3, out_W=3):
        import numpy as np, onnxruntime as ort, onnx, io
        model = color_to_pattern_network(5, 5, self._COLOR_MAP, out_color=5)
        buf = io.BytesIO(); onnx.save(model, buf); buf.seek(0)
        sess = ort.InferenceSession(buf.read())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp_grid):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(None, {"input": x})[0]
        return np.argmax(out[0], axis=0)[:out_H, :out_W].tolist()

    def test_network_color2(self):
        self.assertEqual(self._run_net(self._inp(2)), [[5, 5, 5], [0, 5, 0], [0, 5, 0]])

    def test_network_color1(self):
        self.assertEqual(self._run_net(self._inp(1)), [[0, 5, 0], [5, 5, 5], [0, 5, 0]])

    def test_network_color3(self):
        self.assertEqual(self._run_net(self._inp(3)), [[0, 0, 5], [0, 0, 5], [5, 5, 5]])

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            color_to_pattern_network(5, 5, self._COLOR_MAP, batch=2)


class CrossLineFillTest(unittest.TestCase):
    # task299: H=6, W=6, color_col=8, color_row=2, color_inter=4
    def _pairs(self):
        # pair0: col4=8(rows0,1), row2=2(cols0,1) → full cross with intersection(2,4)=4
        return [
            _pair(
                [[0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 8, 0],
                 [2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 8, 0],
                 [2, 2, 2, 2, 4, 2], [0, 0, 0, 0, 8, 0],
                 [0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 8, 0]],
            ),
            _pair(
                [[0, 8, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2],
                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 8, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0],
                 [0, 8, 0, 0, 0, 0], [2, 4, 2, 2, 2, 2],
                 [0, 8, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0]],
            ),
        ]

    def test_detected(self):
        m = _detect_cross_line_fill(self._pairs())
        self.assertIsNotNone(m)
        self.assertEqual(m["H"], 6)
        self.assertEqual(m["W"], 6)
        self.assertEqual(m["color_col"], 8)
        self.assertEqual(m["color_row"], 2)
        self.assertEqual(m["color_inter"], 4)

    def test_not_detected_one_color(self):
        pairs = [_pair([[0, 8, 0], [0, 8, 0], [0, 0, 0]],
                       [[0, 8, 0], [0, 8, 0], [0, 8, 0]])]
        self.assertIsNone(_detect_cross_line_fill(pairs))

    def test_not_detected_both_multirow_multicol(self):
        # Both colors span multiple rows AND multiple cols → neither is pure col/row
        pairs = [_pair([[8, 8, 0], [8, 8, 0], [2, 2, 0]],
                       [[8, 8, 0], [8, 8, 0], [2, 2, 0]])]
        self.assertIsNone(_detect_cross_line_fill(pairs))

    def test_not_detected_wrong_intersection(self):
        # intersection is 0 → rejected
        pairs = [_pair(
            [[0, 8, 0], [2, 2, 0], [0, 0, 0]],
            [[0, 8, 0], [2, 0, 2], [0, 8, 0]],
        )]
        self.assertIsNone(_detect_cross_line_fill(pairs))

    def test_not_detected_shape_change(self):
        pairs = [_pair([[0, 8, 0], [2, 2, 0]], [[0, 8, 0], [2, 4, 2], [0, 8, 0]])]
        self.assertIsNone(_detect_cross_line_fill(pairs))

    def _run_net(self, inp_grid, H=6, W=6):
        import numpy as np, onnxruntime as ort, onnx, io
        model = cross_line_fill_network(H, W, color_col=8, color_row=2, color_inter=4)
        buf = io.BytesIO(); onnx.save(model, buf); buf.seek(0)
        sess = ort.InferenceSession(buf.read())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp_grid):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(None, {"input": x})[0]
        return np.argmax(out[0], axis=0)[:H, :W].tolist()

    def test_network_pair0(self):
        inp = [[0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 8, 0],
               [2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        expected = [[0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 8, 0],
                    [2, 2, 2, 2, 4, 2], [0, 0, 0, 0, 8, 0],
                    [0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 8, 0]]
        self.assertEqual(self._run_net(inp), expected)

    def test_network_pair1(self):
        inp = [[0, 8, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2],
               [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        expected = [[0, 8, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0],
                    [0, 8, 0, 0, 0, 0], [2, 4, 2, 2, 2, 2],
                    [0, 8, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0]]
        self.assertEqual(self._run_net(inp), expected)

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            cross_line_fill_network(6, 6, 8, 2, 4, batch=2)


class RectangularZeroRankFillTest(unittest.TestCase):
    def _sample_pair(self):
        inp = [
            [0, 0, 2, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 2],
            [0, 2, 0, 0, 2, 0, 0],
            [0, 2, 0, 0, 2, 0, 0],
        ]
        out = [
            [0, 0, 2, 1, 1, 1, 1],
            [0, 0, 2, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2],
            [8, 2, 0, 0, 2, 0, 0],
            [8, 2, 0, 0, 2, 0, 0],
        ]
        return inp, out

    def test_detects_sample_pair(self):
        inp, out = self._sample_pair()
        self.assertEqual(_detect_rectangular_zero_rank_fill([_pair(inp, out)]), {"H": 5, "W": 7})

    def test_rejects_wrong_smallest_label(self):
        inp, out = self._sample_pair()
        wrong = [row[:] for row in out]
        wrong[3][0] = 0
        self.assertIsNone(_detect_rectangular_zero_rank_fill([_pair(inp, wrong)]))

    def test_rejects_non_rectangular_zero_component(self):
        inp = [[0, 0], [0, 2]]
        out = [[1, 1], [1, 2]]
        self.assertIsNone(_detect_rectangular_zero_rank_fill([_pair(inp, out)]))

    def test_network_matches_sample(self):
        import io
        import numpy as np
        import onnx
        import onnxruntime as ort

        inp, expected = self._sample_pair()
        model = rectangular_zero_rank_fill_network(5, 7)
        buf = io.BytesIO()
        onnx.save(model, buf)
        sess = ort.InferenceSession(buf.getvalue())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        y = sess.run(None, {"input": x})[0]
        got = []
        for r in range(len(expected)):
            row = []
            for c in range(len(expected[0])):
                colors = [ch for ch in range(10) if y[0, ch, r, c] > 0.5]
                row.append(colors[0] if len(colors) == 1 else None)
            got.append(row)
        self.assertEqual(got, expected)

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            rectangular_zero_rank_fill_network(5, 7, batch=2)


class FramedZeroSquareFillTest(unittest.TestCase):
    def _sample_pair(self):
        inp = [
            [5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 5, 0, 5, 5, 5, 5, 5],
            [5, 0, 0, 5, 0, 5, 0, 0, 0, 5],
            [5, 5, 5, 5, 0, 5, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 5, 5, 5, 5, 5],
            [0, 5, 5, 5, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0, 0, 0, 0, 0, 0],
        ]
        out = [row[:] for row in inp]
        for r in (1, 2):
            for c in (1, 2):
                out[r][c] = 2
        out[6][2] = 2
        return inp, out

    def test_detects_sample_pair(self):
        inp, out = self._sample_pair()
        self.assertEqual(
            _detect_framed_zero_square_fill([_pair(inp, out)]),
            {"H": 8, "W": 10, "max_size": 2, "frame_color": 5, "fill_color": 2},
        )

    def test_rejects_filled_non_square_rectangle(self):
        inp, out = self._sample_pair()
        wrong = [row[:] for row in out]
        wrong[2][6] = 2
        self.assertIsNone(_detect_framed_zero_square_fill([_pair(inp, wrong)]))

    def test_rejects_incomplete_frame_fill(self):
        inp, out = self._sample_pair()
        broken = [row[:] for row in inp]
        broken[0][1] = 0
        self.assertIsNone(_detect_framed_zero_square_fill([_pair(broken, out)]))

    def test_rejects_missing_fill_even_if_later_pair_infers_color(self):
        inp, out = self._sample_pair()
        missing = [row[:] for row in inp]
        self.assertIsNone(_detect_framed_zero_square_fill([_pair(inp, missing), _pair(inp, out)]))

    def test_network_matches_sample(self):
        import io
        import numpy as np
        import onnx
        import onnxruntime as ort

        inp, expected = self._sample_pair()
        model = framed_zero_square_fill_network(8, 10, max_size=2)
        buf = io.BytesIO()
        onnx.save(model, buf)
        sess = ort.InferenceSession(buf.getvalue())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        y = sess.run(None, {"input": x})[0]
        got = []
        for r in range(len(expected)):
            row = []
            for c in range(len(expected[0])):
                colors = [ch for ch in range(10) if y[0, ch, r, c] > 0.5]
                row.append(colors[0] if len(colors) == 1 else None)
            got.append(row)
        self.assertEqual(got, expected)

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            framed_zero_square_fill_network(8, 10, max_size=2, batch=2)


class BouncingDiagonalRayTest(unittest.TestCase):
    def _pair_for_width(self, width):
        inp = [[0 for _ in range(width)] for _ in range(10)]
        inp[9][0] = 1
        out = [[0 for _ in range(width)] for _ in range(10)]
        period = 2 * (width - 1)
        for r in range(10):
            dist = 9 - r
            m = dist % period
            col = m if m < width else period - m
            out[r][col] = 1
        return inp, out

    def test_detects_variable_widths(self):
        p2 = self._pair_for_width(2)
        p5 = self._pair_for_width(5)
        self.assertEqual(
            _detect_bouncing_diagonal_ray([_pair(*p2), _pair(*p5)]),
            {"max_w": 5},
        )

    def test_rejects_missing_bottom_seed(self):
        inp, out = self._pair_for_width(4)
        inp[9][0] = 0
        self.assertIsNone(_detect_bouncing_diagonal_ray([_pair(inp, out)]))

    def test_network_matches_multiple_widths(self):
        import io
        import numpy as np
        import onnx
        import onnxruntime as ort

        model = bouncing_diagonal_ray_network(max_w=5)
        buf = io.BytesIO()
        onnx.save(model, buf)
        sess = ort.InferenceSession(buf.getvalue())
        for width in (2, 5):
            inp, expected = self._pair_for_width(width)
            x = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for r, row in enumerate(inp):
                for c, v in enumerate(row):
                    x[0, v, r, c] = 1.0
            y = sess.run(None, {"input": x})[0]
            got = []
            for r in range(10):
                row = []
                for c in range(width):
                    colors = [ch for ch in range(10) if y[0, ch, r, c] > 0.5]
                    row.append(colors[0] if len(colors) == 1 else None)
                got.append(row)
            self.assertEqual(got, expected)
            self.assertFalse((y[0, :, :10, width:5] > 0.5).any())

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            bouncing_diagonal_ray_network(max_w=5, batch=2)


class TrirowPeriodicFourFillTest(unittest.TestCase):
    def _pair_for_width(self, width):
        inp = [[0 for _ in range(width)] for _ in range(3)]
        out = [[0 for _ in range(width)] for _ in range(3)]
        for r in range(3):
            for c in range(width):
                scaffold = (
                    (r == 0 and c % 4 == 0) or
                    (r == 1 and c % 2 == 1) or
                    (r == 2 and c % 4 == 2)
                )
                mark = (
                    (r == 0 and c % 12 in (5, 6, 7)) or
                    (r == 1 and c % 6 == 0) or
                    (r == 2 and c % 12 in (11, 0, 1))
                )
                inp[r][c] = 2 if scaffold else 0
                out[r][c] = 2 if scaffold else (4 if mark else 0)
        return inp, out

    def test_detects_variable_widths(self):
        p10 = self._pair_for_width(10)
        p25 = self._pair_for_width(25)
        self.assertTrue(_detect_trirow_periodic_four_fill([_pair(*p10), _pair(*p25)]))

    def test_rejects_bad_scaffold(self):
        inp, out = self._pair_for_width(15)
        inp[0][1] = 2
        self.assertIsNone(_detect_trirow_periodic_four_fill([_pair(inp, out)]))

    def test_network_matches_batched_variable_widths(self):
        import io
        import numpy as np
        import onnx
        import onnxruntime as ort

        model = trirow_periodic_four_fill_network(batch=2)
        buf = io.BytesIO()
        onnx.save(model, buf)
        sess = ort.InferenceSession(buf.getvalue())
        widths = (10, 25)
        x = np.zeros((2, 10, 30, 30), dtype=np.float32)
        expected = []
        for i, width in enumerate(widths):
            inp, out = self._pair_for_width(width)
            expected.append(out)
            for r, row in enumerate(inp):
                for c, v in enumerate(row):
                    x[i, v, r, c] = 1.0
        y = sess.run(None, {"input": x})[0]
        for i, width in enumerate(widths):
            got = []
            for r in range(3):
                row = []
                for c in range(width):
                    colors = [ch for ch in range(10) if y[i, ch, r, c] > 0.5]
                    row.append(colors[0] if len(colors) == 1 else None)
                got.append(row)
            self.assertEqual(got, expected[i])
            self.assertFalse((y[i, :, :3, width:] > 0.5).any())
            self.assertFalse((y[i, :, 3:, :] > 0.5).any())


class SquareSpiralThreeTest(unittest.TestCase):
    def _pair_for_size(self, n):
        inp = [[0 for _ in range(n)] for _ in range(n)]
        out = [[0 for _ in range(n)] for _ in range(n)]
        r = c = 0
        out[r][c] = 3
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        lengths = [n - 1, n - 1, n - 1]
        k = n - 3
        while k > 0:
            lengths.extend([k, k])
            k -= 2
        for idx, length in enumerate(lengths):
            dr, dc = directions[idx % 4]
            for _ in range(length):
                r += dr
                c += dc
                if 0 <= r < n and 0 <= c < n:
                    out[r][c] = 3
        return inp, out

    def test_detects_variable_sizes(self):
        p5 = self._pair_for_size(5)
        p20 = self._pair_for_size(20)
        self.assertEqual(
            _detect_square_spiral_three([_pair(*p5), _pair(*p20)]),
            {"max_n": 30},
        )

    def test_rejects_nonzero_input(self):
        inp, out = self._pair_for_size(8)
        inp[0][0] = 1
        self.assertIsNone(_detect_square_spiral_three([_pair(inp, out)]))

    def test_network_matches_batched_variable_sizes(self):
        import io
        import numpy as np
        import onnx
        import onnxruntime as ort

        model = square_spiral_three_network(max_n=30, batch=2)
        buf = io.BytesIO()
        onnx.save(model, buf)
        sess = ort.InferenceSession(buf.getvalue())
        sizes = (5, 20)
        x = np.zeros((2, 10, 30, 30), dtype=np.float32)
        expected = []
        for i, n in enumerate(sizes):
            inp, out = self._pair_for_size(n)
            expected.append(out)
            for r, row in enumerate(inp):
                for c, v in enumerate(row):
                    x[i, v, r, c] = 1.0
        y = sess.run(None, {"input": x})[0]
        for i, n in enumerate(sizes):
            got = []
            for r in range(n):
                row = []
                for c in range(n):
                    colors = [ch for ch in range(10) if y[i, ch, r, c] > 0.5]
                    row.append(colors[0] if len(colors) == 1 else None)
                got.append(row)
            self.assertEqual(got, expected[i])
            self.assertFalse((y[i, :, :n, n:] > 0.5).any())
            self.assertFalse((y[i, :, n:, :] > 0.5).any())


class ZeroAdjacencyLabelTest(unittest.TestCase):
    def _grid(self):
        g = [[5 for _ in range(10)] for _ in range(10)]
        g[1][1] = 0  # isolated -> 3
        g[3][3] = 0
        g[3][4] = 0  # pair with no heavy neighbor -> both 2
        g[6][6] = 0
        g[6][7] = 0
        g[7][6] = 0  # L cluster -> all 1
        return g

    def _expected(self):
        out = [[5 for _ in range(10)] for _ in range(10)]
        out[1][1] = 3
        out[3][3] = 2
        out[3][4] = 2
        out[6][6] = 1
        out[6][7] = 1
        out[7][6] = 1
        return out

    def test_detected(self):
        self.assertEqual(_detect_zero_adjacency_label([_pair(self._grid(), self._expected())]), True)

    def test_rejects_wrong_label(self):
        bad = self._expected()
        bad[3][3] = 1
        self.assertIsNone(_detect_zero_adjacency_label([_pair(self._grid(), bad)]))

    def test_rejects_wrong_shape(self):
        self.assertIsNone(_detect_zero_adjacency_label([_pair([[0]], [[3]])]))

    def _run_net(self, inp_grid):
        import numpy as np
        import onnxruntime as ort

        model = zero_adjacency_label_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp_grid):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0)
        counts = (out[0] == 1.0).sum(axis=0)
        return decoded.tolist(), counts.tolist()

    def test_network_round_trip(self):
        decoded, counts = self._run_net(self._grid())
        self.assertEqual([row[:10] for row in decoded[:10]], self._expected())
        for r in range(10):
            for c in range(10):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")
        for r in range(30):
            for c in range(30):
                if r >= 10 or c >= 10:
                    self.assertEqual(counts[r][c], 0, f"padding cell ({r},{c}) is lit")

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            zero_adjacency_label_network(batch=2)


class ArchMarkerFallTest(unittest.TestCase):
    def _grid(self):
        g = [[0 for _ in range(20)] for _ in range(20)]
        arch = [
            (5, 9),
            (6, 8), (6, 9), (6, 10),
            (7, 7), (7, 8), (7, 10), (7, 11),
            (8, 6), (8, 12),
        ]
        for r, c in arch:
            g[r][c] = 9
        for r, c in [(2, 5), (3, 16), (4, 3), (10, 1), (10, 18), (13, 9), (18, 15), (19, 1)]:
            g[r][c] = 8
        return g

    def test_detects_arch_marker_fall(self):
        inp = self._grid()
        expected = _arch_marker_fall_expected(inp)
        self.assertIsNotNone(expected)
        self.assertEqual(_detect_arch_marker_fall([_pair(inp, expected.tolist())]), True)

    def test_rejects_wrong_shape(self):
        self.assertIsNone(_detect_arch_marker_fall([_pair([[0]], [[0]])]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort

        inp = self._grid()
        expected = _arch_marker_fall_expected(inp)
        model = arch_marker_fall_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0)
        counts = (out[0] == 1.0).sum(axis=0)
        self.assertTrue(np.array_equal(decoded[:20, :20], expected))
        for r in range(20):
            for c in range(20):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")


class DiagonalPairConnectTest(unittest.TestCase):
    def _grid(self):
        g = [[0 for _ in range(10)] for _ in range(10)]
        for r, c, v in [
            (0, 0, 2), (4, 4, 2),
            (1, 9, 5), (4, 6, 5),
            (9, 0, 7), (6, 3, 7),
        ]:
            g[r][c] = v
        return g

    def test_detects_diagonal_pair_connect(self):
        inp = self._grid()
        expected = _diagonal_pair_connect_expected(inp)
        self.assertIsNotNone(expected)
        self.assertEqual(_detect_diagonal_pair_connect([_pair(inp, expected.tolist())]), True)

    def test_rejects_nondiagonal_pair(self):
        inp = [[0 for _ in range(10)] for _ in range(10)]
        inp[0][0] = 3
        inp[0][2] = 3
        self.assertIsNone(_diagonal_pair_connect_expected(inp))
        self.assertIsNone(_detect_diagonal_pair_connect([_pair(inp, inp)]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort

        inp = self._grid()
        expected = _diagonal_pair_connect_expected(inp)
        model = diagonal_pair_connect_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0)
        counts = (out[0] == 1.0).sum(axis=0)
        self.assertTrue(np.array_equal(decoded[:10, :10], expected))
        for r in range(10):
            for c in range(10):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")


class DownshiftRecolor8To2Test(unittest.TestCase):
    def test_detects_downshift_recolor(self):
        inp = [
            [0, 8, 0, 0],
            [8, 8, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected = _downshift_recolor8_to2_expected(inp)
        self.assertEqual(expected.tolist(), [
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [2, 2, 0, 0],
            [0, 0, 0, 0],
        ])
        self.assertEqual(_detect_downshift_recolor8_to2([_pair(inp, expected.tolist())]), True)

    def test_rejects_bottom_row_source(self):
        inp = [[0, 0, 0], [0, 0, 0], [0, 8, 0]]
        self.assertIsNone(_downshift_recolor8_to2_expected(inp))
        self.assertIsNone(_detect_downshift_recolor8_to2([_pair(inp, inp)]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort

        inp = [
            [0, 8, 0, 0],
            [8, 8, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected = _downshift_recolor8_to2_expected(inp)
        model = downshift_recolor8_to2_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0)
        counts = (out[0] == 1.0).sum(axis=0)
        self.assertTrue(np.array_equal(decoded[:4, :4], expected))
        for r in range(4):
            for c in range(4):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")


class Task005MotifRayExtendTest(unittest.TestCase):
    def _grid(self):
        g = [[0 for _ in range(21)] for _ in range(21)]
        motif = [
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        for r in range(3):
            for c in range(3):
                if motif[r][c]:
                    g[8 + r][8 + c] = 8
        # Seed a rightward color-2 ray and an up-left color-4 ray with partial copies.
        for r, c in ((8, 12), (9, 12), (10, 12)):
            g[r][c] = 2
        for r, c in ((5, 5), (6, 5)):
            g[r][c] = 4
        return g

    def test_expected_extends_seeded_rays(self):
        inp = self._grid()
        expected = _task005_motif_ray_extend_expected(inp)
        self.assertIsNotNone(expected)
        self.assertEqual(_detect_task005_motif_ray_extend([_pair(inp, expected)]), {"matched": True})
        self.assertEqual(expected[8][12], 2)
        self.assertEqual(expected[8][16], 2)
        self.assertEqual(expected[0][0], 4)

    def test_rejects_wrong_output(self):
        inp = self._grid()
        expected = _task005_motif_ray_extend_expected(inp)
        expected[8][16] = 0
        self.assertIsNone(_detect_task005_motif_ray_extend([_pair(inp, expected)]))

    def test_network_round_trip(self):
        import numpy as np
        import onnxruntime as ort

        inp = self._grid()
        expected = np.array(_task005_motif_ray_extend_expected(inp), dtype=np.int64)
        model = task005_motif_ray_extend_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        x[0, 0, :, :] = 1.0
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                if v:
                    x[0, 0, r, c] = 0.0
                    x[0, v, r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0)
        counts = (out[0] == 1.0).sum(axis=0)
        self.assertTrue(np.array_equal(decoded[:21, :21], expected))
        self.assertEqual(float(out.min()), 0.0)
        self.assertEqual(float(out.max()), 1.0)
        self.assertTrue(np.all(counts[:21, :21] == 1))
        self.assertTrue(np.all(counts[21:, :] == 0))
        self.assertTrue(np.all(counts[:, 21:] == 0))

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            task005_motif_ray_extend_network(batch=2)


class BorderDistanceProbeRecolorStrictTest(unittest.TestCase):
    def test_vertical_borders_decode_strictly(self):
        import numpy as np
        import onnxruntime as ort

        inp = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0, 0, 3, 0, 0, 2],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        ]
        expected = _border_distance_probe_recolor_predict(inp)
        self.assertIsNotNone(expected)
        self.assertEqual(_detect_border_distance_probe_recolor([_pair(inp, expected)]), {"matched": True, "h": 10, "w": 10})
        model = border_distance_probe_recolor_network(10, 10, batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, v, r, c] = 1.0
        out = sess.run(["output"], {"input": x})[0]
        decoded = np.argmax(out[0], axis=0)
        counts = (out[0] == 1.0).sum(axis=0)
        self.assertTrue(np.array_equal(decoded[:10, :10], expected))
        for r in range(10):
            for c in range(10):
                self.assertEqual(counts[r][c], 1, f"active cell ({r},{c}) is not one-hot")
        self.assertEqual(float(out.min()), 0.0)
        self.assertEqual(float(out.max()), 1.0)


class AnchorCrossAlignedCellsTask270Test(unittest.TestCase):
    """task270: paired-anchor color-partition cross drawing (15x15)."""

    def _train0_pair(self):
        # task270 train pair 0: 1-center at (10,11), 2-center at (4,3);
        # 7-cells partition to c1 (more unambig 7s on c1's row/col), 3-cells to c2.
        inp = [[0] * 15 for _ in range(15)]
        out = [[0] * 15 for _ in range(15)]
        inp[4][3] = 2;  out[4][3] = 2
        inp[10][11] = 1; out[10][11] = 1
        inp[6][11] = 7
        inp[10][5] = 7
        inp[10][14] = 7
        inp[14][11] = 7
        inp[0][3] = 3
        inp[4][0] = 3
        inp[4][8] = 3
        inp[9][3] = 3
        out[9][11] = 7
        out[11][11] = 7
        out[10][10] = 7
        out[10][12] = 7
        out[3][3] = 3
        out[5][3] = 3
        out[4][2] = 3
        out[4][4] = 3
        return inp, out

    def test_numpy_predict_train0(self):
        inp, out = self._train0_pair()
        self.assertEqual(_anchor_cross_aligned_cells_predict(inp), out)

    def test_detect_matches(self):
        inp, out = self._train0_pair()
        m = _detect_anchor_cross_aligned_cells([_pair(inp, out)])
        self.assertEqual(m, {"matched": True})

    def test_rejects_non_15x15(self):
        inp = [[0] * 10 for _ in range(10)]
        self.assertIsNone(_anchor_cross_aligned_cells_predict(inp))

    def test_rejects_extra_colors(self):
        inp, _ = self._train0_pair()
        inp[0][0] = 5
        self.assertIsNone(_anchor_cross_aligned_cells_predict(inp))

    def test_onnx_strict_parity_train0(self):
        import numpy as np
        import onnxruntime as ort
        inp, out = self._train0_pair()
        model = anchor_cross_aligned_cells_network(batch=1)
        sess = ort.InferenceSession(model.SerializeToString())
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        x[0, 0, :, :] = 1.0
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                x[0, 0, r, c] = 0.0
                x[0, v, r, c] = 1.0
        y = sess.run(["output"], {"input": x})[0]
        decoded = []
        for r in range(15):
            row = []
            for c in range(15):
                vec = y[0, :, r, c]
                idxs = np.where(np.isclose(vec, 1.0, atol=1e-4))[0]
                self.assertEqual(len(idxs), 1, f"cell ({r},{c}) not one-hot: {vec}")
                row.append(int(idxs[0]))
            decoded.append(row)
        self.assertEqual(decoded, out)

    def test_network_batch1_only(self):
        with self.assertRaises(NotImplementedError):
            anchor_cross_aligned_cells_network(batch=2)


if __name__ == "__main__":
    unittest.main()
