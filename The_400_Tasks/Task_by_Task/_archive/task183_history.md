# task183 — archived history (moved 2026-06-22 w-183 for task-md-size lint)

Per-rung Cost-analysis prose for the superseded 998→841 ladder. The CURRENT best (839 `pad18_axes`)
proof + the dissolve floor proof + kill table remain in `task183.md`. Attempts table (one row per rung)
also remains in `task183.md`.

## Cost analysis (superseded rungs)

LIVE/current before the 2026-06-08 rewrite: cost 15,138 (memory 15,092 + params 46). Dominant tensors were the 10x10 F32 crop (4,000), F16 crop (2,000), two `[1,10,6,10]` reads (1,200 each), and repeated `[1,10,6,6]` F16 branch products (720 each).

Staged `coloridx_einsum` fallback cost: 998 (memory 872 + params 126), profile cost 998. Dominant remaining tensors:
- 360 B: small `[1,10,6,6]` BOOL one-hot before Pad. This is the #81 small-grid inversion output tail.
- 144 B + 36 B: cyan channel slice and bool cast.
- 120 B: dynamic `[1,30]` F32 OneHot selector for the right/bottom corner.
- 3 x 36 B UINT8 color-index canvases and 2 x 36 B bool masks.

Staged `fixedcorner_einsum` cost: 896 (memory 744 + params 152), profile cost 896. It removes `corner_pos`, `corner_idx`, and `[1,30] corner_sel` (-128 B memory) plus the `three/depth/onehot_vals` params (-4), then adds one shared `[30]` F32 candidate-position initializer (+30 params). The corner read vector is `[0,0,2,3,4,5,6,7,0,9]`, so black, blue, cyan, and off-grid candidate cells contribute zero; the fixed row/col candidate mask `[5,7,9]` sums to the true corner color for TL/TR/BL/BR on all `size in {2,4,6}` cases. Gates: score 896; visible 265/265; profile real 896; fresh ARC-GEN 5000/5000 and 200000/200000; A/B live 200265/200265.

Staged `halfsize_einsum` cost: 866 (memory 744 + params 122), profile cost 866. The graph is structurally identical to `fixedcorner_einsum` except the size detector now computes `half_size=size/2` by summing active cells at row 0, columns `[5,7,9]`, reusing the existing `corner_pos30` selector. Quadrant mask is `coord6 < half_size`; active mask is `[0,0,1,1,2,2] < half_size`, which yields 2/4/6 active cells for generator sizes 2/4/6. This deletes the old 30-param `lower_rows` selector and keeps memory unchanged. Gates: score 866; visible 265/265; profile real 866 on 265 inputs; fresh ARC-GEN 200000/200000; A/B live 200265/200265.

Staged `halfsize_novec_einsum` cost: 856 (memory 744 + params 112), profile cost 856. It keeps the 866 graph and removes the 10-param `ones10` count vector: the half-size detector is `Einsum(input, zero30, corner_pos30, equation="bchw,h,w->b")`, so the channel label is reduced in-kernel. Gates: score 856; visible 265/265; profile real 856 on 265 inputs; fresh ARC-GEN 200000/200000; A/B live 200265/200265.

Staged `halfsize_novec_addactive` cost: 853 (memory 748 + params 105), profile cost 853. It keeps the 856 graph but derives `active6` as `coord6 < (half_size + half_size)`, removing the 6-param `pair6` initializer at the cost of one 4-byte scalar `Add` output. The two row `Reshape`s are replaced by `Unsqueeze` with one shared 1-param axes initializer, saving one more param. Gates: score 853; visible 265/265; profile real 853 on 265 inputs; fresh ARC-GEN 200000/200000; A/B live 200265/200265.

Staged `count_cornersum` cost: 841 (memory 760 + params 81), profile cost 841. It replaces the half-size/corner-position selector path with two count facts: total active input cells are exactly 36/64/100 for size 2/4/6, and the four distinct corner colors are the only nonzero weighted cells in row0/col0/full-grid corner sums. Threshold tensors derive `half6` and `active6`; `tl`, `tr`, `bl`, and `br` are recovered from top-row, left-col, and all-corner sums. This removes the 10-param blue selector and the 30-param corner-position selector while adding 12 B of scalar outputs. Gates: score 841; visible 265/265; profile real 841 on 265 inputs; fresh ARC-GEN 200000/200000; A/B live 200265/200265.

Superseded intermediate `bluecount_cornersum_v2` cost: 851 (memory 760 + params 91), smoke clean on fresh ARC-GEN 5000/5000 and A/B live 5265/5265. It used the blue-frame count 20/28/36 to infer size, but the total-active-count variant removed the 10-param blue-channel selector before full staging.

Post-841 probes: `count_cornersum_f16scalar` verifies clean on fresh ARC-GEN 5000/5000 and A/B live 5265/5265 but still scores 841. `count_cornersum_const_u8` verifies on a 1000-case smoke/A-B check but scores 843 (`memory=762 params=81`), so `Constant` nodes are not a scalar-param discount. `polish_candidate.py` is a no-op on the 841 ONNX.

## Spread-knowledge scan (20260518)

Current canonical cost at scan time: **29361** (memory=29324, params=37, points≈14.712577).

Pattern sweep verdicts (each measured INDEPENDENTLY vs the original; patterns overlap so the sum overstates real cumulative savings): `clip_mul_to_min` —, `const_to_init` —, `crop_to_size_advisory` ? -32000 (max active grid 10×10 but graph has 1 [...,30,30] intermediate), `fp16_selective` ✗ verify, `onnxsim` —, `safe_opt` —, `scalar_broadcast_ones` —, `slice_axis_collapse` —. Sequential pipeline: 0 bytes saved — already at floor for the patterns in `tools/spread_patterns/`. Advisory crop_to_size est ~32000 B is a false lead for this task (the actual rewrite reached 839, not a 30×30-crop saving).

- 2026-05-20 [OB_BIG worker]: pattern=cast_after_first_slice delta=-7032 verify=265/265 cand=task183_cast_after_first_slice_t_8c2e281d_on_5835_30
