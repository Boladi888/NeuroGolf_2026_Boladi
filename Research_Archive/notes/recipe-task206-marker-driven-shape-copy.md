# Recipe: task206 — marker-driven shape copy (color 5 = marker)

> `[B→A]` handoff — 2026-05-04 draft; **rule cracked 2026-05-05 (266/266)**.
>
> **Status:** truly-unsolved per `tools/blend_diff.py` (full LB-gain target).
> Numpy rule: **266/266, 0 mismatches** — fully verified.
> **Output shape:** same as input (variable; input.shape == output.shape always).
> **Estimated cost:** ~300-600k MACs (~11-12 pts).

---

## True rule (266/266)

**Color 5 (gray) is ALWAYS the marker.** It is always a single isolated cell.
The non-5 pattern (all other non-bg cells) is copied so its bbox center aligns
with the marker position. The marker cell (color 5) disappears from output.
The original pattern stays in place.

```python
def predict_task206(inp):
    a = np.array(inp); H, W = a.shape
    in_colors = sorted(set(a.flatten().tolist()) - {0})
    if 5 not in in_colors:
        return None   # always has color 5 (all 266 pairs verified)
    marker_pos = np.argwhere(a == 5)
    if len(marker_pos) != 1:
        return None   # single-cell marker (all 266 pairs verified)
    marker_pos = marker_pos[0]
    pattern_mask = (a != 0) & (a != 5)
    if not pattern_mask.any():
        return None
    pattern_rows = np.where(pattern_mask.any(axis=1))[0]
    pattern_cols = np.where(pattern_mask.any(axis=0))[0]
    pr0, pr1 = pattern_rows.min(), pattern_rows.max()
    pc0, pc1 = pattern_cols.min(), pattern_cols.max()
    # Center with round-half-even (Python default round())
    cr = (pr0 + pr1) / 2
    cc = (pc0 + pc1) / 2
    dr = round(marker_pos[0] - cr)
    dc = round(marker_pos[1] - cc)
    out = a.copy()
    out[marker_pos[0], marker_pos[1]] = 0   # erase marker
    for r in range(H):
        for c in range(W):
            if pattern_mask[r, c]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr, nc] = a[r, c]
    return out
```

**Verification:** 266/266, 0 mismatches.

## Why the prior draft only got 150/266

Prior draft tried to detect the marker as "the color whose cells all lie outside
the bbox of all other non-bg colors." With color 5 present, TWO colors passed
this test — color 5 (the true marker) AND one pattern-edge color (e.g., an
anchor cell of the pattern that was spatially isolated from the rest). The draft
rejected these 116 ambiguous cases.

With `marker_color = 5` hardcoded, all 266 pass trivially.

**Confirmed:** All 116 multi-candidate cases had color 5 as one of the candidates,
and color 5 ALWAYS disappeared in the output. The other "candidate" was actually
a pattern cell — the prior detection was too aggressive.

## Detector preconditions

For `_detect_task206` in the dispatcher:
1. Input has color 5 (channel 5 active).
2. Color 5 has exactly 1 cell across all pairs (single-cell marker).
3. Pattern = all non-bg non-5 cells; must be non-empty.
4. Verify the rule: for each pair, predict and compare. Target ≥ 95% on train+test.
5. No need for the "outside bbox" isolation test — color 5 is always isolated
   (but this is not required for detection; just verify the rule output matches).

## ONNX implementation sketch (opset 10)

`marker_driven_shape_copy_network(batch=1)`:

**Key simplification vs prior sketch:** Marker channel is fixed = 5. No need for
per-channel isolation tests or ArgMax-over-isolation-mask.

**Build-time constants:**
- All patterns have a single color-5 marker cell → marker bbox in each pair is
  a single pixel. Extract at build time? No — marker position varies per input.
  This is a RUNTIME pattern (variable marker position).

**Runtime pipeline:**

1. **Extract channel 5 (gray marker):**
   - `ch5 = Slice(input, axes=[1], starts=[5], ends=[6])` → (1,1,H,W)
   - Use `ArgMax` with descending row-weight to find `marker_r` (FLOAT scalar).
   - Use `ArgMax` with descending col-weight to find `marker_c` (FLOAT scalar).

2. **Pattern = input minus channel 5:**
   - `pattern = input * ch5_zero_mask` where ch5_zero_mask zeros channel 5.
   - Or: Concat(input[:,0:5], zeros, input[:,6:10], axis=1) — but complex.
   - Simpler: `pattern_19 = Slice(input, ch1..9) * (1 - marker_oh_19)` where
     `marker_oh_19` is a (1,9,1,1) constant with 1 at position 4 (= ch5 - 1).
   - `pattern_any = ReduceMax(pattern_19, axis=1, keepdims=1)` → (1,1,H,W).

3. **Pattern bbox (runtime):**
   - Same per-channel ArgMax trick as `foreground_bbox_crop_network`:
   - `row_has = ReduceMax(pattern_any, axes=[3], keepdims=1)` → (1,1,H,1).
   - `pr0_f`: ArgMax of `row_has * descending_30` → first row.
   - `pr1_f`: ArgMax of `row_has * ascending_30` → last row.
   - Same for cols: `pc0_f`, `pc1_f`.
   - `cr_f = (pr0_f + pr1_f) / 2` (FLOAT; rounding handled by casting to nearest int
     via Add(0.5) + Cast(INT) in opset 10, or just use FLOAT and apply as-is).

4. **Translation offsets:**
   - `dr_f = marker_r_f - cr_f` (FLOAT scalar).
   - `dc_f = marker_c_f - cc_f` (FLOAT scalar).

5. **Translated pattern:**
   - `src_r[r] = range_30[r] - dr_f` → (30,) FLOAT.
   - `src_c[c] = range_30[c] - dc_f` → (30,) FLOAT.
   - Validity: `0 <= src_r < 30` (Less-based, same as task020).
   - Clip + Cast → INT64 indices.
   - Gather rows of `pattern_19` → (1,9,30,W): row remapping.
   - Gather cols → (1,9,30,30): col remapping.
   - Apply row_valid * col_valid mask.
   - `translated = gathered * valid_mask`.

6. **Union with original pattern:**
   - `nonbg_union_19 = Max(pattern_19, translated)` → (1,9,H,W).

7. **Derive ch0:**
   - `any_nonbg = ReduceMax(nonbg_union_19, axis=1, keepdims=1)` → (1,1,H,W).
   - `is_active = ReduceMax(input, axis=1, keepdims=1) > 0` ... actually:
     `is_active` = cells within the canvas (non-padding). Use:
     `is_active = 1 - Slice(input, ch0)` = 1 where any channel is active.
     Actually: `ReduceMax(input[:,1:10], axis=1)` gives 1 at any non-bg cell.
     Canvas active = anything in the grid, even bg. Approach: use the grid's
     bounding box from `pattern_any` and the marker to define `is_active`.
     Simpler: just let `ch0 = 1 - any_nonbg` — correct for ARC cells within
     grid extent; canvas-padding cells already have all channels=0 so ch0=1
     naturally. No `is_active` mask needed if the translated pattern doesn't
     bleed into canvas-padding. Guard via validity check in step 5.
   - `ch0_out = 1 - any_nonbg`.

8. **Concat([ch0_out, nonbg_union_19], axis=1)** → output (1,10,H,W).

## Cost estimate

Key intermediates per [1, 10, 30, 30]:
- `pattern_19`: 9×30×30×4 = 32.4 KB
- Gather (row): ~32.4 KB × 2 (input + output) = 64.8 KB
- Gather (col): same = 64.8 KB
- `nonbg_union_19`: 32.4 KB
- `ch0_out`: 3.6 KB
- Misc (ArgMax outputs, scalars): ~2 KB

Total memory: ~200-300 KB.
MACs: ReduceMax (9×900)×2 = 16.2k, Gathers (0), multiplies (~50k), total ~70k.
**Cost: ~70k MACs + ~250k memory ≈ 320k → ~12.5 pts.**

(Prior estimate was 300-400k — consistent.)

## Half-integer center handling in ONNX

Python `round(1.5)` = 2 (banker's rounding toward even). In ONNX opset 10:
- `Cast(Add(x, 0.5), INT64)` uses floor-truncation after +0.5, not banker's.
- For most pairs this is fine (the center is an integer or half-integer doesn't
  matter for small offsets).
- Verified 266/266 using Python's `round()` — so the rounding mode matches the
  data. In ONNX, `Cast(Add(x, 0.5))` gives the same result as banker's rounding
  for `.5` cases: Cast truncates → `Add(1.5, 0.5) = 2.0` → Cast to 2; same as
  `round(1.5) = 2`. Check both conventions agree on the 266 pairs before
  committing.

## Verification path

1. Numpy 266/266 verified (this file). ✓
2. ONNX builder: see sketch. Use `mdc_` prefix convention.
3. Test: 5-8 unit tests covering small pattern (single-color), large pattern
   (multi-color), integer center, half-integer center, near-edge translation.
4. Add to `_collect_matches` after `marker-section-extract`.
5. Add `MarkerDrivenShapeCopyTest` to `tests/test_detect.py`.
6. Estimated Alpha effort: ~3-4 hr (much simpler than prior estimate since
   marker detection is trivial — just use channel 5).

## Related patterns

- Sister to `rotational_symmetry_completion_network` (same bbox + Gather +
  Max union + ch0 derivation).
- `marker_corners_network`: also uses a fixed marker color convention?
  (Check — if marker_corners also uses ch5 as the anchor, cost may differ.)

---

*Draft: 2026-05-04. Rule cracked + verified 266/266: 2026-05-05 Beta session.*
