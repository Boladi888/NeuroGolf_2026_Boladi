# algorithm_patterns.md — recurring ARC algorithms with minimal-cost ONNX templates

Each pattern: plain-English description, the rule it captures, the minimal-cost ONNX template, citation of tasks that used it.

Use this as a candidate-shortlist when you've discovered your task's rule. Match the rule to a pattern; use the template as your starting point.

---

## 1. Color permutation

**Rule**: for every cell of color c in the input, output color π(c). Identity outside the grid.

**Detection**: train pair colors satisfy a bijection: for every (input_color → output_color) pair observed, π is consistent.

**ONNX template** (single op, cost ~10):
```
input [1,10,30,30] → Gather(input, indices=π_inv, axis=1) → output [1,10,30,30]
```

- `π_inv` is an INT64 initializer of shape `[10]`: the inverse permutation.
- Cost: ~10 params (the indices array).
- Max points: ~22.7 / task.

**Threshold**: require ≥ 7 of 10 colors observed across visible pairs to confidently infer π. Below that, unseen colors default to identity and the inferred Gather may produce wrong outputs for cells whose color is in the unseen set.

**Tasks**: task016 (8 of 10 colors observed), task337 (9 of 10).

---

## 2. Bounding-box extraction

**Rule**: find the smallest rectangle containing all non-bg cells.

**Detection**: tasks involving "the object" or "the shape" — typically single-object inputs where the output references the object's extent.

**ONNX template**:
```
active = Greater(ReduceSum(input, axes=[1], keepdims=1), 0)   # BOOL [1,1,30,30]
# OR for non-bg only (excluding padding inside the bg region):
nonbg = Greater(Sub(ReduceSum(input, axes=[1], keepdims=1), Gather(input, [0], axis=1)), 0)

# Row presence vector [1,1,30,1] — any cell in this row is active
row_present = ReduceMax(active, axes=[3], keepdims=1)

# Col presence vector [1,1,1,30]
col_present = ReduceMax(active, axes=[2], keepdims=1)

# Bbox: outer product gives the bounding rectangle
bbox = Mul(Cast(row_present, FLOAT), Cast(col_present, FLOAT))   # [1,1,30,30]
```

Reorder for `mul_mask_to_where` (rule #2 of bool-first): keep row_present and col_present as BOOL through an `And` then Cast once at the end.

- Cost band: 5,000-30,000 depending on intermediates.
- Tasks: task049 (least_color_solid_bbox, fresh rebuild -51,523), object_based family generally.

**Variants**:
- "Largest object" → bbox + size comparison via ReduceSum then ArgMax.
- "Smallest object" → similar with ArgMin.
- "Color of the largest" → bbox + Gather of input color at the bbox center.

---

## 3. Active-cell / non-bg detection

The encoder ambiguity (ch0=1 means bg cell INSIDE grid, all-0 means padding outside grid) makes "is this cell a foreground cell?" tricky. Use the correct formulation:

```
# Active (any color, including bg INSIDE grid — excludes padding outside):
active = Greater(ReduceSum(input, axes=[1], keepdims=1), 0)   # BOOL [1,1,30,30]

# Non-bg (excludes both bg cells AND padding):
full_sum = ReduceSum(input, axes=[1], keepdims=1)
ch0_val  = Gather(input, [0], axis=1)
nonbg    = Greater(Sub(full_sum, ch0_val), 0)
```

DO NOT use `Sub(1, ch0)` — the "1-ch0 trap" in `core.md`.

---

## 4. Periodic / tile fill

**Rule**: input has periodic pattern with period P along H and/or W; some cells are missing (bg) and need to be filled per the period.

**Detection**: rH = period along height, rW = period along width. Test by checking `input[r, c] == input[(r + P) % H, c]` for various P.

**ONNX template** (period scoring + selection + iterative fill):
```python
# Phase 1: convert one-hot to color index (cheap)
# Conv binds X/W/Y to the same type parameter (T) — F32 input + F16 weights FAILS load.
# Either: (a) Cast input to F16 first, then Conv with F16 weights; or (b) keep F16 input
# at the entry boundary (input→Cast→F16); or (c) use F32 weights with F32 intermediates.
input_f16 = Cast(input, F16)                       # required if input is F32
input_idx = Conv(input_f16, weights=[0,1,2,...,9]) # F16 weights → F16 idx [1,1,H,W]

# Phase 2: for each candidate period k, score validity SEPARATELY for V and H
# Don't joint-ArgMax — test pair may have asymmetric period (rH ≠ rW)
for k in [2, 4, 5, 6, 7, 8, 9]:
    shifted = Gather(input_idx, shift_indices_k_V)  # shifted by k along H
    bg_mask = Equal(input_idx, 0)                    # missing cells
    mvp     = boundary_validity_mask_for_k_V         # avoid wrap
    ok      = Or(Equal(input_idx, shifted), Or(bg_mask, Not(mvp)))
    mismatch = Not(ok)
    valid_V[k] = Equal(ReduceSum(Cast(mismatch, INT32)), 0)

# Pick smallest valid k via Where chain
selected_k_V = Where(valid_V[2], 2, Where(valid_V[4], 4, ...))

# Phase 3: iterative fill (2 iters typically converges)
filled = input_idx
for _ in range(2):
    # fill vp/vn/hp/hn directions
    filled = ...

# Phase 4: padding detection — set fi=10 (out-of-range OneHot index) so OneHot produces all-zero on padding cells
is_padding = Equal(ReduceSum(input, axes=[1], keepdims=1), 0)  # TRUE for padding (all channels are 0)
fi = Where(is_padding, 10, filled)  # write 10 INTO padding; keep `filled` for active cells
# Alternative form: input_active = Greater(ReduceSum(...), 0); fi = Where(input_active, filled, 10)

# Phase 5: OneHot expand to [1,10,H,W]
output = OneHot(fi, depth=10)
```

- Cost band: 200,000+ for full 7-candidate × 2-direction scoring.
- Lower bound for this algorithm: ~210k. Below 100k needs row-signature hashing instead of per-cell mismatch tensors.

**Tasks**: task110 (radical rewrite 2026-05-20, -94,303 bytes, +12.73 LB).

**Critical**: separate V and H scoring. Joint ArgMax over combined V+H scores systematically favors V when V pair count ≥ H pair count, masking H periodicity.

---

## 5. Symmetry completion

**Rule**: input is half (or 3/4) of a symmetric pattern; fill in the missing half.

**Detection**: input + its mirror (h-flip, v-flip, transpose) equals the output in non-empty cells.

**ONNX template** (h-mirror as Gather along W):
```
mirrored = Gather(input, indices=[29,28,...,1,0], axis=3)
output   = max(input, mirrored)   # or input OR mirrored, in bool
```

For sub-grid mirroring (not full canvas), use Slice→Gather→Pad:
```
sub      = Slice(input, ends=[1,10,H,W])
mirrored = Gather(sub, indices=[W-1,W-2,...,0], axis=3)
result   = Concat([sub, mirrored], axis=3)
output   = Pad(result, [0,0,0,0, 0,0,0, 30-2W])
```

Cost: ~1,500-5,000 (shape-only family).
Tasks: mirror-concat family (5 tasks), mirror-grid-2x2 (3 tasks).

---

## 6. Single-fixed-shape task (hardcoded transform)

**Rule**: every visible pair has the same input/output shape, and the transform is some fixed rearrangement (rotation, transpose, specific permutation).

**Detection**: `len(set(input_shape for pair in pairs)) == 1` AND output is a fixed-indices Gather of the input.

**ONNX template**:
```
output = Transpose(input, perm=[0,1,3,2])    # Cost: 0 — pure shape-op, no initializer
# OR
output = Gather(input, fixed_indices, axis=K) # Cost: element count of `fixed_indices` initializer
```

- **`Transpose`/`Identity` can achieve cost 0** when they are the only node and have no initializers — pure shape-op with the named graph output.
- **`Gather` is NOT cost 0** — the index initializer counts in `params`. A full-canvas 30-element Gather index costs 30 params; a 900-element index costs 900 params.
- Mining heuristic from 2026-05-01: 4/400 tasks (053, 087, 140, 380) had all-pairs-(3,3) and were solvable by hardcoded Gather indices.

**Caveat**: full-canvas Gather (cost 30) vs sub-grid Slice→Gather→Pad sandwich (cost 1,091): for already-cost-30 patterns the sandwich adds ~36k bytes via the intermediate. Measure before refactoring.

---

## 7. Connected components (limited form)

**Rule**: identify which cells belong to the same connected region.

**Constraint**: NeuroGolf bans `Unique` op, so label compaction is blocked. Full CC is structurally hard. Approximations:

### 7a. Rectangular-partition CC approximation
```
# Treat walls as "padding"; compute bbox per cell as the largest rectangle of bg-0 cells containing it.
# Use cumulative min/max of wall positions along H and W.
```
- Approximation: rooms are bounding-boxes, NOT true CC. L-shaped rooms get over-counted.
- Risk: silent-fail trap. task145 ONNX passed 267/267 visible but lost -1.18 LB on Kaggle.

### 7b. 4-connectivity vs 8-connectivity Conv kernel
For "isolated cell" detection:
```
# 4-conn (up/down/left/right):
kernel_4 = [[0,1,0],[1,0,1],[0,1,0]]
# 8-conn:
kernel_8 = [[1,1,1],[1,0,1],[1,1,1]]

neighbor_count = Conv(mask, kernel_K, padding=[1,1])
isolated = Equal(neighbor_count, 0)
```
**Always verify which connectivity by inspecting ≥ 3 pairs.** task272 lesson: 4-conn vs 8-conn isn't determinable from one pair; the diagonal-only neighbor edge case decides.

**Tasks**: task145 (CC approx), task272 (isolated-cell-4conn).

---

## 8. Color counting / mode finding

**Rule**: output depends on which color is most/least common, or how many cells have a specific color.

**ONNX template**:
```
# Per-channel histogram via spatial reduction
hist = ReduceSum(input, axes=[2, 3], keepdims=False)  # [1, 10]
mode_color = ArgMax(hist, axis=1)                       # [1] INT64

# To compute "most common foreground color" (exclude ch0):
hist_no_bg = Slice(hist, starts=[1], ends=[10], axes=[1])  # [1, 9]
mode_fg = Add(ArgMax(hist_no_bg, axis=1), 1)               # [1] INT64, adjusted for slice
```

Cost: ~50 (one small intermediate).
Use OneHot to convert the scalar color back to a `[1, 10, H, W]` output if needed: **+36,000 F32 or +18,000 F16**. **OneHot CANNOT produce BOOL or UINT8 output** — ORT's OneHot kernel rejects BOOL/UINT8 `values` (verified opset 11/13). If you need a BOOL multi-channel mask, replace OneHot with `Equal(label, channel_constants) → BOOL` which produces `[1, 10, H, W] BOOL` = 9,000 bytes at the same semantic outcome. See `ort_reality.md §3`.

---

## 9. Whole-grid summary via single Conv (bounded by F32 24-bit mantissa)

**Rule**: a per-cell decision depends on the entire grid pattern (a hash of all 900 cells).

**Hard constraint — F32 mantissa**: F32 represents integers exactly only up to **2^24 = 16,777,216** (24-bit mantissa). A power-of-two-weight Conv hash is **only collision-free for binary signatures with ≤ 24 active positions**. A 30×30 (900-bit) "whole-grid" hash overflows F32 long before it covers the grid — the resulting hash collides arbitrarily on visible inputs that differ only in bits ≥ 24.

Use one of these instead:
- **For ≤ 24 binary positions**: F32 power-of-two Conv hash is exact (3×3=9 bits, 5×5=25 bits is already too many).
- **For larger grids**: multiple smaller hashes + Concat, OR INT64-safe encoding (`BitShift`/`BitwiseAnd`), OR structured row/column reductions, OR explicit `Equal`-against-templates (accept that this is a memorized lookup — see `traps.md §14e` zero-weight family).

**ONNX template (≤ 24-position case)**:
```python
# Build weights: 2^(i*W + j) at active grid positions, 0 elsewhere
# CONSTRAINT: in_h * in_w <= 24 for collision-free F32 hash
assert in_h * in_w <= 24, "F32 hash collides above 24 active bits — use INT64 or structured reduction"
weights = np.zeros((1, 1, in_h, in_w), dtype=np.float32)
for i in range(in_h):
    for j in range(in_w):
        weights[0, 0, i, j] = 2 ** (i * in_w + j)

# Single Conv with no padding produces [1, 1, 1, 1] = the summed key
hash_key = Conv(input_ch1plus, weights, no_padding)  # [1, 1, 1, 1] F32

# Then Cast→Reshape→Gather into a per-task LUT
```

Drives `grid_pattern_to_color_network`. Combined with `OneHot` (0 MACs, but output memory still counts when intermediate), the LUT can be INT64 colors per output cell instead of one-hot floats.

**Cost band**: depends on the LUT size. ~5,000-50,000 typical. **WARNING**: if the LUT has N = 266 entries (the visible+arc-gen pair count), this is in the zero-public-weight family — cherry13b MANDATORY before staging. See `traps.md §14e`.

---

## 10. Static-table → dynamic compute (rank reduction)

**Rule**: the algorithm uses a lookup table that's a computable function of position (ring index, distance from center, parity, etc.).

**ONNX template** (rank-2 dynamic compute):
```python
# For a ring-index table at [9, 9]:
range9 = init [0, 1, ..., 8] INT64 [9]
ring_h = Min(range9, Sub(2L, range9))   # 2L = 2*(H-1)
ring_w = same
ring   = Min(Unsqueeze(ring_h, [1]),    # [9, 1]
             Unsqueeze(ring_w, [0]))    # [1, 9]   →   [9, 9]
gathered = Gather(frame_colors, ring)
```

- **Avoid materializing extra broadcast/channel/template dimensions.** `calculate_memory` charges `num_elements × dtype.itemsize`; rank itself has no per-element multiplier. What matters is total element count.
  - `[H, W]` = H·W elements
  - `[1, 1, H, W]` = H·W elements (SAME COST — rank-2 vs rank-4 makes no difference if element counts match)
  - `[1, 10, H, W]` = 10× larger (the channel dimension blows it up)
  - `[5, 1, 1, H, W]` = 5× larger (the template dimension)
  - Rank-1 `[W]` broadcast against `[H, 1]` = `H + W` storage vs full materialized `[H, W]` = `H·W` storage (this is the separable-mask gain, see `techniques.md` separable_mask_collapse)
- Trade-off (task071): 23 dynamic compute nodes producing per-color `[9, 9]` intermediates beats a static `[5, 9, 9]` table because the 5-template dimension is eliminated. The win is dimensional reduction, NOT rank reduction per se.

Tasks: task071 (ring-mask dynamic, -2,709 bytes, params 2,140 → 133).

---

## 11. Padding detection (for masked-output construction)

**Rule**: distinguish "active grid cells" from "padding cells outside the active region".

**ONNX template** (use index-10 trick with OneHot):
```
# 1. Sum across channels: padding sums to 0; active cells sum to 1
input_active = Greater(ReduceSum(input, axes=[1], keepdims=1), 0)

# 2. Compute color index for active cells; set padding to 10 (out-of-range)
fi = Where(input_active, computed_color_idx, 10)

# 3. OneHot with depth=10 produces all-zero rows for index=10 (no channel matches)
output = OneHot(fi, depth=10)
```

The "set padding index to 10" trick exploits OneHot's all-zero behavior on out-of-range indices. Cleaner than masking the output post-hoc.

Tasks: task110 padding handling.

---

## 12. Pure-constant output (rare but cheap)

**Rule**: the task's output is purely determined by the task ID (not the input). I.e., every test pair has the same output regardless of input.

**Detection**: all visible+arc-gen pairs in the task have IDENTICAL output tensors. Cross-check input variation is real.

**ONNX template**:
```
input → Identity → output  (and output is set by an initializer)
```

Or more practically:
```python
# Constant model
output_const_init = numpy_helper.from_array(known_output, name="output_value")
node = helper.make_node("Identity", inputs=["output_value"], outputs=["output"])
# input is provided but ignored
```

- Cost: params = element count of the known output. `[1, 10, 30, 30] BOOL` = 9,000.
- 25 pts max if cost=0 (only achievable for genuinely constant outputs).

Almost no NeuroGolf tasks have this property in practice — listed for completeness.

---

## 13. Anti-pattern catalog (when NOT to use these)

- **DO NOT use a bool reformulation of a float algorithm** if the original used Where(cond, float_x, float_y). Risk of silent-fail-trap on hidden inputs (task080 v5_bool_slices: 231/231 visible + 450 adversarial → -13.32 LB on Kaggle). Bool reformulations are ONLY safe when the underlying values are provably 0/1 (one-hot inputs, threshold outputs).

- **DO NOT use rectangular-partition as a CC approximation** unless you've verified on adversarial inputs against LIVE. Approximations of CC analysis pass visible 267/267 but fail private (task145).

- **DO NOT use joint V+H ArgMax for periodic-fill**. Separate per-direction scoring is required (task110 v4 lesson).

- **DO NOT use `Sub(1, ch0)` for foreground detection**. Padding-cell encoding trap (see `core.md`).
