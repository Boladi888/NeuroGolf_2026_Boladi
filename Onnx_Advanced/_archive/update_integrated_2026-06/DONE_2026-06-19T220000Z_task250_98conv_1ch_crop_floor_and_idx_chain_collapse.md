# DONE: task250 — #98 Conv 1-channel crop floor + target_idx chain collapse

## Filing info
- Task: task250 (3092 → 2700, -392 cost, +0.136 raw ln)
- Date: 2026-06-19
- Worker: worker-250
- Sorter action: EXTRACT to cost_floor_reference.md (#98 boundary condition) + potential new technique for index-chain collapse

---

## Finding 1: #98 dilated Conv DOES NOT HELP for 1-channel→1-channel crops

**The boundary condition for #98 that the library does not state explicitly:**

#98 saves memory ONLY when the SLICE intermediate is LARGER than the Conv output. The saving = eliminated_Slice_bytes - Conv_output_bytes. When both have the same size (1-channel input crop), there is ZERO memory saving.

**Concrete task250 proof:**
- Current: `Slice(input[1,10,30,30], ch5) → ch5 [1,1,10,10] F32 = 400B` + `Cast → ch5u U8 = 100B` = 500B
- Conv version: `Conv(input, W[1,10,2,2], dilations=[20,20]) → ch5f [1,1,10,10] F32 = 400B` + `Cast → ch5u U8 = 100B` = 500B
- Params: Slice uses 16p (4 init vectors); Conv uses 40p (kernel). Net: +24p.
- Actual score: Slice version 3,092; Conv version 3,116. Conv is WORSE.

**Why task271 worked:** Old graph had `Slice([1,2,9,9] F32 = 648B)` then `Conv → [1,1,9,9] F32 = 324B`. The Slice was 2× larger than the Conv output (different channel count + size). Saving = 648B.

**Rule:** `#98_saving = max(0, slice_bytes - conv_output_bytes)`. When `slice = [1,1,H,W]` (single channel) and Conv output is `[1,1,H,W]` (same size), saving is exactly 0. The #98 carve-out in cost_floor_reference.md should add this boundary condition.

---

## Finding 2: Index chain collapse via offsets + broadcast Add

**The technique (new; not in library):**

When building a `[N,D]` INT64 dynamic index by:
1. Computing components (row offsets, col offsets, batch=0, channel=0 columns)
2. Unsqueezing each to [N,1]  
3. Concatenating to [N,D]

...this costs `D×N×8` bytes for the final index PLUS `N×8` bytes for EACH of the `D` intermediate [N,1] tensors.

**Collapse to:** Precompute `offsets [N,D] INT64` constant (static, params) with the known-column entries filled in. Compute `r0c0_vec [1,D] INT64` from the dynamic scalar components. Then `target_idx = Add(offsets, r0c0_vec)` (broadcast). Chain cost: `D×8` bytes for r0c0_vec + `N×D×8` for target_idx. Saving: all the [N,1] unsqueezed intermediates (D-2 of them, each `N×8B`).

**task250 concrete saving:**
- Old: `target_rows [12]=96B`, `target_cols [12]=96B`, `target_rows_u [12,1]=96B`, `target_cols_u [12,1]=96B`, `target_idx [12,4]=384B` = 768B + 49p
- New: `r0c0_4 [4]=32B`, `r0c0_vec [1,4]=32B`, `target_idx [12,4]=384B` = 448B + 52p
- Net: -320B memory, +3p = -317B cost
- (r0 and c0 are [1] not scalar, so Concat(zeros2[2], r0[1], c0[1]) → [4] works without Unsqueeze)

**Applicability:** Any graph building a [N,D] scatter/gather index from dynamic scalar parts where some columns are constant (e.g., batch=0, channel=0). Replace the per-column Add+Unsqueeze chain with one precomputed constant [N,D] + one dynamic [1,D] broadcast Add.

---

## Finding 3: Box mask → ScatterND (mask chain collapse for small fixed-size N objects)

**The technique:**

A 2×2 red block mask built as `row_mask [H] BOOL × col_mask [W] BOOL → outer_product [H,W] BOOL → U8` costs: 2×compare (BOOL [H]) + 2×compare (BOOL [W]) + 2×And-chain + 2×Unsqueeze + And → [1,1,H,W] BOOL = H×W bytes + U8 = H×W bytes. Total ~2HW + several small.

**Alternative:** Use `ScatterND(zeros[1,1,H,W], box_idx[4,4], ones[4])` where `box_idx = box_offsets[4,4] + r0c0_vec[1,4]` (the same r0c0_vec already computed for target_idx). The only new intermediate is `box_idx [4,4] INT64 = 128B`.

**task250 saving:**
- Old box chain: 296B intermediates + 27p (two_i, row10_i, col10_i, axes_row_mask, axes_col_mask)
- New box path: box_idx 128B + box2 100B = 228B + 20p (box_offsets 16p, box_ones 4p)
- Net: -68B memory, -7p params = -75B cost
- REQUIRES: r0c0_vec already available (from target_idx chain collapse above)

**Applicability:** Any graph that builds a small N-object mask (N cells) using row/col outer-product mask chains, when the object positions can be described as base_position + fixed_offsets.

---

## Gate summary
- task250 cost 3,092 → 2,700 (-392, +0.136 raw ln)
- visible 265/265, AB 200265/200265 byte_equal, truth 200000/200000, profile==static==2,700
