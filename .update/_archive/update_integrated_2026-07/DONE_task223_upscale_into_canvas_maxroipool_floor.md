# Transferable insight — integer pixel-upscale into a fixed canvas: MaxRoiPool rois=5 is the forced floor

**Source**: task223 batch 53 (`bo_ng_cld_worker2`, 2026-07-07). LIVE 5 (Aditya) confirmed at forced floor; no improvement. Perfect-25 "zero-cost direct-output" rumor **disproven**.

**Task family**: "×k nearest-neighbour pixel upscale, placed top-left in the one-hot `[1,10,30,30]` canvas" (Spatial_and_Geometric / Tiling). Rule here: `output[c,i,j] = input[c, i//3, j//3]`, i.e. `np.kron(grid, ones(3,3))`. Verified 265/265 visible.

**Why it is NOT a perfect-25 zero-cost slot**: the output ≠ any permutation/flip/rotate of the input (3×3 → 9×9 upscale is a genuine resample). A single `Identity`/`Transpose` visible gate would return 0/265 — do not grind the one-node sweep for this family. The "forced floor ≤1" framing only applies to true no-op tasks; an upscale is not one.

**The winning cost-5 architecture (live)**: a **single** `MaxRoiPool` node.
- rois `[[0,0,0,2.9,2.9]]` (5 F32), `spatial_scale=3.0`, `pooled_shape=[30,30]`. Maps the input's top-left window into the 30×30 output at the correct upscaled resolution — no intermediate, so memory 0; params = rois = **5**.

**Why 5 is the forced floor (forced-component argument)**:
Credited cost = memory (Σ bytes of every intermediate; `input`/`output` exempt) + params (Σ initializer elements + Constant-node attr payloads — **op-level attributes are FREE**). To beat 5 you need mem+params ≤ 4.
- **mem = 0 ⇒ single node** doing a spatially-varying ×k resample that keeps size 30.
  - Uniform resamplers (Resize/Upsample/ConvTranspose/Pool) force output = k·30 = 90, not 30. A 90×90 output fails visible **0/265** (harness `mode_visible` does full-tensor `array_equal` vs the encoded `[1,10,30,30]`).
  - Sub-window→fixed-size resamplers need a region spec, and its element count is the floor: **MaxRoiPool rois `[K,5]` = 5**, RoiAlign rois`[K,4]`+batch_idx`[K]` = 5, Resize `tf_crop_and_resize` roi = 2·rank = 8. → params ≥ 5.
  - Index/matrix maps (Gather repeat-index, MatMul selection) need ≥ output-dim (30) elements. → ≥ 30.
- **mem > 0 ⇒ ≥1 intermediate**; cheapest carried tensor is `[1,10,10,10]`≈4,000 B or a `[1,10,90,90]` upscale = **324,000 B**. Both ≫ 4.

∴ min cost = 5.

**The lever / anti-pattern (measured)**:
- Attribute-only **Upsample-7 ×3 + Slice-1 crop** → correct 265/265 but the `[1,10,90,90]` intermediate = **324,000** memory. Op attributes cost nothing (params 0), but the *materialized upscale* is the whole cost. **Never upscale-then-crop; use a window-resample op that writes the final resolution directly.**
- Single **Resize ×3** declared straight to a `[1,10,90,90]` output (params 4, mem 0) is cheap but **visible 0/265** — output shape must be exactly `[1,10,30,30]`.

**Guards**:
- Output shape is hard-checked by full-tensor `array_equal`; a bigger-than-30×30 output never matches regardless of top-left content.
- `MaxRoiPool` rois shape is fixed `[num_rois, 5]` → 5 is the hard minimum; no dtype trick helps (params count *elements*, not bytes).

**Applies to**: any integer-scale pixel-upscale / block-replicate task on the fixed 30×30 canvas. When the current best is a `MaxRoiPool`/`RoiAlign` at cost 5, it is already at the floor — record WALL, do not re-explore uniform-resize or gather routes.
