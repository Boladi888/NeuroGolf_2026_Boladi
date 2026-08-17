# Transferable insight — two ~1-tensor 1-D morphology primitives that gut CumSum/Pad-Slice cost: wide-kernel MaxPool = prefix/suffix-OR of a contiguous band, and QLinearConv(y_scale=k) = run-threshold erosion

**Source**: task255 batch (`bo_ng_cld_worker1`, Claude lane, 2026-07-12). A from-scratch projection-fill
rewrite of the corridor/box-fill rule went **10331 → 6179** (−40%, beats LIVE 7530 by −1351) with NO
change to the detection logic — purely by swapping two verbose 1-D op-chains for these primitives.
Both verified byte-identical on 265/265 official + 3000 fresh vs the naive numpy/ONNX reference.

## Primitive 1 — prefix/suffix-OR of a CONTIGUOUS 1-D band via ONE wide MaxPool
When you have a contiguous 0/1 band `b[ra0..ra1]` along a size-30 axis and need
`prefixOR[c] = (c >= ra0)` or `suffixOR[c] = (c <= ra1)` (the classic "fill from the band to the grid
edge" masks — vein/corridor col-ranges, half-plane selectors, `cols<=right`/`cols>=left`):

- **Naive** (what most graphs do): `Cast(u8->f32)` + `CumSum(f32, reverse=?)` + `Greater(0.5)` + `Cast`
  = **4 tensors, ~300 B** each (the f32 CumSum + its cast are 120 B apiece).
- **Cheap**: a single `MaxPool` with `kernel_shape=[1,30]` (full-axis) and an **asymmetric pad of 29**:
  - prefixOR (`c>=ra0`): pad the BEGIN of that axis by 29  → `pads=[0,29,0,0]` for W, `[29,0,0,0]` for H.
  - suffixOR (`c<=ra1`): pad the END by 29                → `pads=[0,0,0,29]` for W, `[0,0,29,0]` for H.
  - `MaxPool(mask_u8, kernel [1,30] or [30,1], those pads, strides=1)` → `[.,.,.,30]` u8. **1 tensor, 30 B.**
  - Why it works: output[c] = max over the length-30 window ending/starting at c = OR of the band cells
    reachable within 30 = exactly `c>=ra0` (or `c<=ra1`) because a contiguous band on a 30-axis is always
    within 30 of any c on its filled side. **Only valid for a CONTIGUOUS band** (one run); a multi-run
    mask gives per-run fills, not a global prefix. MaxPool supports uint8 (opset 12+).
- Saving here: 6 such scans × ~270 B = **~1620 B** (the single biggest cut, 8341 → 6719).

## Primitive 2 — 1-D erosion (run-threshold) via ONE QLinearConv, no Pad+Slice+And
`erode1d(m)[i] = m[i-1] & m[i] & m[i+1]` (keep cells whose k-neighborhood is all-1 = interior of a box):

- **Naive**: `Pad` + 3×`Slice` + 2×`And` = **6 tensors** (and `And`/`Mul` on u8 needs opset 14).
- **Cheap**: `QLinearConv(x_u8, x_scale=1, x_zp=0, w=ones[1,1,1,3], w_scale=1, w_zp=0, y_scale=5, y_zp=0)`
  → the conv computes the window sum (0..3); `y_scale=5` quantizes 3→round(0.6)=1 and 0/1/2→0, i.e. a
  hard "all-3 → 1" threshold. **1 tensor, 30 B.** Kernel `[1,1,1,3]` for W-axis, `[1,1,3,1]` for H.
  - Pad-FALSE erode (edge sees 0): give the pad to the conv (`pads=[0,1,0,1]` W / `[1,0,1,0]` H).
  - Pad-TRUE erode (box extends off-grid → keep edge): `Pad(mask, value=1)` first, then conv `pads=0`.
  - Generalizes: window length K (kernel `[1,K]`) + `y_scale` chosen so `(K)/y_scale` rounds to 1 and
    `(K-1)/y_scale` rounds to 0 (i.e. `2(K-1) < y_scale <= 2K`) = a "run>=K" detector in one op. This is
    the same family as Sho's documented `QLinearConv y_scale=23`/`y_scale=13`/`y_scale=3` threshold tricks
    (task255 structural v10/v11/v17) — generalized to arbitrary run length.
- Dilation dually: `MaxPool(kernel [1,3], pads)` = 1-D dilate (max over window), 1 tensor.

## Rule of thumb
For any projection/corridor/box detector that reasons in 1-D masks along a fixed-size axis: never emit
a CumSum+Cast chain for prefix/suffix fills, and never Pad+Slice+And for erosions. A wide MaxPool and a
threshold-quantized QLinearConv each collapse the op to one small u8 tensor. Keep masks u8 end-to-end
(Min=And, Max=Or, Sub(1,x)=Not; all opset-13/14 u8) and cast to f32 only for the Einsum weights and the
final QLinearMatMul union. Reconstruction floor for a Where-rendered mask stays ~1800 B (QLinearMatMul
union 900 + BOOL mask 900); everything else is compressible to near-projection-count.
