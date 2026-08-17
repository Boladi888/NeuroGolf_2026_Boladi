# DONE 2026-06-22T1540Z — patch-onehot PAD-LAST floor (max-count bbox-crop family, task300)

**Topic:** when a small per-channel one-hot is the dominant charge, padding the U8 canvas
BEFORE the channel-expanding Equal/OneHot always loses — the canvas reaches full 30x30 size.

**Context (task300, max-count-color bbox-crop):** the output is `[1,10,30,30]` decoded by the
`>0` thresholder. The cheap representation slices a fixed `[1,1,4,3]` patch of the selected
channel, builds a U8 index canvas (selected color inside the bbox, 0 elsewhere, sentinel 10
outside the bbox), and must expand it to the 10-channel one-hot. Two orderings:

- **EXPAND-THEN-PAD (the floor):** `Equal(idx_crop_u8 [1,1,4,3], ids_u8 [1,10,1,1])` ->
  `patch_onehot [1,10,4,3] BOOL` = **120 B**, then `Pad` to `[1,10,30,30]` as the FREE graph
  output (I/O tensors named `output` are skipped by `calculate_memory`). Charged tail ~132 B.
- **PAD-THEN-EXPAND (the tempting "U8-patch-canvas" lever):** `Pad(idx_crop_u8 [1,1,4,3])` ->
  `[1,1,30,30] U8` = **900 B** charged, then `Equal(ids_u8)` -> `[1,10,30,30]` free output.
  Charged tail ~900 B. **LOSES by ~780 B.**

**Rule:** the 10-channel expansion is mandatory (the decoder reads 10 channels). Do it at the
SMALL patch scale and pad LAST. Padding the U8 canvas first inflates a single-channel tensor to
the full 30x30 (900 B) — strictly worse than the 120 B 10-channel one-hot at 4x3. This is the
PAD-LAST boundary (#81 / WORKER_PROTOCOL verdict #1) applied to a per-channel one-hot whose width
is tiny: the asymmetry is `10 * patch_area` (120) vs `1 * canvas_area` (900). PAD-LAST wins
whenever `patch_area < canvas_area / channels`, i.e. `12 < 900/10 = 90`. Generalizes: a U8
sentinel canvas only beats the BOOL one-hot when it already operates at (or near) full canvas
scale; at a bounded sub-patch scale the one-hot-then-pad inverts the win.

**Corollary (dtype-slice trap):** `selected_patch [1,1,4,3] F32` = 48 B looks shrinkable to 12 B
by slicing a BOOL/U8 input, but `Slice` preserves dtype and casting the `[1,10,30,30] F32` input
to a smaller dtype charges the full 9000 B canvas. The F32 slice is irreducible here.

**Outcome:** task300 floor confirmed at cost 500 (v18: memory 440 + params 60), dominant charge
`patch_onehot` 120 B. Beats team-best 544 by 44 B. All gates green (visible 267/267, truth
200000/200000, AB byte-equal 200267/200267, profile static==profile==500 over 267 real inputs).
