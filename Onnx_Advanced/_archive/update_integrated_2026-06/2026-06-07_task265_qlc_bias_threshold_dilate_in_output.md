# task265: int32-bias thresholding in QLinearConv + morphological dilate INSIDE the free output conv

**Ship-class result** (9,584 → 5,107 byte-equal-SAFE, all gates gold incl. the protrusion
landmine pair; sibling 6,844): the v2 F16 morph_open+template port rebuilt on u8 QLC lanes
with the output tail and the dilate plane both absorbed into a free output QLinearConv.
Three reusable facts beyond #72/#74/#77/#78:

1. **QLC int32 BIAS = free threshold/binarizer.** With scales 1, zps 0:
   `y = sat(conv_int32 + B)`. Any `count == K` test on a bounded count becomes
   `B = -(K-1)` → y ∈ {0,1} (u8 saturation floors negatives = free Relu). This deletes
   the Conv→Greater pair per detector: erode `sat(Σ2×2 − 3)`, exact-template-match
   `sat(ΣKdiff − 12)` ⇔ ==13 (an exact 25-cell 2-channel template fits a SINGLE-channel
   i8 Kdiff kernel: match ⟺ sum hits its structural max). Each binarized plane is u8
   324 B vs F16 648+BOOL 324.

2. **Spatial window taps inside the free output conv replace a whole morphology stage.**
   task125 banked center-taps + 3×3-count taps; extension: give the output QLC a 2×2
   kernel and tap the BINARIZED erode channel at all 4 positions → `covc` (dilate counts)
   never exists as a tensor. Output channels then ride sign-correct integer expressions
   under the official `(out>0)` decode: `ch0 = sat(z − covc + 4·excl)`,
   `ch2 = sat(covc − 4·excl)` (the ±4 dominates because covc ≤ 4). Boundary-leak rule
   refined: positive off-center taps are safe IFF the tapped channel is STRUCTURALLY ZERO
   on its last row/col (here: erode row/col 17 ≡ 0 because its 2×2 window reads pad) —
   audited with a 3,014-grid border-stress suite, 0 divergences.

3. **The bias-leak law (free-output convs).** The conv padding region ≡ all-input-channels-0,
   so EVERY output channel must be constant-free: any per-channel bias or folded "+1"
   lights the whole out-of-grid canvas (decode non-empty → certain fail). Consequence:
   complement channels (five = 1 − z) can NOT be derived inside the output conv — both a
   plane and its complement must arrive as separate x channels if both are consumed.
   Also dead (priced or proven): packing two u8 planes into one channel when any consumer
   taps a spatial window (neighbor values pollute the sum); #72 rounding gadgets for one
   channel when other channels need integer identity (y_scale is per-TENSOR); per-row
   masks inside any conv (rowmask stays a [1,1,H,1] u8 Mul).

4. **CROP-instead-of-mask for exact-template detectors (v4, −384 B more).** When an exact
   template partitions its k×k window across complementary channels (here V6c0+V6c1 =
   all-ones 5×5), a TRUE fire needs the full window in-grid ⇒ fires live only at centers
   ≥⌈k/2⌉ from every edge. Give the detector QLC pads=0 so it EMITS only that interior
   ([1,1,14,14] = 196 B vs 324) — boundary-spurious fires of a folded single-channel form
   land outside the crop by construction — then ONE shared `Pad`(default 0) after the
   detector-Add restores alignment AND **is** the boundary mask (the [1,1,H,1] mask init
   + Mul die). Audit per-axis before cropping: count the lost "+1" cells per padded
   row/col of the folded kernel — fire impossible iff ≥1 is lost (t6/t7: every boundary
   col loses ≥1; rows lose 0 at the documented spurious rows 1/16 — which the crop also
   removes, replicating the v2 row-mask exactly).

5. **Formulation-break audit residue (multi-formulation floor method).** 11 decompositions
   skeleton-priced against the 4,713 kill-number (full table: task265.md). The one
   sub-kill skeleton worth re-using elsewhere: **pack-entry + BitwiseAnd-as-output** —
   `Conv(input, W[1,10,1,30]=2^c ch0-row weights)` reads the input FREE and emits packed
   rows `[1,1,30,1]` F32 **120 B** (sums ≤2^18 < 2^24 F32-exact; NO 1,296 slice!), and
   `BitwiseAnd(rowcodes [1,10,30,1] i32, colpows 1<<c)` unpacks AT the free output node
   (i32 decodes >0, task345). On task265 it died at ~7.5k because bit-parallel EXACT-5×5
   matching costs ~30 charged [30,1] tensors (BitwiseAnd is binary-only; vertical shifts
   are 2-op). **Use the pair on tasks whose per-row logic is a few shifts/ANDs** —
   skeleton is ~1.8k incl. assembly. Blockers to know: no wide-weight integer pack op
   exists (ConvInteger w∈i8, QLC y≤255, Conv needs float) — packed compute can't be
   reached FROM u8 planes; commit to packed end-to-end or not at all.

*Floors note (#68 again): this killed task265's 2026-06-01 "byte-equal floor ≈ 11,237" and
"sub-5k impossible" proofs — their premises were §8f F16-Conv pricing and a mandatory
Pad/Equal tail. Builders: `task265/prototypes/build_task265_v3.py` / `_v4.py`
(+`_v3b.py` classic-tail sibling). Staged ladder: `task265_v4_qlc_freeout_crop_cost4713`
(primary) + `task265_v3_qlc_freeout_cost5107` + `task265_v3b_qlc_classictail_cost6844`.
Spread target: any Conv→Greater/Equal threshold pair on bounded integer counts, any
erode→dilate (opening) pipeline whose dilate output feeds only sign-consumers, and any
boundary-mask Mul beside an exact-template Conv (crop usually beats the mask).*
