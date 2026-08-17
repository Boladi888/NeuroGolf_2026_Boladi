# LESSON (task182, 2026-07-02, bo_ng_cld_worker1) — bit-packed integer match beats spatial box-conv for framed-shape-recolor

**Transferable insight (candidate for library §op_cost_patterns / by_symptom "shape-match on one-hot input"):**

For "find every sprite congruent to a key shape and recolor it" on a one-hot [1,10,H,W] input, **do NOT reach for a spatial hash Conv** (my prior task182 floor proof wrongly asserted the one-hot F32 input *forces* a box-conv + F32 score plane). The cheaper paradigm (Vamshi's 5448 `task182_factored_integer`, −28.8% vs the 7649 box-conv):

1. **Localize** structural anchors (e.g. a fixed 7×7 gray frame) with **1D profiles**: `Einsum(input, channel_sel, ones)` → row/col profile [1,1,W], then `Conv(profile, run_kernel=[1×k,0×…])` + `ArgMax`. Cheap (120B per axis) and safe **for localization only**.
2. **Pack** the target channel to per-row integers: `Einsum(input, channel_sel, pow2_W)` → F32 row-codes → Cast u32 [1,1,H]. A whole HxW plane collapses to H integers (900B → 120B).
3. **Verify shape identity in the integer/bit domain**, not spatially: precompute a per-shape table of `(row<<4 | col<<1 | expected_bit)` cell-checks; `Slice`+`BitShift`+`BitwiseXor`+`BitwiseAnd`-accumulate over the packed rows. The number of check-blocks = the max distinct discriminating cells any single shape needs (pad shorter shapes with **repeated bytes = idempotent no-ops** under BitwiseAnd).
4. **Render** by bit-unpacking packed row-masks: split each n-bit row into **≤8-bit u8 lanes** (`BitShift` by {0,5,10,15}) → `And bit_masks` → BOOL → Reshape → Pad → `Where`. Chunked u8-lane unpack beats a direct `BitShift[1,1,1,W]` (which materializes a full HxW **u32** plane).

**Two false "floor" claims this corrects (watch for them elsewhere):**
- "One-hot F32 input forces an F32 spatial score plane" — FALSE; pack to integers first.
- "1D profiles are unsafe for this class" — HALF-TRUE: unsafe for shape *identity*, fine for *localization*. The v49–v53 hidden-0 failure was using a profile as the shape *hash*, not the localization step.

**Still forced (real floor drivers):** the 30×30 output `Where` mask (900B BOOL); reading a per-instance attribute (color C) from *inside* the safe region when decoys exist outside (TR3); a run-detector Conv per frame axis.

Cross-refs: relates to library #157 (U8-lane bit-unpack narrowing) and #109 (one-hot-dissolve rank gate). If a sorter picks this up, route to op_cost_patterns (bit-pack shape-match) + by_symptom (one-hot shape match / spatial-conv smell).
