# task304 — archived history (compressed out of task304.md per the task233 convention)

## v3's RETRACTED floor argument (2026-06-07; broken same-day by the contraction-final class, v4/v5)

Claimed "pre-output ≥ 810 B forced". Correct ONLY for broadcast-op finals —
Einsum-as-output with selector inits (v4/v5) synthesizes the one-hot in the
free output node and lands at 369. The per-family disproofs below remain
valid AS A KILL-LIST for broadcast/data-movement finals on one-hot-expansion
tasks; only the "≥810 floor" conclusion is retracted.

1. **Pre-output expansion = 810 B** (#68 final-op family enumeration):
   output must be the one-hot [1,10,30,30]. Pad-final ⇒ [1,10,9,9] BOOL = 810 B
   (cheapest broadcast-class). Priced/disproven alternatives: Equal-final on
   padded u8 canvas [1,1,30,30] = 900 B; **Equal-final with rank-1 operands**
   (x[c,R]==y[c,C]) — equality patterns are rectangle-closed, an L-shaped mode
   mask (M[0,0]=M[0,1]=M[1,0]=1, M[1,1]=0) violates closure ⇒ cannot express
   arbitrary data-dependent M; **broadcast-Add/Sum/Mul/And/Where finals** —
   the pattern factor `u[c,R%3,C%3] ∧ v[R/3,C/3]` needs PERIODIC tiling,
   broadcasting only repeats size-1 axes ⇒ structurally dead (NOTE: einsum
   contraction with [x%3==m]-style selector inits sidesteps exactly this —
   the v4/v5 break); **BitwiseAnd-final (#67 205-style)** — needs 9 bits
   (pairs (dr,k)) > u8, u16 operands with channel axis = 1,200 B;
   **ConvTranspose-final** (stride-3 kernel-3 IS the stamp semantics: out =
   ConvTranspose(Mpad, inOH-slice as runtime W, strides=3)) — needs X
   [1,2,10,10] F32 800 B + runtime W via the 360 B slice + Concat 720 B ≈
   1,900 B, and the G=1 form has NO ch0 path; **MatMul-final** rank-9
   shared-factor — A [1,10,30,9] F32 10,800 B (F16 5,400, #77) — the 9 lanes
   that u16-BitwiseAnd packs into 2 B/row cost 36 B/row in MatMul;
   **BitwiseAnd-final u16** (9 bits (dc,k); x[c,R] = q[c,R%3]·m[R/3]
   place-value factorization, q<8 ⇒ carry-free): constructed end-to-end at
   ≈1,297 mem + ~95p ≈ 1,390 — the padded x [1,10,30,1] u16 600 B + q/m
   construction outprice the 972 they replace; OneHot-final (INT32 idx
   [1,30,30] 3,600 B), Gather-final (index tensor [1,10,30,30] I32 36,000 B),
   Tile-final (padding breaks periodicity), Resize-final (block-constant only;
   though Resize tf_crop_and_resize + extrapolation_value=0 CAN zero-pad as a
   free output node — roi-end 29/8 keeps scale exactly 1 — an alternative to
   Pad, never a saving), Reshape/Concat/Expand finals — all dead.
2. **Canvas assembly 162 B**: w 81 + c9 81. DepthToSpace(CRD) is exactly the
   block-stamp interleave (v3) — saves the v2 6D lifts (−9 B, −3p) but the two
   81-elem tensors are both irreducible in the broadcast class: no single ONNX
   op fuses outer-product+interleave at integer dtypes (ConvTranspose does,
   but F32: [1,1,9,9] 324 B > 162). Gather-tiling ≥ 243 B; Einsum outer
   product F32-only (324 B + cast). Equal in 6D/block layout pre-interleave ⇒
   a second 810 B tensor — dead.
3. **Detection 94 B**: P 36 (Conv F32 output forced; §41 negpad deletes the
   360 B Slice) + cnt 40 (free-input einsum; per-channel counts cannot come
   from P alone — pairwise-Equal counting ≈ 200 B, LIVE's per-cell-count trick
   needs the [1,10,3,3] slice) + mode 9 + casts 19. Also priced: Hardmax +
   runtime-weight negpad-Conv mode chain 165 B; percell-count Conv (ReduceSum
   [1,10,1,1] as runtime Conv weights + Equal vs ReduceMax) 89 B vs the
   58 B ArgMax chain; TopK (two outputs, 12 B) vs ArgMax (8 B).

The "no probe-class candidate exists" conclusion also carried over: every
formulation is fully static, §39 static==exact ⇒ submits teach nothing about
PRICE; the v4/v5 submit-unknowns are about grader-ORT op SUPPORT (12-operand
einsum, einsum-as-output), not cost.
