# DepthToSpace(CRD) = free block-stamp interleave + the fractal-task recipe (task304, 2026-06-07)

**Candidate addition to techniques.md #44 (spatial data-movement quick table).**

## Pattern

Any **block-stamp / fractal layout** `out[b·r+dr][b·k+dc] = f(r,k) · g(dr,dc)`
(tile-where-mask, self-similar stamps, periodic block fills) is ONE op:

```
w = Where(MB [1,1,b,b] BOOL, gR [b*b,1,1] U8, 0u8)     # outer product, ch = b*dr+dc
canvas = DepthToSpace(w, blocksize=b, mode='CRD')       # [1,1,b*b,b*b]
```

CRD semantics with C=1: `out[0,0,h*b+i,w*b+j] = in[0,i*b+j,h,w]` — exactly the
interleave that otherwise costs two Unsqueeze lifts + a 6D broadcast + a Reshape
(task304 v2 vs v3: −9 B, −3 params, −1 node). The two `b²`-element tensors
(pre-interleave product, post-interleave canvas) are irreducible — no single ONNX
op fuses outer-product+interleave at integer dtypes (ConvTranspose stride-b does,
but F32-only: 4 B/elem loses to u8 2×1 B/elem).

- **DepthToSpace UINT8 + mode='CRD' has a working ORT CPU kernel** — verified on
  200k real inferences (task304 v3 gates). Grader-unshipped as of 2026-06-07;
  task304 staged v3 primary (D2S) + v2 boring sibling (no D2S) — if v3 ships
  clean, mark D2S(u8, CRD) grader-validated here.
- SpaceToDepth is the inverse read (block → channel); priced WORSE than negpad-Conv
  for 3×3 extraction (S2D output is F32-sized when fed F32: 36 B + cast 9 vs
  Pu8-reshape 9+9).

## The fractal mode-stamp recipe (task304, 11 nodes, LIVE 11,465 → 1,114 = 9.7%)

ARC family "stamp the input at every mode-colored cell" (`task_c3e719e8`-like):
negpad-Conv arange → [1,1,3,3] idx map (§41, 36 B beats 360 B Slice) ·
`Einsum('bchw->bc', input)` free-input counts (outside active = all-zero) ·
ArgMax→Cast-u8 mode · Equal mode mask · Where outer product · D2S(CRD) ·
`Equal(canvas, ids [1,10,1,1])` one-hot [1,10,9,9] BOOL · **Pad = free output**
(#67). ch0 falls out free: colors 1..9 ⇒ canvas==0 ⟺ background, and
Pad(False) satisfies the outside-grid all-channels-0 contract with NO sentinel
(unlike task124's pre-Equal 255-pad).

## Floor lemma worth reusing (#68 template)

For one-hot [1,10,30,30] outputs from a 9×9 active grid, the pre-output is
FORCED ≥ 810 B ([1,10,9,9] BOOL, Pad-final). Disproof kit (full numbers in
task304.md): Equal-final padded canvas 900; rank-1 Equal-final killed by
RECTANGLE-CLOSURE (equality patterns can't express an L-shaped data mask);
broadcast-Add/Sum/Mul/Where finals killed by PERIODICITY (broadcast repeats
size-1 axes, never period-b structure); BitwiseAnd-final needs bits = b²=9 > 8
⇒ u16 carrier ≈ 1,390 even with carry-free place-value packing (q<8 ⇒ q·8^k
never carries — the packing trick itself is sound and reusable); MatMul-final
rank-9 shared-factor 10,800 (lanes cost 36 B/row in F32 vs 2 B/row packed);
ConvTranspose-final w/ runtime stamp weights ~1,900; Resize tf_crop_and_resize
+ extrapolation_value=0 CAN zero-pad as a free output node (roi-end 29/8 keeps
scale exactly 1) but its pre-output is the same 810 — an alternative to Pad,
never a saving.
