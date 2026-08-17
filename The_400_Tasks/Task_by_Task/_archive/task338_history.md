# task338 — archived history (superseded families: BFS, column-parity)

Moved from `task338.md` 2026-06-30 (size-cap hygiene, WORKER_PROTOCOL staging rule).
The CURRENT family (2D-imos, v7/v8/v9) floor proof + the FINAL post-ship audit
(the wall) remain in `task338.md`. This file holds the superseded BFS and
column-parity rungs.

## BFS family (v2→v4) — superseded by column-parity

The original live graph used a 12-iteration `MaxPool(3x3, pad=1, stride=1)`
BFS interleaved with `Mul(free_f)` to spread reach 1 cell per iteration
through free cells while blocked by walls. For a 25×25 active grid the max
border-to-interior distance is 12, so 12 iterations are required.

Why 12 iterations cannot be reduced via larger kernel:
- `MaxPool(5x5)` would spread 2 cells per iter (halving the count), BUT a 5×5
  kernel reads cells past walls — the wall-block invariant of BFS-with-barriers
  REQUIRES single-cell propagation per iteration. A 5×5 pool marks cells not
  actually reachable through free cells (it ignores barriers within the window).

Original memory profile 74375 B: the 12 MaxPool + 12 Mul outputs are all
`(1,1,25,25) F32` (10000 B nominal each) but ORT buffer-reuse keeps the peak
working set far below 24×10000; structural ops (Slice/Cast/And/Or/Xor/Concat/Pad)
add small intermediates.

**2026-06-04 BFS line (v2→v4)**: UINT8 BFS K=12 (proven minimum: 226/267 pairs
need K=12), mp_11 trick (skip last Mul), arithmetic canvas
`10*free − ch0*(7+3*mp_11)`. Floor of the BFS family: **26,562 B** (v4). The
BFS chain (23 × 625 B) + 2 F32 slices (5,000 B) dominate.

## Column-parity family (v5/v6) — superseded by 2D-imos

**2026-06-04 BREAKTHROUGH (v5_parity, 17,212 B)**: the BFS is NOT needed at all.
**Crossing-parity rule** (ray-casting point-in-polygon for complete rectangular
rings): a cell is frame-interior ⟺ the count of *horizontal-edge wall cells*
strictly above it in its column is odd, where
`H(r,c) = is2(r,c) ∧ is2(r,c−1) ∧ is2(r,c+1)`. A frame fully above contributes
top+bottom = 2 crossings (even); the cell's own frame contributes only its top
edge = 1 (odd). Validated **200,000/200,000** vs ARC-GEN truth in numpy.

v5 ONNX realization (opset 14, memory 16,525 + params 687):
- H-cells: `1−MaxPool(free, [1,3], pads[0,1,0,1])` erosion (1,250 B) + edge-column
  mask Mul (625 B; MaxPool 0-pads = virtual wall outside, spurious H at cols
  0/24 — true H is provably 0 there, mask kills it; 25 params).
- Column prefix-sum: **F16 triangular MatMul** `cum = tril(1)[25,25] @ H_f16`
  (1,250+1,250 B; sums ≤25 exact in F16; 625 params). CumSum has NO F16 kernel
  and INT32 CumSum costs 2× — the MatMul saves 1,875 B net.
- Parity: `Mod(Cast(cum, U8), 2)` — Mod(u8) probed OK at opset 14.
- Canvas: `10·free − ch0·(10−3·parity)`, Pad sentinel-10, Equal-expand.

**The 16/200k LIVE-vs-truth divergences decoded (pockets)**: cells enclosed by
the UNION of several rings but inside no single ring's bbox. LIVE's flood-fill
marks them 3; the generator's per-ring bbox-fill truth says 0. All 16 fails are
1-2 cell diffs, all frames complete rings, no nesting in corpus. v5/v6 (parity =
exact generator semantics) get all 16 RIGHT — more-correct-than-LIVE (task279
precedent). v4/v3/v2 (flood semantics) reproduce LIVE exactly, including these 16.

**v6 floor analysis (16,586 = column-parity family floor; superseded by v7)**:
- 2 ORT-forced F32 input reads `[1,1,25,25]` = 5,000 B + 2 U8 casts 1,250 B.
  Measured alternatives: §35 Conv-code single-read with asym kernel = 6,250 B TIE
  (+60 params, worse); 1×1 Conv-code = 7,000 B; `[1,3,25,25]` slice = 7,500 B.
  The 2-slice read is the floor.
- F16 tri-MatMul prefix block 2,500 B (INT32 CumSum = 5,000 B; MatMulInteger U8 =
  tie; XOR-scan = 7,500 B).
- H-chain 1,250 B + parity extraction 1,250 B + canvas tail 3,125 B (every tail
  reformulation measured ties at 5×625) + padded canvas 900 B (single-channel
  canvas output floor) + free 625 B.
- Hard lower bound: forced input reads (5,000) + output canvas (900) + ≥1
  aggregation plane ≈ 8,000-9,000 B with zero glue; realistic floor ≈ 15-16k.
  **16,586 achieved.**
- Max grid: `corpus.build(338, want=200000)['in_shapes'].max(0)` = **[25, 25]**
  (out same; n=200,000) — the 25×25 crop is corpus-proven.

**Additional dead-ends priced 2026-06-04 (column-parity third pass — do not retry)**:
- `Mod(3·cum, 6)` fold (scale L_tri by 3, skip po): TIES — still needs the edge
  mask as a separate Mul, 3 tensors either way.
- `MatMulInteger(L_u8, H_u8)` → INT32: 2,500+625 = 3,125 — TIES the F16 route.
- Sparse-initializer L_tri (325 nnz params): ORT cannot execute sparse MatMul
  inputs — model won't load.
- Single strided `Slice(ch0:ch2:2)` → `[1,2,25,25]` + shared u8 ops: same
  5,000 B read, then +1,250 B to re-split; shared 2-channel planes double every
  downstream plane. WORSE.
- Negative-Pad channel crop instead of Slice: same 2,500 B/plane, no param win,
  adds negative-Pad ORT risk. No gain.
- Narrow parity planes `[25,23]` (mask cols pre-dropped): broadcast mismatch at
  t3 forces a Pad-back (625 B) that eats the 4×50 B savings. WORSE.
- 4-op canvas: every factoring of `f(free,ch0,po) → {0,3,10}` needs ≥1 nonlinear
  Mul + 2 selectors + 2 combiners = 5 tensors. `free·(10−ch0·(10−3po))` lands at 5.
- Conv+Relu-bias H detection (`Relu(Conv(is2,1×3)−2)`): ORT forces Conv F32 ⇒
  5,000 B for what the u8 erosion chain does in 1,875 B.
