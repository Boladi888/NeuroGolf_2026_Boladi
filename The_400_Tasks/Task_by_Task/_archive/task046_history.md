# task046 — full rewrite history (archived 2026-06-08; current best + floor proof live in `../task046.md`)

LIVE was a shared-blend graph (cost 25,984, src `thisray4808`, shared with 117 tasks; 100% truth but never a bespoke task046 solver). From-scratch generator-decoded SIEGE rewrite (snake-reassembly, ARC-GEN `task_234bbc79`).

## Full ladder (each: 267/267 official + 1000–3000 fresh oracle; staged rungs also 200000/200000 truth + 200267/200267 byte-equal LIVE)

| v | cost | key change |
|---|---:|---|
| LIVE | 25984 | shared blend, 8-channel [1,8,3,20], opset10 |
| v1 | 17253 | from-scratch, F32, opset11 (correctness-first) |
| v2 | 7801 | F16 working tensors + GatherElements vertical roll (vs 5 Pad/Slice branches) + bool output |
| v3 | 7128 | drop input ch0 (Conv reads ch1–9) + [20,16] pack |
| v4 | 6815 | F16 cumsum chain (cast→F32 only at the 2 CumSum ops) |
| v5 | 6475 | OneHot pack-select (drops bool sel_b + reshape scaffolding) |
| v6 | 6374 | drop sentinel mask — gap cols carry CGroll=0, harmless OneHot collision (sum-of-products, not UB) |
| v7 | 6214 | Where-fuse gray-row & delta (drop Cast+Mul pairs) |
| v8 | 4675 | **tall-kernel Conv [1,10,28,1] kills the 2160 B crop** — valid conv reads free input, crops rows 30→3 + reduces channels |
| v9 | 4555 | cast-before-slice (drop F32 M) + drop P f16 (occ from ReduceMax(Mh), CGorig via Where) |
| v10 | 4395 | int32 CumSum (drop the F32 + separate-int casts; S/target come straight out as indices) |
| v11 | 4135 | **dilated kernel** [1,10,2,1] dil27 — 2 taps span 28 rows, 280→20 params |
| v12 | 3853 | **dual-dilated** [1,10,2,2] dil[27,10] — crops cols 30→20 too, output [1,1,3,20] directly |
| v13 | 3272 | **TopK pack** — score=100·occ−c, TopK(k=16).idx = src; replaces OneHot+cumsum entirely (−581) |
| v14 | 3072 | **roll the BOOL mask** (Pad bool 140 vs F16 280), colour applied post-roll via Where; drops CGorig (−200) |

## Three wrong "floor" claims (the meta-lesson)

1. **"Colour read crop-floored at 2160 B"** (held v1–v7). FALSE: a dilated valid Conv does spatial-crop (both axes) + channel-reduce in one op for ~40 params, reading the free input — no multi-channel crop. To crop axis L→k with 2 taps use `dilation = L−k`, real weight on tap 0, zero on tap 1. (techniques.md #94)
2. **"Pack floored at OneHot Sel 640 B"** (held v5–v12). FALSE: TopK directly finds occupied-column positions. `score = BIG·occ − arange` (BIG≥20) makes occupied cols outrank unoccupied and sort ascending by column; `TopK(largest, k=W_out_max, sorted=1).indices` = src(t); plain `Gather(grid, idx, axis=cols)`. Trailing idx land on CGroll=0 cols → zeroed downstream. No cumsum, no select-matrix, no UB.
3. **"Roll needs an F16 colour grid"**. FALSE: roll the BOOL snake mask (GatherElements supports bool), apply per-column colour AFTER via `Where(rolled_mask, segcolor, 0)` — valid because the vertical roll doesn't move columns and segcolor is column-invariant.

## Rejected (don't retry)
- **ScatterElements pack** (~−147 vs OneHot, or as a TopK alternative): ONNX duplicate-index = UNDEFINED behaviour (gap cols collide on occupied targets); a hidden-eval landmine even if it passes locally. TopK+Gather is the safe equivalent and cheaper anyway.
- **Conv-first colour read** ([1,1,30,30]=3600 B) worse than crop-first; both obsoleted by the dilated conv.
- **Triangular-matmul cumsum** (400 params/matrix > mem saved); **F16 CumSum** (ORT unsupported — but int32 CumSum works, v10).
- **fp16_selective / onnxsim** on the LIVE blend (pre-rewrite): regressions / no-ops (see SPREAD scan).
