# ngs cheap-op backlog — the untapped techniques (2026-07-10 research)

Einsum + geometry were only two of the families that produce cheap (cost ≤ 100) NeuroGolf graphs.
Two deep research passes (Onnx_Advanced catalog + the teammate's engine) surfaced the full set. Empirical
grounding: of the team's 71 cost-≤100 graphs, ngs solves 21 and misses 50 — the misses are almost all a
**single op with a small LEARNED tensor** (26 Einsum, 11 Conv, 3 ConvTranspose) that ngs can't emit/fit yet.

Cost model: `cost = memory + params`; op **attributes are free**, `input`/`output` are exempt, so the win
is "one op reads input → writes output with the smallest initializer and no charged intermediate."

## Ranked new-op backlog (cheapest × families-unlocked × inference-ease)

| # | Op / family | Cost floor | Unlocks | Arg-inference | Status |
|---|---|---:|---|---|---|
| 1 | **RoiAlign / MaxRoiPool** | **5** | crop, rot180, upscale-into-canvas, sub-region place | bbox-fit ROI (4-5 floats) from I/O; reversed ROI = rot180; `Gather(roi_table, bucket)` for data-dependent | TODO — would cheapen ngs's 16 CoordRemap geometric solves (376/3390 → **5**), creating team-ties/wins |
| 2 | **ConvTranspose** block-upsampler | **C·k·k (40)** | cell→k×k block upscale, stamp-from-seed | k = out/in size ratio; W = all-ones (or color-routed) k×k stamp | TODO |
| 3 | **DepthToSpace / SpaceToDepth** | **0** | block-interleave, fractal stamp, channel↔spatial repack | blocksize divides axis (∈{1,2,3,5,6,10,15,30}); mode by one example | TODO |
| 4 | **Boolean Xor/Not/And/Or** (+opset-18 Bitwise*) | 900 B/plane or 120 B/packed lane | parity/XOR CA, XNOR equality, corner/T-junction logic | fit per-cell boolean fn → minimal XAG; parity = Xor-fold. ORT: logical ops BOOL-only | TODO |
| 5 | **Zero-param generators** (Trilu, ConstantOfShape, Equal(iota,iota), OneHot) | ~17-33 B | triangular/diagonal/band masks, const-fill, one-hot out | mask shape family + offset k from I/O. NOT EyeLike (no bool/u8 kernel) | TODO |
| 6 | **Local-topology 2×2 Conv** `[[8,4],[2,1]]` | ~4 | Euler χ, holes, perimeter, convex/concave corners | regress output scalar vs 2×2-code counts; pick 4- vs 8-conn | TODO |
| 7 | **Concat / Tile / Expand** | tiled bytes (bool ¼) | mirror-tile, m×n repeat, edge-extend | repeat = out/in ratio; mirror by flipping alternate copies (Slice steps=-1) | TODO — cheaper mirror-tile than CoordRemap's 3390 |
| 8 | **Attribute-baking** | −2..34 B/op | shaves every Slice/Reduce/Transpose/Pad graph | mechanical: pick opset where the const is an attribute (Slice≤10, Reduce axes≤17) | TODO — free cost shave on existing ops |
| 9 | **Periodicity** Mod(Range,k)+Gather / orbit-Max | ~9 B | periodic tiling/fill, residue-class inpaint | sweep period P, test `in[r,c]==in[r+P,c]`; per-direction, never joint | TODO |
| 10 | **ConvInteger / QLinearConv** | M·(C/g)·kH·kW | quantized color-sum, threshold-in-conv, U8 morphology | weight stencil fit; y_scale buckets. ORT: U8/I8 input only | TODO |
| 11 | **Div / Mod-by-const** base-N pack | ~9 B | K-label pack-and-decode | base N = #distinct labels (keep packed < 2^24). Divisor MUST be constant | TODO |
| 12 | **Resize** nearest upscale | 4 (but charged upscale intermediate) | k× replicate | k = ratio. Loses to #1/#2 in-canvas — use only feeding a comparison | low priority |

## Systematic Einsum / Conv FIT method (the 26 Einsum + 11 Conv misses)

The teammate's engine fits, not hand-crafts. Ordered attack (cheapest/most-general first), all verified by
the per-channel `>0` gate on argen:
1. **Marginal / diagonal enumeration** (`bchw,c->bh`, `bchh->bch`, …): 0-param, a fixed ~10-equation list (as INTERMEDIATE; a single-op solve needs a 4-axis output).
2. **Color map** `nchw,dc->ndhw`: argmax-infer `W[10,10]` from (in_color→out_color) pairs (general incl. merges), or per-row `2^10` binary enumeration. Cost = |W| (100), so bijections stay on ColorRemap's Gather (10).
3. **Row/col permutation** `nchw,hi,wj->ncij`: infer selectors by occupancy-pattern matching.
4. **Separable low-rank CP** `tk,tr,tc,b->bkrc` (#109): fit least rank T by ascending search — CP lower bound = max unfolding rank; use the **argmax⊇per-channel shortcut** (if held-out argmax acc < 100%, rank T can't pass the `>0` gate → skip). Discretize to ternary, verify. Applies only when the per-channel target masks are low-rank (rank-gate up front).
5. **Dense-conv lstsq** (`fit.py`): stack all 3×3 windows → lstsq → ternarize → Conv. General local rule; floor 900 (or SVD-compressed).
6. **Gradient fallback** (Adam + L1) over a chosen skeleton — factors only, does not discover the equation.

## What's built + measured so far
- `einsum_search.py` — zero-param grammar (#1 partial): solves t67 at cost 0. Ceiling ~3.
- `fit.py` — color-map Einsum (#2) + depthwise/cross-channel lstsq Conv (#5). **Verified correct** (solves the
  recolor tasks t16/276/309/337 via a 1×1 Conv at cost 100), but **0/26 on the hard "same_other" cheap tasks** —
  those are bespoke (marginals/low-rank/conditional), not local-Conv or cell-color-map expressible.
- `batched.py` — GPU engine (B70, 22-65M cand/s) + `TernaryKernel` template: **0 reach naive** (Conv `>0`
  bleeds into padding). Needs the boundary mask.

## Honest reach assessment
The two GENERAL fit methods reach the recolor family (already cheaply solved by ColorRemap) but not the hard
same-shape spatial-logic tasks. The two highest-value UNBUILT levers are:
- **#1 RoiAlign / #2 ConvTranspose / #7 Concat-tile** — deterministic shape-inference ops that would recast
  ngs's existing 16 geometric solves from cost 376/3390 down to **5/40**, matching the team (ties/wins), and
  add the crop/upscale/tile families. Highest ratio of reach-per-effort.
- **Separable low-rank CP fit (#4 above)** — the one un-built general fitter with a shot at the rank-limited
  `same_other` tasks (e.g. t024 = rank-4). Needs torch ALS + the argmax rank-floor shortcut.
The rest of the `same_other` bucket is genuinely per-task (bespoke Einsum equations); a general solver won't
crack them — that's the teammate's per-task-probe territory.
