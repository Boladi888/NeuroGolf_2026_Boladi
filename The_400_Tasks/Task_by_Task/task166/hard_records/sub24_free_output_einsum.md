# task166 — sub-24 free-output Einsum route: WALL (2026-07-07)

**Live bar:** cost **24** (manifest, owner Aditya, updated 2026-06-19T12:21:50Z).
sha256 `5b27fd62c7975ee02404cfb956fcf535d40fba52b6f0b09987d88c744fe4ede8`
(`Team_Best_Onnx/task166.onnx`, 352 B). Confirmed by me: `ng_gate visible 266/266
PASS`; `profile_real_cost --arcgen 266` → **static=24 profile=24 (memory 0 + params 24)**;
fully static ⇒ local==Kaggle exact.

## What the cost-24 team-best actually is (decoded)
A **single free-output Einsum** — the only named tensor is the graph `output`, so
memory=0; the entire charged cost is two initializers:

```
eq   = bihw,bjhx,bkyw,pi,qj,qk,po,pq->bohw          # 3×input, 4×basis, 1×coeff
basis F32 [2,10] = 20 params  [[1,0,-1,0,0,0,0,0,0,0],
                               [0,0,-1,0,0,0,0,0,1,0]]
coeff F32 [2,2]  =  4 params  [[1,-256],[0,1]]
                                    total = 24
```
Decode: with input one-hot in channel (colors 0/8 only), `basis` row0 = bg-indicator,
row1 = shape-indicator. The three `input` factors give (i) per-cell channel projection
`a[p]=B[p,ch]`, (ii) row-sum `f2(q,h)=Σ_x B[q,ch(h,x)]`, (iii) col-sum `f3(q,w)=Σ_y B[q,ch(y,w)]`.
`q=0` = bg-count, `q=1` = shape-count. Output =
`Σ_p B[p,o]·a[p,h,w]·(Σ_q coeff[p,q]·f2(q,h)·f3(q,w))`. The `-256` off-diagonal
(p=0,q=1) subtracts the shape-count product so only cross-product BG cells win the
argmax → color 2; `basis[p,o]` re-embeds to output channels {0,2,8}. This is the
cross-product occupancy fill (our decoded rule) as a degree-3 free-output multilinear.

## Param floor argument (why <24 is a wall)
- **Free-output vehicle is forced.** memory=0 requires one node = graph output, no other
  named intermediate. The rule is degree-3 in input (per-cell mask × row-occupancy ×
  col-occupancy); a degree-2 op (MatMul/Gemm) cannot form the triple product, Conv is
  shift-invariant (no global row/col occupancy). Only Einsum fuses it at memory 0.
- **basis [2,10] = 20 forced.** Channel axis = 10 is the input's channel dim; contracting
  `input` forces a size-10 axis in `basis`. `calculate_params` counts elements dtype-blind
  → the structural zeros still cost. p=2 is minimal: p=1 gives a single output-channel
  vector (can't mix passthrough {0,8} with fill {2}). Shrinking the channel axis needs a
  Slice → a counted `[1,k,30,30]` intermediate ≥ 900·4·k B ≫ any param saving.
- **coeff [2,2] = 4 forced.** Needs both q=0 (bg-count) and q=1 (shape-count) projections
  AND the off-diagonal p≠q coupling (the `-256` correction). A diagonal/rank-lower coupling
  cannot store the cross term.

## Sub-24 builds actually gated (empirical wall)
| variant | eq | params | visible |
|---|---|---|---|
| A_nocoeff_diag | `bihw,bjhx,bkyw,pi,pj,pk,po->bohw` (diagonal, no coeff) | 20 | **0/266 FAIL** |
| B_veccoeff_diag | `…,po,p->bohw` (length-2 diagonal coeff) | 22 | **0/266 FAIL** |

Both drop the off-diagonal p≠q coupling → 0/266, confirming the `-256` cross term is
load-bearing. Builders: `prototypes/build_sub24_variants.py`.

## Cost-1 permutation family — DEAD (prompt's "24→1 perfect-25" premise is a misclassification)
Officials: input colors {0,8}, output colors {0,2,8}. Any zero-param whole-grid
permutation (Identity/Transpose/flip/rotate) preserves the color multiset and can NEVER
introduce color 2. `ng_gate visible` on Identity → **0/266 FAIL**. task166 is a genuine
data-dependent cross-product fill (ARC-GEN 200k truth-verified), not a perfect-25
zero-cost task. Do not re-run the one-node/direct-output sweep.

## Verdict
**WALL at 24.** 24 is the free-output Einsum param floor; the memory-route family (our old
v6 at 1,898) is dominated by ~74×. One remaining (unproven) hypothesis: a 23-param
free-output op that stores the p≠q coupling in <4 elements — no such shape found; treat as
closed unless a new op with sub-4-element 2×2-equivalent coupling appears.
