# task211 — bilinear free-output Einsum floor = cost 92 (WALL below 92)

live bar at audit: cost 96, sha256 c63f5f1ce665a4b91771fcc5c4400f791f5a64c68684f9f1760fecd6fb7b7060
(manifest, owner Vamshi, 2026-07-02). WIN staged this batch: cost **92** (-4).

## Rule (confirmed across train+test+arc-gen)
Input = 3x2 active grid at top-left, one color + zeros, rest of 30x30 all-zero
(encode.py: padding cells are all-zero, even channel 0). Output:
`out[r,s] = in[rowmap[r], colmap[s]]`, rowmap=[2,1,0,0,1,2,2,1,0], colmap=[1,0,0,1],
placed at top-left 9x4 of a 30x30 / 10-channel one-hot tensor.

## Scorer semantics that bound the floor (encode.py:decode_to_grid)
- Output tensor must be **[1,10,30,30]** (ab/truth byte-compare the full tensor).
- decode reads ONLY the top-left h×w = **9×4** region; cells r>=9, s>=4 are never
  checked (DON'T CARE). Correctness = exactly one channel `>0` per cell.

## Winning family: terminal free-output separable Einsum (memory=0)
`out[k,r,s] = sum_{h,w} in[k,h,w] · R[h,r] · C[w,s]`, with a single shared
coordinate basis F[30,2]=[coord,1]; R = product of row bilinears F[h]·Jι·F[r],
C = product of col bilinears F[w]·Jι·F[s]. Cost = params(F)=60 + 4·(#unique J).

## Floor proof (cost 92 = 60 + 8·4)
1. **F = 60 forced.** Output tensor shape [1,10,30,30] ⇒ F must index r,s ∈ 0..29
   ⇒ first dim 30; affine position selection needs [coord,1] ⇒ 2 cols ⇒ 60 params.
   Smaller F ⇒ wrong output shape (crop→Pad is the dead 1520 route / shape-wrong,
   see output9x4_direct.md). Generating F by Range/Concat materializes a [30,2]
   intermediate (≥60 memory) — no net gain.
2. **Rows ≥ 6 bilinear factors (rigorous, binding row = input row h=1).**
   Each bilinear is degree-1 in r, so a product of K row bilinears is degree K in r
   and has ≤K sign changes over r=0..8. Required sign pattern of R[h,·]
   (`+` at rowmap[r]==h, `≤0` else):
   - h=0: `- - + + - - - - +`  (3 sign changes)
   - h=1: `- + - - + - - + -`  (**6 sign changes**)  ← forces degree ≥6 ⇒ ≥6 factors
   - h=2: `+ - - - - + + - -`  (3 sign changes)
   Holds for exact (R=0 off) AND sign-based (R<0 off) constructions.
3. **Cols ≥ 2 bilinear factors.** Single bilinear is linear in s; can't be `>0` on
   interior {1,2} and `≤0` on both ends {0,3} (col w=0 needs 2 sign changes; w=1 too).
4. **Col factors cannot reuse row factors.** Exhaustive search over all 162 exact
   row factors × 26,540 valid 6-factor row-sets (search_sub92_relaxed.py, columns
   relaxed to s∈{0,1,2,3} only): NO pair or triple of row factors sign-selects the
   columns. So the ≥2 col factors are disjoint from the 6 row factors.
   ⇒ #unique J ≥ 6 + 2 = 8 ⇒ cost ≥ 60 + 32 = **92**.

## One remaining (low-EV) hypothesis for a future batch
A *sign-based* 6-factor row set (broader than the 162 exact factors) in which 1–2
factors ALSO sign-select the columns when evaluated at (w,s). Would give 88 / 84.
Not found by exact search; the sign-based joint space is large and the prior batch
(search_sub92_reuse.py) also walled it. Max upside −8 bytes vs the proven 92.
Verdict: **WALL at 92** for this batch.

## Dead routes (do not re-explore)
- Perfect-25 cost≤1 (Identity/Transpose/flip/rotate): DISPROVEN — this is a genuine
  tiling (per-row reversal-concat + row permutation), not any whole-grid geometric
  primitive. The prompt's forced-floor ≤1 was a misclassification.
- Gather/Slice selection: materializes a [1,10,30,30] intermediate ⇒ memory≈9000,
  or crop to [1,10,9,4] ⇒ shape-wrong (visible 0/266, output9x4_direct.md).
- fp16 conversion: +36000 (spread scan).
