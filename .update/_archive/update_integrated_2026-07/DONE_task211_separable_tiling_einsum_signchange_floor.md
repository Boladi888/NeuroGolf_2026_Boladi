# Transferable insight — separable-tiling free-output Einsum: the sign-change lower bound on factor count, the decode-region don't-care, and cross-axis factor reuse

**Source**: task211 batch 53 (`bo_ng_cld_worker1`, 2026-07-07). LIVE 96 (Vamshi, sha `c63f5f1c…7060`) improved to **92** (staged; visible 266/266, profile 92, cost-diff −4). Proven floor 92.

**Task**: 3×2 input → 9×4 output, `out[r,s] = in[rowmap[r], colmap[s]]` with rowmap=[2,1,0,0,1,2,2,1,0], colmap=[1,0,0,1] (per-row reversal-concat `[a,b]→[b,a,a,b]` then a row permutation). One-hot `[1,10,30,30]`. Family = Spatial_and_Geometric / Tiling.

**Winning architecture (same micro-floor family as task262)**: a single terminal free-output separable Einsum, `out[k,r,s] = Σ_{h,w} in[k,h,w]·R[h,r]·C[w,s]`, memory=0. One shared coordinate basis **F[30,2]=[coord,1]** (60 params) reused for h,r,w,s; R = product of row bilinears `F[h]·Jι·F[r]`, C = product of col bilinears. **Cost = 60 + 4·(#unique J[2,2])**.

**Three transferable levers:**

1. **Sign-change lower bound on factor count (the floor tool).** A product of K bilinears sharing F[·,2] is degree-K in the output coordinate, so it has ≤K sign changes along that axis. The selection pattern `+` at target / `≤0` at off-positions therefore needs **#factors ≥ max-over-input-lines(sign-changes)**. Here input row h=1 maps to output rows {1,4,7} with off-rows interleaved → pattern `- + - - + - - + -` = **6 sign changes ⇒ ≥6 row factors** (binding). Columns need ≥2 (a single linear-in-s can't be `>0` on interior {1,2} and `≤0` on both ends {0,3}). Holds for exact (=0 off) AND sign-based (<0 off) constructions. This is a clean forced-component argument for any separable positional selector.

2. **Decode-region DON'T-CARE (the lever the prior batch missed).** `encode.py:decode_to_grid` reads ONLY the top-left h×w = expected-shape region; positions beyond it (here r≥9, s≥4) are **never checked**. So selectors need not suppress out-of-region positions. The prior sub-92 search over-constrained columns over s∈0..8 (demanding negativity at the unused s=4..8) and found nothing; relaxing to s∈{0,1,2,3} is correct — though here it still didn't open a reuse (col factors provably disjoint from the 6 row factors, exhaustive over 162 exact factors × 26,540 row-sets). **Always scope selector constraints to the decoded region, not the full 30×30.**

3. **Cross-axis factor reuse saves 4 params/factor.** The SAME bilinear J[2,2] can appear in both the row-product and the column-product (it evaluates at (h,r) and at (w,s) respectively). Sharing one factor across row+col dropped 9→8 unique J ⇒ 96→92. Cheapest concrete win when a full new axis-selector isn't removable.

**Guards (same as task129/task262):** output MUST stay full-canvas `[1,10,30,30]` — a compact `[1,10,9,4]` gives visible 0/266 (grader byte-compares the full encoded tensor; also forces F to first-dim 30 ⇒ 60 params is a hard floor). Gather/Slice selection loses: it materializes a `[1,10,30,30]` charged intermediate (memory≈9000) or crops shape-wrong. **Perfect-25 cost≤1 disproven** — a tiling with a non-identity row permutation is not any whole-grid Identity/Transpose/Flip/Rotate; don't grind the zero-param sweep.

**Applies to**: separable row×col gather/tiling tasks (rowmap/colmap position remaps). Catalog #119 free-output Einsum micro-floors; complements DONE_task262 (rank floor for column→color remap). Rule of thumb: cost = basis(60) + 4·(#unique bilinear factors); factor count is lower-bounded by the max sign-alternation of any input line's selection pattern over the decoded region.
