# Transferable insight — single-Einsum classifier: symmetric-CP rank reduction

**Source**: task052 batch 2026-07-07 (`bo_ng_cld_worker3`). LIVE 40 → staged 30 (−25%), truth 200000/200000.

**Pattern**: A "classifier-Einsum" of the form
`out[k,h,w] = Q[h,w]·Σ_t U[k,t]·(Σ_a n[a,h]·U[a,t])²`
(equation `bahx,bdhy,bjhw,at,dt,kt->bkhw`) is a **symmetric-CP rank-R** factorization of a routing tensor `C[a,d,k]=Σ_t U[a,t]U[d,t]U[k,t]`. Its entire cost is the `U[C,R]` initializer = `C·R` params (single node → memory 0). When U was **trained** (a learned float classifier), R is usually padded above what the task needs.

**Lever**: The decode gate only needs `out[target]>0` and `out[other]≤0` (binarized `>0`), NOT exact values. So re-fit `U[C,R]` at the **smallest R** that satisfies those *inequalities* with a margin — often R−1 or R−2 works. Each −1 in R saves `C` params directly.

**How to do it safely (recipe)**:
1. Enumerate the generator's true input palette from the cached 200k corpus (colors actually used; special output colors are often absent from input — here {0,5} never appear in input).
2. Build the **complete** combinatorial set of input feature vectors over that palette (all row histograms). For a k-cell row that's just the multiset partitions.
3. Margin-fit `U[C,R]` (scipy L-BFGS, squared hinge: `relu(m−out[tgt])² + relu(out[other]+m)²`, small L2, multi-restart) enforcing the decode inequalities on the *complete* set → provably correct for ANY generator draw, so truth=100% for free.
4. Gate cheapest-first; confirm with a closed-form run over the 200k corpus (binarize `>0`) which is byte-identical to the truth gate and ~instant vs the throttle-serialized official gate.

**Guard**: rank floor is real. Rank R is infeasible once the required sign pattern across classes can't be realized (task052 R=2: a diagonal quadratic form in P² can't give two output channels opposite signs across the three row-classes → 80% strict decode wall). Record the floor.

**Applies to**: any single-Einsum task whose cost is dominated by a learned `U[C,R]` factor (the free-output Einsum micro-floor family, catalog #119). Check `techniques_catalog.md` Einsum entries for candidates.
