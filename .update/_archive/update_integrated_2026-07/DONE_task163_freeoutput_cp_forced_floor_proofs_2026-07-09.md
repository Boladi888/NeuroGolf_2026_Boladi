# Transferable insight — proving a free-output CP Einsum is at its FLOOR (4 tools)

**Source**: task163 batch56 2026-07-09 (`bo_ng_cld_worker3`). LIVE 392 single-Einsum free-output
(`signrank3`, memory=0). Verdict: **WALL at 392** — three from-scratch formulations all hit a
forced floor. `floor_est 156` was STALE (assumed a charged `[1,10,H,W]` prepad long since deleted).
These are the general proof tools that turned "I tried hard" into "it is provably minimal."

Context: block-copy / hollywood-squares family (one Einsum emits `[1,10,30,30]` directly; decode
reads only the active region and thresholds `o>0`). Cost = Σ initializer params. Applies to the
free-output Einsum micro-floor family (catalog #119) and any CP graph whose cost is its factors.

**Tool 1 — Channel-rank floor via colour-circle orthogonality.**
Copy-Einsums encode colour with a circular code (`C[0,c]=cos θc`, `C[1,c]=sin θc`, `+ const offset`)
so `Σ cos/sin` correlation peaks at the matching colour (rank 3 for the whole hue circle). But if
the graph must ALSO emit special colours OFF the circle (black=0, gray=5 have cos=sin=0), the const
those terms need (e.g. `e0−e5`) is **orthogonal** to the circular span → it forces an *extra* rank
component. Prove it: `a,*=lstsq(circ.T, target); err=norm(circ.T@a−target)` — nonzero residual
(here √2) ⇒ that channel rank is mathematically required, not padding. (Contrast task052 where the
learned U was reducible — here it is provably not.)

**Tool 2 — Spatial-rank floor via placement-indexing.**
If the graph places content at a *data-selected* block via a contraction `Factor[x, D]` where `D` is
a decoded coordinate value ∈{0..K−1} (e.g. the destination block index), then `Factor` must be
one-hot-**indexable** by all K values → **rank ≥ K**. This rules out a rank-(K−1) simplex encoding:
a simplex `Factor[x,D]` can't select a single output block per D (it smears). Structural, not just
leakage — the simplex literally has the wrong number of columns to index the placement.

**Tool 3 — Product-placement forces "duplicated" markers.**
When a constant (grid separators/gray lines) is placed by a **product of two** spatial factors
`A[x,·]·B[x,·]` (because the copy needs both a block and a local factor per axis), both factors are
0 at the line positions unless each carries a line-marker column. So the marker appears in BOTH
factors (`A[:,sep]==B[:,sep]`, both ablation-load-bearing). This looks like a wasteful 2× duplication
of a 1-D fact but is FORCED: dedup to a single shared marker needs a `ones[30]` back for the other
slot → net zero. Don't "optimize" it.

**Tool 4 — Free channel-identity does NOT survive folding.**
A channel-preserving copy (source channel index == output channel index) gives a *free* rank-10
identity (no C/S factor) — tempting: 210 spatial only. But it only stays free as a **standalone**
Einsum whose output IS `[1,10,30,30]`. The moment you must add data-independent constant terms
(background/separators) you either (a) use a 2nd node → the copy output becomes a charged
`[1,10,30,30]` intermediate (~18 KB F16) ≫ any saving, or (b) fold into one Einsum → the copy
channel-flow can't coexist with fixed-channel constants under one index, so the channel goes back
through a low-rank factor and the free identity is lost. Verify the math in numpy (it'll pass
267/267 and feel like a win) — then check it can't fold before staging.

**Recipe to close a free-output CP as WALL (or find the win):**
1. `inspect_graph` the LIVE onnx; dump every initializer; decode the Einsum equation index roles.
2. Split the CP by its rank/term index (zero all-but-one row of the term-selector factors) and print
   per-term output planes — you'll see exactly what each term contributes (copy / bg / separators).
3. Ablate each rank component (zero one f-row / term / glue row) → any 0/267 proves it load-bearing.
4. For each factor, apply Tools 1–3: is its rank forced by orthogonality / indexing / product-placement?
5. If a rank looks reducible, build it in numpy FIRST; if it uses free identity, apply Tool 4.
6. Drop a `hard_records/<route>.md` with the forced-component arithmetic so the next batch skips it.

**Applies to**: task068/071/074/093/161/192/222/260 (same Pattern_Recognition/symmetry family) and
any free-output copy/render Einsum. These tools convert `floor_est` optimism into a real proof.
