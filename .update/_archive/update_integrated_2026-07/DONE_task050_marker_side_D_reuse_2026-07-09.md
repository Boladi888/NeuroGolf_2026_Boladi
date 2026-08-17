# Transferable insight — free-output Einsum: reuse the poly-expansion map on BOTH sides

**Source**: task050 batch 2026-07-09 (`bo_ng_cld_worker1`). LIVE 172 → staged 164 (−8, byte-equal to live), truth 16000/16000, ab byte_equal=True vs Team_Best.

**Pattern**: A direct-output single-Einsum builds a degree-2 position response by reading a
`[1,x]` position table `G[W,2]` twice and mapping the pair to `[1,x,x^2]`. There are TWO places
this happens: the **query** side (output position c) and the **marker/source** side (summed input
position d). A prior rung introduced a shared map `D[2,2,3]` ( `[1,x]⊗[1,x] -> [1,x,x^2]` ) to
factor the **query** block, but left the **marker** side folding `x^2` *inline* inside a dense
selector `S[2,2,2,4]=32` (the pre-D pattern: subscript `efFg`, two raw `[1,d]` reads).

**Lever**: The SAME `D` already in the graph expands the marker `[1,d]⊗[1,d]` to `[1,d,d^2]` too.
Route the two marker reads through `D` first (`df,dF,fFN`) and let a smaller selector
`S''[2,3,4]=24` pick features from `(basis e) × (poly N)`. `D` is shared → **zero extra cost**;
selector drops `32 → 24` = **−8 params**, one node, memory stays 0. Because it's an exact algebraic
refactor of the same product, the output is **byte-identical** (max raw diff 0.0; ab byte_equal=True
to the fully-proven live graph → transitively truth 200000/200000).

**Rule of thumb**: whenever a free-output Einsum contains a shared low-rank expansion tensor
(`[1,x]²→[1,x,x²]`, or any `read-twice→D`) used on ONE axis, grep the equation for any OTHER
operand that reconstructs the same power *inline* via a dense `[...,f,F,...]` selector. Re-express
that selector through the shared `D` and it shrinks for free. Cost delta = `(inline_selector_numel −
D-routed_selector_numel)`; here `2·2·2·4 − 2·3·4 = 32 − 24 = 8` per shared axis.

**Floor reached (record it)**: task050 direct-Einsum floor is **164** = `P[10,2]20 + G[30,2]60 +
S''[2,3,4]24 + A[4,4,3]48 + D[2,2,3]12`, every block forced:
- `A[4,4,3]=48` is optimal: mode-ranks (3,4,3); **exact CP-rank = 5** (e1-block needs 3 terms +
  e0 + e2, no non-axis-aligned merge without spurious entries) → CP factor 11·5=55 > 48; 2-way
  factor needs k≥3 (N-mode rank 3) → 19·3=57 > 48; Tucker (3,4,3)-core = 36+12 = 48 (ties). No gain.
- 4 features `{active,count,sum,sqsum}` forced; dropping `active` to 3 features is the recorded-dead
  marker-only route (direct155 probe visible 0/271) — the negative `active` baseline is required so
  background cells preserve under the multiplicative `H·V·Pre` argmax decode.
- `G[30,·]=60` pinned by the 30×30 output canvas (crop+Pad materializes a charged intermediate).

**Applies to**: the free-output Einsum micro-floor family (catalog #119) — any task whose direct
one-node output reads a position/feature table more than once and folds a power inline on a
secondary axis while a shared expansion map already exists elsewhere in the same Einsum.
