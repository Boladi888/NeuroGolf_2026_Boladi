# CONTRACTION-FINAL class: Einsum AS the output node + selector inits (task304, 2026-06-07)

**Candidate addition to techniques.md #67 (FREE-OUTPUT catalog) — a NEW final-op
family that #68 enumerations MUST include from now on.** Broke task304's
"proven" 810 B floor by 3×: 1,114 → 424 → **369** (3.2% of LIVE, +3.44 raw;
`staged_submits/task304_v5_margin_double_read_cost369/`, full gates byte-equal).

## The class

Every #67 pattern so far (Equal-tail, BitwiseAnd, broadcast-Add, BOOL tails)
is a BROADCAST final — and broadcast finals provably cannot express periodic
(`x%3`-indexed) or arbitrary-2D-mask structure (rectangle-closure / periodicity
disproofs, `_archive/task304_history.md`). An **Einsum final** is a contraction:
with one-hot **positional selector inits** it synthesizes structure broadcast
can't, and since the output node is never charged, the whole expansion is free.
Memory→params conversion at a steep discount.

Core selector kit (all [30,3] F32, 90p each, reusable across slots/axes):
- `Ddiv[a,j] = [a//3==j]` — block coordinate; rows ≥9 all-zero ⇒ ALSO the free
  outside-the-active-grid zeroing (no Pad needed!).
- `Dmod[a,j] = [a%3==j]` — cell coordinate. Factorization
  `[x==3r+h] = [x//3==r]·[x%3==h]` turns a [30,3,3] (270p) or [30,30] (900p)
  join into 90+90, and ONE init serves rows and cols (same tensor, many
  equation slots — initializer params count ONCE).
- "Leak" rows (e.g. `Dmod[h,m]` for h≥3) are safe whenever they multiply input
  rows the ENCODER guarantees zero (outside active grid) — fold the crop into
  the contraction instead of slicing.

## Force multipliers stacked on top (task304 v5, 5-node graph)

1. **Double free-input read**: `input` can appear MULTIPLE times as einsum
   operands (§42/#74 extended; ORT-verified through full gates). A per-COLOR
   vector then acts per-CELL: `Σ_e in[e,g,v]·sgn[e] = sgn[color at (g,v)]`
   (one-hot channels) — a data-dependent 2D mask read with NO mask tensor.
2. **ArgMax-free unique-max selection (#69)**: `sgn = cnt − (max(cnt) − 0.5)`
   ∈ {+0.5 winner, ≤−0.5 rest}, F32-exact (integer counts) — replaces
   ArgMax/Equal/Where/iota with ReduceMax + 2 Sub. Only signs survive the
   `>0` decode; magnitudes are free (verified raw outputs: exact halves).
3. **Channel-mixer matrix for the bg/ch0 term**: H2 = eye(10) with row0 = −1:
   `Σ_d H2[c,d]·in[d] = in[c]` for c≥1 and `−Σin = −1` for ch0 (encoder: ch0≡0
   on colored cells). Rank-forced at [10,10]=100p (params count ELEMENTS —
   eye(10) is 100p despite 90 zeros; sparse inits scorer-rejected).

## THE TRAP — ORT einsum operand order (850× measured)

ORT Einsum reduces operands **strictly sequentially, pairwise, left-to-right**
(greedy label elimination). The SAME 9-operand graph measured **476 ms/pair
(≈26 h per 200k gate) vs 0.56 ms/pair** purely on operand order. Rule: order
small-first so summed labels die immediately; the big input axes (h,w) enter
LAST and contract in the next step; cost/score is order-invariant, runtime is
not. Always per-pair-benchmark (64 pairs) BEFORE launching 200k gates at
operand counts ≥4.

## Where it applies

Any task whose output is a fixed positional rearrangement/expansion of small
data-dependent factors: block-stamps/fractals (304), tilings with periodic
index maps (#55c/#62 family), one-hot expansions, separable products,
upsample-by-integer layouts. The selector kit replaces: Tile, Resize-nearest,
DepthToSpace, Pad-final, Equal-tail — whenever total params (joins + mixers)
undercut the cheapest charged pre-output tensor (here: 281p vs 810 B).

Floor template for the class (task304 numbers): mem = only the nonlinearity
bridge (counts 40 + max 4 + thr 4 + margin-vector 40 = 88 — data-dependent
selection can't be params-only; Gather-from-bank priced 468, loses); params =
mixers (rank-forced) + joins (element-count-forced). Grader-unproven pieces
to confirm on first ship: einsum-as-output, 12-operand einsum (per §39 static
⇒ price is exact 369 if it runs; local ORT == grader ORT class).
