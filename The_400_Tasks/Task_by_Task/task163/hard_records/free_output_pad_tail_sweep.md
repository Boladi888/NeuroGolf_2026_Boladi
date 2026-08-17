# task163 — FreeOutputPadTailSweep: WALL at 392 (batch 56, 2026-07-09)

**Live bar:** cost **392** (memory 0 + params 392), profile_cost **392** (static==profile,
single Einsum, no dynamic shapes). Source: manifest (owner Aditya, 2026-06-13).
`live_sha256 = 10fc7b7adca638eb17f7ad03f990a9be34a43af217a27423a3d6435af917df80`
(matches `Team_Best_Onnx/task163.onnx`). Confirmed live via `ng_current_bar.py --task 163`.

## The live graph is ALREADY a clean single-Einsum free-output
`graph = task163_signrank3_signed_baseline`, ONE `Einsum` node, output `[1,10,30,30]`
directly (memory=0). 9 initializers = 392 params total:
`D4[30,4]=120, M4[30,4]=120, S[4,10]=40, C[4,10]=40, B[5,4]=20, T[3,5]=15, P[3,5]=15, A[3,4]=12, Y[10]=10`.

Equation: `bepq,e,pr,pu,qs,qv,bdhw,hr,hl,ws,wm,tf,fc,fd,tg,gu,ga,xa,to,ol,oz,xz,tn,nv,ni,yi,tj,jm,jk,yk->bcxy`

Decoded structure (verified by per-`t` decomposition, 3 CP terms):
- **t=0 COPY** — circular colour code: `C[0,1]=10·cos/sin(hue)`, `C[2]`=offset. Copies the
  8 non-black/non-gray source colours; pushes ch0/ch5 to −3 at coloured dest cells.
- **t=1 BG** — `+2·(ch0−ch5)` at all block cells (background black; suppresses gray there).
- **t=2 CONST** — `−1·(ch0−ch5)` across the whole active 11×11 grid. Separators emerge as
  (whole-grid gray, t=2) − (block-cell gray, t=1); no explicit sep term needed.

The **direction in the prompt is stale**: there is NO charged 10-channel prepad/output tensor
to delete — Aditya already did the free-output collapse. `floor_est 156` was computed against
the old 23,679/744 lineage that still had a `[1,10,H,W]` charged canvas. It does NOT apply here.
`392` is NOT "a formulation artifact above the floor"; it is at the floor (proof below).

## Why 392 is the floor (forced-component argument, each part proven)
Grader facts used: decode reads only the **11×11 active region** and thresholds `o>0`
(positivity, not argmax) — this is the sign-slack the graph exploits. Padding rows 11–29 are
garbage-tolerant (random rows still 267/267) but still stored (dense [30,·] init) — any
Pad/Concat to reconstruct them costs a charged intermediate ≥ what it saves.

- **Channel rank-4 (C,S = 80) is minimal.** The copy needs the rank-3 circular code
  (cos, sin, offset). The constant terms must emit `ch0−ch5 = e0−e5`. Black(0) and gray(5) are
  OFF the colour circle (cos=sin=0), so `e0−e5` is **orthogonal** to the rank-3 circular span —
  lstsq reconstruction error = √2 = 1.4142. ⇒ a 4th channel component (f=3) is mathematically
  required. Ablation: zeroing ANY of the 4 f-rows → 0/267.
- **Spatial rank-4 in BOTH D4,M4 (=240) is minimal.**
  - Block/local rank ≥3: the dest-block placement contracts `BL[x, D]` where `D` = marker-local
    value ∈{0,1,2}; BL must be one-hot-indexable by the 3 values → rank ≥3. A rank-2 simplex is
    **structurally incompatible** (can't index the placement) — confirmed, not just leakage.
  - Sep column in BOTH factors: separators are placed by the **product** `D4[x,·]·M4[x,·]`; both
    factors are 0 at sep rows (3,7) unless they carry a sep entry, so the product is 0 there
    unless sep ∈ both. `D4[:,3] == M4[:,3] == SEP` (identical 30-vectors); ablation: zeroing
    either col3 → 0/267. The apparent "duplication" (60 params for a 30-value fact) is FORCED by
    the 2-factor placement — you cannot dedup to a single SEP[30] without adding a ones[30]
    (=30) back for the other factor slot. Net: no win.
- **Glue B,T,P (=50)** — every row load-bearing (ablation: zeroing any B row → ≤17/267).
- **Y[10]** — yellow selector, forced.
- **Total 392.** No ablatable slack anywhere.

## Three from-scratch formulations tried — all confirm the floor
1. **Free-identity rank-3 copy** (`scratch1.py`, VERIFIED 267/267 in numpy): BL[30,3]+LO[30,3]
   +SEP[30]=210 spatial, channel-preserving copy (c flows from source ⇒ free identity, no C/S),
   sign override `+2·copy − ch0·copysum`. Beautiful & correct — but the free identity only holds
   when the copy is a standalone Einsum whose output is `[1,10,30,30]`. Adding the required
   constant bg/sep terms needs either (a) a 2nd node ⇒ the copy output becomes a charged
   `[1,10,30,30]` intermediate (~18 KB F16) ⇒ dead, or (b) folding into one Einsum ⇒ the channel
   must go through a factor (copy channel-flow can't coexist with fixed-channel constants under
   one index) ⇒ rank-4 C/S returns ⇒ back to 392. **Free identity does not survive folding.**
2. **Rank-3 spatial + explicit SEP/ACTIVE**: the 2-factor separator placement forces sep into
   both output factors ⇒ 240 (identical to current). No win.
3. **Rank-2 simplex block** (`scratch_simplex.py`): structurally incompatible with dest-placement
   indexing (needs rank-3) + mismatch leakage (−0.5) corrupts the copy sum. Dead.

## One remaining (low-confidence) hypothesis for a future batch
A **non-Einsum topology** that avoids the block×local *product* placement (the thing that forces
2×rank-4 spatial). E.g. a data-independent constant bias fused into a free-output tail so the
copy could keep its free channel identity (190 spatial-only) without a charged intermediate.
No concrete cheaper construction found; the charged-intermediate barrier (any `[1,10,30,30]`
non-output tensor ≈ 18–36 KB) has killed every additive-bias variant tried. Treat as speculative.

## Do-not-re-explore
- Do NOT chase `floor_est 156` — stale (assumed a charged prepad already removed).
- Do NOT try to dedup the SEP column (D4/M4 col3): forced by 2-factor placement.
- Do NOT try rank-3 channel: `ch0−ch5` is provably orthogonal to the circular code (err √2).
- Do NOT try rank-2 simplex spatial: dest-placement needs rank-3 indexing.
- Do NOT add bg/sep via a second node: charged `[1,10,30,30]` intermediate ≫ any saving.
