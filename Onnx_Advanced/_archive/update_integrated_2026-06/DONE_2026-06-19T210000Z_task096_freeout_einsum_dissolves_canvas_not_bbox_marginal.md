# task096 — free-output separable Einsum dissolves canvas+drawing, but the bbox-extent F32 marginal is a SEPARATE locked floor

## What I discovered

A **free-output multi-term separable Einsum (#109/#119) successfully dissolved BOTH the 900 B
Equal-tail canvas AND the ~1450 B per-cell drawing planes** for a nested-square-ring output rule,
but it does NOT touch the F32 row/col bbox-extent marginals — those are a distinct, formulation-locked
floor that the free-output rewrite leaves untouched. Net 6443 -> 6119 (-324), gated byte-equal vs LIVE.

The validated decomposition (worth recording as a transfer pattern): a D4-symmetric output where each
"shell" is an L-band at edge-distance `s` decomposes as, per shell,
`paint_s = E[s] (x) ColBand[s] + ColBandStrict[s] (x) E[s]`, with `E[d,r] = (min(r,S-1-r)==d)` the
edge-distance one-hot basis and `ColBand = B @ E` (`B[s,d]=(s<=d<reach[s])`) computed IN-KERNEL inside
the final Einsum. Using `ColBandStrict[s,d]=(s<d<reach[s])` for one of the two pieces removes the
corner double-count, so you avoid a whole set of `-` correction terms (18 terms -> 13). The channel
selector `K[s] = onehot(color[s]) - onehot(bg)` makes painted cells cancel exactly to the shell color
while a single `BGk·active` term fills the background, so there is NO sentinel/canvas. Full equation:
`output = Einsum('tk,td,te,dr,ec,b->bkrc', KT[T,10], RF[T,N], CF[T,N], E[N,30], E[N,30], ones[1])`.
Gate-5 confirmed the `[T,1,10,30,30]` contraction temp is uncharged (static==profile==6119).

## The dead-end that matters most for the library

**Exact bbox-extent ranking is F32-marginal-locked.** The rule ranks colors by `max(H_extent, W_extent)`
where extent = last-occupied minus first-occupied along an axis. That needs `ArgMax` over a per-(color,
position) occupancy marginal = `ReduceMax(input, axis)` -> `[1,10,30]` F32 = 1200 B each (row + col = 2400).
Every attempt to narrow it failed:
- **F32->F16 cast**: the F32 plane is still produced by ReduceMax and charged; the cast ADDS a plane.
- **Slice input to the active 19 width before reduce**: materializes `[1,10,19,30]` = 22.8 KB (measured).
  Slicing the 4-D input is catastrophic; never slice the input to narrow a marginal.
- **Moment-based extent (#115, `sqrt(12·var)`)**: mismatches the true-bbox RANK ORDER on 325/2000
  ARC-GEN samples (16%). Moments give spread, not first/last-occupied extent; they break exact ranking
  whenever shapes are clipped/asymmetric. #115 is fine for "how big is this blob" but NOT for an
  exact tie-sensitive rank that must be byte-equal.
- **F16 graph input**: not permitted — the NeuroGolf harness (`tools/ng_harness/encode.py` +
  `session.run_raw`) always feeds `x.astype(np.float32)`, so the input is F32 and so is any ReduceMax of it.

## Why it matters

The library's #109/#67/#81 "free-output dissolves the canvas" doctrine is real and powerful, but on
tasks where the SAME graph also needs an exact data-dependent bbox for ranking/ordering, the F32
occupancy marginal is an independent floor the free output cannot remove. Worth a one-line caveat next
to #109 / #63: "free-output Einsum dissolves the OUTPUT canvas; it does NOT dissolve an F32 bbox-extent
marginal needed for exact-rank ordering (that is dtype-locked to [1,C,W] F32 by the F32 input)."

## Where

- Builder: `staged_submits/task096_freeout_einsum_cost6119/build_task096_v6_freeout_bitrun.py`
- Staged: `staged_submits/task096_freeout_einsum_cost6119/` (cost 6119, sha 2128415...c77360)
- Task notes: `The_400_Tasks/Task_by_Task/task096.md` (2026-06-19 attempt)

## Library snapshot check

#109 (techniques.md ~L4026) and #119 (~L4198) already cover the deep free-output Einsum and the
in-kernel-temp/gate-5 doctrine well — this drop is an EXTENSION (the marginal-floor caveat + the
strict-band corner-correction-elimination trick), not a new technique. #63 (selected-channel
marginals) does not apply here because ranking needs ALL colors' extents before selection.
