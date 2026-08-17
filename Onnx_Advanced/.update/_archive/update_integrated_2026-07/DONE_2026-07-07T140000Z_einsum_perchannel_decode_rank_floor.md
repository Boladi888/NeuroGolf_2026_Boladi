# Free-output Einsum rank floor is set by the `>0` per-channel decode, not by argmax

Encountered on **task024** (batch53, ZeroopsweepCeilingAudit). Live bar cost 72,
sha `e56f9abe…`. Build/evidence: `The_400_Tasks/Task_by_Task/task024/prototypes/`
and `.../hard_records/zeroparam_and_rank3.md`.

## What I discovered

Two reusable insights:

### 1. The harness decode is per-channel `(out > 0.0)`, NOT argmax.
`tools/ng_harness/session.run_raw` returns `(out > 0.0).astype(f32)` and the gate
`array_equal`s that against the one-hot expected grid. So for a `[1,10,30,30]`
colour-channel output, correctness at each in-grid cell requires the TRUE colour
channel `>0` **AND all 9 other channels `≤0`**. This is a much stronger constraint
than argmax. If you fit/optimise a candidate with a softmax/argmax objective it can
score ~100% "argmax-correct" and still gate **0/266** because the non-max channels
have the wrong sign. I burned a full build cycle on an argmax-fit rank-3 Einsum that
was argmax-perfect but visible 0/266. Fit the actual decode: per-channel
BCE-with-logits (target = the full one-hot, threshold 0), or check
`(out>0)==onehot` directly.
(Padding cells auto-satisfy: an off-grid row/col has RowCount/ColCount 0, so a
row/col-contraction Einsum outputs exactly 0 there ⇒ decodes to all-zero.)

### 2. Rank floor of a free-output Einsum is provable via the decode subset relation.
For a factored free-output Einsum `...,sc->nchw` (latent `s`), the credited cost is
`~10k + 2k²` at latent rank `k`. To prove rank `k` is the floor without exhaustive
gating: `{per-channel-`>0`-correct} ⊆ {argmax-correct}` (true ch >0 & all others ≤0
⇒ true ch is the unique max). Hence **per-channel exact-accuracy ≤ argmax
exact-accuracy**. So fit the EASIER argmax objective (fast, convex-ish, converges);
if the best argmax fit at rank k is `< 100%` on a large held-out corpus, rank k
*cannot* pass the real 100%-truth gate — no need to fit the harder per-channel
objective or run `--full`. On task024: rank-2 = hard wall (0% exact), rank-3 argmax
tops at 99.98% held-out (fails ~1/6000) ⇒ dead, rank-4 exact = shipped floor 72.
Structurally the `>0` decode of a row/col cross-fill needs 4 tied rank-1 patterns
(`r1⊗1`, `r3⊗1`, `nonfill⊗c2`, `nonfill⊗noncol`), which independently pins rank at 4.

## Why it matters
Saves the next agent from (a) shipping/grinding an argmax-fit graph that gates 0,
and (b) a full rank-reduction search when a fast argmax fit already proves the floor.

## What's missing from the library
Checked `traps.md §VISIBLE-FAIL/§14e/§39` and INDEX verdict list: the `>0`
per-channel decode contract and the argmax⊇per-channel rank-floor shortcut are not
stated explicitly. #119 (free-output Einsum micro-floors) covers cost mechanics but
not the decode-driven rank floor.

## Sorter note: traps.md — add as new section "Per-channel `>0` decode ≠ argmax; fit the real threshold, and use argmax⊇per-channel as a rank-floor shortcut for free-output Einsums"
## Sorter note: techniques_catalog.md #119 — extend with the argmax-fit rank-floor test (per-channel exact ≤ argmax exact)
