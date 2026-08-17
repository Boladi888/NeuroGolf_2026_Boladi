# Free-output multi-term Einsum LOSES to the 900-B canvas for a FULL [1,10,30,30] banded output (task295)

## What I discovered
For task295 (left-anchored staircase, output = full `[1,10,30,30]` one-hot, active region 9×18) I built the
prompt-directed free-output multi-term separable Einsum with the #109 negative-channel fold:
`output = Einsum('tk,tr,tc,b->bkrc', K[T,10], R[T,30], C[T,30], ones[1])`, T = 1 grid term + 9 band terms.
It is **functionally correct (exhaustive 441/441 generator support; gate-5 profile==static==2756, so the
contraction temp is genuinely uncharged)** — but it priced **2756, LOSING to the existing 900-B U8 canvas +
Equal-tail (1539)**.

## Why it loses — the generalizing law (extends #109)
The prompt estimated the K/R/C factors as "~700 params, ~700-900". **That conflates params with memory.**
When `length`/`color`/`width` vary per input (the normal case for a from-scratch rule rewrite), K/R/C are
**data-dependent INTERMEDIATES (memory @ 2 B/elem F16), NOT initializers (params @ dtype-free element count).**
Because the Einsum must EMIT the full `[1,10,30,30]` output, its r-axis and c-axis are pinned at 30:
`K[T,10] + R[T,30] + C[T,30]` F16 = `(T·10 + T·30 + T·30)·2`. At T=10 that's 1400 B, plus the `Less` BOOL
`[T,30]`=300 B, plus the scalar chain → **~1700 floor**. You cannot shrink R/C to `[T,9]/[T,18]`: a smaller
Einsum output `[1,10,9,18]` becomes a charged 3240-B intermediate that then needs a Pad (Pad output is free,
but the Einsum result is no longer the output node). Verified ladder: direct-30 5967 → small+Pad 5377 →
fold-30 3905 → min-construction 2756.

This is the SAME conclusion as cost_floor_reference.md "Single-channel color-index canvas = 900 B is the
floor for a [1,10,30,30] output" + its "do NOT try to eliminate the canvas" corollary, but reached from the
free-output-Einsum side: **for a full `[1,10,30,30]` output whose active region ≥ ~9×10, the data-dependent
F16 factors of a full-extent free-output Einsum (~1400-1700 B) lose to the 900-B U8 canvas.** Free-output
Einsum/broadcast wins only when the output (or active crop) is genuinely small (≤ ~9×10) OR the factors can
be baked as initializers (rule is position-only, color/length fixed) — neither holds for a varying-color,
varying-length, full-canvas band.

## Suggested library action
- Extend techniques.md #109 (or its "Output canvas disappears; upstream extent floors do not" sub-note) with
  ONE line: *the free-output factors are MEMORY (data-dependent F16 @ 2 B/elem), not params; at full
  30-output-extent K+R+C ≈ 1400 B and lose to the 900-B canvas for active regions ≥ ~9×10. Price factor-bytes
  = (T·10 + 2·T·D)·2 against the 900-B canvas BEFORE building.*
- cost_floor_reference.md §canvas-floor already states the canvas floor; could add the free-output-Einsum
  factor-byte formula as a third row in its "price both" comparison.

## Where
- task295.md (Attempts 2026-06-21 row + FLOOR PROOF in Cost analysis + Don't-do entry).
- Builders: `The_400_Tasks/Task_by_Task/task295/prototypes/build_task295_v13_freeout_min.py` (+v10/v11/v12 ladder).
- Standing best stays `staged_submits/task295_packscalar_cost1539/` (canvas route, 1539 < team-best 1604).
