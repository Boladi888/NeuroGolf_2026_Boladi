# Transferable insight — a per-channel MOMENT detector that ranks classes by "spatial spread" via the UN-normalized variance NUMERATOR (`n²·variance = n·m2 − m1²`) mis-ranks toward HIGH-COUNT classes; normalize by count (`/n`) before the argmax

**Source**: task054 (`bo_ng_cld_worker1`, 2026-07-11). LIVE bar **13,454** (Boladi, `v26`,
sha `5bbf53b7…`, the Einsum poly-moment lineage). This is a **NEW correctness win + root-cause**
that 6+ prior sessions missed by filing the truth gap as "detector-debug, out of scope."
Applies to the whole moment-detector family (task064/task054 and any per-channel
`Einsum(input, iota^k)` classifier that picks a class by spread/variance).

## The trap it kills
Moment detectors identify "the color whose pixels are most spread out" (e.g. the center/marker
color, scattered across boxes, vs a clustered legend/arm color) by:
`row_varnum = counts*m2 - m1^2` then `argmax` over special channels (`m1=Σr, m2=Σr²`, per channel).
That expression is **`n² · variance`** (variance = `(n·m2 − m1²)/n²`), so it keeps a hidden **n²
count factor**. A class with MANY moderately-spread cells can outrank the truly-most-spread class
with FEW cells. On task054 the arm color (8 length-2 legend-arm cells when both vert&horiz active)
outranked the true center color (2 clustered markers + legend center) → the detector locked onto
the WRONG channel → the entire downstream (line-drawing, legend-erase, root-finding) collapsed.
This produced **exactly** the 173/200k truth misses (all "2-marker both-orientation" cases).

## The fix (one Div, essentially free)
`argmax(row_varnum / max(counts,1))` (= `n·variance`; the `max(.,1)` guards 0/0 on absent-color
channels). Empirical on 40,000 fresh ARC-GEN center-color id:
- current `argmax(n²·variance)`: **39968/40000 (99.920%)** — miss-set IDENTICAL to the graph's fail-set.
- **`argmax(n·variance)`: 40000/40000 (100.000%)** (col-normalised and 2d-normalised also 100%; the
  UN-normalised 2d sum still fails once → **normalisation is the essential ingredient, not the axis**).
Graph cost of the fix: +1 `Div`, +1 `Max`, +1 `one_f16` init on `[1,10]` F16 vectors = **+41 B**.
Regression scan vs live over 20,000 seeds: **FIXED 13, REGRESSED 0** → clean §DUAL-TRUTH improvement
(v29 @ 13,342, still 112 B under the 13,454 bar, staged PROBE_READY).

## Why it mattered / how to spot it
The un-normalised numerator is TEMPTING because it avoids a divide (and `n·m2 − m1²` is already the
`Sub` a variance computation lands on). But it is a **count-biased** discriminator, wrong whenever the
"target" class is the sparse one. Symptom: a moment-detector that passes ~99.9% but fails a specific
STRUCTURED subfamily (here: the layouts that maximise the competing class's cell count — both-orientation
legends have the most arm cells). **Before filing a moment-detector's residual truth gap as "needs a
different detector," dump the per-channel discriminator on a failing seed** (`diag_dump_detector.py`
pattern: shape-infer the onnx, append the intermediate as a graph output, run on the ARC-GEN input) and
check whether the CLASS PICK (center/box/bg channel) is already wrong. If the class pick is wrong, the
cheap fix is almost always a normalisation of the ranking statistic, not a new detection architecture.

## Residual / boundary (do not overclaim)
Fixing the class-pick does NOT reach 100%: task054 has a second, INHERENT residual (~0.015%) — the
degree-4 Vieta root-finder's double-root-at-the-marker-mean can crowd a real root out of TopK-4 for
2-marker layouts. That one needs double-root deflation or exact legend-block moments (unproven) or mask
detection (~2-3k > the sub-bar headroom). So: normalise the class discriminator = big cheap win; the
root-finder multiplicity is a separate, harder wall. See
`Task_by_Task/task054/hard_records/centercolor_moment_detector_rootcause.md`.
