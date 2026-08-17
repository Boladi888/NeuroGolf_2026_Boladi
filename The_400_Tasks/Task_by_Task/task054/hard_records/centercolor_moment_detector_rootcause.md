# task054 — ROOT CAUSE of the 173 truth misses + why 100%-under-bar is (near-)walled

**Recorded:** 2026-07-11 by `bo_ng_cld_worker1` (Claude lane).
**Supersedes** the vague prior note "repairing correctness likely needs detector/marker
logic" (v28 second_pass_floor_attack.md) with a dumped, probe-backed root cause.

## Bars in play (live, re-checked 2026-07-11)
- Authoritative bar: **13,454** (manifest, owner Boladi, sha `5bbf53b7…184e75`,
  `ng_current_bar.py --task 54`). = the v26 no-sp function, byte-identical to
  `prototypes/task054_v26_no_sp.onnx`. It is **99.87% truth** (the cheap moment-detector).
- Best byte-equal cost shave: **v28 @ 13,301** (−152 B, byte-equal to live over 200266).
- New correctness shave: **v29 @ 13,342** (−111 B, +13/20000 fixes, 0 regressions).
- Best known **100%-truth** function: **26,003** (v7-lineage, byte-equal to submission.zip).

## The detector is moment-based on the ONE-HOT (nodes 0-100) — that is why it is cheap
It NEVER materialises a per-cell marker mask. Per-COLOR-CHANNEL it computes:
- `counts` [1,10] and row/col moments `m1..m4 = Einsum(input, iota^k)` [1,10] each.
- Center color = `argmax` over special channels of **row-variance NUMERATOR**
  `row_varnum = counts*m2 - m1^2` (node 23-25).
- Marker positions = **roots of a degree-4 polynomial** built (Newton's identities,
  nodes 80-95) from the center-color power sums with the legend's moment contribution
  subtracted (`marker_axis_pows = center_axis_pows - legend_axis_pows`, node 79), then
  `TopK(|poly|, k=4, smallest)` over 30 positions (nodes 96-100).

## ROOT CAUSE #1 (dominant, CHEAPLY FIXED by v29) — center-color mis-id
`row_varnum = n^2 * variance` keeps an **n^2 count factor**. When BOTH vert&horiz are
active the legend has 8 length-2 arm cells (arm color, spread); the true center color has
only the few clustered marker cells + legend center. The arm color's `n^2*variance` can
exceed the center color's -> `argmax` picks the **arm color as the center** -> the whole
detection runs on the wrong channel and lines/legend break.

- Dumped seed 770044: `center_u8=8` (WRONG; true center=3); missed both V-lines + left the
  legend unerased. All 8/6000 v28 fails were exactly this.
- Discriminator test (`prototypes/diag_discriminator.py`, 40000 seeds):
  - current `argmax(row_varnum)` = 39968/40000 (99.920%), miss-set == the graph's fail-set.
  - **`argmax(row_varnum / count)` = 40000/40000 (100.000%)** (also col/2d-normalised = 100%).
  - `2d_varnum` (unnormalised) = 99.998% -> normalisation is the essential ingredient.
- Fix = one `Div` (+ `Max(counts,1)` guard) before the center argmax. v29 = 13,342 (+41 B),
  cuts misses ~8x, 0 regressions over 20000.

## ROOT CAUSE #2 (residual ~0.015%, NOT cheaply fixable) — poly multiplicity
With the center color correct, ~3/20000 still fail (all 2/3/4-marker BOTH-orientation).
Dumped seed 770881: center now correct (7), both marker ROWS found (9,19), but the COL
poly missed col 18. The col |poly| is near-zero at the true roots 11 (0.009) and 18 (0.036)
**and** at the marker mean ~14-16 (the `x^2` double root of `x^2*(x-a)*(x-b)` for 2 markers).
TopK-4 picks 11 + three mean-adjacent positions, crowding col 18 out. This is an **inherent
multiplicity limit** of a single fixed-degree-4 root-finder, not a numeric/overflow bug
(moments are position-normalised and in-range; verified crm3/crm4 ~ tens, not inf).

## Why 100%-under-bar is (near-)WALLED
- Fixing #2 reliably needs per-cell marker detection = the mask approach
  `markers = special & adj_box` -> ≥2-3k of [30,30] BOOL planes (that is exactly the cost the
  moment approach was built to avoid). v29 has only 112 B of headroom under the bar; adding
  mask planes blows it. Best measured 100%-correct function stays at **26,003** (≈2x the bar).
- TopK k=5/6 or double-root suppression both need the [.,4]-capacity strip machinery widened
  and marker-count knowledge -> more cost + fragile. Not attempted (predicted over-bar).

## Verdict / disposition
- **Cost floor for the current function: ~13,301** (v28, byte-equal) — structure walled
  (see `striprunscatter_7000_route_dead.md`).
- **Correctness/cost frontier: v29 @ 13,342** — strictly more correct than live, still under
  bar, 0 regressions. Ship-decision (v28 vs v29) is a cost-vs-correctness call for the captain.
- **100%-truth under 13,454: effectively WALL** (needs mask planes > headroom). The only
  100%-correct base remains 26,003.

## One remaining hypothesis (if anyone insists on 100% under bar)
Make `legend_same_pows` (node 77, the Gemm that approximates the legend 3x3-surround block
moments) EXACT, and/or deflate the poly's mean/double-root before TopK. If the residual is
purely the multiplicity artifact, an exact deflation of the known double-root at
`p1/N_markers` (N is derivable from the center-color count minus the legend cells) could
recover the crowded-out real root **without** mask planes. Unproven; ~tens of bytes if it
works. This is the single most promising sub-bar path to 100% and is NOT yet explored.
